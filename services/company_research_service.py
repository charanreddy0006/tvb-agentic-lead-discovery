"""
Company research and profile extraction service for TVB Agentic Company Lead Discovery.

Fetches webpage content from discovered SearchResult items and uses Groq LLM
for strict, zero-hallucination entity extraction into structured
CompanyProfile models.
"""

import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

from core.models import CompanyProfile, SearchResult
from services.search_service import TavilySearchService


# ---------------------------------------------------------------------------
# Windows console handling
# ---------------------------------------------------------------------------

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(
            encoding="utf-8",
            errors="replace",
        )
        getattr(sys.stderr, "reconfigure")(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("tvb.research_service")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Source quality filters
# ---------------------------------------------------------------------------

THIRD_PARTY_DOMAINS = {
    "techcrunch.com",
    "eu-startups.com",
    "sifted.eu",
    "venturebeat.com",
    "forbes.com",
    "bloomberg.com",
    "crunchbase.com",
    "pitchbook.com",
    "dealroom.co",
    "ycombinator.com",
    "linkedin.com",
    "medium.com",
    "news.ycombinator.com",
    "uktech.news",
    "entrackr.com",
    "yourstory.com",
}

SOURCE_URL_REJECT_TERMS = (
    "/investor",
    "/search/",
    "/label/",
    "/sitemap",
    "/in/",
    "/tag/",
    "/category/",
    "/investors",
    "/directory",
    "/directories",
    "/database",
    "/lists/",
    "/list/",
    "list-of-",
    "top-50",
    "top-100",
    "top-20",
    "funding-database",
    "startup-database",
    "/funds/",
)

SOURCE_TITLE_REJECT_TERMS = (
    "latest news",
    "latest news, reports",
    "social media —",
    "sitemap",
    "top 50",
    "top 100",
    "top 20",
    "investor list",
    "investors",
    "vc funds",
    "venture capital funds",
    "investor guide",
    "funding database",
    "startup database",
    "complete database",
    "directory",
    "list of funded",
    "funded startups",
)

GENERIC_SOURCE_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "bbc.com",
    "thehackernews.com",
}


def _normalize_text(value: str) -> str:
    """Normalize whitespace and case for conservative comparisons."""
    return " ".join((value or "").split()).lower()


def _is_primary_company_source(result: SearchResult) -> bool:
    """
    Reject obvious investor/list/database pages before invoking Groq.

    This does not identify or hardcode any company names.
    """
    url = _normalize_text(result.url)
    title = _normalize_text(result.title)
    snippet = _normalize_text(result.content or "")
    parsed = urlparse(result.url)
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()

    if any(term in url for term in SOURCE_URL_REJECT_TERMS):
        return False

    # Generic social/news profile pages are rarely company-specific sources.
    # Allow credible company-specific articles from news domains only when the
    # title/snippet contains an explicit funding or revenue signal.
    if domain in GENERIC_SOURCE_DOMAINS:
        financial_signal = re.search(
            r"\b(raise|raises|raised|funding|investment|invested|secured|revenue|arr|annual recurring revenue|seed round|series [a-c])\b.*(?:\$|€|£|usd|eur|gbp|million|m\b|billion)",
            snippet,
            re.I,
        )
        if not financial_signal:
            return False

    if any(term in title for term in SOURCE_TITLE_REJECT_TERMS):
        return False

    snippet_reject_terms = (
        "complete database",
        "investor database",
        "venture capital database",
        "list of startups",
        "list of funded startups",
        "top investors",
        "investor directory",
        "browse startups",
    )

    if any(term in snippet for term in snippet_reject_terms):
        return False

    return True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CompanyResearchError(Exception):
    """Base exception for company research service errors."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class CompanyResearchService:
    """Extract structured CompanyProfile entities from search results using Groq."""

    DEFAULT_MODEL = "openai/gpt-oss-20b"

    # Do not silently switch to larger models with different rate limits.
    FALLBACK_MODELS = ["openai/gpt-oss-20b"]

    # Keep the source compact enough for Groq.
    MAX_PAGE_CHARS = 2600
    MAX_SEARCH_CONTEXT_CHARS = 1000

    FETCH_TIMEOUT = 10.0

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: Optional[str] = None,
        fetch_timeout: float = FETCH_TIMEOUT,
        max_workers: int = 1,
    ):
        """Initialize the Company Research Service."""

        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        configured_model = os.getenv("GROQ_MODEL")

        # Blank GROQ_MODEL values must not override the safe default.
        self.model = (
            model
            or (configured_model.strip() if configured_model else "")
            or self.DEFAULT_MODEL
        )

        self.fetch_timeout = fetch_timeout
        self.client: Optional[Groq] = None
        self.search_service = TavilySearchService()

        # Research calls stay sequential in production.
        self.max_workers = max(1, min(max_workers, 5))

        self._http_client: Optional[httpx.Client] = None
        self._http_client_lock = threading.Lock()

        self._page_cache: Dict[str, Optional[str]] = {}
        self._page_cache_lock = threading.Lock()

        self._diagnostics = {
            "research_fetch_failures": 0,
            "research_insufficient_text": 0,
            "research_non_company": 0,
            "research_extraction_failures": 0,
            "duplicate_companies": 0,
            "research_api_failures": 0,
            "research_json_failures": 0,
        }

        self._diagnostics_lock = threading.Lock()
        self._last_research_error: Optional[str] = None

        if not self.groq_api_key or self.groq_api_key.strip() in (
            "",
            "your_groq_api_key_here",
        ):
            logger.warning(
                "GROQ_API_KEY is not configured or contains placeholder text."
            )
        else:
            try:
                self.client = Groq(
                    api_key=self.groq_api_key,
                    timeout=25.0,
                )
            except Exception as exc:
                logger.error(
                    "Failed to initialize Groq client: %s",
                    exc,
                )

    # -----------------------------------------------------------------------
    # Client helpers
    # -----------------------------------------------------------------------

    def _validate_client(self) -> None:
        """Ensure Groq client is ready."""
        if not self.groq_api_key or self.groq_api_key.strip() in (
            "",
            "your_groq_api_key_here",
        ):
            raise CompanyResearchError(
                "GROQ_API_KEY is missing. Please configure a valid Groq API key."
            )

        if self.client is None:
            try:
                self.client = Groq(
                    api_key=self.groq_api_key,
                    timeout=25.0,
                )
            except Exception as exc:
                raise CompanyResearchError(
                    f"Could not connect to Groq client: {exc}"
                ) from exc

    def _increment_diagnostic(self, key: str) -> None:
        with self._diagnostics_lock:
            if key in self._diagnostics:
                self._diagnostics[key] += 1

    def consume_diagnostics(self) -> Dict[str, int]:
        """Return and reset research diagnostics for the latest batch."""
        with self._diagnostics_lock:
            diagnostics = dict(self._diagnostics)

            for key in self._diagnostics:
                self._diagnostics[key] = 0

        return diagnostics

    def consume_last_research_error(self) -> Optional[str]:
        """Return and clear the latest research error."""
        with self._diagnostics_lock:
            error = self._last_research_error
            self._last_research_error = None

        return error

    def _record_research_error(
        self,
        exc: Exception,
        diagnostic: str,
    ) -> None:
        status = getattr(exc, "status_code", None)
        status_text = f" HTTP {status}" if status is not None else ""

        message = " ".join(str(exc).split())[:300]
        if not message:
            message = exc.__class__.__name__

        with self._diagnostics_lock:
            if diagnostic in self._diagnostics:
                self._diagnostics[diagnostic] += 1

            self._last_research_error = (
                f"{exc.__class__.__name__}{status_text}: {message}"
            )

    # -----------------------------------------------------------------------
    # Web fetching
    # -----------------------------------------------------------------------

    def fetch_webpage_text(self, url: str) -> Optional[str]:
        """
        Fetch and clean readable webpage text.

        Returns text truncated to MAX_PAGE_CHARS.
        """
        with self._page_cache_lock:
            if url in self._page_cache:
                return self._page_cache[url]

        logger.info(
            "Fetching webpage content | URL: %s",
            url,
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
        }

        try:
            with self._http_client_lock:
                if self._http_client is None:
                    self._http_client = httpx.Client(
                        timeout=self.fetch_timeout,
                        follow_redirects=True,
                    )

                http_client = self._http_client

            response = http_client.get(
                url,
                headers=headers,
            )

            if response.status_code != 200:
                self._increment_diagnostic(
                    "research_fetch_failures"
                )

                logger.warning(
                    "Page fetch failed (HTTP %d) | URL: %s",
                    response.status_code,
                    url,
                )

                with self._page_cache_lock:
                    self._page_cache[url] = None

                return None

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            for tag in soup(
                [
                    "script",
                    "style",
                    "noscript",
                    "nav",
                    "footer",
                    "header",
                    "aside",
                ]
            ):
                tag.decompose()

            raw_text = soup.get_text(
                separator=" ",
                strip=True,
            )

            clean_text = " ".join(raw_text.split())

            if len(clean_text) < 100:
                logger.warning(
                    "Page returned minimal useful text (<100 chars) | URL: %s",
                    url,
                )

                with self._page_cache_lock:
                    self._page_cache[url] = None

                return None

            truncated_text = clean_text[: self.MAX_PAGE_CHARS]

            logger.info(
                "Page fetch successful | URL: %s | Extracted Length: %d chars",
                url,
                len(truncated_text),
            )

            with self._page_cache_lock:
                self._page_cache[url] = truncated_text

            return truncated_text

        except httpx.TimeoutException:
            self._increment_diagnostic(
                "research_fetch_failures"
            )

            logger.warning(
                "Page fetch timed out after %.1fs | URL: %s",
                self.fetch_timeout,
                url,
            )

            with self._page_cache_lock:
                self._page_cache[url] = None

            return None

        except httpx.RequestError as exc:
            self._increment_diagnostic(
                "research_fetch_failures"
            )

            logger.warning(
                "Page fetch network error: %s | URL: %s",
                exc,
                url,
            )

            with self._page_cache_lock:
                self._page_cache[url] = None

            return None

        except Exception as exc:
            self._increment_diagnostic(
                "research_fetch_failures"
            )

            logger.warning(
                "Unexpected error fetching page: %s | URL: %s",
                exc,
                url,
            )

            with self._page_cache_lock:
                self._page_cache[url] = None

            return None

    # -----------------------------------------------------------------------
    # Evidence helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_source_text(value: str) -> str:
        return " ".join((value or "").split())

    @classmethod
    def _is_source_backed_evidence(
        cls,
        evidence: str,
        source_text: str,
    ) -> bool:
        return (
            cls._normalize_source_text(evidence)
            in cls._normalize_source_text(source_text)
        )

    @staticmethod
    def _location_from_evidence(text: str) -> Optional[tuple[str, str]]:
        """Extract explicit headquarters/location wording without geographic inference."""
        normalized = " ".join((text or "").split())
        patterns = (
            r"\b(?:headquartered|headquarters|based|located)\s+(?:in|at)\s+([^.;|]{2,100})",
            r"\b(?:based|headquartered|located)\s+out\s+of\s+([^.;|]{2,100})",
            r"\b([^,.;|]{2,60})[- ](?:based|headquartered)\b",
            r"\b(?:hq|h\.q\.)\s+(?:in|at)\s+([^.;|]{2,100})",
            r"\b(?:headquartered|based|located)\s+(?:in|at)\s+([^.;|]{2,100})\s*,\s*(United Kingdom|UK|Germany|Sweden|Norway|Denmark|Finland|Netherlands|France|Spain|Italy|Ireland|India|Singapore|Indonesia|Vietnam|Australia|New Zealand|Canada|Japan|South Korea|Brazil|Mexico|Colombia|UAE|Saudi Arabia|Egypt|South Africa)",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip()
            candidate = re.split(
                r"\s+(?:and|with|where|while|but|operating|serving|founded|established)\s+",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" ,:-")
            if candidate and not re.match(r"^(?:the )?(?:united states|usa|u\.s\.|us)$", candidate, re.I):
                evidence = match.group(0).strip()
                return candidate, evidence
        return None

    @staticmethod
    def _currency_from_evidence(
        evidence: Optional[str],
    ) -> Optional[str]:
        """Determine currency only when explicitly supported."""
        if not evidence:
            return None

        if re.search(
            r"€|\bEUR\b|\beuros?\b",
            evidence,
            flags=re.IGNORECASE,
        ):
            return "EUR"

        if re.search(
            r"£|\bGBP\b|\bpounds?\b",
            evidence,
            flags=re.IGNORECASE,
        ):
            return "GBP"

        if re.search(
            r"\$|\bUSD\b|\bUS\s*dollars?\b|\bdollars?\b",
            evidence,
            flags=re.IGNORECASE,
        ):
            return "USD"

        return None

    @classmethod
    def _is_explicit_website(
        cls,
        website: str,
        source_text: str,
    ) -> bool:
        candidate = (
            website
            if website.startswith(("http://", "https://"))
            else f"https://{website}"
        )

        hostname = urlparse(
            candidate
        ).netloc.split(":")[0].lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return bool(
            hostname
            and hostname in source_text.lower()
        )

    @staticmethod
    def _parse_json_object(
        raw_content: object,
    ) -> Dict[str, object]:
        """
        Parse JSON from plain, fenced, or prose-wrapped model output.
        """
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ValueError(
                "Model returned empty or non-text extraction output."
            )

        candidate = raw_content.strip()

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            candidate,
            re.IGNORECASE | re.DOTALL,
        )

        if fenced:
            candidate = fenced.group(1)

        decoder = json.JSONDecoder()

        for start, character in enumerate(candidate):
            if character != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(
                    candidate[start:]
                )
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return parsed

        raise ValueError(
            "Model output did not contain a valid JSON object."
        )

    @staticmethod
    def _sentences(text: str) -> List[str]:
        """Return compact sentence-like chunks for deterministic evidence recovery."""
        normalized = " ".join((text or "").split())
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]

    @classmethod
    def _recover_financial_evidence(cls, text: str) -> Optional[tuple[str, float, str, str]]:
        """Recover an explicit funding/revenue claim and amount from source text."""
        money = r"(?P<prefix>US\$|\$|USD|EUR|GBP|€|£)?\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>million|billion|m|bn|b|thousand|k)?\s*(?P<suffix>USD|EUR|GBP)?"
        for sentence in cls._sentences(text):
            if not re.search(r"\b(raised|raises|has raised|funding|funding round|secured|secures|investment|revenue|arr|annual recurring revenue)\b", sentence, re.I):
                continue
            # A sentence may begin with a year (for example, "In 2026, ...").
            # Search all money matches and keep only a match with explicit
            # currency so the year cannot be mistaken for the financial amount.
            for match in re.finditer(money, sentence, re.I):
                raw_amount = float(match.group("amount"))
                unit = (match.group("unit") or "").lower()
                multiplier = {"m": 1_000_000, "million": 1_000_000, "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000, "k": 1_000, "thousand": 1_000}.get(unit, 1)
                currency_tokens = {(match.group("prefix") or "").strip().upper(), (match.group("suffix") or "").strip().upper()}
                currency = "USD" if currency_tokens & {"US$", "$", "USD"} else "EUR" if currency_tokens & {"EUR", "€"} else "GBP" if currency_tokens & {"GBP", "£"} else ""
                if not currency:
                    continue
                kind = "revenue" if re.search(r"\b(revenue|arr|annual recurring revenue)\b", sentence, re.I) else "funding"
                return kind, raw_amount * multiplier, currency, sentence
        return None

    @classmethod
    def _recover_technology_evidence(cls, text: str) -> Optional[str]:
        """Recover an explicit technology-product statement from source text."""
        patterns = (
            r"\b(?:saas|software|platform|api|artificial intelligence|ai|cybersecurity|fintech|healthtech|digital health|developer tools|cloud|operating system)\b",
        )
        for sentence in cls._sentences(text):
            if any(re.search(pattern, sentence, re.I) for pattern in patterns):
                if re.search(r"\b(platform|software|saas|api|operating system|product|tool|technology|ai|artificial intelligence|cybersecurity|healthtech|digital health|fintech)\b", sentence, re.I):
                    return sentence
        return None

    @classmethod
    def _recover_company_name(cls, text: str) -> Optional[str]:
        """Recover a company name from explicit funding/news headline grammar."""
        for sentence in cls._sentences(text):
            if not re.search(r"\b(?:raised|raises|has raised|secured|secures|lands|closes|announces)\b", sentence, re.I):
                continue
            patterns = (
                r"\b(?:startup|company|firm|platform|business)\s+(?P<name>[A-Z][A-Za-z0-9.&'/-]*(?:\s+[A-Z][A-Za-z0-9.&'/-]*){0,5}?)\s+(?:has\s+)?(?i:raised|raises|secured|secures|lands|closes|announces)\b",
                r"^(?P<name>[A-Z][A-Za-z0-9.&'/-]*(?:\s+[A-Z][A-Za-z0-9.&'/-]*){0,5}?)\s+(?:has\s+)?(?i:raised|raises|secured|secures|lands|closes|announces)\b",
                r"(?P<name>[A-Z][A-Za-z0-9.&'/-]*(?:\s+[A-Z][A-Za-z0-9.&'/-]*){0,5}?)\s+(?:has\s+)?(?i:raised|raises|secured|secures|lands|closes|announces)\b",
            )
            for pattern in patterns:
                match = re.search(pattern, sentence)
                if match:
                    name = re.sub(r"\s+", " ", match.group("name")).strip(" ,:-")
                    if 2 <= len(name) <= 100 and len(name.split()) <= 6:
                        return name
        return None

    @classmethod
    def _deterministic_profile_from_text(cls, text: str, source_url: str, fallback_domain: str) -> Optional[CompanyProfile]:
        """Create a profile from explicit source evidence without consuming Groq tokens."""
        financial = cls._recover_financial_evidence(text)
        if not financial:
            return None
        location = cls._location_from_evidence(text)
        technology = cls._recover_technology_evidence(text)
        company_name = cls._recover_company_name(text)
        if not company_name:
            return None
        kind, amount, currency, financial_evidence = financial
        evidence = {
            "funding" if kind == "funding" else "revenue": financial_evidence,
        }
        if technology:
            evidence["technology"] = technology
        if location:
            evidence["headquarters"] = location[1]
            evidence["location"] = location[1]
        return CompanyProfile(
            company_name=company_name,
            website=None,
            description=technology,
            industry=None,
            location=location[0] if location else None,
            funding_amount=amount if kind == "funding" else None,
            funding_currency=currency if kind == "funding" else None,
            revenue_amount=amount if kind == "revenue" else None,
            revenue_currency=currency if kind == "revenue" else None,
            source_urls=[source_url],
            evidence=evidence,
        )

    # -----------------------------------------------------------------------
    # LLM extraction
    # -----------------------------------------------------------------------

    def _extract_from_text(
        self,
        text: str,
        source_url: str,
        fallback_domain: str,
    ) -> Optional[CompanyProfile]:
        """
        Extract a structured CompanyProfile from source text.

        Missing or unsupported facts remain null.
        """
        self._validate_client()
        assert self.client is not None

        parsed = urlparse(source_url)
        source_domain = parsed.netloc.split(":")[0].lower()

        system_prompt = (
            "You are an expert corporate intelligence analyst for "
            "The Venture Build (TVB).\n"
            "Analyze the supplied text and extract factual information "
            "about the PRIMARY TECHNOLOGY COMPANY described.\n\n"

            "ZERO-HALLUCINATION RULES:\n"
            "1. Only extract facts explicitly stated in the supplied text.\n"
            "2. Missing facts MUST be null.\n"
            "3. Never guess, infer, convert, or fabricate.\n"
            "4. Distinguish FUNDING from REVENUE.\n"
            "5. Funding/revenue currency must be explicitly supported.\n"
            "6. Website must be explicitly stated in the text.\n"
            "7. Evidence excerpts must be VERBATIM and present in the text.\n"
            "8. If the text is a list, directory, database, investor page, "
            "or generic report rather than a primary commercial company profile, "
            "set is_company=false.\n\n"

            f"Current source domain: {source_domain}\n\n"

            "Return valid JSON ONLY using exactly this structure:\n"
            "{\n"
            '  "is_company": true,\n'
            '  "company_name": "string or null",\n'
            '  "website": "string or null",\n'
            '  "description": "string or null",\n'
            '  "industry": "string or null",\n'
            '  "location": "City, Country or null",\n'
            '  "funding_amount": number or null,\n'
            '  "funding_currency": "USD" or "EUR" or "GBP" or null,\n'
            '  "funding_stage": "Seed or null",\n'
            '  "revenue_amount": number or null,\n'
            '  "revenue_currency": "USD" or "EUR" or "GBP" or null,\n'
            '  "evidence": {\n'
            '    "funding": "verbatim quote or null",\n'
            '    "revenue": "verbatim quote or null",\n'
            '    "technology": "verbatim quote or null",\n'
            '    "headquarters": "verbatim quote or null",\n'
            '    "location": "verbatim quote or null",\n'
            '    "us_operations": "verbatim quote or null"\n'
            "  }\n"
            "}"
        )

        user_prompt = (
            f"Source URL: {source_url}\n\n"
            f"Webpage Text Content:\n{text}"
        )

        try:
            # IMPORTANT:
            # Do not use response_format={"type": "json_object"} here.
            # Some gpt-oss-20b requests are rejected with HTTP 400
            # json_validate_failed before the model produces output.
            #
            # We already have a defensive JSON parser below.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
                include_reasoning=False,
                reasoning_effort="low",
                max_completion_tokens=1000,
            )

        except Exception as exc:
            self._record_research_error(
                exc,
                "research_api_failures",
            )

            logger.warning(
                "Research model API failed | URL: %s | %s",
                source_url,
                self._last_research_error,
            )

            return None

        try:
            raw = response.choices[0].message.content

            if not raw:
                raise ValueError(
                    "Model returned empty extraction output."
                )

            data = self._parse_json_object(raw)

        except (ValueError, json.JSONDecodeError) as exc:
            self._record_research_error(
                exc,
                "research_json_failures",
            )

            logger.warning(
                "Research JSON parsing failed | URL: %s | %s",
                source_url,
                str(exc)[:300],
            )

            return None

        try:
            if not data.get("is_company", True):
                self._increment_diagnostic(
                    "research_non_company"
                )

                logger.info(
                    "Page evaluated as not describing a primary company | URL: %s",
                    source_url,
                )

                return None

            company_name = data.get("company_name")

            if not company_name or str(company_name).strip().lower() in (
                "null",
                "none",
                "",
            ):
                self._increment_diagnostic(
                    "research_extraction_failures"
                )

                logger.info(
                    "No clear company name could be extracted | URL: %s",
                    source_url,
                )

                return None

            company_name = str(company_name).strip()

            # A lead for this assignment must have an explicit funding or
            # revenue signal. Reject generic articles/profiles early so they
            # cannot become false candidates such as Meta or unrelated hacks.
            if not re.search(
                r"\b(raised|raises|has raised|funding|funding round|secured|secures|investment|revenue|arr|annual recurring revenue)\b",
                text,
                re.I,
            ):
                self._increment_diagnostic("research_extraction_failures")
                logger.info(
                    "Rejected company profile without explicit financial signal | URL: %s",
                    source_url,
                )
                return None

            # ---------------------------------------------------------------
            # Website
            # ---------------------------------------------------------------

            website = data.get("website")

            if website and str(website).strip().lower() not in (
                "null",
                "none",
                "",
            ):
                website = str(website).strip()

                if not website.startswith(
                    ("http://", "https://")
                ):
                    website = f"https://{website}"

                if not self._is_explicit_website(
                    website,
                    text,
                ):
                    website = None
            else:
                website = None

            # ---------------------------------------------------------------
            # Numeric values
            # ---------------------------------------------------------------

            funding_amt = data.get(
                "funding_amount"
            )

            if funding_amt is not None:
                try:
                    funding_amt = float(
                        funding_amt
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    funding_amt = None

            revenue_amt = data.get(
                "revenue_amount"
            )

            if revenue_amt is not None:
                try:
                    revenue_amt = float(
                        revenue_amt
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    revenue_amt = None

            # ---------------------------------------------------------------
            # Evidence
            # ---------------------------------------------------------------

            raw_evidence = data.get(
                "evidence",
                {},
            ) or {}

            if not isinstance(
                raw_evidence,
                dict,
            ):
                raw_evidence = {}

            clean_evidence = {
                key: str(value).strip()
                for key, value in raw_evidence.items()
                if value
                and str(value).strip().lower()
                not in (
                    "null",
                    "none",
                    "",
                )
                and self._is_source_backed_evidence(
                    str(value).strip(),
                    text,
                )
            }

            # If the model misses a clearly stated HQ phrase, recover it
            # deterministically from the same source text. This never guesses
            # a country or converts an unsupported location.
            explicit_location = self._location_from_evidence(text)
            if explicit_location:
                detected_location, detected_location_evidence = explicit_location
                if not data.get("location"):
                    data["location"] = detected_location
                if "headquarters" not in clean_evidence:
                    clean_evidence["headquarters"] = detected_location_evidence
                if "location" not in clean_evidence:
                    clean_evidence["location"] = detected_location_evidence

            # Deterministically recover critical evidence the model may omit.
            # This prevents valid funding/technology/location facts from being
            # lost merely because the LLM returned a partial JSON object.
            financial_recovery = self._recover_financial_evidence(text)
            if financial_recovery:
                kind, amount, currency, evidence_text = financial_recovery
                # Source-backed deterministic parsing is authoritative for the
                # numeric fact. This prevents an LLM typo such as "$10" from
                # replacing an explicit "$10 million" statement in the source.
                if kind == "funding":
                    funding_amt = amount
                    funding_currency = currency
                    clean_evidence["funding"] = evidence_text
                elif kind == "revenue":
                    revenue_amt = amount
                    revenue_currency = currency
                    clean_evidence["revenue"] = evidence_text

            technology_recovery = self._recover_technology_evidence(text)
            if technology_recovery and "technology" not in clean_evidence:
                clean_evidence["technology"] = technology_recovery

            funding_currency = self._currency_from_evidence(
                clean_evidence.get("funding")
            )

            revenue_currency = self._currency_from_evidence(
                clean_evidence.get("revenue")
            )

            profile = CompanyProfile(
                company_name=company_name,
                website=website,
                description=data.get("description"),
                industry=data.get("industry"),
                location=data.get("location"),
                funding_amount=funding_amt,
                funding_currency=funding_currency,
                funding_stage=data.get("funding_stage"),
                revenue_amount=revenue_amt,
                revenue_currency=revenue_currency,
                source_urls=[source_url],
                evidence=clean_evidence,
            )

            profile = self._enrich_profile_from_search(profile)

            logger.info(
                "Extracted CompanyProfile | Name: '%s' | "
                "Location: %s | Funding: %s %s",
                profile.company_name,
                profile.location,
                profile.funding_amount,
                profile.funding_currency,
            )

            return profile

        except Exception as exc:
            self._increment_diagnostic(
                "research_extraction_failures"
            )

            logger.error(
                "LLM profile extraction failed for URL '%s': %s",
                source_url,
                exc,
            )

            return None

    def _enrich_profile_from_search(self, profile: CompanyProfile) -> CompanyProfile:
        """Fill missing qualification evidence with targeted Tavily searches.

        Enrichment is deterministic: only facts literally present in returned
        search-result text are accepted. No currency conversion or geographic
        inference is performed.
        """
        queries: List[str] = []
        name = profile.company_name.strip()
        if not profile.website:
            queries.append(f'"{name}" official website')
        if not profile.location or not profile.evidence.get("headquarters"):
            queries.append(f'"{name}" (headquarters OR headquartered OR "based in" OR "located in")')
        if profile.funding_amount is None and profile.revenue_amount is None:
            queries.append(f'"{name}" (funding OR raised OR "funding round" OR revenue OR ARR)')
        if not profile.evidence.get("technology"):
            queries.append(f'"{name}" (platform OR software OR SaaS OR technology OR API)')
        if not queries:
            return profile

        for query in queries[:3]:
            try:
                results = self.search_service.search(query, max_results=3)
            except Exception as exc:
                logger.warning("Profile enrichment search failed | Company: %s | Error: %s", name, exc)
                continue

            for result in results:
                fetched = self.fetch_webpage_text(result.url) or ""
                source = " ".join(part for part in (result.title, result.content, fetched) if part)
                if not source or name.lower() not in source.lower():
                    continue

                # Resolve the official website from an explicit "official website"
                # search result. Never treat social, directory, investor, or
                # contact-data domains as the company website.
                if not profile.website and re.search(r"\bofficial\s+website\b", query, re.I):
                    result_domain = urlparse(result.url).netloc.lower().removeprefix("www.")
                    blocked_domains = THIRD_PARTY_DOMAINS | GENERIC_SOURCE_DOMAINS | {
                        "rocketreach.co", "signalhire.com", "apollo.io", "zoominfo.com",
                        "lead411.com", "theorg.com", "wellfound.com", "crunchbase.com",
                        "startupfundraising.com",
                    }
                    if result_domain and result_domain not in blocked_domains and name.lower() in source.lower():
                        profile.website = result.url

                location = self._location_from_evidence(source)
                if location and not profile.location:
                    profile.location, evidence = location
                    profile.evidence.setdefault("headquarters", evidence)
                    profile.evidence.setdefault("location", evidence)

                financial = self._recover_financial_evidence(source)
                if financial and profile.funding_amount is None and profile.revenue_amount is None:
                    kind, amount, currency, evidence = financial
                    if kind == "funding":
                        profile.funding_amount = amount
                        profile.funding_currency = currency
                        profile.evidence["funding"] = evidence
                    else:
                        profile.revenue_amount = amount
                        profile.revenue_currency = currency
                        profile.evidence["revenue"] = evidence

                technology = self._recover_technology_evidence(source)
                if technology:
                    profile.evidence.setdefault("technology", technology)

                if result.url and result.url not in profile.source_urls:
                    profile.source_urls.append(result.url)

            if (profile.location and profile.evidence.get("technology") and
                    (profile.funding_amount is not None or profile.revenue_amount is not None)):
                break

        return profile

    # -----------------------------------------------------------------------
    # Single-result research
    # -----------------------------------------------------------------------

    def research_search_result(
        self,
        result: SearchResult,
    ) -> Optional[CompanyProfile]:
        """
        Process one SearchResult.

        Obvious investor/list/database sources are rejected before
        fetching or calling Groq.
        """
        logger.info(
            "Researching SearchResult | URL: %s | Title: '%s'",
            result.url,
            result.title,
        )

        if not _is_primary_company_source(result):
            self._increment_diagnostic(
                "research_non_company"
            )

            logger.info(
                "Skipping obvious non-company source | URL: %s | Title: %s",
                result.url,
                result.title,
            )

            return None

        page_text = self.fetch_webpage_text(result.url)

        # Always preserve the Tavily title/snippet alongside the fetched page.
        # Important company facts such as headquarters or funding can appear
        # outside the first section of a page. The search result is still
        # source-backed evidence, so it is safe to provide it to extraction.
        snippet = (result.content or "").strip()[: self.MAX_SEARCH_CONTEXT_CHARS]
        search_context = (
            f"Search result title: {result.title}\n"
            f"Search result snippet: {snippet}"
        )

        if not page_text or len(page_text) < 150:
            if len(snippet) >= 80:
                logger.info(
                    "Using Tavily title/snippet as fallback content for: %s",
                    result.url,
                )
                page_text = search_context
            else:
                self._increment_diagnostic(
                    "research_insufficient_text"
                )
                logger.warning(
                    "Insufficient text content for analysis | URL: %s",
                    result.url,
                )
                return None
        else:
            page_text = f"{search_context}\n\nFetched webpage content:\n{page_text}"

        # Deterministic extraction is attempted first. This is essential when
        # Groq reaches its daily token limit: explicit funding/location/technology
        # facts in Tavily results remain usable without another model call.
        deterministic = self._deterministic_profile_from_text(
            page_text, result.url, result.domain
        )
        if deterministic:
            return self._enrich_profile_from_search(deterministic)

        profile = self._extract_from_text(
            text=page_text,
            source_url=result.url,
            fallback_domain=result.domain,
        )
        if profile:
            return profile

        # Groq may have failed because of a transient/API rate-limit error.
        # Retry the no-LLM path before dropping the candidate.
        deterministic = self._deterministic_profile_from_text(
            page_text, result.url, result.domain
        )
        if deterministic:
            return self._enrich_profile_from_search(deterministic)
        return None

    # -----------------------------------------------------------------------
    # Batch research
    # -----------------------------------------------------------------------

    def research_candidates(
        self,
        search_results: List[SearchResult],
    ) -> List[CompanyProfile]:
        """
        Research multiple candidates with deduplication and evidence merging.
        """
        dedup_store: Dict[str, CompanyProfile] = {}

        logger.info(
            "Beginning batch research on %d search results",
            len(search_results),
        )

        def research_one(
            item: SearchResult,
        ) -> Optional[CompanyProfile]:
            try:
                return self.research_search_result(
                    item
                )
            except Exception as exc:
                logger.warning(
                    "Candidate research failed | URL: %s | Error: %s",
                    item.url,
                    exc,
                )
                return None

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            profiles = list(
                executor.map(
                    research_one,
                    search_results,
                )
            )

        for idx, profile in enumerate(
            profiles,
            1,
        ):
            logger.info(
                "--- Processing Candidate [%d/%d] ---",
                idx,
                len(search_results),
            )

            if not profile:
                continue

            dedup_key = profile.get_dedup_key()

            if dedup_key in dedup_store:
                self._increment_diagnostic(
                    "duplicate_companies"
                )

                existing = dedup_store[
                    dedup_key
                ]

                logger.info(
                    "Duplicate company detected ('%s' -> key: %s). "
                    "Merging evidence and sources.",
                    profile.company_name,
                    dedup_key,
                )

                # Merge source URLs
                for url in profile.source_urls:
                    if url not in existing.source_urls:
                        existing.source_urls.append(
                            url
                        )

                # Enrich missing fields
                if (
                    not existing.website
                    and profile.website
                ):
                    existing.website = profile.website

                if (
                    not existing.description
                    and profile.description
                ):
                    existing.description = (
                        profile.description
                    )

                if (
                    not existing.industry
                    and profile.industry
                ):
                    existing.industry = (
                        profile.industry
                    )

                if (
                    not existing.location
                    and profile.location
                ):
                    existing.location = (
                        profile.location
                    )

                if (
                    existing.funding_amount is None
                    and profile.funding_amount is not None
                ):
                    existing.funding_amount = (
                        profile.funding_amount
                    )

                    existing.funding_currency = (
                        profile.funding_currency
                    )

                    existing.funding_stage = (
                        profile.funding_stage
                    )

                if (
                    existing.revenue_amount is None
                    and profile.revenue_amount is not None
                ):
                    existing.revenue_amount = (
                        profile.revenue_amount
                    )

                    existing.revenue_currency = (
                        profile.revenue_currency
                    )

                # Merge evidence
                for key, value in profile.evidence.items():
                    if key not in existing.evidence:
                        existing.evidence[key] = value

            else:
                dedup_store[
                    dedup_key
                ] = profile

        final_profiles = list(
            dedup_store.values()
        )

        logger.info(
            "Batch research complete | Input Results: %d | "
            "Unique Companies Discovered: %d",
            len(search_results),
            len(final_profiles),
        )

        return final_profiles


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def run_research_demo(
    query: Optional[str] = None,
    max_results: int = 3,
) -> List[CompanyProfile]:
    """Run a small manual research pipeline demonstration."""

    demo_query = (
        query
        or "European B2B SaaS startup raised $2M seed funding"
    )

    print("=" * 75)
    print("TVB Company Research Pipeline Demo")
    print(f"Discovery Query: '{demo_query}'")
    print("=" * 75)

    try:
        from services.search_service import TavilySearchService

        print("\n[1] Executing Tavily Search...")

        search_service = TavilySearchService()

        results = search_service.search(
            demo_query,
            max_results=max_results,
        )

        print(
            f"    Retrieved {len(results)} search results."
        )

    except Exception as exc:
        print(
            f"    [ERROR] Search failed: {exc}"
        )
        return []

    print(
        "\n[2] Researching and extracting company profiles..."
    )

    research_service = CompanyResearchService()

    try:
        profiles = research_service.research_candidates(
            results
        )

        print(
            f"\n[3] Research complete. Extracted "
            f"{len(profiles)} unique company profile(s):"
        )

        for idx, profile in enumerate(
            profiles,
            1,
        ):
            print("\n" + "-" * 60)

            print(
                f"Company [{idx}]: {profile.company_name}"
            )

            print(
                f"  Website:     "
                f"{profile.website or '[Not Provided]'}"
            )

            print(
                f"  Industry:    "
                f"{profile.industry or '[Not Provided]'}"
            )

            print(
                f"  Location:    "
                f"{profile.location or '[Not Provided]'}"
            )

            print(
                f"  Description: "
                f"{profile.description or '[Not Provided]'}"
            )

            if profile.funding_amount is not None:
                funding_text = (
                    f"{profile.funding_amount:,.0f} "
                    f"{profile.funding_currency or '[Currency Not Disclosed]'} "
                    f"({profile.funding_stage or 'N/A'})"
                )
            else:
                funding_text = "[Not Disclosed / Null]"

            print(
                f"  Funding:     {funding_text}"
            )

            if profile.revenue_amount is not None:
                revenue_text = (
                    f"{profile.revenue_amount:,.0f} "
                    f"{profile.revenue_currency or '[Currency Not Disclosed]'}"
                )
            else:
                revenue_text = "[Not Disclosed / Null]"

            print(
                f"  Revenue:     {revenue_text}"
            )

            print(
                f"  Sources ({len(profile.source_urls)}):"
            )

            for url in profile.source_urls:
                print(
                    f"    - {url}"
                )

            if profile.evidence:
                print(
                    "  Evidence Quotes:"
                )

                for key, value in profile.evidence.items():
                    print(
                        f'    * {key.upper()}: "{value}"'
                    )

        print("\n" + "=" * 75)
        print(
            "[SUCCESS] COMPANY RESEARCH DEMO FINISHED"
        )
        print("=" * 75)

        return profiles

    except Exception as exc:
        print(
            f"    [ERROR] Company research failed: {exc}"
        )
        return []


if __name__ == "__main__":
    run_research_demo()