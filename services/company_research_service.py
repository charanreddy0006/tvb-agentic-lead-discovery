"""Company research and profile extraction service for TVB Agentic Company Lead Discovery.

Fetches webpage content from discovered SearchResult items and uses Groq LLM
for strict, zero-hallucination entity extraction into structured CompanyProfile models.
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

# Ensure Windows console handles UTF-8 characters without CharMap errors
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
        getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
import httpx

from core.models import CompanyProfile, SearchResult
from services.search_service import TavilySearchService

# Load environment variables from project root .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

# Configure module-level logger
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


class CompanyResearchError(Exception):
    """Base exception for company research service errors."""
    pass


# Domains commonly representing news, publications, or directories (not company official sites)
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


class CompanyResearchService:
    """Extracts structured CompanyProfile entities from search results using Groq LLM."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    FALLBACK_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    MAX_PAGE_CHARS = 6000  # Context truncation for clean, fast extraction
    FETCH_TIMEOUT = 10.0   # HTTP request timeout per page

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: Optional[str] = None,
        fetch_timeout: float = FETCH_TIMEOUT,
        max_workers: int = 4,
    ):
        """Initialize the Company Research Service.

        Args:
            groq_api_key: Optional Groq API key. Defaults to GROQ_API_KEY from .env.
            model: Groq model name for structured extraction.
            fetch_timeout: Timeout in seconds for fetching individual webpages.
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self.fetch_timeout = fetch_timeout
        self.client: Optional[Groq] = None
        self.max_workers = max(1, min(max_workers, 5))
        self._http_client: Optional[httpx.Client] = None
        self._http_client_lock = threading.Lock()
        self._page_cache: Dict[str, Optional[str]] = {}
        self._page_cache_lock = threading.Lock()

        if not self.groq_api_key or self.groq_api_key.strip() in ("", "your_groq_api_key_here"):
            logger.warning(
                "GROQ_API_KEY is not configured or contains placeholder text. "
                "Profile extraction will require a valid key in .env."
            )
        else:
            try:
                self.client = Groq(api_key=self.groq_api_key, timeout=25.0)
            except Exception as e:
                logger.error("Failed to initialize Groq client: %s", e)

    def _validate_client(self) -> None:
        """Ensure Groq client is ready."""
        if not self.groq_api_key or self.groq_api_key.strip() in ("", "your_groq_api_key_here"):
            raise CompanyResearchError(
                "GROQ_API_KEY is missing. Please configure a valid Groq API key in your .env file."
            )
        if self.client is None:
            try:
                self.client = Groq(api_key=self.groq_api_key, timeout=25.0)
            except Exception as e:
                raise CompanyResearchError(f"Could not connect to Groq client: {e}") from e

    def fetch_webpage_text(self, url: str) -> Optional[str]:
        """Fetch and clean readable text from a webpage.

        Args:
            url: Target webpage URL.

        Returns:
            Clean text content truncated to MAX_PAGE_CHARS, or None if fetch fails.
        """
        with self._page_cache_lock:
            if url in self._page_cache:
                return self._page_cache[url]

        logger.info("Fetching webpage content | URL: %s", url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        try:
            with self._http_client_lock:
                if self._http_client is None:
                    self._http_client = httpx.Client(timeout=self.fetch_timeout, follow_redirects=True)
                http_client = self._http_client
            resp = http_client.get(url, headers=headers)

            if resp.status_code != 200:
                logger.warning(
                    "Page fetch failed (HTTP %d) | URL: %s", resp.status_code, url
                )
                with self._page_cache_lock:
                    self._page_cache[url] = None
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove unneeded elements
            for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
                tag.decompose()

            raw_text = soup.get_text(separator=" ", strip=True)
            clean_text = " ".join(raw_text.split())

            if len(clean_text) < 100:
                logger.warning("Page returned minimal useful text (<100 chars) | URL: %s", url)
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
            logger.warning("Page fetch timed out after %.1fs | URL: %s", self.fetch_timeout, url)
            with self._page_cache_lock:
                self._page_cache[url] = None
            return None
        except httpx.RequestError as exc:
            logger.warning("Page fetch network error: %s | URL: %s", exc, url)
            with self._page_cache_lock:
                self._page_cache[url] = None
            return None
        except Exception as exc:
            logger.warning("Unexpected error fetching page: %s | URL: %s", exc, url)
            with self._page_cache_lock:
                self._page_cache[url] = None
            return None

    @staticmethod
    def _normalize_source_text(value: str) -> str:
        """Normalize whitespace for conservative source-text comparisons."""
        return " ".join(value.split())

    @classmethod
    def _is_source_backed_evidence(cls, evidence: str, source_text: str) -> bool:
        """Return whether an evidence excerpt is present in the supplied source text."""
        return cls._normalize_source_text(evidence) in cls._normalize_source_text(source_text)

    @staticmethod
    def _currency_from_evidence(evidence: Optional[str]) -> Optional[str]:
        """Determine a funding currency only when the supporting excerpt states one."""
        if not evidence:
            return None

        if re.search(r"\u20ac|\bEUR\b|\beuros?\b", evidence, flags=re.IGNORECASE):
            return "EUR"
        if re.search(r"\u00a3|\bGBP\b|\bpounds?\b", evidence, flags=re.IGNORECASE):
            return "GBP"
        if re.search(r"\$|\bUSD\b|\bUS\s*dollars?\b|\bdollars?\b", evidence, flags=re.IGNORECASE):
            return "USD"

        if re.search(r"€|\bEUR\b|\beuros?\b", evidence, flags=re.IGNORECASE):
            return "EUR"
        if re.search(r"£|\bGBP\b|\bpounds?\b", evidence, flags=re.IGNORECASE):
            return "GBP"
        if re.search(r"\$|\bUSD\b|\bUS\s*dollars?\b|\bdollars?\b", evidence, flags=re.IGNORECASE):
            return "USD"
        return None

    @classmethod
    def _is_explicit_website(cls, website: str, source_text: str) -> bool:
        """Return whether the candidate website's hostname is stated in source text."""
        candidate = website if website.startswith(("http://", "https://")) else f"https://{website}"
        hostname = urlparse(candidate).netloc.split(":")[0].lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return bool(hostname and hostname in source_text.lower())

    def _extract_from_text(
        self, text: str, source_url: str, fallback_domain: str
    ) -> Optional[CompanyProfile]:
        """Extract a structured CompanyProfile from text using Groq LLM.

        Enforces zero-hallucination rules: missing values are returned as null.
        """
        self._validate_client()
        assert self.client is not None

        # Determine the source domain for prompt context only. A source URL is never
        # promoted to a company website without an explicit website statement.
        parsed = urlparse(source_url)
        source_domain = parsed.netloc.split(":")[0].lower()

        system_prompt = (
            "You are an expert corporate intelligence analyst for The Venture Build (TVB).\n"
            "Analyze the provided text from a web search result and extract factual information "
            "about the primary technology company described.\n\n"
            "CRITICAL ZERO-HALLUCINATION DIRECTIVES:\n"
            "1. ONLY extract information that is explicitly stated in the text.\n"
            "2. If any piece of information is NOT explicitly stated, set that field to null. "
            "NEVER invent, guess, or extrapolate values.\n"
            "3. FUNDING VS REVENUE: Strictly distinguish between FUNDING RAISED (VC/equity/debt) "
            "and ANNUAL REVENUE (commercial sales/ARR). Never categorize funding as revenue or revenue as funding.\n"
            "   - For funding currency, return EUR when the evidence uses the euro symbol or EUR; return GBP "
            "when it uses the pound symbol or GBP; return USD only when it uses a dollar sign or USD. "
            "Return null when the funding claim does not state a currency. Never convert currencies.\n"
            "4. NUMERIC AMOUNTS AND CURRENCY: Extract funding_amount and revenue_amount as plain numbers "
            "(e.g., €2.5M -> 2500000). If no exact number is given, return null. Extract a currency "
            "ONLY when it is explicitly stated with that financial claim: € or EUR means EUR; £ or GBP means "
            "GBP; $ or USD means USD. If currency is absent, return null. Never assume USD and never convert.\n"
            "5. OFFICIAL WEBSITE HANDLING:\n"
            f"   - Current source domain is: '{source_domain}'.\n"
            "   - Extract website ONLY when the text explicitly provides the company's official domain or URL. "
            "If not found, return null. Do NOT derive a website from the company name or use the source/news/directory URL.\n"
            "6. EVIDENCE: In the 'evidence' object, include VERBATIM excerpts from the text supporting "
            "funding, revenue, and location claims.\n"
            "7. NON-COMPANY FILTER: If the text is a generic report, list of 50 tools, or does not focus "
            "on a specific commercial company, set 'is_company' to false.\n\n"
            "Return valid JSON ONLY matching this schema:\n"
            "{\n"
            '  "is_company": true/false,\n'
            '  "company_name": "string or null",\n'
            '  "website": "string or null",\n'
            '  "description": "string or null",\n'
            '  "industry": "string or null",\n'
            '  "location": "string (City, Country) or null",\n'
            '  "funding_amount": number or null,\n'
            '  "funding_currency": "USD", "EUR", "GBP", or null,\n'
            '  "funding_stage": "Seed" or null,\n'
            '  "revenue_amount": number or null,\n'
            '  "revenue_currency": "USD", "EUR", "GBP", or null,\n'
            '  "evidence": {\n'
            '    "funding": "verbatim quote or null",\n'
            '    "revenue": "verbatim quote or null",\n'
            '    "location": "verbatim quote or null"\n'
            "  }\n"
            "}"
        )

        user_prompt = f"Source URL: {source_url}\n\nWebpage Text Content:\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for strict factual extraction
            )

            raw = response.choices[0].message.content
            if not raw:
                logger.warning("Empty response received from LLM for URL: %s", source_url)
                return None

            data = json.loads(raw)

            if not data.get("is_company", True):
                logger.info("Page evaluated as not describing a primary company | URL: %s", source_url)
                return None

            company_name = data.get("company_name")
            if not company_name or str(company_name).strip().lower() in ("null", "none", ""):
                logger.info("No clear company name could be extracted | URL: %s", source_url)
                return None

            company_name = str(company_name).strip()

            # Keep a website only when its domain is explicitly present in this source.
            # Never infer it from a company name or promote the result URL as a fallback.
            website = data.get("website")
            if website and str(website).strip().lower() not in ("null", "none", ""):
                website = str(website).strip()
                if not website.startswith(("http://", "https://")):
                    website = f"https://{website}"
                if not self._is_explicit_website(website, text):
                    website = None
            else:
                website = None

            # Clean numeric values
            funding_amt = data.get("funding_amount")
            if funding_amt is not None:
                try:
                    funding_amt = float(funding_amt)
                except (ValueError, TypeError):
                    funding_amt = None

            revenue_amt = data.get("revenue_amount")
            if revenue_amt is not None:
                try:
                    revenue_amt = float(revenue_amt)
                except (ValueError, TypeError):
                    revenue_amt = None

            # Clean evidence dict
            raw_evidence = data.get("evidence", {}) or {}
            clean_evidence = {
                k: str(v).strip()
                for k, v in raw_evidence.items()
                if v
                and str(v).strip().lower() not in ("null", "none", "")
                and self._is_source_backed_evidence(str(v).strip(), text)
            }

            # Trust the original funding excerpt—not an inferred/default LLM currency.
            # This deliberately leaves currency null when the evidence has no explicit marker.
            funding_currency = self._currency_from_evidence(clean_evidence.get("funding"))

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
                revenue_currency=data.get("revenue_currency"),
                source_urls=[source_url],
                evidence=clean_evidence,
            )

            logger.info(
                "Extracted CompanyProfile | Name: '%s' | Location: %s | Funding: %s %s",
                profile.company_name,
                profile.location,
                profile.funding_amount,
                profile.funding_currency,
            )
            return profile

        except Exception as e:
            logger.error("LLM profile extraction failed for URL '%s': %s", source_url, e)
            return None

    def research_search_result(self, result: SearchResult) -> Optional[CompanyProfile]:
        """Process a single SearchResult: fetch webpage text and extract CompanyProfile.

        Falls back to result.content snippet if live webpage fetching fails.
        """
        logger.info("Researching SearchResult | URL: %s | Title: '%s'", result.url, result.title)

        # 1. Attempt live page fetch
        page_text = self.fetch_webpage_text(result.url)

        # 2. Fallback to Tavily snippet if live fetch yields insufficient text
        if not page_text or len(page_text) < 150:
            if result.content and len(result.content) >= 80:
                logger.info("Using search snippet as fallback content for: %s", result.url)
                page_text = f"Title: {result.title}\n\nSnippet: {result.content}"
            else:
                logger.warning("Insufficient text content for analysis | URL: %s", result.url)
                return None

        # 3. Extract structured profile via Groq LLM
        return self._extract_from_text(
            text=page_text,
            source_url=result.url,
            fallback_domain=result.domain,
        )

    def research_candidates(
        self, search_results: List[SearchResult]
    ) -> List[CompanyProfile]:
        """Research multiple search results, with deduplication and evidence merging.

        Args:
            search_results: List of SearchResult items from search service.

        Returns:
            Deduplicated list of structured CompanyProfile models.
        """
        dedup_store: Dict[str, CompanyProfile] = {}

        logger.info("Beginning batch research on %d search results", len(search_results))

        def research_one(item: SearchResult) -> Optional[CompanyProfile]:
            try:
                return self.research_search_result(item)
            except Exception as exc:
                logger.warning("Candidate research failed | URL: %s | Error: %s", item.url, exc)
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            profiles = list(executor.map(research_one, search_results))

        for idx, profile in enumerate(profiles, 1):
            logger.info("--- Processing Candidate [%d/%d] ---", idx, len(search_results))

            if not profile:
                continue

            dedup_key = profile.get_dedup_key()

            if dedup_key in dedup_store:
                existing = dedup_store[dedup_key]
                logger.info(
                    "Duplicate company detected ('%s' -> key: %s). Merging evidence and sources.",
                    profile.company_name,
                    dedup_key,
                )

                # Merge source URLs
                for u in profile.source_urls:
                    if u not in existing.source_urls:
                        existing.source_urls.append(u)

                # Enrich missing fields
                if not existing.website and profile.website:
                    existing.website = profile.website
                if not existing.description and profile.description:
                    existing.description = profile.description
                if not existing.industry and profile.industry:
                    existing.industry = profile.industry
                if not existing.location and profile.location:
                    existing.location = profile.location
                if existing.funding_amount is None and profile.funding_amount is not None:
                    existing.funding_amount = profile.funding_amount
                    existing.funding_currency = profile.funding_currency
                    existing.funding_stage = profile.funding_stage
                if existing.revenue_amount is None and profile.revenue_amount is not None:
                    existing.revenue_amount = profile.revenue_amount
                    existing.revenue_currency = profile.revenue_currency

                # Merge evidence
                for k, v in profile.evidence.items():
                    if k not in existing.evidence:
                        existing.evidence[k] = v

            else:
                dedup_store[dedup_key] = profile

        final_profiles = list(dedup_store.values())
        logger.info(
            "Batch research complete | Input Results: %d | Unique Companies Discovered: %d",
            len(search_results),
            len(final_profiles),
        )
        return final_profiles


def run_research_demo(
    query: Optional[str] = None,
    max_results: int = 3,
) -> List[CompanyProfile]:
    """Verification function demonstrating the Step 3 research pipeline:

    Query -> Tavily Search -> Webpage Fetch -> Groq Extraction -> CompanyProfile.

    Args:
        query: Search query string. Defaults to a generic technology discovery query.
        max_results: Number of search results to retrieve and research.

    Returns:
        List of extracted CompanyProfile objects.
    """
    demo_query = query or "European B2B SaaS startup raised $2M seed funding"

    print("=" * 75)
    print("TVB Company Research Pipeline Demo (Step 3)")
    print(f"Discovery Query: '{demo_query}'")
    print("=" * 75)

    # 1. Execute search using TavilySearchService (reusing Step 2 service)
    print("\n[1] Executing Tavily Search...")
    search_service = TavilySearchService()

    try:
        results = search_service.search(demo_query, max_results=max_results)
        print(f"    Retrieved {len(results)} search results.")
    except Exception as e:
        print(f"    [ERROR] Search failed: {e}")
        return []

    # 2. Process search results via CompanyResearchService
    print("\n[2] Researching and Extracting Company Profiles via Groq...")
    research_service = CompanyResearchService()

    try:
        profiles = research_service.research_candidates(results)
        print(f"\n[3] Research complete. Extracted {len(profiles)} unique company profile(s):")

        for idx, cp in enumerate(profiles, 1):
            print(f"\n" + "-" * 60)
            print(f"Company [{idx}]: {cp.company_name}")
            print(f"  Website:     {cp.website or '[Not Provided]'}")
            print(f"  Industry:    {cp.industry or '[Not Provided]'}")
            print(f"  Location:    {cp.location or '[Not Provided]'}")
            print(f"  Description: {cp.description or '[Not Provided]'}")

            funding_str = (
                f"{cp.funding_amount:,.0f} {cp.funding_currency or '[Currency Not Disclosed]'} "
                f"({cp.funding_stage or 'N/A'})"
                if cp.funding_amount is not None
                else "[Not Disclosed / Null]"
            )
            print(f"  Funding:     {funding_str}")

            revenue_str = (
                f"{cp.revenue_amount:,.0f} {cp.revenue_currency or 'USD'}"
                if cp.revenue_amount is not None
                else "[Not Disclosed / Null]"
            )
            print(f"  Revenue:     {revenue_str}")
            print(f"  Sources ({len(cp.source_urls)}):")
            for u in cp.source_urls:
                print(f"    - {u}")

            if cp.evidence:
                print("  Evidence Quotes:")
                for k, v in cp.evidence.items():
                    print(f"    * {k.upper()}: \"{v}\"")

        print("\n" + "=" * 75)
        print("[SUCCESS] STEP 3 PIPELINE DEMO FINISHED")
        print("=" * 75)
        return profiles

    except Exception as e:
        print(f"    [ERROR] Company research failed: {e}")
        return []


if __name__ == "__main__":
    run_research_demo()
