"""Step 6: source-backed professional email discovery for qualified founders."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from groq import Groq

from core.email_models import EmailCandidate, EmailDiscoveryStatus
from core.founder_models import FounderDiscoveryStatus, FounderProfile
from core.models import CompanyProfile, SearchResult
from services.company_research_service import CompanyResearchService
from services.qualification_service import QualificationService
from services.search_service import TavilySearchService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger("tvb.email_discovery_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
GENERIC_LOCAL_PARTS = {"info", "support", "sales", "hello", "contact", "team", "press", "privacy"}


class EmailDiscoveryService:
    """Discovers only emails explicitly attributed to a qualified founder or CEO."""

    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(self, groq_api_key: Optional[str] = None, model: Optional[str] = None,
                 search_service: Optional[TavilySearchService] = None,
                 page_fetcher: Optional[Callable[[str], Optional[str]]] = None,
                 max_workers: int = 1) -> None:
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        configured_model = os.getenv("GROQ_MODEL")
        self.model = model or (configured_model.strip() if configured_model else "") or self.DEFAULT_MODEL
        self.search_service = search_service or TavilySearchService()
        self.page_fetcher = page_fetcher or CompanyResearchService().fetch_webpage_text
        self.max_workers = max(1, min(max_workers, 4))
        # Keep Groq extraction sequential to stay within the current
        # Groq TPM limit. Search/fetch are also bounded by max_workers.
        self.max_extraction_workers = 1
        self.client: Optional[Groq] = None
        if self.groq_api_key and self.groq_api_key.strip() not in ("", "your_groq_api_key_here"):
            self.client = Groq(api_key=self.groq_api_key, timeout=25.0)

    @staticmethod
    def _domain(website: Optional[str]) -> Optional[str]:
        if not website:
            return None
        return urlparse(website if "://" in website else f"https://{website}").netloc.lower().removeprefix("www.") or None

    def build_queries(self, company: CompanyProfile, founder: FounderProfile) -> List[str]:
        name, founder_name = company.company_name, founder.founder_name or ""
        queries = [
            f'"{founder_name}" "{name}" email',
            f'"{founder_name}" "{name}" contact',
            f'"{founder_name}" CEO email',
            f'"{founder_name}" founder email',
            f'"{founder_name}" "@"',
        ]
        if domain := self._domain(company.website):
            queries.insert(0, f'"{founder_name}" "{domain}"')
        return queries

    def _validate_client(self) -> None:
        if not self.groq_api_key or self.groq_api_key.strip() in ("", "your_groq_api_key_here"):
            raise RuntimeError("GROQ_API_KEY is missing; email extraction cannot run.")
        if self.client is None:
            self.client = Groq(api_key=self.groq_api_key, timeout=25.0)

    @staticmethod
    def _source_type(url: str, company: CompanyProfile) -> str:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        company_domain = EmailDiscoveryService._domain(company.website)
        if domain and domain == company_domain:
            return "official_company_website"
        if "linkedin.com" in domain:
            return "professional_profile"
        if any(word in domain for word in ("press", "newsroom", "prnewswire", "businesswire")):
            return "press_release"
        return "third_party_public_source"

    @staticmethod
    def _source_contains(excerpt: str, source: str) -> bool:
        return " ".join(excerpt.split()).lower() in " ".join(source.split()).lower()

    def _candidate_from_data(self, data: Dict[str, object], source: str, url: str,
                             company: CompanyProfile, founder: FounderProfile) -> Optional[EmailCandidate]:
        email = str(data.get("email") or "").strip().lower()
        evidence = str(data.get("evidence") or "").strip()
        founder_name = founder.founder_name or ""
        if not email or not evidence or not EMAIL_PATTERN.fullmatch(email):
            return None
        if email.split("@", 1)[0] in GENERIC_LOCAL_PARTS:
            return None
        if founder_name.lower() not in evidence.lower() or email not in evidence.lower():
            return None
        if not self._source_contains(evidence, source):
            return None
        company_domain = self._domain(company.website)
        email_domain = email.split("@", 1)[1].lower()
        if company_domain and email_domain != company_domain:
            return None
        if not company_domain:
            source_domain = urlparse(url).netloc.lower().removeprefix("www.")
            blocked_domains = {
                "rocketreach.co", "signalhire.com", "apollo.io", "zoominfo.com",
                "lead411.com", "theorg.com", "wellfound.com", "crunchbase.com",
                "startupfundraising.com", "linkedin.com", "facebook.com",
                "instagram.com", "x.com", "twitter.com",
            }
            # Without an independently known company domain, only accept a
            # founder email from the same non-social, non-contact-data domain
            # as the source page. This blocks false positives such as a
            # fundraising agency email attached to a founder's name.
            if source_domain in blocked_domains or email_domain != source_domain:
                return None
        confidence = "high" if company_domain and email_domain == company_domain else "medium"
        return EmailCandidate(company_name=company.company_name, founder_name=founder_name,
                              founder_role=founder.role or "", email=email, source_url=url,
                              evidence=evidence, source_type=self._source_type(url, company),
                              discovery_status=EmailDiscoveryStatus.FOUND, confidence=confidence)

    @staticmethod
    def _explicit_emails(source: str) -> List[str]:
        """Return unique email addresses that are literally present in source."""
        seen = set()
        emails: List[str] = []
        for match in EMAIL_PATTERN.findall(source or ""):
            email = match.strip().lower()
            if email not in seen:
                seen.add(email)
                emails.append(email)
        return emails

    @staticmethod
    def _relevant_excerpt(source: str, founder_name: str, max_chars: int = 2500) -> str:
        """Keep a compact source window around the founder/email evidence."""
        text = " ".join((source or "").split())
        if len(text) <= max_chars:
            return text

        lower = text.lower()
        positions = [match.start() for match in EMAIL_PATTERN.finditer(text)]

        if founder_name:
            founder_pos = lower.find(founder_name.lower())
            if founder_pos >= 0:
                positions.append(founder_pos)

        if not positions:
            return text[:max_chars]

        center = min(positions)
        start = max(0, center - max_chars // 2)
        end = min(len(text), start + max_chars)
        return text[start:end]

    def _deterministic_candidates(self, source: str, url: str, company: CompanyProfile, founder: FounderProfile) -> List[EmailCandidate]:
        """Extract explicit founder emails using sentence and proximity evidence; never invent an address."""
        name = (founder.founder_name or "").strip()
        if not name:
            return []
        text = " ".join((source or "").split())
        candidates: List[EmailCandidate] = []

        # First: strongest evidence — founder name and email in the same sentence/block.
        blocks = re.split(r"(?<=[.!?])\s+|\n+|(?<=:)\s+", text)
        for block in blocks:
            if name.lower() not in block.lower():
                continue
            for email in self._explicit_emails(block):
                candidate = self._candidate_from_data(
                    {"email": email, "evidence": block}, block, url, company, founder
                )
                if candidate:
                    candidates.append(candidate)

        if candidates:
            return candidates

        # Second: many public team/contact pages put the person's name, role, and
        # email in adjacent HTML/text blocks rather than one sentence. Accept only
        # an explicit email within a tight window around the exact founder name.
        lower = text.lower()
        name_pos = lower.find(name.lower())
        if name_pos < 0:
            return []

        window_start = max(0, name_pos - 700)
        window_end = min(len(text), name_pos + len(name) + 700)
        window = text[window_start:window_end]
        for email in self._explicit_emails(window):
            evidence = window.strip()
            candidate = self._candidate_from_data(
                {"email": email, "evidence": evidence}, evidence, url, company, founder
            )
            if candidate:
                candidates.append(candidate)

        return candidates

    def _extract_candidates(self, source: str, url: str, company: CompanyProfile,
                            founder: FounderProfile) -> List[EmailCandidate]:
        # Never call the LLM when the source contains no explicit email.
        # This saves Groq tokens and prevents inferred/guessed addresses.
        explicit_emails = self._explicit_emails(source)
        if not explicit_emails:
            return []

        deterministic = self._deterministic_candidates(source, url, company, founder)
        if deterministic:
            return deterministic

        self._validate_client()
        assert self.client is not None

        prompt = (
            "Extract only an email address explicitly present in the supplied content. "
            "The exact email must be explicitly associated with the named founder. "
            "Never construct, infer, guess, normalize, complete, or invent an address. "
            "If there is no explicit founder-email association, return an empty list. "
            'Return JSON only: {"candidates":[{"email":"string","evidence":"verbatim source excerpt"}]}.'
        )

        excerpt = self._relevant_excerpt(source, founder.founder_name or "")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Company: {company.company_name}\n"
                        f"Founder: {founder.founder_name}\n"
                        f"Explicit emails in source: {', '.join(explicit_emails)}\n"
                        f"Source URL: {url}\n\n"
                        f"Source excerpt:\n{excerpt}"
                    ),
                },
            ],
            include_reasoning=False,
            reasoning_effort="low",
            max_completion_tokens=800,
            temperature=0.0,
        )

        raw = response.choices[0].message.content
        try:
            payload = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            logger.warning("Email extraction returned invalid JSON | Source: %s", url)
            return []

        if not isinstance(payload, dict):
            return []

        candidates: List[EmailCandidate] = []
        items = payload.get("candidates", [])
        if not isinstance(items, list):
            return []

        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = self._candidate_from_data(
                item, source, url, company, founder
            )
            if candidate:
                candidates.append(candidate)

        return candidates

    @staticmethod
    def _select(candidates: List[EmailCandidate], company: CompanyProfile) -> Optional[EmailCandidate]:
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            return None
        company_domain = EmailDiscoveryService._domain(company.website)
        return sorted(candidates, key=lambda item: (
            0 if company_domain and item.email and item.email.endswith(f"@{company_domain}") else 1,
            0 if item.source_type == "official_company_website" else 1, item.email or "",
        ))[0]

    def _search_queries(self, queries: List[str]) -> List[SearchResult]:
        pending = []
        seen_queries = set()
        for query in queries:
            normalized = query.strip().lower()
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                pending.append(query)

        def search(query: str) -> List[SearchResult]:
            return self.search_service.search(query, max_results=2)

        results: List[SearchResult] = []
        seen_urls = set()
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(pending)))) as executor:
            futures = [executor.submit(search, query) for query in pending]
            for query, future in zip(pending, futures):
                try:
                    for result in future.result():
                        if result.url in seen_urls:
                            continue
                        seen_urls.add(result.url)
                        results.append(result)
                except Exception as exc:
                    logger.warning("Email search failed | Query: %s | Error: %s", query, exc)
            return results

    def _fetch_sources(self, results: List[SearchResult]) -> List[tuple[SearchResult, str]]:
        def fetch(result: SearchResult) -> tuple[SearchResult, str]:
            fetched_text = self.page_fetcher(result.url) or ""
            source = "\n".join(
                part for part in (result.title, result.content, fetched_text) if part
            )
            return result, source

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(results)))) as executor:
            fetched = list(executor.map(fetch, results))
        return [(result, source) for result, source in fetched if source]

    def _extract_sources(
        self,
        sources: List[tuple[SearchResult, str]],
        company: CompanyProfile,
        founder: FounderProfile,
    ) -> tuple[List[EmailCandidate], int]:
        """Extract sequentially and only from sources containing explicit emails."""
        candidates: List[EmailCandidate] = []
        errors = 0

        eligible_sources = [
            item for item in sources
            if self._explicit_emails(item[1])
        ]
        logger.info("Email discovery sources with explicit emails | Company: %s | Eligible: %d/%d", company.company_name, len(eligible_sources), len(sources))

        for result, source in eligible_sources:
            try:
                candidates.extend(
                    self._extract_candidates(
                        source, result.url, company, founder
                    )
                )
            except Exception as exc:
                errors += 1
                logger.warning(
                    "Email extraction failed | Source: %s | Error: %s",
                    result.url,
                    exc,
                )

        return candidates, errors

    def discover_email(self, company: CompanyProfile, founder: FounderProfile) -> EmailCandidate:
        if not QualificationService().qualify_profile(company).is_qualified or founder.discovery_status != FounderDiscoveryStatus.FOUND or not founder.founder_name:
            return EmailCandidate(company_name=company.company_name, founder_name=founder.founder_name or "", founder_role=founder.role or "", discovery_status=EmailDiscoveryStatus.NOT_FOUND)
        results = self._search_queries(self.build_queries(company, founder))
        sources = self._fetch_sources(results)
        candidates, extraction_errors = self._extract_sources(sources, company, founder)
        errors = extraction_errors
        logger.info("Email discovery candidates | Company: %s | Candidates: %d | Errors: %d", company.company_name, len(candidates), errors)
        selected = self._select(candidates, company)
        if selected:
            logger.info("Email discovered | Company: %s | Founder: %s", company.company_name, founder.founder_name)
            return selected
        return EmailCandidate(company_name=company.company_name, founder_name=founder.founder_name,
                              founder_role=founder.role or "", discovery_status=EmailDiscoveryStatus.UNCERTAIN if errors else EmailDiscoveryStatus.NOT_FOUND)
