"""Step 6: source-backed professional email discovery for qualified founders."""

import json
import logging
import os
import re
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

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, groq_api_key: Optional[str] = None, model: Optional[str] = None,
                 search_service: Optional[TavilySearchService] = None,
                 page_fetcher: Optional[Callable[[str], Optional[str]]] = None) -> None:
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self.search_service = search_service or TavilySearchService()
        self.page_fetcher = page_fetcher or CompanyResearchService().fetch_webpage_text
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
            f'"{founder_name}" "{name}" email', f'"{founder_name}" "{name}" contact',
            f'"{founder_name}" CEO email', f'"{founder_name}" founder email',
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
        confidence = "high" if company_domain and email.endswith(f"@{company_domain}") else "medium"
        return EmailCandidate(company_name=company.company_name, founder_name=founder_name,
                              founder_role=founder.role or "", email=email, source_url=url,
                              evidence=evidence, source_type=self._source_type(url, company),
                              discovery_status=EmailDiscoveryStatus.FOUND, confidence=confidence)

    def _extract_candidates(self, source: str, url: str, company: CompanyProfile,
                            founder: FounderProfile) -> List[EmailCandidate]:
        self._validate_client()
        assert self.client is not None
        prompt = (
            "Extract only email addresses explicitly present in the supplied content. Never construct, infer, "
            "guess, normalize into a new address, or complete a missing email address. Return only email addresses "
            "that the source explicitly associates with the named founder at the named company. Return JSON: "
            '{"candidates": [{"email": "string", "evidence": "verbatim source excerpt"}]}. '
            "Use an empty list if none qualify."
        )
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Company: {company.company_name}\nFounder: {founder.founder_name}\n\nSource: {source}"}],
            response_format={"type": "json_object"}, temperature=0.0,
        )
        raw = response.choices[0].message.content
        payload = json.loads(raw or "{}")
        return [candidate for item in payload.get("candidates", []) if isinstance(item, dict)
                if (candidate := self._candidate_from_data(item, source, url, company, founder))]

    @staticmethod
    def _select(candidates: List[EmailCandidate], company: CompanyProfile) -> Optional[EmailCandidate]:
        if not candidates:
            return None
        company_domain = EmailDiscoveryService._domain(company.website)
        return sorted(candidates, key=lambda item: (
            0 if company_domain and item.email and item.email.endswith(f"@{company_domain}") else 1,
            0 if item.source_type == "official_company_website" else 1, item.email or "",
        ))[0]

    def discover_email(self, company: CompanyProfile, founder: FounderProfile) -> EmailCandidate:
        if not QualificationService().qualify_profile(company).is_qualified or founder.discovery_status != FounderDiscoveryStatus.FOUND or not founder.founder_name:
            return EmailCandidate(company_name=company.company_name, founder_name=founder.founder_name or "", founder_role=founder.role or "", discovery_status=EmailDiscoveryStatus.NOT_FOUND)
        results: List[SearchResult] = []
        seen = set()
        for query in self.build_queries(company, founder):
            try:
                for result in self.search_service.search(query, max_results=2):
                    if result.url not in seen:
                        seen.add(result.url); results.append(result)
            except Exception as exc:
                logger.warning("Email search failed | Company: %s | Error: %s", company.company_name, exc)
        candidates: List[EmailCandidate] = []
        errors = 0
        for result in results:
            source = self.page_fetcher(result.url) or result.content
            if not source:
                continue
            try:
                candidates.extend(self._extract_candidates(source, result.url, company, founder))
            except Exception as exc:
                errors += 1; logger.warning("Email extraction failed | Source: %s | Error: %s", result.url, exc)
        selected = self._select(candidates, company)
        if selected:
            logger.info("Email discovered | Company: %s | Founder: %s", company.company_name, founder.founder_name)
            return selected
        return EmailCandidate(company_name=company.company_name, founder_name=founder.founder_name,
                              founder_role=founder.role or "", discovery_status=EmailDiscoveryStatus.UNCERTAIN if errors else EmailDiscoveryStatus.NOT_FOUND)
