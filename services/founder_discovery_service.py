"""Evidence-backed Step 5 founder/CEO discovery using the existing Tavily and Groq services."""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from groq import Groq

from core.founder_models import FounderConfidence, FounderDiscoveryStatus, FounderProfile
from core.models import CompanyProfile, SearchResult
from services.company_research_service import CompanyResearchService
from services.qualification_service import QualificationService
from services.search_service import TavilySearchService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

logger = logging.getLogger("tvb.founder_discovery_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


VALID_ROLE_PATTERNS = (
    (r"\b(?:co[- ]?founder|founder)\s*(?:&|and)\s*(?:chief executive officer|ceo)\b", 0),
    (r"\b(?:chief executive officer|ceo)\b", 1),
    (r"\bco[- ]?founder\b", 2),
    (r"\bfounder\b", 3),
)
INVALID_ROLE_PATTERN = r"\b(?:cto|chief technology officer|coo|cmo|advisor|investor|board member|employee)\b"
NEGATED_ROLE_PATTERN = r"\b(?:not|former|previous)\s+(?:the\s+)?(?:ceo|chief executive officer|co[- ]?founder|founder)\b"


@dataclass(frozen=True)
class _FounderCandidate:
    name: str
    role: str
    linkedin_url: Optional[str]
    source_url: str
    evidence: str
    role_rank: int
    source_rank: int


class FounderDiscoveryService:
    """Researches one evidence-supported CEO or founder after Step 4 qualification."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: Optional[str] = None,
        search_service: Optional[TavilySearchService] = None,
        page_fetcher: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self.search_service = search_service or TavilySearchService()
        self.page_fetcher = page_fetcher or CompanyResearchService().fetch_webpage_text
        self.client: Optional[Groq] = None
        if self.groq_api_key and self.groq_api_key.strip() not in ("", "your_groq_api_key_here"):
            try:
                self.client = Groq(api_key=self.groq_api_key, timeout=25.0)
            except Exception as exc:
                logger.error("Failed to initialize Groq client: %s", exc)

    def _validate_client(self) -> None:
        if not self.groq_api_key or self.groq_api_key.strip() in ("", "your_groq_api_key_here"):
            raise RuntimeError("GROQ_API_KEY is missing; founder extraction cannot run.")
        if self.client is None:
            self.client = Groq(api_key=self.groq_api_key, timeout=25.0)

    @staticmethod
    def _domain(website: Optional[str]) -> Optional[str]:
        if not website:
            return None
        parsed = urlparse(website if "://" in website else f"https://{website}")
        return parsed.netloc.lower().removeprefix("www.") or None

    def build_queries(self, profile: CompanyProfile) -> List[str]:
        """Build diverse source-discovery queries from the actual company identity."""
        name = profile.company_name.strip()
        domain = self._domain(profile.website)
        queries = [
            f'"{name}" CEO founder',
            f'"{name}" co-founder',
            f'"{name}" founder LinkedIn',
            f'"{name}" leadership CEO',
            f'"{name}" about founder',
        ]
        if domain:
            queries.insert(0, f'site:{domain} "{name}" founder OR CEO')
        return queries

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.split()).lower()

    @classmethod
    def _source_contains(cls, excerpt: str, source_text: str) -> bool:
        return bool(excerpt and cls._normalize(excerpt) in cls._normalize(source_text))

    @staticmethod
    def _role_rank(role: str) -> Optional[int]:
        normalized = role.lower()
        if re.search(INVALID_ROLE_PATTERN, normalized, re.I) and not any(
            re.search(pattern, normalized, re.I) for pattern, _ in VALID_ROLE_PATTERNS
        ):
            return None
        for pattern, rank in VALID_ROLE_PATTERNS:
            if re.search(pattern, normalized, re.I):
                return rank
        return None

    def _source_rank(self, source_url: str, profile: CompanyProfile) -> int:
        domain = self._domain(source_url)
        if domain and domain == self._domain(profile.website):
            return 0
        if domain and "linkedin.com" in domain:
            return 3
        if domain and any(token in domain for token in ("newsroom", "press", "prnewswire", "businesswire")):
            return 2
        return 4

    def _candidate_from_data(
        self, data: Dict[str, object], source_text: str, source_url: str, profile: CompanyProfile
    ) -> Optional[_FounderCandidate]:
        """Accept only a company-matched, source-backed name and valid role."""
        name = str(data.get("founder_name") or "").strip()
        role = str(data.get("role") or "").strip()
        evidence = str(data.get("evidence") or "").strip()
        if not name or not role or not evidence:
            return None
        if profile.company_name.lower() not in source_text.lower():
            return None
        if not self._source_contains(evidence, source_text):
            return None
        if re.search(NEGATED_ROLE_PATTERN, evidence, re.I):
            return None
        role_rank = self._role_rank(role)
        if role_rank is None:
            return None
        linkedin_url = data.get("linkedin_url")
        linkedin_url = str(linkedin_url).strip() if linkedin_url else None
        if linkedin_url and linkedin_url not in source_text and "linkedin.com" not in source_url:
            linkedin_url = None
        return _FounderCandidate(
            name=name, role=role, linkedin_url=linkedin_url, source_url=source_url,
            evidence=evidence, role_rank=role_rank, source_rank=self._source_rank(source_url, profile),
        )

    def _extract_candidates(self, source_text: str, source_url: str, profile: CompanyProfile) -> List[_FounderCandidate]:
        self._validate_client()
        assert self.client is not None
        prompt = (
            "Extract only information explicitly supported by the supplied source content. "
            "Do not guess, infer, or complete missing founder information. The person must be explicitly "
            "associated with the named company and explicitly be a CEO, founder, or co-founder. Return JSON only: "
            '{"company_match": true/false, "candidates": [{"founder_name": "string", "role": "string", '
            '"linkedin_url": "string or null", "evidence": "verbatim source excerpt"}]}. '
            "Return an empty candidates list if no valid person is stated."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Company: {profile.company_name}\nSource URL: {source_url}\n\nSource content:\n{source_text}"},
            ],
            response_format={"type": "json_object"}, temperature=0.0,
        )
        raw = response.choices[0].message.content
        if not raw:
            return []
        payload = json.loads(raw)
        if not payload.get("company_match"):
            return []
        return [
            candidate for item in payload.get("candidates", [])
            if isinstance(item, dict)
            if (candidate := self._candidate_from_data(item, source_text, source_url, profile))
        ]

    @staticmethod
    def _select_candidate(candidates: List[_FounderCandidate]) -> Optional[_FounderCandidate]:
        """Prefer CEO roles, then stronger source types, with deterministic name ordering."""
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item.role_rank, item.source_rank, item.name.lower()))[0]

    def _not_found(self, profile: CompanyProfile, status: FounderDiscoveryStatus, evidence: Optional[str] = None) -> FounderProfile:
        return FounderProfile(
            company_name=profile.company_name, evidence=evidence, discovery_status=status,
            confidence=FounderConfidence.LOW,
        )

    def discover_founder(self, profile: CompanyProfile) -> FounderProfile:
        """Discover a CEO/co-founder only if the profile continues to pass Step 4."""
        qualification = QualificationService().qualify_profile(profile)
        if not qualification.is_qualified:
            logger.info("Skipping founder discovery for unqualified company: %s", profile.company_name)
            return self._not_found(profile, FounderDiscoveryStatus.NOT_FOUND, "Company did not pass Step 4 qualification.")

        logger.info("Researching founder/CEO for qualified company: %s", profile.company_name)
        results: List[SearchResult] = []
        seen_urls = set()
        for query in self.build_queries(profile):
            logger.info("Founder discovery query | Company: %s | Query: %s", profile.company_name, query)
            try:
                for result in self.search_service.search(query, max_results=2):
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        results.append(result)
            except Exception as exc:
                logger.warning("Founder search failed for '%s': %s", query, exc)
        logger.info("Founder discovery sources | Company: %s | Count: %d", profile.company_name, len(results))

        candidates: List[_FounderCandidate] = []
        extraction_errors = 0
        contradictory_evidence = False
        for result in results:
            source_text = self.page_fetcher(result.url) or result.content
            if not source_text:
                continue
            if re.search(NEGATED_ROLE_PATTERN, source_text, re.I):
                contradictory_evidence = True
            try:
                candidates.extend(self._extract_candidates(source_text, result.url, profile))
            except Exception as exc:
                extraction_errors += 1
                logger.warning("Founder extraction failed | Source: %s | Error: %s", result.url, exc)

        selected = self._select_candidate(candidates)
        if contradictory_evidence:
            logger.info("Founder discovery uncertain due to contradictory evidence | Company: %s", profile.company_name)
            return self._not_found(profile, FounderDiscoveryStatus.UNCERTAIN)
        if not selected:
            status = FounderDiscoveryStatus.UNCERTAIN if (extraction_errors and results) or contradictory_evidence else FounderDiscoveryStatus.NOT_FOUND
            logger.info("Founder discovery %s | Company: %s", status.value, profile.company_name)
            return self._not_found(profile, status)
        confidence = FounderConfidence.HIGH if selected.source_rank <= 2 else FounderConfidence.MEDIUM
        logger.info("Founder discovered | Company: %s | Founder: %s | Role: %s", profile.company_name, selected.name, selected.role)
        return FounderProfile(
            company_name=profile.company_name, founder_name=selected.name, role=selected.role,
            linkedin_url=selected.linkedin_url, source_urls=[selected.source_url], evidence=selected.evidence,
            confidence=confidence, discovery_status=FounderDiscoveryStatus.FOUND,
        )

    def discover_founders(self, companies: List[CompanyProfile]) -> List[FounderProfile]:
        """Process every supplied profile while isolating individual company failures."""
        profiles: List[FounderProfile] = []
        for company in companies:
            try:
                profiles.append(self.discover_founder(company))
            except Exception as exc:
                logger.error("Founder discovery failed | Company: %s | Error: %s", company.company_name, exc)
                profiles.append(self._not_found(company, FounderDiscoveryStatus.UNCERTAIN))
        return profiles


def run_founder_demo() -> List[FounderProfile]:
    """Run discovery -> Step 3 -> Step 4 -> founder discovery for live qualified results."""
    qualified_profiles: List[CompanyProfile] = []
    # Reuse the existing dynamic query and services; do not maintain a company list here.
    try:
        search = TavilySearchService().search("European B2B SaaS startup raised $2M seed funding", max_results=3)
        discovered = CompanyResearchService().research_candidates(search)
        qualified_profiles = [profile for profile in discovered if QualificationService().qualify_profile(profile).is_qualified]
    except Exception as exc:
        print(f"[ERROR] Qualification pipeline failed: {exc}")
        return []
    if not qualified_profiles:
        print("No dynamically discovered companies passed Step 4; founder discovery was not run.")
        return []
    profiles = FounderDiscoveryService().discover_founders(qualified_profiles)
    for founder in profiles:
        print(f"Company: {founder.company_name}")
        print(f"Status: {founder.discovery_status.value}")
        print(f"Founder: {founder.founder_name or '[Not Found]'}")
        print(f"Role: {founder.role or '[Not Found]'}")
        print(f"Evidence: {founder.evidence or '[None]'}")
        print("Sources:")
        for url in founder.source_urls:
            print(f"  - {url}")
    return profiles


if __name__ == "__main__":
    run_founder_demo()
