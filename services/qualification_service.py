"""Deterministic, evidence-first Step 4 company qualification service."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from core.models import CompanyProfile
from core.validation_models import ConfidenceLevel, QualificationResult, ValidationStatus
from services.company_research_service import CompanyResearchService
from services.search_service import TavilySearchService

logger = logging.getLogger("tvb.qualification_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


USD_MINIMUM = 1_000_000
USD_MAXIMUM = 5_000_000
TECHNOLOGY_PATTERNS = (
    r"\bsaas\b", r"\bsoftware\b", r"\b(?:ai|artificial intelligence)[ -]?(?:powered )?(?:platform|software)\b",
    r"\b(?:cloud|data|cybersecurity|fintech|healthcare technology|education technology|developer|api|workflow automation) platform\b",
    r"\benterprise software\b", r"\bplatform\b",
)
CONSULTING_PATTERNS = (r"\bconsulting\b", r"\bit services\b", r"\bagency\b", r"\bprofessional services\b")
US_OPERATION_PATTERNS = (
    r"\b(?:headquartered|based) in (?:the )?(?:united states|usa|u\.s\.)\b",
    r"\b(?:united states|usa|u\.s\.)[ -](?:based|headquartered)\b", r"\bus headquarters?\b",
    r"\b(?:new york|san francisco|california|boston|austin|seattle),? (?:usa|us|united states)\b",
    r"\bmajor us office\b", r"\bprimary operations? in the (?:us|united states)\b",
)
NON_US_LOCATION_PATTERNS = (
    r"\b(?:paris|france|london|united kingdom|uk|berlin|germany|europe|india|singapore|indonesia|vietnam|uae|saudi arabia|egypt|brazil|mexico|colombia|australia|new zealand|canada|japan|south korea)\b",
)
MEDIUM_QUALITY_DOMAINS = {"techcrunch.com", "sifted.eu", "tech.eu", "venturebeat.com", "eu-startups.com", "uktech.news"}
LOW_QUALITY_DOMAINS = {"crunchbase.com", "pitchbook.com", "dealroom.co", "startupintros.com"}


@dataclass
class _FinancialAssessment:
    status: ValidationStatus
    basis: Optional[str]
    amount: Optional[float]
    currency: Optional[str]
    financial_type: str
    evidence: Optional[str]
    reason: Optional[str]


class QualificationService:
    """Qualifies Step 3 profiles without web calls, currency conversion, or inference."""

    @staticmethod
    def _has_text(value: Optional[str]) -> bool:
        return bool(value and value.strip())

    @staticmethod
    def _currency_label(currency: Optional[str]) -> str:
        return currency or "not disclosed"

    @staticmethod
    def _amount_in_usd_range(amount: float) -> ValidationStatus:
        if amount < USD_MINIMUM:
            return ValidationStatus.FAIL
        if amount > USD_MAXIMUM:
            return ValidationStatus.FAIL
        return ValidationStatus.PASS

    def _assess_amount(
        self, amount: Optional[float], currency: Optional[str], evidence: Optional[str], financial_type: str,
        requires_annual_revenue: bool = False,
    ) -> _FinancialAssessment:
        basis = f"{financial_type.title()} reported in profile"
        if amount is None:
            return _FinancialAssessment(ValidationStatus.UNKNOWN, basis, None, currency, financial_type, evidence, None)
        if not self._has_text(evidence):
            return _FinancialAssessment(
                ValidationStatus.UNKNOWN, basis, amount, currency, financial_type, evidence,
                f"{financial_type.title()} amount has no supporting evidence.",
            )
        if requires_annual_revenue and not re.search(r"\b(?:annual (?:revenue|recurring revenue)|arr)\b", evidence, re.I):
            return _FinancialAssessment(
                ValidationStatus.UNKNOWN, basis, amount, currency, financial_type, evidence,
                "Revenue evidence does not clearly identify annual revenue or ARR.",
            )
        if currency != "USD":
            return _FinancialAssessment(
                ValidationStatus.UNKNOWN, basis, amount, currency, financial_type, evidence,
                f"Financial currency is {self._currency_label(currency)} and cannot be compared directly with the USD range.",
            )
        status = self._amount_in_usd_range(amount)
        if status == ValidationStatus.PASS:
            return _FinancialAssessment(status, basis, amount, currency, financial_type, evidence, None)
        comparison = "below the $1M minimum" if amount < USD_MINIMUM else "exceeds the $5M maximum"
        return _FinancialAssessment(status, basis, amount, currency, financial_type, evidence, f"{financial_type.title()} {comparison}.")

    @staticmethod
    def _explicit_total_funding(evidence: Optional[str]) -> Optional[Tuple[float, str]]:
        """Find an explicitly stated total funding amount, never a selected historical round."""
        if not evidence:
            return None
        money = (
            r"(?P<prefix>US\$|\$|USD|EUR|GBP|€|£)?\s*"
            r"(?P<amount>\d+(?:\.\d+)?)\s*"
            r"(?P<unit>[mMbBkK]|million|billion|thousand)?\s*"
            r"(?P<suffix>USD|EUR|GBP)?"
        )
        patterns = (
            rf"(?:raised|total funding|total raised)\b[^.\n]{{0,40}}?{money}[^.\n]{{0,20}}\btotal\b",
            rf"\btotal (?:funding|raised)\b[^.\n]{{0,40}}?{money}",
        )
        for pattern in patterns:
            match = re.search(pattern, evidence, re.I)
            if not match:
                continue
            raw_amount = float(match.group("amount"))
            unit = (match.group("unit") or "").lower()
            multiplier = 1_000_000 if unit in {"m", "million"} else 1_000_000_000 if unit in {"b", "billion"} else 1_000 if unit in {"k", "thousand"} else 1
            currency_tokens = {
                (match.group("prefix") or "").strip().upper(),
                (match.group("suffix") or "").strip().upper(),
            }
            currency = "USD" if currency_tokens & {"US$", "$", "USD"} else next(
                (token for token in ("EUR", "GBP") if token in currency_tokens), ""
            )
            return raw_amount * multiplier, currency
        return None

    def validate_financials(self, profile: CompanyProfile) -> _FinancialAssessment:
        """Validate funding or explicitly annual revenue against the USD target range."""
        funding_evidence = profile.evidence.get("funding")
        funding_amount = profile.funding_amount
        funding_currency = profile.funding_currency
        total_funding = self._explicit_total_funding(funding_evidence)
        funding_basis = "Funding reported in profile"
        if total_funding:
            funding_amount, funding_currency = total_funding
            funding_basis = "Explicit total funding in evidence"

        funding = self._assess_amount(funding_amount, funding_currency, funding_evidence, "funding")
        funding.basis = funding_basis
        revenue = self._assess_amount(
            profile.revenue_amount, profile.revenue_currency, profile.evidence.get("revenue"), "revenue", True
        )

        assessments = [item for item in (funding, revenue) if item.amount is not None]
        for assessment in assessments:
            if assessment.status == ValidationStatus.PASS:
                return assessment
        for assessment in assessments:
            if assessment.status == ValidationStatus.UNKNOWN:
                return assessment
        if assessments:
            return assessments[0]
        return _FinancialAssessment(
            ValidationStatus.UNKNOWN, None, None, None, "unknown", None,
            "No qualifying funding or annual revenue evidence found.",
        )

    def validate_technology(self, profile: CompanyProfile) -> Tuple[ValidationStatus, Optional[str], Optional[str]]:
        """Require evidence that the company itself sells a technology product or platform."""
        candidates = [profile.description, profile.industry, profile.evidence.get("technology")]
        for text in candidates:
            if not self._has_text(text):
                continue
            normalized = text.lower()
            if any(re.search(pattern, normalized, re.I) for pattern in TECHNOLOGY_PATTERNS):
                return ValidationStatus.PASS, text, None
            if any(re.search(pattern, normalized, re.I) for pattern in CONSULTING_PATTERNS):
                return ValidationStatus.UNKNOWN, text, "Consulting or services evidence does not establish a technology platform product."
        return ValidationStatus.UNKNOWN, None, "No clear evidence that the company provides a technology platform."

    def validate_geography(self, profile: CompanyProfile) -> Tuple[ValidationStatus, Optional[str], Optional[str]]:
        """Pass a known non-US HQ only when no significant US operations are evidenced."""
        headquarters_evidence = profile.evidence.get("headquarters") or profile.evidence.get("location") or profile.location
        us_operations_evidence = profile.evidence.get("us_operations")
        if not self._has_text(headquarters_evidence) and not self._has_text(us_operations_evidence):
            return ValidationStatus.UNKNOWN, None, "Insufficient geographic evidence."
        headquarters_text = (headquarters_evidence or "").lower()
        us_operations_text = (us_operations_evidence or "").lower()
        combined_evidence = " | ".join(value for value in (headquarters_evidence, us_operations_evidence) if value)
        if any(re.search(pattern, text, re.I) for text in (headquarters_text, us_operations_text) for pattern in US_OPERATION_PATTERNS):
            return ValidationStatus.FAIL, combined_evidence, "Significant US operational presence detected."
        if any(re.search(pattern, headquarters_text, re.I) for pattern in NON_US_LOCATION_PATTERNS):
            return ValidationStatus.PASS, combined_evidence, None
        return ValidationStatus.UNKNOWN, combined_evidence, "Geographic evidence does not establish a non-US headquarters."

    @staticmethod
    def _source_warnings(profile: CompanyProfile) -> List[str]:
        warnings: List[str] = []
        for url in profile.source_urls:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            if domain in LOW_QUALITY_DOMAINS:
                warnings.append(f"Lower-confidence directory or aggregator source: {domain}.")
            elif domain not in MEDIUM_QUALITY_DOMAINS and not (profile.website and domain in profile.website):
                warnings.append(f"Source quality is unclassified: {domain}.")
        return warnings

    @staticmethod
    def _confidence(statuses: List[ValidationStatus], warnings: List[str]) -> ConfidenceLevel:
        if all(status == ValidationStatus.PASS for status in statuses) and not warnings:
            return ConfidenceLevel.HIGH
        if sum(status == ValidationStatus.PASS for status in statuses) >= 2:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def qualify_profile(self, profile: CompanyProfile) -> QualificationResult:
        """Apply all deterministic TVB qualification rules to one profile."""
        financial = self.validate_financials(profile)
        technology_status, technology_evidence, technology_reason = self.validate_technology(profile)
        geography_status, geographic_evidence, geography_reason = self.validate_geography(profile)
        statuses = [financial.status, technology_status, geography_status]
        qualified = all(status == ValidationStatus.PASS for status in statuses)
        reasons = [reason for reason in (financial.reason, technology_reason, geography_reason) if reason]
        warnings = self._source_warnings(profile)
        result = QualificationResult(
            company_name=profile.company_name,
            website=profile.website,
            is_financially_qualified=financial.status,
            is_technology_qualified=technology_status,
            is_geographically_qualified=geography_status,
            is_qualified=qualified,
            financial_basis=financial.basis,
            financial_amount=financial.amount,
            financial_currency=financial.currency,
            financial_type=financial.financial_type,
            financial_evidence=financial.evidence,
            technology_evidence=technology_evidence,
            geographic_evidence=geographic_evidence,
            validation_sources=list(profile.source_urls),
            rejection_reasons=reasons,
            warnings=warnings,
            confidence=self._confidence(statuses, warnings),
        )
        logger.info(
            "Qualification | Company: %s | Financial: %s | Technology: %s | Geographic: %s | Overall: %s",
            result.company_name, financial.status.value, technology_status.value, geography_status.value,
            "QUALIFIED" if qualified else "NOT QUALIFIED",
        )
        for reason in reasons:
            logger.info("Qualification reason | Company: %s | %s", result.company_name, reason)
        return result

    def qualify_candidates(self, profiles: List[CompanyProfile]) -> List[QualificationResult]:
        """Qualify every already-deduplicated Step 3 profile exactly once."""
        seen = set()
        results: List[QualificationResult] = []
        for profile in profiles:
            key = profile.get_dedup_key()
            if key not in seen:
                seen.add(key)
                results.append(self.qualify_profile(profile))
        return results


def run_qualification_demo(query: Optional[str] = None, max_results: int = 3) -> List[QualificationResult]:
    """Run the existing live Step 3 pipeline followed by Step 4 qualification."""
    demo_query = query or "European B2B SaaS startup raised $2M seed funding"
    print("=" * 75)
    print("TVB Qualification Pipeline Demo (Step 4)")
    print(f"Discovery Query: '{demo_query}'")
    print("=" * 75)
    try:
        profiles = CompanyResearchService().research_candidates(
            TavilySearchService().search(demo_query, max_results=max_results)
        )
    except Exception as exc:
        print(f"[ERROR] Step 3 pipeline failed: {exc}")
        return []

    results = QualificationService().qualify_candidates(profiles)
    for result in results:
        print(f"\nCompany: {result.company_name}")
        print(f"  Website: {result.website or '[Not Provided]'}")
        print(f"  Financial: {result.is_financially_qualified.value}")
        print(f"  Technology: {result.is_technology_qualified.value}")
        print(f"  Geographic: {result.is_geographically_qualified.value}")
        print(f"  Overall: {'QUALIFIED' if result.is_qualified else 'NOT QUALIFIED'}")
        print(f"  Reasons: {result.rejection_reasons or ['[None]']}")
        print(f"  Evidence: financial={result.financial_evidence or '[None]'} | technology={result.technology_evidence or '[None]'} | geographic={result.geographic_evidence or '[None]'}")
        print("  Sources:")
        for url in result.validation_sources:
            print(f"    - {url}")

    print("\nSummary:")
    print(f"  Total candidates: {len(results)}")
    print(f"  Financially qualified: {sum(r.is_financially_qualified == ValidationStatus.PASS for r in results)}")
    print(f"  Technology qualified: {sum(r.is_technology_qualified == ValidationStatus.PASS for r in results)}")
    print(f"  Geographically qualified: {sum(r.is_geographically_qualified == ValidationStatus.PASS for r in results)}")
    print(f"  Fully qualified: {sum(r.is_qualified for r in results)}")
    return results


if __name__ == "__main__":
    run_qualification_demo()
