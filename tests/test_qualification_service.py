"""Deterministic Step 4 edge-case tests; no external company data or API calls."""

import unittest

from core.models import CompanyProfile
from core.validation_models import ValidationStatus
from services.qualification_service import QualificationService


def profile(**overrides: object) -> CompanyProfile:
    values = {
        "company_name": "Fixture Company",
        "description": "AI SaaS platform for business workflows",
        "industry": "Enterprise software",
        "location": "Paris, France",
        "funding_amount": 2_000_000,
        "funding_currency": "USD",
        "source_urls": ["https://publication.example/article"],
        "evidence": {"funding": "The company raised $2M in a seed round.", "location": "The company is Paris, France-based."},
    }
    values.update(overrides)
    return CompanyProfile(**values)


class QualificationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = QualificationService()

    def test_financial_boundaries_and_currency(self) -> None:
        self.assertEqual(self.service.validate_financials(profile()).status, ValidationStatus.PASS)
        self.assertEqual(self.service.validate_financials(profile(funding_amount=500_000)).status, ValidationStatus.FAIL)
        self.assertEqual(self.service.validate_financials(profile(funding_amount=7_000_000)).status, ValidationStatus.FAIL)
        self.assertEqual(self.service.validate_financials(profile(funding_currency="EUR", evidence={"funding": "The company raised EUR 2M.", "location": "Paris, France-based."})).status, ValidationStatus.UNKNOWN)
        self.assertEqual(self.service.validate_financials(profile(funding_amount=None, funding_currency=None, evidence={"location": "Paris, France-based."})).status, ValidationStatus.UNKNOWN)

    def test_annual_revenue_can_qualify(self) -> None:
        candidate = profile(funding_amount=None, funding_currency=None, revenue_amount=2_000_000, revenue_currency="USD", evidence={"revenue": "Annual revenue reached $2M.", "location": "Paris, France-based."})
        assessment = self.service.validate_financials(candidate)
        self.assertEqual(assessment.status, ValidationStatus.PASS)
        self.assertEqual(assessment.financial_type, "revenue")

    def test_total_funding_and_monthly_revenue_do_not_overqualify(self) -> None:
        total_funding = profile(evidence={"funding": "The company has raised $20M total, including a $2M seed round.", "location": "Paris, France-based."})
        self.assertEqual(self.service.validate_financials(total_funding).status, ValidationStatus.FAIL)
        monthly_revenue = profile(funding_amount=None, funding_currency=None, revenue_amount=200_000, revenue_currency="USD", evidence={"revenue": "Monthly revenue is $200K.", "location": "Paris, France-based."})
        self.assertEqual(self.service.validate_financials(monthly_revenue).status, ValidationStatus.UNKNOWN)

    def test_technology_and_geography_rules(self) -> None:
        self.assertEqual(self.service.validate_technology(profile())[0], ValidationStatus.PASS)
        self.assertNotEqual(self.service.validate_technology(profile(description="Technology consulting company", industry="IT services"))[0], ValidationStatus.PASS)
        self.assertEqual(self.service.validate_geography(profile())[0], ValidationStatus.PASS)
        self.assertEqual(self.service.validate_geography(profile(location="San Francisco, USA", evidence={"funding": "Raised $2M.", "location": "US-headquartered company in San Francisco, USA."}))[0], ValidationStatus.FAIL)
        self.assertEqual(self.service.validate_geography(profile(location=None, evidence={"funding": "Raised $2M."}))[0], ValidationStatus.UNKNOWN)

    def test_us_investor_is_not_us_operations(self) -> None:
        candidate = profile(evidence={"funding": "The Paris-based company raised $2M from a US investor.", "location": "Paris, France-based."})
        self.assertEqual(self.service.validate_geography(candidate)[0], ValidationStatus.PASS)

    def test_unknown_never_qualifies(self) -> None:
        result = self.service.qualify_profile(profile(funding_currency="EUR", evidence={"funding": "Raised EUR 2M.", "location": "Paris, France-based."}))
        self.assertFalse(result.is_qualified)
        self.assertEqual(result.is_financially_qualified, ValidationStatus.UNKNOWN)

    def test_explicit_usd_total_funding_formats(self) -> None:
        for evidence in (
            "The company raised $2M total.",
            "The company raised USD 2M total.",
            "The company raised 2M USD total.",
            "The company raised US$2M total.",
            "The company raised $2 million total.",
            "The company raised USD 2 million total.",
        ):
            assessment = self.service.validate_financials(profile(evidence={"funding": evidence, "location": "Paris, France-based."}))
            self.assertEqual(assessment.status, ValidationStatus.PASS, evidence)
            self.assertEqual(assessment.currency, "USD", evidence)

    def test_technology_and_geography_use_preserved_evidence(self) -> None:
        candidate = profile(
            description="A business company",
            industry="Operations",
            location=None,
            evidence={
                "funding": "Raised $2M.",
                "technology": "The company sells a cloud platform for data teams.",
                "headquarters": "Headquartered in Paris, France.",
                "us_operations": "No US operations are stated.",
            },
        )
        self.assertEqual(self.service.validate_technology(candidate)[0], ValidationStatus.PASS)
        self.assertEqual(self.service.validate_geography(candidate)[0], ValidationStatus.PASS)

    def test_us_operations_evidence_fails_geography(self) -> None:
        candidate = profile(evidence={
            "funding": "Raised $2M.",
            "headquarters": "Headquartered in Paris, France.",
            "us_operations": "The company has a major US office in Boston, USA.",
        })
        self.assertEqual(self.service.validate_geography(candidate)[0], ValidationStatus.FAIL)


if __name__ == "__main__":
    unittest.main()
