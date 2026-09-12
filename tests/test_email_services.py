"""Deterministic Step 6/7 tests; all search, Groq, and DNS work is mocked."""

import unittest

from core.email_models import EmailCandidate, EmailDiscoveryStatus, EmailVerificationStatus
from core.founder_models import FounderDiscoveryStatus, FounderProfile
from core.models import CompanyProfile
from services.email_discovery_service import EmailDiscoveryService
from services.email_verification_service import EmailVerificationService


def company() -> CompanyProfile:
    return CompanyProfile(company_name="Fixture Platform", website="https://fixture.example", description="AI SaaS platform", industry="software", location="Paris, France", funding_amount=2_000_000, funding_currency="USD", source_urls=["https://fixture.example"], evidence={"funding": "Fixture Platform raised $2M.", "location": "Fixture Platform is Paris, France-based."})


def founder() -> FounderProfile:
    return FounderProfile(company_name="Fixture Platform", founder_name="Jane Doe", role="CEO", discovery_status=FounderDiscoveryStatus.FOUND)


class EmailServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = EmailDiscoveryService(groq_api_key="test-key", page_fetcher=lambda _: None)
        self.company, self.founder = company(), founder()
        self.url = "https://fixture.example/team"

    def candidate(self, email: str, evidence: str):
        return self.service._candidate_from_data({"email": email, "evidence": evidence}, evidence, self.url, self.company, self.founder)

    def test_explicit_founder_email_is_extracted(self) -> None:
        candidate = self.candidate("jane@fixture.example", "Jane Doe, CEO of Fixture Platform, can be reached at jane@fixture.example.")
        self.assertEqual(candidate.discovery_status, EmailDiscoveryStatus.FOUND)

    def test_missing_or_guessed_email_is_rejected(self) -> None:
        self.assertIsNone(self.candidate("jane@fixture.example", "Jane Doe is CEO of Fixture Platform."))
        self.assertIsNone(self.candidate("jane@fixture.example", "Jane Doe is CEO of Fixture Platform and no email is listed."))

    def test_generic_email_is_not_founder_email(self) -> None:
        self.assertIsNone(self.candidate("info@fixture.example", "Jane Doe, CEO of Fixture Platform: info@fixture.example."))

    def test_multiple_candidates_prefer_company_domain(self) -> None:
        external = self.candidate("jane@professional.example", "Jane Doe, CEO of Fixture Platform: jane@professional.example.")
        company_domain = self.candidate("jane@fixture.example", "Jane Doe, CEO of Fixture Platform: jane@fixture.example.")
        self.assertEqual(self.service._select([external, company_domain], self.company).email, "jane@fixture.example")

    def test_verification_outcomes(self) -> None:
        invalid = EmailVerificationService(lambda _: True).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="not-an-email"))
        self.assertEqual(invalid.status, EmailVerificationStatus.INVALID)
        no_mx = EmailVerificationService(lambda _: False).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="jane@example.com", evidence="Jane Doe: jane@example.com"))
        self.assertEqual(no_mx.status, EmailVerificationStatus.INVALID)
        unknown = EmailVerificationService(lambda _: True).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="jane@example.com", evidence="Contact jane@example.com"))
        self.assertEqual(unknown.status, EmailVerificationStatus.UNKNOWN)
        verified = EmailVerificationService(lambda _: True).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="jane@example.com", evidence="Jane Doe can be reached at jane@example.com."))
        self.assertEqual(verified.status, EmailVerificationStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main()
