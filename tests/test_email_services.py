"""Deterministic Step 6/7 tests; all search, Groq, and DNS work is mocked."""

import unittest
from threading import Barrier

from core.email_models import EmailCandidate, EmailDiscoveryStatus, EmailVerificationStatus
from core.founder_models import FounderDiscoveryStatus, FounderProfile
from core.models import CompanyProfile, SearchResult
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

    def test_external_only_email_is_rejected(self) -> None:
        external = self.candidate("jane@professional.example", "Jane Doe, CEO of Fixture Platform: jane@professional.example.")
        self.assertIsNone(self.service._select([external], self.company))

    def test_verification_outcomes(self) -> None:
        invalid = EmailVerificationService(lambda _: True).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="not-an-email"))
        self.assertEqual(invalid.status, EmailVerificationStatus.INVALID)
        no_mx = EmailVerificationService(lambda _: False).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="jane@example.com", evidence="Jane Doe: jane@example.com"))
        self.assertEqual(no_mx.status, EmailVerificationStatus.INVALID)
        unknown = EmailVerificationService(lambda _: True).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="jane@example.com", evidence="Contact jane@example.com"))
        self.assertEqual(unknown.status, EmailVerificationStatus.UNKNOWN)
        verified = EmailVerificationService(lambda _: True).verify(EmailCandidate(company_name="x", founder_name="Jane Doe", founder_role="CEO", email="jane@example.com", evidence="Jane Doe can be reached at jane@example.com."))
        self.assertEqual(verified.status, EmailVerificationStatus.VERIFIED)

    def test_email_search_is_bounded_concurrent_and_ordered(self) -> None:
        class ConcurrentSearch:
            def __init__(self):
                self.barrier = Barrier(3)

            def search(self, query, max_results=2):
                self.barrier.wait(timeout=2)
                return [SearchResult(title=query, url=f"https://{query}.example", content="source")]

        service = EmailDiscoveryService(
            groq_api_key="test-key", search_service=ConcurrentSearch(), max_workers=3,
        )
        results = service._search_queries(["one", "two", "three"])
        self.assertEqual([result.title for result in results], ["one", "two", "three"])

    def test_email_search_isolates_failures_and_deduplicates_urls(self) -> None:
        class Search:
            def search(self, query, max_results=2):
                if query == "bad":
                    raise RuntimeError("search failed")
                return [SearchResult(title=query, url="https://same.example", content="source")]

        service = EmailDiscoveryService(groq_api_key="test-key", search_service=Search())
        results = service._search_queries(["first", "bad", "second"])
        self.assertEqual([result.title for result in results], ["first"])

    def test_search_failures_preserve_not_found_status(self) -> None:
        class FailingSearch:
            def search(self, query, max_results=2):
                raise RuntimeError("search unavailable")

        service = EmailDiscoveryService(
            groq_api_key="test-key", search_service=FailingSearch(), page_fetcher=lambda _: None,
        )
        result = service.discover_email(self.company, self.founder)
        self.assertEqual(result.discovery_status, EmailDiscoveryStatus.NOT_FOUND)

    def test_email_worker_count_is_clamped(self) -> None:
        self.assertEqual(EmailDiscoveryService(groq_api_key="test-key", max_workers=0).max_workers, 1)
        self.assertEqual(EmailDiscoveryService(groq_api_key="test-key", max_workers=20).max_workers, 4)


if __name__ == "__main__":
    unittest.main()
