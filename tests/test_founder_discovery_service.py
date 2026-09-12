"""Deterministic Step 5 tests using source content, not live Tavily or Groq calls."""

import unittest
from unittest.mock import patch

from core.founder_models import FounderDiscoveryStatus
from core.models import CompanyProfile, SearchResult
from services.founder_discovery_service import FounderDiscoveryService, _FounderCandidate


def qualified_profile(**overrides: object) -> CompanyProfile:
    values = {
        "company_name": "Fixture Platform",
        "website": None,
        "description": "AI SaaS platform for workflow teams",
        "industry": "Enterprise software",
        "location": "Paris, France",
        "funding_amount": 2_000_000,
        "funding_currency": "USD",
        "source_urls": ["https://news.example/fixture"],
        "evidence": {"funding": "Fixture Platform raised $2M in a seed round.", "location": "Fixture Platform is Paris, France-based."},
    }
    values.update(overrides)
    return CompanyProfile(**values)


class FounderDiscoveryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FounderDiscoveryService(groq_api_key="test-key", page_fetcher=lambda _: None)
        self.profile = qualified_profile()
        self.source_url = "https://fixture.example/about"

    def candidate(self, name: str, role: str, evidence: str):
        return self.service._candidate_from_data(
            {"founder_name": name, "role": role, "evidence": evidence}, evidence, self.source_url, self.profile
        )

    def test_valid_ceo_extraction(self) -> None:
        candidate = self.candidate("Jane Doe", "CEO", "Fixture Platform CEO Jane Doe leads the company.")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.role, "CEO")

    def test_valid_cofounder_extraction(self) -> None:
        candidate = self.candidate("John Doe", "Co-Founder", "John Doe is Co-Founder of Fixture Platform.")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.role, "Co-Founder")

    def test_cto_only_person_is_rejected(self) -> None:
        candidate = self.candidate("Sam Doe", "CTO", "Sam Doe is CTO of Fixture Platform.")
        self.assertIsNone(candidate)

    def test_missing_evidence_is_not_found(self) -> None:
        result = self.service._not_found(self.profile, FounderDiscoveryStatus.NOT_FOUND)
        self.assertEqual(result.discovery_status, FounderDiscoveryStatus.NOT_FOUND)
        self.assertIsNone(result.founder_name)

    def test_contradictory_or_negative_evidence_is_rejected(self) -> None:
        candidate = self.candidate("Jane Doe", "CEO", "Jane Doe is not the CEO of Fixture Platform.")
        self.assertIsNone(candidate)

    def test_multiple_founders_prefers_ceo(self) -> None:
        founder = self.candidate("Avery Doe", "Co-Founder", "Avery Doe is Co-Founder of Fixture Platform.")
        ceo = self.candidate("Jordan Doe", "Founder & CEO", "Jordan Doe is Founder & CEO of Fixture Platform.")
        selected = self.service._select_candidate([founder, ceo])
        self.assertEqual(selected.name, "Jordan Doe")

    def test_missing_website_does_not_crash(self) -> None:
        self.assertNotIn("site:", self.service.build_queries(self.profile)[0])
        self.assertEqual(len(self.service.build_queries(self.profile)), 5)

    def test_contradictory_sources_are_uncertain_even_with_positive_candidate(self) -> None:
        class Search:
            def search(self, query, max_results=2):
                return [SearchResult(title="source", url="https://fixture.example/about", content="Fixture Platform says Jane Doe is not the CEO.")]

        service = FounderDiscoveryService(
            search_service=Search(),
            page_fetcher=lambda _: "Fixture Platform says Jane Doe is not the CEO.",
        )
        candidate = _FounderCandidate(
            name="Jane Doe", role="CEO", linkedin_url=None, source_url="https://fixture.example/about",
            evidence="Jane Doe is CEO of Fixture Platform.", role_rank=1, source_rank=0,
        )
        with patch.object(service, "_extract_candidates", return_value=[candidate]):
            result = service.discover_founder(self.profile)
        self.assertEqual(result.discovery_status, FounderDiscoveryStatus.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
