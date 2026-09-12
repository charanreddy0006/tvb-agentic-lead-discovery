"""Deterministic tests for source-backed company research extraction."""

import json
import unittest

from core.models import SearchResult
from services.company_research_service import CompanyResearchService


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content: str):
        self.content = content

    def create(self, **kwargs):
        return _Response(self.content)


class _Client:
    def __init__(self, content: str):
        self.chat = type("Chat", (), {"completions": _Completions(content)})()


class CompanyResearchServiceTests(unittest.TestCase):
    def _service_for_payload(self, payload: object) -> CompanyResearchService:
        service = CompanyResearchService(groq_api_key="test-key")
        service.client = _Client(payload if isinstance(payload, str) else json.dumps(payload))
        return service

    def test_preserves_source_backed_evidence_and_validates_revenue_currency(self) -> None:
        source = (
            "Acme Platform raised USD 2 million total. It sells a cloud platform. "
            "Acme Platform is headquartered in Paris, France. It has no US operations. "
            "Annual revenue reached $2M."
        )
        payload = {
            "is_company": True,
            "company_name": "Acme Platform",
            "website": None,
            "description": "Cloud platform",
            "industry": "Software",
            "location": "Paris, France",
            "funding_amount": 2_000_000,
            "funding_currency": "EUR",
            "revenue_amount": 2_000_000,
            "revenue_currency": "EUR",
            "evidence": {
                "funding": "Acme Platform raised USD 2 million total.",
                "revenue": "Annual revenue reached $2M.",
                "technology": "It sells a cloud platform.",
                "headquarters": "Acme Platform is headquartered in Paris, France.",
                "location": "Acme Platform is headquartered in Paris, France.",
                "us_operations": "It has no US operations.",
            },
        }
        service = CompanyResearchService(groq_api_key="test-key")
        service.client = _Client(json.dumps(payload))
        profile = service._extract_from_text(source, "https://source.example/article", "source.example")

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.funding_currency, "USD")
        self.assertEqual(profile.revenue_currency, "USD")
        self.assertEqual(profile.evidence["technology"], "It sells a cloud platform.")
        self.assertEqual(profile.evidence["headquarters"], "Acme Platform is headquartered in Paris, France.")
        self.assertEqual(profile.evidence["us_operations"], "It has no US operations.")

    def test_accepts_fenced_json_with_extra_text_and_missing_evidence(self) -> None:
        source = "Acme Platform sells a SaaS platform and is headquartered in Berlin, Germany."
        payload = {
            "is_company": True,
            "company_name": "Acme Platform",
            "description": "SaaS platform",
            "industry": "Software",
            "location": "Berlin, Germany",
            "funding_amount": None,
            "funding_currency": None,
            "revenue_amount": None,
            "revenue_currency": None,
            "evidence": {
                "funding": None,
                "revenue": None,
                "technology": "Acme Platform sells a SaaS platform",
                "headquarters": "headquartered in Berlin, Germany",
                "location": "headquartered in Berlin, Germany",
                "us_operations": None,
            },
        }
        fenced = "The structured result is below:\n```json\n" + json.dumps(payload) + "\n```\nEnd."
        profile = self._service_for_payload(fenced)._extract_from_text(source, "https://source.example", "source.example")

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.company_name, "Acme Platform")
        self.assertNotIn("funding", profile.evidence)
        self.assertEqual(profile.evidence["technology"], "Acme Platform sells a SaaS platform")

    def test_missing_evidence_object_does_not_fail_company_extraction(self) -> None:
        source = "Acme Platform is a SaaS company headquartered in Berlin, Germany."
        payload = {"is_company": True, "company_name": "Acme Platform", "description": "SaaS company"}
        profile = self._service_for_payload(payload)._extract_from_text(source, "https://source.example", "source.example")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.evidence, {})

    def test_non_company_extraction_increments_diagnostic(self) -> None:
        service = CompanyResearchService(groq_api_key="test-key")
        service.client = _Client(json.dumps({"is_company": False}))
        self.assertIsNone(service._extract_from_text("A generic list of tools.", "https://source.example", "source.example"))
        self.assertEqual(service.consume_diagnostics()["research_non_company"], 1)

    def test_insufficient_snippet_increments_diagnostic(self) -> None:
        service = CompanyResearchService(groq_api_key="test-key")
        service.fetch_webpage_text = lambda _: None
        result = service.research_search_result(SearchResult(title="short", url="https://source.example", content="short"))
        self.assertIsNone(result)
        self.assertEqual(service.consume_diagnostics()["research_insufficient_text"], 1)


if __name__ == "__main__":
    unittest.main()