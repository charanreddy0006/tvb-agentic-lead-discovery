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