"""Deterministic tests for bounded local performance optimizations."""

import unittest

from services.company_research_service import CompanyResearchService
from services.search_service import TavilySearchService


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload=None, text=""):
        self._payload = payload or {"results": []}
        self.text = text

    def json(self):
        return self._payload


class _SearchClient:
    def __init__(self):
        self.calls = 0

    def post(self, url, json):
        self.calls += 1
        return _Response({"results": [{"title": "Result", "url": "https://example.com", "content": "content"}]})


class _PageClient:
    def __init__(self):
        self.calls = 0

    def get(self, url, headers):
        self.calls += 1
        return _Response(text="<html><body>" + ("Useful company research text. " * 10) + "</body></html>")


class PerformanceOptimizationTests(unittest.TestCase):
    def test_search_reuses_cached_query_result(self):
        service = TavilySearchService(api_key="test-key")
        client = _SearchClient()
        service._client = client

        first = service.search("same query")
        second = service.search(" SAME QUERY ")

        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)

    def test_page_fetch_reuses_cached_url_result(self):
        service = CompanyResearchService(groq_api_key="test-key")
        client = _PageClient()
        service._http_client = client

        first = service.fetch_webpage_text("https://example.com/page")
        second = service.fetch_webpage_text("https://example.com/page")

        self.assertEqual(client.calls, 1)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()