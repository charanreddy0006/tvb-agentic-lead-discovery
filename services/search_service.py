"""Web search service for TVB Agentic Company Lead Discovery.

Integrates with Tavily Search API to retrieve structured web results
without relying on static company lists.
"""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# Ensure Windows console handles UTF-8 characters without CharMap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx
from dotenv import load_dotenv

from core.models import SearchResult

# Load environment variables from project root .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

# Configure module-level logger
logger = logging.getLogger("tvb.search_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SearchServiceError(Exception):
    """Base exception for search service errors."""
    pass


class TavilySearchService:
    """Reusable search client for the Tavily API."""

    TAVILY_API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 20.0):
        """Initialize the Tavily search service.

        Args:
            api_key: Optional Tavily API key. If not provided, reads from TAVILY_API_KEY.
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._client_lock = threading.Lock()
        self._cache: dict[tuple[str, int, str, tuple[str, ...], tuple[str, ...]], List[SearchResult]] = {}
        self._cache_lock = threading.Lock()

        if not self.api_key or self.api_key.strip() in ("", "your_tavily_api_key_here"):
            logger.warning(
                "TAVILY_API_KEY is not configured or contains placeholder text. "
                "Web search queries will fail until a valid key is provided in .env."
            )

    def _validate_credentials(self) -> None:
        """Validate that a usable API key is present."""
        if not self.api_key or self.api_key.strip() in ("", "your_tavily_api_key_here"):
            raise SearchServiceError(
                "TAVILY_API_KEY is missing or invalid. Please configure a valid key "
                "in your .env file (e.g. TAVILY_API_KEY=tvly-xxxxxxxx)."
            )

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Execute a search query via Tavily and return structured SearchResult objects.

        Args:
            query: The search query string.
            max_results: Maximum number of search results to retrieve (default: 5).
            search_depth: 'basic' or 'advanced' depth.
            include_domains: Optional list of domains to restrict search to.
            exclude_domains: Optional list of domains to exclude.

        Returns:
            A list of SearchResult models containing title, URL, snippet, and domain.

        Raises:
            SearchServiceError: If the API key is missing, invalid, or an API error occurs.
        """
        self._validate_credentials()

        if not query or not query.strip():
            logger.warning("Empty search query provided to SearchService. Returning empty list.")
            return []

        cache_key = (
            query.strip().lower(), max_results, search_depth,
            tuple(include_domains or ()), tuple(exclude_domains or ()),
        )
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        payload = {
            "api_key": self.api_key,
            "query": query.strip(),
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        logger.info(
            "Executing search | Provider: Tavily | Query: '%s' | Max Results: %d",
            query,
            max_results,
        )

        try:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=self.timeout)
                client = self._client
            response = client.post(self.TAVILY_API_URL, json=payload)

            if response.status_code == 401:
                err_msg = "Tavily API Authentication Failed (401 Unauthorized). Check your TAVILY_API_KEY."
                logger.error(err_msg)
                raise SearchServiceError(err_msg)

            if response.status_code == 429:
                err_msg = "Tavily API Rate Limit Exceeded (429 Too Many Requests)."
                logger.error(err_msg)
                raise SearchServiceError(err_msg)

            if response.status_code != 200:
                err_msg = (
                    f"Tavily API Error (HTTP {response.status_code}): {response.text[:300]}"
                )
                logger.error(err_msg)
                raise SearchServiceError(err_msg)

            data = response.json()
            raw_results = data.get("results", [])

            structured_results: List[SearchResult] = []
            for item in raw_results:
                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", "")
                score = item.get("score")

                domain = ""
                if url:
                    parsed = urlparse(url)
                    domain = parsed.netloc.split(":")[0].lower()

                result_obj = SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    domain=domain,
                    score=score,
                )
                structured_results.append(result_obj)

            if not structured_results:
                logger.warning(
                    "Search returned 0 results | Provider: Tavily | Query: '%s'", query
                )
            else:
                logger.info(
                    "Search completed successfully | Provider: Tavily | Results Found: %d",
                    len(structured_results),
                )

            with self._cache_lock:
                self._cache[cache_key] = list(structured_results)
            return structured_results

        except httpx.TimeoutException as exc:
            err_msg = f"Tavily search request timed out after {self.timeout}s: {exc}"
            logger.error(err_msg)
            raise SearchServiceError(err_msg) from exc
        except httpx.RequestError as exc:
            err_msg = f"Network connection error contacting Tavily API: {exc}"
            logger.error(err_msg)
            raise SearchServiceError(err_msg) from exc
        except Exception as exc:
            if isinstance(exc, SearchServiceError):
                raise
            err_msg = f"Unexpected error during Tavily search: {exc}"
            logger.error(err_msg)
            raise SearchServiceError(err_msg) from exc


# Module entry point for direct testing
if __name__ == "__main__":
    print("=== Testing Tavily Search Service Directly ===")
    test_query = "European B2B SaaS startup raised $2M seed funding"
    service = TavilySearchService()
    try:
        results = service.search(test_query, max_results=3)
        print(f"\nFound {len(results)} results for query: '{test_query}'\n")
        for i, res in enumerate(results, 1):
            print(f"[{i}] {res.title}")
            print(f"    URL:    {res.url}")
            print(f"    Domain: {res.domain}")
            print(f"    Snippet: {res.content[:150]}...\n")
    except Exception as e:
        print(f"\n[Test Result] Expected handling: {e}")
