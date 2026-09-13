"""
Query planning component for TVB Agentic Company Lead Discovery.

Uses Groq LLM to dynamically generate targeted search queries based on
technology sectors, non-US geographies, and TVB's $1M-$5M funding/revenue
parameters.

The planner does NOT hardcode any company names. Companies must be
discovered dynamically through web search.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from core.models import SearchResult
from services.search_service import TavilySearchService


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("tvb.planner")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Query / source safety helpers
# ---------------------------------------------------------------------------

DISALLOWED_QUERY_PHRASES = (
    "investor list",
    "investor lists",
    "vc firms",
    "venture capital firms",
    "top investors",
    "funding list",
    "funding lists",
    "investor directory",
    "investor directories",
    "fund database",
    "startup database",
    "funded startups database",
)

FINANCIAL_QUERY_TERMS = (
    "raised",
    "funding",
    "secured",
    "revenue",
    "arr",
    "seed round",
    "funding round",
)

# These are source-level filters. They reject obvious list/directory/database
# pages before we spend a Groq research call on them.
DISALLOWED_SOURCE_URL_TERMS = (
    "/investor",
    "/investors",
    "/directory",
    "/directories",
    "/database",
    "/lists/",
    "/list/",
    "list-of-",
    "top-50",
    "top-100",
    "top-20",
    "funds",
    "funding-database",
    "/sitemap",
    "sitemap",
    "/search/",
    "/tag/",
    "/category/",
    "youtube.com/",
    "youtu.be/",
    "merriam-webster.com/",
)

DISALLOWED_SOURCE_TITLE_TERMS = (
    "top 50",
    "top 100",
    "top 20",
    "investor list",
    "investors",
    "vc funds",
    "venture capital funds",
    "investor guide",
    "funding database",
    "startup database",
    "complete database",
    "directory",
    "list of funded",
    "funded startups",
    "definition",
    "meaning",
    "latest news, reports & analysis",
)


def _normalize(value: str) -> str:
    """Normalize whitespace and case for conservative matching."""
    return " ".join((value or "").lower().split())


def _is_company_discovery_query(query: str) -> bool:
    """Return True when a query is suitable for discovering real companies."""
    normalized = _normalize(query)

    if any(phrase in normalized for phrase in DISALLOWED_QUERY_PHRASES):
        return False

    return any(term in normalized for term in FINANCIAL_QUERY_TERMS)


def _is_company_search_result(result: SearchResult) -> bool:
    """
    Return True when a search result looks like a primary company/source page.

    This filter is deliberately conservative. It is not a company list and
    does not identify specific companies; it only removes obvious investor,
    directory, list, and database sources.
    """
    url = _normalize(result.url)
    title = _normalize(result.title)
    content = _normalize(result.content or "")

    if any(term in url for term in DISALLOWED_SOURCE_URL_TERMS):
        return False

    if any(term in title for term in DISALLOWED_SOURCE_TITLE_TERMS):
        return False

    # Discovery should not spend research calls on social posts, videos,
    # personal profiles, or dictionary pages. Founder discovery has its own
    # targeted search stage, so these sources are intentionally excluded here.
    if any(domain in url for domain in (
        "youtube.com/", "youtu.be/", "facebook.com/", "instagram.com/",
        "tiktok.com/", "x.com/", "twitter.com/", "merriam-webster.com/",
    )):
        return False
    if "/in/" in url and "linkedin.com" in url:
        return False

    # Strong generic database/list wording in the snippet is also a warning.
    generic_result_terms = (
        "browse startups",
        "complete database",
        "list of startups",
        "list of funded startups",
        "investor database",
        "venture capital database",
        "funding database",
        "top investors",
        "investor directory",
    )

    if any(term in content for term in generic_result_terms):
        return False

    return True


def _dedupe_queries(queries: List[str], count: int) -> List[str]:
    """Normalize and deduplicate query strings."""
    clean: List[str] = []
    seen = set()

    for query in queries:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue

        key = normalized.lower()
        if key in seen:
            continue

        if not _is_company_discovery_query(normalized):
            continue

        seen.add(key)
        clean.append(normalized)

        if len(clean) >= max(1, count):
            break

    return clean


# ---------------------------------------------------------------------------
# Deterministic fallback planner
# ---------------------------------------------------------------------------

def _fallback_queries(
    sector: str,
    geography: str,
    count: int,
) -> List[str]:
    """Generate diverse company-focused queries when Groq planning is unavailable."""
    geography_text = " ".join(geography.replace("&", " ").replace("(", " ").replace(")", " ").split())
    expanded = {
        "Germany Nordics": ["Germany", "Sweden", "Norway", "Denmark", "Finland"],
        "Southeast Asia Singapore Indonesia Vietnam": ["Singapore", "Indonesia", "Vietnam"],
        "MENA UAE Saudi Arabia Egypt": ["UAE", "Saudi Arabia", "Egypt"],
        "Latin America Brazil Mexico Colombia": ["Brazil", "Mexico", "Colombia"],
        "Australia New Zealand": ["Australia", "New Zealand"],
    }
    geo_terms = expanded.get(geography_text, [geography_text])
    # Split grouped regions so each query has a strong geographic intent.
    geo = f'"{geo_terms[0]}"' if len(geo_terms) == 1 else f'"{geo_terms[0]}"'
    alternatives = [
        f'"{sector}" {geo} "raised $3 million" startup founder',
        f'"{sector}" {geo} "raised $4 million" startup CEO',
        f'"{sector}" {geo} "raised $2 million" software platform',
        f'"{sector}" {geo} "raised $1 million" technology company founder',
        f'"{sector}" {geo} "raised $5 million" SaaS founder',
        f'"{sector}" {geo} "annual revenue" "$3 million" software',
        f'"{sector}" {geo} "ARR" "$2 million" platform',
        f'"{sector}" {geo} "seed round" "$4M" founder',
        f'"{sector}" {geo} "pre-seed" "$1M" technology',
    ]
    # Rotate the selected country for grouped geographies to avoid repeatedly
    # querying only the first country.
    expanded_queries: List[str] = []
    for country in geo_terms[:3]:
        expanded_queries.extend([
            f'"{sector}" "{country}" startup "raised $3 million" founder',
            f'"{sector}" "{country}" startup "raised $4 million" CEO',
            f'"{sector}" "{country}" software platform "raised $2 million"',
        ])
    return _dedupe_queries(expanded_queries + alternatives, count)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PlannerError(Exception):
    """Base exception for Query Planner errors."""


# ---------------------------------------------------------------------------
# Search strategy configuration
# ---------------------------------------------------------------------------

TARGET_SECTORS = [
    "B2B SaaS",
    "AI platform",
    "developer tools",
    "data platforms",
    "fintech infrastructure",
    "cybersecurity",
    "healthcare technology",
    "education technology",
    "cloud management & DevOps",
]

TARGET_GEOGRAPHIES = [
    "United Kingdom",
    "Germany & Nordics",
    "Western Europe",
    "India",
    "Southeast Asia Singapore Indonesia Vietnam",
    "MENA UAE Saudi Arabia Egypt",
    "Latin America Brazil Mexico Colombia",
    "Australia & New Zealand",
]


# ---------------------------------------------------------------------------
# Query Planner
# ---------------------------------------------------------------------------

class QueryPlanner:
    """Generate dynamic web-search queries using Groq."""

    DEFAULT_MODEL = "openai/gpt-oss-20b"
    FALLBACK_MODEL = "openai/gpt-oss-20b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 25.0,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        configured_model = os.getenv("GROQ_MODEL")
        self.model = model or (configured_model.strip() if configured_model else "") or self.DEFAULT_MODEL
        self.timeout = timeout

        self.client: Optional[Groq] = None

        if self.api_key and self.api_key.strip() not in (
            "",
            "your_groq_api_key_here",
        ):
            try:
                self.client = Groq(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Groq planner client: %s",
                    exc,
                )

        self.search_service = TavilySearchService()
        self._planning_index = 0

    def _validate_client(self) -> None:
        """Ensure the Groq planner client is available."""
        if not self.api_key or self.api_key.strip() in (
            "",
            "your_groq_api_key_here",
        ):
            raise PlannerError(
                "GROQ_API_KEY is missing. Configure it in .env."
            )

        if self.client is None:
            try:
                self.client = Groq(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                raise PlannerError(
                    f"Could not initialize Groq planner client: {exc}"
                ) from exc

    @staticmethod
    def _clean_model_queries(value: object) -> List[str]:
        """
        Extract a clean list of query strings from common JSON layouts.
        """
        if isinstance(value, list):
            raw_queries = value
        elif isinstance(value, dict):
            raw_queries = (
                value.get("queries")
                or value.get("search_queries")
                or value.get("items")
                or []
            )
        else:
            raw_queries = []

        queries: List[str] = []

        for item in raw_queries:
            if isinstance(item, str):
                queries.append(item.strip())
            elif isinstance(item, dict):
                query = (
                    item.get("query")
                    or item.get("search_query")
                    or item.get("text")
                )
                if isinstance(query, str):
                    queries.append(query.strip())

        return queries

    @staticmethod
    def _parse_json(raw: object) -> Dict[str, object]:
        """Parse plain or fenced JSON returned by Groq."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Planner returned empty output.")

        candidate = raw.strip()

        if candidate.startswith("```"):
            candidate = candidate.replace("```json", "", 1)
            candidate = candidate.replace("```", "", 1).strip()

        decoder = json.JSONDecoder()

        for index, char in enumerate(candidate):
            if char != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Planner output did not contain a valid JSON object.")

    def _generate_with_groq(
        self,
        sector: str,
        geography: str,
        count: int,
    ) -> List[str]:
        """Generate discovery queries with Groq."""
        self._validate_client()
        assert self.client is not None

        system_prompt = (
            "You are a search-query planner for The Venture Build (TVB).\n"
            "Generate search queries that discover REAL TECHNOLOGY COMPANIES, "
            "not investors, funds, directories, databases, jobs, or list articles.\n\n"
            "TARGET COMPANY REQUIREMENTS:\n"
            "- Technology platform/company.\n"
            "- Non-US headquarters or minimal US presence.\n"
            "- Funding OR revenue between approximately USD $1M and $5M.\n"
            "- Search should find primary company pages or credible company-specific coverage.\n"
            "- Do not return company names as a fixed list.\n"
            "- Do not fabricate company names.\n\n"
            "GOOD SEARCH INTENT:\n"
            "- funding announcement for a specific startup/company\n"
            "- revenue/ARR announcement for a specific company\n"
            "- founder/CEO plus funding\n"
            "- technology company plus funding round\n\n"
            "BAD SEARCH INTENT:\n"
            "- investor list\n"
            "- VC funds\n"
            "- startup database\n"
            "- funded startup directory\n"
            "- top startups list\n"
            "- investor directory\n\n"
            "Return ONLY valid JSON in this exact structure:\n"
            '{\"queries\":[\"query 1\",\"query 2\",\"query 3\"]}\n'
        )

        user_prompt = (
            f"Sector: {sector}\n"
            f"Geography: {geography}\n"
            f"Number of queries: {count}\n"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Parse JSON defensively below instead of using provider-side
            # JSON mode, which can intermittently reject gpt-oss requests.
            temperature=0.2,
            include_reasoning=False,
            reasoning_effort="low",
            max_completion_tokens=1000,
        )

        content = response.choices[0].message.content

        data = self._parse_json(content)
        queries = self._clean_model_queries(data)

        validated = _dedupe_queries(queries, count)

        if not validated:
            raise ValueError(
                "Groq returned no usable company-discovery queries."
            )

        return validated

    def generate_queries(
        self,
        sector: Optional[str] = None,
        geography: Optional[str] = None,
        count: int = 3,
    ) -> List[str]:
        """
        Generate dynamic discovery queries.

        Groq is attempted first. If it fails or returns invalid JSON,
        deterministic company-focused fallback queries are used.
        """
        if sector:
            selected_sector = sector
        else:
            selected_sector = TARGET_SECTORS[self._planning_index % len(TARGET_SECTORS)]

        if geography:
            selected_geography = geography
        else:
            selected_geography = TARGET_GEOGRAPHIES[self._planning_index % len(TARGET_GEOGRAPHIES)]

        if sector is None or geography is None:
            self._planning_index += 1

        safe_count = max(1, min(count, 6))

        logger.info(
            "Generating discovery queries via Groq | Model: %s | Sector: '%s' | Geography: '%s'",
            self.model,
            selected_sector,
            selected_geography,
        )

        try:
            queries = self._generate_with_groq(
                selected_sector,
                selected_geography,
                safe_count,
            )

            logger.info(
                "Planner generated %d validated company-discovery queries",
                len(queries),
            )
            return queries

        except Exception as exc:
            logger.warning(
                "Groq planner call failed; using deterministic fallback queries: %s",
                exc,
            )

            fallback = _fallback_queries(
                selected_sector,
                selected_geography,
                safe_count,
            )

            if not fallback:
                raise PlannerError(
                    "Could not generate any valid company-discovery queries."
                )

            return fallback

    def search_discovered_companies(
        self,
        sector: Optional[str] = None,
        geography: Optional[str] = None,
        query_count: int = 3,
        max_results_per_query: int = 5,
    ) -> List[SearchResult]:
        """
        Generate queries, execute Tavily search, and retain only promising
        company-oriented source results.
        """
        queries = self.generate_queries(
            sector=sector,
            geography=geography,
            count=query_count,
        )

        collected: List[SearchResult] = []
        seen_urls = set()

        for query in queries:
            try:
                results = self.search_service.search(
                    query,
                    max_results=max_results_per_query,
                )
            except Exception as exc:
                logger.warning(
                    "Search failed for planner query '%s': %s",
                    query,
                    exc,
                )
                continue

            accepted = 0

            for result in results:
                if not _is_company_search_result(result):
                    logger.info(
                        "Skipping non-company source | URL: %s | Title: %s",
                        result.url,
                        result.title,
                    )
                    continue

                normalized_url = _normalize(result.url)

                if not normalized_url or normalized_url in seen_urls:
                    continue

                seen_urls.add(normalized_url)
                collected.append(result)
                accepted += 1

            logger.info(
                "Planner query source filtering | Query: %s | Accepted: %d / %d",
                query,
                accepted,
                len(results),
            )

        logger.info(
            "Planner discovery complete | Queries: %d | Accepted Sources: %d",
            len(queries),
            len(collected),
        )

        return collected