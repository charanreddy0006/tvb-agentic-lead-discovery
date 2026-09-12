"""Query planning component for TVB Agentic Company Lead Discovery.

Uses Groq LLM to dynamically generate targeted search queries based on
technology sectors, non-US geographies, and TVB's $1M-$5M funding/revenue
parameters.

The planner does NOT hardcode any company names. Companies must be
discovered dynamically through web search.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure Windows console handles UTF-8 characters without CharMap errors.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from groq import Groq

from core.models import SearchResult
from services.search_service import TavilySearchService


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load project-level .env first, then the default environment.
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
# Query safety / deterministic fallback helpers
# ---------------------------------------------------------------------------

# These are query-level exclusions, not company lists. They reduce wasted
# Tavily calls on pages that are clearly about investors rather than companies.
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
)

FINANCIAL_QUERY_TERMS = (
    "raised",
    "funding",
    "secured",
    "revenue",
    "arr",
)


def _is_company_discovery_query(query: str) -> bool:
    """Return True when a generated query is suitable for company discovery."""
    normalized = " ".join(query.lower().split())

    if any(phrase in normalized for phrase in DISALLOWED_QUERY_PHRASES):
        return False

    return any(term in normalized for term in FINANCIAL_QUERY_TERMS)


def _fallback_queries(
    sector: str,
    geography: str,
    count: int,
) -> List[str]:
    """Generate deterministic discovery queries when Groq cannot return JSON.

    The fallback intentionally contains no company names. It is only a set of
    search strategies that lets the autonomous pipeline continue when the LLM
    planner has a transient/API/JSON failure.
    """
    templates = [
        (
            f'"{sector}" platform SaaS startup '
            f'("raised $2 million" OR "raised $3 million" OR "raised $4 million" '
            f'OR "raised $5 million") "{geography}" '
            '-investor -investors -"venture capital" -jobs -careers -directory'
        ),
        (
            f'"{sector}" software company platform '
            f'("secured $2 million" OR "secured $3 million" OR "secured $4 million" '
            f'OR "secured $5 million") "{geography}" '
            '-investor -investors -"venture capital" -jobs -careers -directory'
        ),
        (
            f'"{sector}" SaaS platform "{geography}" '
            f'("annual revenue $2 million" OR "annual revenue $3 million" '
            f'OR "annual revenue $4 million" OR "annual revenue $5 million" '
            f'OR "ARR $2 million" OR "ARR $3 million" OR "ARR $4 million" '
            f'OR "ARR $5 million") '
            '-investor -investors -"venture capital" -jobs -careers -directory'
        ),
        (
            f'"{sector}" technology platform "{geography}" '
            f'("funding" OR "raised" OR "revenue" OR "ARR") '
            f'company startup SaaS '
            '-investor -investors -"venture capital" -jobs -careers -directory'
        ),
        (
            f'"{sector}" software platform "{geography}" '
            f'("seed round" OR "pre-Series A" OR "funding round") '
            f'("$1M" OR "$2M" OR "$3M" OR "$4M" OR "$5M") '
            '-investor -investors -"venture capital" -jobs -careers -directory'
        ),
    ]

    clean: List[str] = []
    for query in templates:
        if query not in clean and _is_company_discovery_query(query):
            clean.append(query)
        if len(clean) >= max(1, count):
            break

    return clean


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """Base exception for Query Planner errors."""


# ---------------------------------------------------------------------------
# Search strategy configuration
# ---------------------------------------------------------------------------

# Curated strategic themes for autonomous search diversity.
# These are sector themes, NOT company names.
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
    "Southeast Asia (Singapore, Indonesia, Vietnam)",
    "MENA (UAE, Saudi Arabia, Egypt)",
    "Latin America (Brazil, Mexico, Colombia)",
    "Australia & New Zealand",
]


# ---------------------------------------------------------------------------
# Query Planner
# ---------------------------------------------------------------------------


class QueryPlanner:
    """Generate dynamic web-search queries using Groq.

    The planner intentionally does not contain a fixed company list.
    Each invocation can use different sector/geography combinations and
    asks the LLM to produce multiple search strategies.
    """

    # 20B is the safer default for the current Groq usage budget.
    DEFAULT_MODEL = "openai/gpt-oss-20b"

    # Kept for compatibility with existing code/tests.
    FALLBACK_MODEL = "openai/gpt-oss-20b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 25.0,
    ):
        """Initialize the Query Planner.

        Args:
            api_key: Optional Groq API key. Defaults to GROQ_API_KEY.
            model: Optional Groq model override.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        # Respect an explicitly supplied model or GROQ_MODEL.
        # If GROQ_MODEL is blank, fall back to the safe 20B default.
        configured_model = os.getenv("GROQ_MODEL")

        self.model = (
            model
            or configured_model
            or self.DEFAULT_MODEL
        )

        self.timeout = timeout
        self.client: Optional[Groq] = None

        if not self.api_key or self.api_key.strip() in (
            "",
            "your_groq_api_key_here",
        ):
            logger.warning(
                "GROQ_API_KEY is not configured or contains placeholder text. "
                "Query generation via LLM will require a valid key."
            )
        else:
            try:
                self.client = Groq(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                logger.error(
                    "Failed to initialize Groq client: %s",
                    exc,
                )
                self.client = None

    # -----------------------------------------------------------------------
    # Client validation
    # -----------------------------------------------------------------------

    def _validate_client(self) -> None:
        """Validate that the Groq client is properly configured."""

        if not self.api_key or self.api_key.strip() in (
            "",
            "your_groq_api_key_here",
        ):
            raise PlannerError(
                "GROQ_API_KEY is missing or unconfigured. "
                "Please set a valid Groq API key."
            )

        if self.client is None:
            try:
                self.client = Groq(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                raise PlannerError(
                    f"Could not connect to Groq client: {exc}"
                ) from exc

    # -----------------------------------------------------------------------
    # Dynamic query generation
    # -----------------------------------------------------------------------

    def generate_queries(
        self,
        sector: Optional[str] = None,
        geography: Optional[str] = None,
        count: int = 3,
    ) -> List[str]:
        """Generate targeted web-search queries dynamically using Groq.

        Args:
            sector:
                Target technology domain, such as B2B SaaS,
                cybersecurity, AI platform, etc.

            geography:
                Non-US geographic market.

            count:
                Number of distinct search queries requested.

        Returns:
            A list of cleaned search query strings.

        Raises:
            PlannerError:
                If the Groq API is unavailable or returns invalid data.
        """

        self._validate_client()
        assert self.client is not None

        # Randomized themes provide discovery diversity when the pipeline
        # doesn't explicitly provide a sector or geography.
        selected_sector = (
            sector
            or random.choice(TARGET_SECTORS)
        )

        selected_geography = (
            geography
            or random.choice(TARGET_GEOGRAPHIES)
        )

        # -------------------------------------------------------------------
        # System prompt
        # -------------------------------------------------------------------

        system_prompt = (
            "You are an autonomous research strategist for The Venture Build "
            "(TVB), an AI-powered venture catalyst and venture operating "
            "platform that helps startups and scale-ups grow through execution, "
            "market access, operator support, partner ecosystems, and capital "
            "readiness.\n\n"

            "Your task is to generate high-yield web search queries that "
            "DISCOVER previously unknown technology companies. "
            "Do not provide company names yourself. "
            "The search engine must discover the companies dynamically.\n\n"

            "TARGET COMPANY REQUIREMENTS:\n"
            "1. Financial: total funding raised OR annual revenue/ARR must be "
            "explicitly between $1,000,000 and $5,000,000 USD.\n"
            "2. Business: the company must operate a real technology product, "
            "software platform, SaaS, cloud platform, AI platform, data platform, "
            "developer tool, cybersecurity platform, or fintech infrastructure.\n"
            "3. Geography: headquarters must be outside the United States, with "
            "minimal or no meaningful US operations.\n\n"

            "QUERY DESIGN RULES:\n"
            "- Generate distinct queries using different search strategies.\n"
            "- At least one query must strongly target explicit funding evidence.\n"
            "- At least one query must strongly target revenue or ARR evidence.\n"
            "- At least one query must strongly target the technology/platform "
            "and non-US headquarters.\n"
            "- Prefer exact financial phrases such as '$2 million', "
            "'$3 million', '$4 million', '$5 million', 'USD 2 million', "
            "'USD 3 million', 'USD 4 million', 'USD 5 million', '$2M', '$3M', "
            "'$4M', '$5M'.\n"
            "- Include terms such as 'raised', 'funding', 'seed round', "
            "'pre-seed', 'pre-Series A', 'annual revenue', 'ARR', or 'revenue'.\n"
            "- Include technology terms such as 'platform', 'SaaS', 'software', "
            "'cloud platform', 'AI platform', 'data platform', or "
            "'cybersecurity platform'.\n"
            "- Include geographic terms from the supplied region.\n"
            "- Prefer company-specific evidence pages, funding announcements, "
            "press releases, official company pages, investor announcements, "
            "and reputable startup/company databases.\n"
            "- Avoid queries primarily intended to find investor lists, VC lists, "
            "directories, jobs pages, rankings, comparison pages, or generic "
            "articles.\n"
            "- Avoid results about individual investors rather than operating "
            "companies.\n"
            "- Use negative search terms when useful, such as "
            "-investors -VC -venture -jobs -careers -directory -list.\n"
            "- Do not rely on a single website, database, or source.\n"
            "- Do not hardcode company names.\n"
            "- Do not invent funding, revenue, company names, or URLs.\n"
            "- Queries must be natural web-search strings, not questions requiring "
            "the search engine to answer.\n"
            "- Return ONLY valid JSON in this exact format: "
            "{\"queries\": [\"query 1\", \"query 2\", \"query 3\"]}"
        )

        # -------------------------------------------------------------------
        # User prompt
        # -------------------------------------------------------------------

        user_prompt = (
            f"Generate {count} distinct, high-yield discovery queries.\n\n"

            f"SECTOR THEME: {selected_sector}\n"
            f"GEOGRAPHIC REGION: {selected_geography}\n"
            "FINANCIAL RANGE: $1M-$5M USD total funding OR annual revenue/ARR\n\n"

            "Create a balanced set of queries using different approaches:\n"
            "A. Funding-first discovery: find companies with explicit "
            "$1M-$5M USD funding announcements.\n"
            "B. Revenue-first discovery: find companies with explicit "
            "$1M-$5M USD annual revenue or ARR evidence.\n"
            "C. Platform-first discovery: find non-US technology platforms "
            "and combine the search with funding/revenue evidence.\n\n"

            "The queries should favor actual operating companies and "
            "evidence-rich pages rather than investor lists or generic "
            "startup articles.\n\n"

            "Return ONLY valid JSON with a 'queries' array."
        )

        logger.info(
            "Generating discovery queries via Groq | "
            "Model: %s | Sector: '%s' | Geography: '%s'",
            self.model,
            selected_sector,
            selected_geography,
        )

        # -------------------------------------------------------------------
        # Groq request
        # -------------------------------------------------------------------

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_object",
                },
                temperature=0.3,
                include_reasoning=False,
                max_completion_tokens=700,
            )

            raw_content = response.choices[0].message.content

            if not raw_content:
                raise PlannerError(
                    "Empty response received from Groq LLM."
                )

            # ---------------------------------------------------------------
            # Parse JSON
            # ---------------------------------------------------------------

            parsed = json.loads(raw_content)

            queries = parsed.get("queries", [])

            if not isinstance(queries, list) or not queries:
                raise PlannerError(
                    "Unexpected JSON structure from Groq: "
                    f"{raw_content[:200]}"
                )

            # ---------------------------------------------------------------
            # Clean and validate queries
            # ---------------------------------------------------------------

            clean_queries: List[str] = []

            for query in queries:
                query_text = str(query).strip().strip('"')

                if not query_text:
                    continue

                # Reject query strategies that are clearly aimed at investor
                # lists/directories or lack any financial evidence signal.
                if not _is_company_discovery_query(query_text):
                    logger.info(
                        "Discarding unsuitable planner query: %s",
                        query_text,
                    )
                    continue

                # Avoid accidental duplicates while preserving order.
                if query_text not in clean_queries:
                    clean_queries.append(query_text)

            if not clean_queries:
                logger.warning(
                    "Groq returned no suitable company-discovery queries; "
                    "using deterministic fallback."
                )
                clean_queries = _fallback_queries(
                    selected_sector,
                    selected_geography,
                    count,
                )

            if not clean_queries:
                raise PlannerError(
                    "Unable to generate usable company-discovery queries."
                )

            logger.info(
                "Successfully generated %d search queries via Groq/fallback",
                len(clean_queries),
            )

            for index, query in enumerate(
                clean_queries,
                start=1,
            ):
                logger.info(
                    "  Generated Query [%d]: %s",
                    index,
                    query,
                )

            return clean_queries

        # -------------------------------------------------------------------
        # Error handling
        # -------------------------------------------------------------------

        except json.JSONDecodeError as exc:
            logger.warning(
                "Groq planner returned invalid JSON; using deterministic "
                "fallback queries: %s",
                exc,
            )
            fallback = _fallback_queries(
                selected_sector,
                selected_geography,
                count,
            )
            if fallback:
                return fallback

            raise PlannerError(
                f"Failed to parse JSON response from Groq: {exc}"
            ) from exc

        except Exception as exc:
            logger.warning(
                "Groq planner call failed; using deterministic fallback "
                "queries: %s",
                exc,
            )
            fallback = _fallback_queries(
                selected_sector,
                selected_geography,
                count,
            )
            if fallback:
                return fallback

            raise PlannerError(
                f"Groq API call failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Standalone discovery test
# ---------------------------------------------------------------------------


def run_discovery_test(
    sector: Optional[str] = "B2B SaaS",
    geography: Optional[str] = "Europe",
    max_results: int = 3,
) -> Dict[str, object]:
    """Run a small planner -> Tavily discovery verification.

    This demonstrates:

        Groq → dynamic query generation → Tavily → structured results

    Args:
        sector:
            Target technology sector.

        geography:
            Target non-US geography.

        max_results:
            Number of Tavily results to retrieve.

    Returns:
        Dictionary containing test status, query, and search results.
    """

    print("=" * 70)
    print(
        "TVB Discovery Pipeline Test: "
        "Groq Query Generation -> Tavily Search"
    )
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Initialize planner and generate dynamic queries
    # -----------------------------------------------------------------------

    print(
        f"\n[1] Initializing Groq Query Planner for: "
        f"Sector='{sector}', Geography='{geography}'..."
    )

    planner = QueryPlanner()

    try:
        queries = planner.generate_queries(
            sector=sector,
            geography=geography,
            count=2,
        )

        print(
            f"    Generated {len(queries)} dynamic queries successfully:"
        )

        for index, query in enumerate(
            queries,
            start=1,
        ):
            print(
                f"      [{index}] {query}"
            )

    except Exception as exc:
        print(
            f"    [ERROR] Groq Planner Error: {exc}"
        )

        return {
            "status": "error",
            "step": "planner",
            "error": str(exc),
        }

    # -----------------------------------------------------------------------
    # 2. Execute Tavily search using first generated query
    # -----------------------------------------------------------------------

    selected_query = queries[0]

    print(
        f"\n[2] Executing Tavily Search for Query: "
        f"'{selected_query}'..."
    )

    search_service = TavilySearchService()

    try:
        results: List[SearchResult] = search_service.search(
            query=selected_query,
            max_results=max_results,
        )

        print(
            f"    Tavily search returned "
            f"{len(results)} structured results:"
        )

        for index, item in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n    --- Result [{index}] ---"
            )

            print(
                f"    Title:   {item.title}"
            )

            print(
                f"    URL:     {item.url}"
            )

            print(
                f"    Domain:  {item.domain}"
            )

            snippet_preview = (
                item.content
                .replace("\n", " ")
                [:140]
            )

            print(
                f"    Snippet: {snippet_preview}..."
            )

        print(
            "\n" + "=" * 70
        )

        print(
            "[SUCCESS] STEP 2 PIPELINE TEST COMPLETED"
        )

        print(
            "=" * 70
        )

        return {
            "status": "success",
            "query": selected_query,
            "results": [
                result.model_dump()
                for result in results
            ],
        }

    except Exception as exc:
        print(
            f"    [ERROR] Tavily Search Error: {exc}"
        )

        return {
            "status": "error",
            "step": "search",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_discovery_test()