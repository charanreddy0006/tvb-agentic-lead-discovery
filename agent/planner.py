"""Query planning component for TVB Agentic Company Lead Discovery.

Uses Groq LLM to dynamically generate targeted search queries based on
technology sectors, non-US geographies, and TVB's $1M–$5M funding parameters.
Does NOT hardcode any company names.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure Windows console handles UTF-8 characters without CharMap errors
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

# Load environment variables from project root .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

# Configure module-level logger
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


class PlannerError(Exception):
    """Base exception for Query Planner errors."""
    pass


# Curated strategic themes for autonomous search diversity (no hardcoded company names)
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


class QueryPlanner:
    """Generates dynamic search queries using Groq LLM without fixed company lists."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    FALLBACK_MODEL = "openai/gpt-oss-20b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 25.0,
    ):
        """Initialize the Query Planner.

        Args:
            api_key: Optional Groq API key. Defaults to GROQ_API_KEY from environment.
            model: Groq model name to use. Defaults to llama-3.3-70b-versatile.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout
        self.client: Optional[Groq] = None

        if not self.api_key or self.api_key.strip() in ("", "your_groq_api_key_here"):
            logger.warning(
                "GROQ_API_KEY is not configured or contains placeholder text. "
                "Query generation via LLM will require a valid key in .env."
            )
        else:
            try:
                self.client = Groq(api_key=self.api_key, timeout=self.timeout)
            except Exception as e:
                logger.error("Failed to initialize Groq client: %s", e)
                self.client = None

    def _validate_client(self) -> None:
        """Validate that Groq client is properly configured."""
        if not self.api_key or self.api_key.strip() in ("", "your_groq_api_key_here"):
            raise PlannerError(
                "GROQ_API_KEY is missing or unconfigured. Please set a valid Groq API key "
                "in your .env file (e.g. GROQ_API_KEY=gsk_xxxxxxxx)."
            )
        if self.client is None:
            try:
                self.client = Groq(api_key=self.api_key, timeout=self.timeout)
            except Exception as e:
                raise PlannerError(f"Could not connect to Groq client: {e}") from e

    def generate_queries(
        self,
        sector: Optional[str] = None,
        geography: Optional[str] = None,
        count: int = 3,
    ) -> List[str]:
        """Generate targeted web search queries dynamically using Groq LLM.

        Args:
            sector: Target technology domain (e.g., 'B2B SaaS', 'AI platform').
            geography: Non-US geographic market (e.g., 'Europe', 'India', 'MENA').
            count: Number of diverse search queries to generate (default: 3).

        Returns:
            List of generated search query strings designed for web search engines.

        Raises:
            PlannerError: If the API key is missing or Groq API call fails.
        """
        self._validate_client()
        assert self.client is not None

        selected_sector = sector or random.choice(TARGET_SECTORS)
        selected_geography = geography or random.choice(TARGET_GEOGRAPHIES)

        system_prompt = (
            "You are an autonomous research strategist for The Venture Build (TVB), "
            "an AI-powered venture catalyst and venture operating platform that helps startups "
            "and scale-ups grow through execution, market access, operator support, partner "
            "ecosystems, and capital readiness.\n"
            "TVB seeks early-stage technology platform companies meeting these strict parameters:\n"
            "1. Financials: $1,000,000 to $5,000,000 USD in funding raised or annual revenue.\n"
            "2. Business Model: Technology platform (SaaS, PaaS, Cloud, Developer Tooling, Data, AI).\n"
            "3. Location: Strictly NON-US based (UK, Europe, India, Southeast Asia, MENA, LATAM, etc.).\n\n"
            "CRITICAL RULES:\n"
            "- Do NOT output or name specific known companies.\n"
            "- Generate search engine queries designed to find recent funding announcements, "
            "press releases, startup directories, or seed rounds matching this profile.\n"
            "- Use search operators where effective (e.g. quotes, OR, site: filters, exclusion -USA).\n"
            "- Output valid JSON ONLY in this exact format: {\"queries\": [\"query 1\", \"query 2\", ...]}"
        )

        user_prompt = (
            f"Generate {count} distinct, high-yield web search queries for finding target companies:\n"
            f"- Sector Theme: {selected_sector}\n"
            f"- Geographic Region: {selected_geography}\n"
            f"- Funding Bracket: $1M to $5M USD (Seed, Seed+, Pre-Series A)\n"
            f"- Exclusion: Minimal to no US presence\n\n"
            "Remember: Return ONLY valid JSON with key 'queries'."
        )

        logger.info(
            "Generating discovery queries via Groq | Model: %s | Sector: '%s' | Geography: '%s'",
            self.model,
            selected_sector,
            selected_geography,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                raise PlannerError("Empty response received from Groq LLM.")

            parsed = json.loads(raw_content)
            queries = parsed.get("queries", [])

            if not isinstance(queries, list) or not queries:
                raise PlannerError(f"Unexpected JSON structure from Groq: {raw_content[:200]}")

            # Clean and validate queries
            clean_queries = [str(q).strip().strip('"') for q in queries if str(q).strip()]

            logger.info("Successfully generated %d search queries via Groq", len(clean_queries))
            for i, q in enumerate(clean_queries, 1):
                logger.info("  Generated Query [%d]: %s", i, q)

            return clean_queries

        except json.JSONDecodeError as exc:
            err_msg = f"Failed to parse JSON response from Groq: {exc}"
            logger.error(err_msg)
            raise PlannerError(err_msg) from exc
        except Exception as exc:
            err_msg = f"Groq API call failed: {exc}"
            logger.error(err_msg)
            raise PlannerError(err_msg) from exc


def run_discovery_test(
    sector: Optional[str] = "B2B SaaS",
    geography: Optional[str] = "Europe",
    max_results: int = 3,
) -> Dict[str, object]:
    """Verification function demonstrating the Step 2 discovery pipeline:

    Groq generates a query -> Tavily executes search -> Structured results returned.

    Args:
        sector: Target technology sector.
        geography: Target non-US geography.
        max_results: Number of search results to retrieve from Tavily.

    Returns:
        Dictionary containing generated queries and search results.
    """
    print("=" * 70)
    print("TVB Discovery Pipeline Test: Groq Query Generation -> Tavily Search")
    print("=" * 70)

    # 1. Initialize Planner and generate dynamic query
    print(f"\n[1] Initializing Groq Query Planner for: Sector='{sector}', Geography='{geography}'...")
    planner = QueryPlanner()

    try:
        queries = planner.generate_queries(sector=sector, geography=geography, count=2)
        print(f"    Generated {len(queries)} dynamic queries successfully:")
        for idx, q in enumerate(queries, 1):
            print(f"      [{idx}] {q}")
    except Exception as e:
        print(f"    [ERROR] Groq Planner Error: {e}")
        return {"status": "error", "step": "planner", "error": str(e)}

    # 2. Execute Tavily search with the first generated query
    selected_query = queries[0]
    print(f"\n[2] Executing Tavily Search for Query: '{selected_query}'...")
    search_service = TavilySearchService()

    try:
        results: List[SearchResult] = search_service.search(
            query=selected_query, max_results=max_results
        )
        print(f"    Tavily search returned {len(results)} structured results:")
        for idx, item in enumerate(results, 1):
            print(f"\n    --- Result [{idx}] ---")
            print(f"    Title:   {item.title}")
            print(f"    URL:     {item.url}")
            print(f"    Domain:  {item.domain}")
            snippet_preview = item.content.replace("\n", " ")[:140]
            print(f"    Snippet: {snippet_preview}...")

        print("\n" + "=" * 70)
        print("[SUCCESS] STEP 2 PIPELINE TEST COMPLETED")
        print("=" * 70)
        return {
            "status": "success",
            "query": selected_query,
            "results": [r.model_dump() for r in results],
        }

    except Exception as e:
        print(f"    [ERROR] Tavily Search Error: {e}")
        return {"status": "error", "step": "search", "error": str(e)}


if __name__ == "__main__":
    run_discovery_test()
