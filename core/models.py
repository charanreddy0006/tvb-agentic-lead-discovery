"""Core Pydantic data models for TVB Agentic Company Lead Discovery.

Step 2: SearchResult data structures.
Step 3: CompanyProfile structured research model.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, model_validator


class SearchResult(BaseModel):
    """Structured representation of a web search result item."""

    title: str = Field(..., description="Title of the search result page")
    url: str = Field(..., description="Full URL of the search result")
    content: str = Field(default="", description="Snippet, summary, or extracted text content")
    domain: str = Field(default="", description="Parsed domain or netloc of the result")
    score: Optional[float] = Field(default=None, description="Relevance score from the search provider")

    @model_validator(mode="after")
    def populate_domain_from_url(self) -> "SearchResult":
        """Ensure domain is populated from URL if not explicitly provided."""
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            # Strip port if present
            netloc = parsed.netloc.split(":")[0]
            self.domain = netloc.lower()
        return self


class CompanyProfile(BaseModel):
    """Structured profile for a discovered candidate company."""

    company_name: str = Field(..., description="Name of the company")
    website: Optional[str] = Field(default=None, description="Official company website URL")
    description: Optional[str] = Field(default=None, description="Brief description of the platform/product")
    industry: Optional[str] = Field(default=None, description="Industry or sector of the company")
    location: Optional[str] = Field(default=None, description="Headquarters city and/or country")
    funding_amount: Optional[float] = Field(default=None, description="Total funding raised or recent round amount (numeric)")
    funding_currency: Optional[str] = Field(default=None, description="Currency of funding (e.g. USD, EUR, GBP)")
    funding_stage: Optional[str] = Field(default=None, description="Funding round stage (e.g. Seed, Pre-Series A)")
    revenue_amount: Optional[float] = Field(default=None, description="Annual revenue or ARR amount (numeric)")
    revenue_currency: Optional[str] = Field(default=None, description="Currency of revenue (e.g. USD, EUR)")
    source_urls: List[str] = Field(default_factory=list, description="URLs where information was discovered")
    evidence: Dict[str, str] = Field(
        default_factory=dict,
        description="Exact supporting text quotes or excerpts for funding, revenue, and location claims",
    )

    def get_dedup_key(self) -> str:
        """Return a normalized key for deduplication based on domain or company name."""
        if self.website:
            try:
                parsed = urlparse(self.website)
                netloc = parsed.netloc.split(":")[0].lower()
                if netloc.startswith("www."):
                    netloc = netloc[4:]
                if netloc:
                    return f"domain:{netloc}"
            except Exception:
                pass

        # Fallback: normalized alphanumeric company name
        normalized_name = re.sub(r"[^a-zA-Z0-9]", "", self.company_name.lower())
        return f"name:{normalized_name}"
