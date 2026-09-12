"""Pydantic models for Step 5 founder and CEO discovery."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class FounderDiscoveryStatus(str, Enum):
    """Outcome of evidence-backed founder research."""

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"


class FounderConfidence(str, Enum):
    """Confidence describes source quality and never fills missing evidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FounderProfile(BaseModel):
    """One selected CEO or founder supported by an actual research source."""

    company_name: str
    founder_name: Optional[str] = None
    role: Optional[str] = None
    linkedin_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None
    confidence: FounderConfidence = FounderConfidence.LOW
    discovery_status: FounderDiscoveryStatus = FounderDiscoveryStatus.NOT_FOUND
