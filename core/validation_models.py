"""Pydantic models for deterministic Step 4 company qualification."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Three-state result used by every hard qualification criterion."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    """Evidence-quality indicator that never overrides qualification rules."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QualificationResult(BaseModel):
    """Evidence-first validation outcome for one Step 3 company profile."""

    company_name: str
    website: Optional[str] = None
    is_financially_qualified: ValidationStatus
    is_technology_qualified: ValidationStatus
    is_geographically_qualified: ValidationStatus
    is_qualified: bool
    financial_basis: Optional[str] = None
    financial_amount: Optional[float] = None
    financial_currency: Optional[str] = None
    financial_type: str = "unknown"
    financial_evidence: Optional[str] = None
    technology_evidence: Optional[str] = None
    geographic_evidence: Optional[str] = None
    validation_sources: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
