"""Pydantic models for evidence-backed email discovery and verification."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EmailDiscoveryStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"


class EmailVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class EmailCandidate(BaseModel):
    company_name: str
    founder_name: str
    founder_role: str
    email: Optional[str] = None
    source_url: Optional[str] = None
    evidence: Optional[str] = None
    source_type: Optional[str] = None
    discovery_status: EmailDiscoveryStatus = EmailDiscoveryStatus.NOT_FOUND
    confidence: str = "low"


class EmailVerificationResult(BaseModel):
    email: Optional[str] = None
    status: EmailVerificationStatus
    reason: str
    evidence: Optional[str] = None
    verification_method: str
