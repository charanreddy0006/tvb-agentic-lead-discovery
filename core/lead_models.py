"""Final, verified-lead model used by the bounded autonomous pipeline."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QualifiedLead(BaseModel):
    company_name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    funding_amount: Optional[float] = None
    funding_currency: Optional[str] = None
    funding_stage: Optional[str] = None
    revenue_amount: Optional[float] = None
    revenue_currency: Optional[str] = None
    founder_name: str
    founder_role: str
    founder_linkedin_url: Optional[str] = None
    email: str
    email_verification_status: str
    company_source_urls: List[str] = Field(default_factory=list)
    founder_source_urls: List[str] = Field(default_factory=list)
    email_source_url: Optional[str] = None
    evidence: Dict[str, str] = Field(default_factory=dict)
    qualification_reasons: List[str] = Field(default_factory=list)
