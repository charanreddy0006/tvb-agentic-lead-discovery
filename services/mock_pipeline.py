"""Deterministic local-only data adapter for the Step 9 interface."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.lead_models import QualifiedLead


@dataclass(frozen=True)
class MockPipelineResult:
    leads: List[QualifiedLead]
    stats: Dict[str, int]
    rejection_counts: Dict[str, int]


def validate_pipeline_config(target: int, iterations: int, candidates: int) -> Tuple[bool, str]:
    if not 1 <= target <= 50:
        return False, "Target leads must be between 1 and 50."
    if iterations < 1:
        return False, "Maximum iterations must be at least 1."
    if candidates < target:
        return False, "Maximum candidates must be at least the target lead count."
    return True, ""


def _leads() -> List[QualifiedLead]:
    return [
        QualifiedLead(company_name="NovaGrid AI [DEMO]", description="Fictional AI workflow platform for energy operations.", industry="AI workflow software", website="https://novagrid.demo", location="Berlin, Germany", funding_amount=2_400_000, funding_currency="USD", funding_stage="Seed", founder_name="Maya Chen", founder_role="Co-Founder & CEO", founder_linkedin_url="https://linkedin.example/maya-chen", email="maya.chen@novagrid.demo", email_verification_status="VERIFIED", company_source_urls=["https://novagrid.demo/news"], founder_source_urls=["https://novagrid.demo/team"], email_source_url="https://novagrid.demo/team", evidence={"funding": "[DEMO] NovaGrid AI raised $2.4M in a seed round.", "founder": "[DEMO] Maya Chen is Co-Founder & CEO.", "email": "[DEMO] Maya Chen can be reached at maya.chen@novagrid.demo."}),
        QualifiedLead(company_name="SecureFlow [DEMO]", description="Fictional cybersecurity SaaS platform for access workflows.", industry="Cybersecurity SaaS", website="https://secureflow.demo", location="London, UK", funding_amount=3_100_000, funding_currency="USD", funding_stage="Pre-Seed", founder_name="Alex Morgan", founder_role="Founder & CEO", email="alex@secureflow.demo", email_verification_status="VERIFIED", company_source_urls=["https://secureflow.demo/press"], founder_source_urls=["https://secureflow.demo/about"], email_source_url="https://secureflow.demo/about", evidence={"funding": "[DEMO] SecureFlow raised $3.1M.", "founder": "[DEMO] Alex Morgan is Founder & CEO.", "email": "[DEMO] Alex Morgan: alex@secureflow.demo."}),
    ]


def run_mock_pipeline(target: int, iterations: int, candidates: int) -> MockPipelineResult:
    """Return fictional final-lead structures without API, DNS, or environment use."""
    valid, message = validate_pipeline_config(target, iterations, candidates)
    if not valid:
        raise ValueError(message)
    leads = _leads()[:target]
    return MockPipelineResult(leads, {"discovered": 8, "researched": 6, "qualified": 4, "founders": 3, "emails": 2, "verified": len(leads), "duplicates": 1, "failures": 0}, {"financial": 1, "technology": 1, "geography": 1, "founder_not_found": 1, "email_not_found": 1, "email_verification_unknown_or_invalid": 1})
