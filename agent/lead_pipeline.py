"""Steps 6-8 bounded autonomous pipeline for final verified TVB leads."""

import argparse
import csv
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from agent.planner import QueryPlanner
from core.email_models import EmailDiscoveryStatus, EmailVerificationStatus
from core.founder_models import FounderDiscoveryStatus
from core.lead_models import QualifiedLead
from core.models import CompanyProfile
from services.company_research_service import CompanyResearchService
from services.email_discovery_service import EmailDiscoveryService
from services.email_verification_service import EmailVerificationService
from services.founder_discovery_service import FounderDiscoveryService
from services.qualification_service import QualificationService
from services.search_service import TavilySearchService

logger = logging.getLogger("tvb.lead_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LeadPipeline:
    """Safety-limited orchestration that counts only fully verified founder leads."""

    def __init__(self, target_leads: int = 15, max_iterations: int = 20, max_candidates: int = 150,
                 planner: Optional[QueryPlanner] = None, search_service: Optional[TavilySearchService] = None,
                 research_service: Optional[CompanyResearchService] = None,
                 qualification_service: Optional[QualificationService] = None,
                 founder_service: Optional[FounderDiscoveryService] = None,
                 email_service: Optional[EmailDiscoveryService] = None,
                 verification_service: Optional[EmailVerificationService] = None,
                 max_workers: int = 4) -> None:
        self.target_leads = target_leads
        self.max_iterations = max_iterations
        self.max_candidates = max_candidates
        self.max_workers = max(1, min(max_workers, 5))
        self.planner = planner or QueryPlanner()
        self.search_service = search_service or TavilySearchService()
        self.research_service = research_service or CompanyResearchService(max_workers=self.max_workers)
        self.qualification_service = qualification_service or QualificationService()
        self.founder_service = founder_service or FounderDiscoveryService(search_service=self.search_service)
        self.email_service = email_service or EmailDiscoveryService(search_service=self.search_service)
        self.verification_service = verification_service or EmailVerificationService()
        self.stats: Dict[str, int] = {key: 0 for key in (
            "discovered", "researched", "qualified", "founders", "emails", "verified",
            "duplicates", "failures", "research_fetch_failures", "research_insufficient_text",
            "research_non_company", "research_extraction_failures", "duplicate_companies",
            "research_api_failures", "research_json_failures",
            "financial_unknown", "financial_fail", "technology_unknown", "technology_fail",
            "geography_unknown", "geography_fail",
        )}
        self.last_research_error: Optional[str] = None

    @staticmethod
    def _lead_key(profile: CompanyProfile) -> str:
        return profile.get_dedup_key()

    @staticmethod
    def _eligible_founder_role(role: Optional[str]) -> bool:
        return bool(role and re.search(r"\b(?:ceo|chief executive officer|co[- ]?founder|founder)\b", role, re.I))

    def _to_lead(self, company: CompanyProfile, founder, email, verification, qualification) -> QualifiedLead:
        return QualifiedLead(
            company_name=company.company_name, description=company.description, industry=company.industry,
            website=company.website, location=company.location, funding_amount=company.funding_amount,
            funding_currency=company.funding_currency, funding_stage=company.funding_stage,
            revenue_amount=company.revenue_amount, revenue_currency=company.revenue_currency,
            founder_name=founder.founder_name or "", founder_role=founder.role or "",
            founder_linkedin_url=founder.linkedin_url, email=email.email or "",
            email_verification_status=verification.status.value, company_source_urls=company.source_urls,
            founder_source_urls=founder.source_urls, email_source_url=email.source_url,
            evidence={key: value for key, value in {
                "funding": qualification.financial_evidence or company.evidence.get("funding"),
                "technology": qualification.technology_evidence,
                "geography": qualification.geographic_evidence,
                "founder": founder.evidence, "email": email.evidence,
            }.items() if value}, qualification_reasons=qualification.rejection_reasons,
        )

    def _search_queries(self, queries: List[str], seen_queries: set[str], seen_search_urls: set[str]):
        pending = []
        for query in queries:
            normalized_query = query.strip().lower()
            if normalized_query and normalized_query not in seen_queries:
                seen_queries.add(normalized_query)
                pending.append(query)

        def search(query: str):
            return self.search_service.search(query, max_results=5)

        search_results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(pending)))) as executor:
            futures = [executor.submit(search, query) for query in pending]
            for query, future in zip(pending, futures):
                try:
                    for result in future.result():
                        if result.url and result.url in seen_search_urls:
                            continue
                        if result.url:
                            seen_search_urls.add(result.url)
                        search_results.append(result)
                except Exception as exc:
                    logger.warning("Search failed for query '%s': %s", query, exc)
                    self.stats["failures"] += 1
        return search_results

    def _merge_research_diagnostics(self) -> None:
        consume = getattr(self.research_service, "consume_diagnostics", None)
        if not callable(consume):
            return
        for key, value in consume().items():
            self.stats[key] = self.stats.get(key, 0) + value
        consume_last_error = getattr(self.research_service, "consume_last_research_error", None)
        if callable(consume_last_error):
            self.last_research_error = consume_last_error() or self.last_research_error

    def run(self) -> List[QualifiedLead]:
        """Discover/process until target or configured safety limits are reached."""
        leads: List[QualifiedLead] = []
        seen_companies, seen_founder_emails = set(), set()
        seen_queries, seen_search_urls = set(), set()
        for iteration in range(1, self.max_iterations + 1):
            if len(leads) >= self.target_leads or self.stats["researched"] >= self.max_candidates:
                break
            try:
                queries = self.planner.generate_queries(count=2)
            except Exception as exc:
                self.stats["failures"] += 1; logger.warning("Iteration %d planner failed: %s", iteration, exc); continue
            search_results = self._search_queries(queries, seen_queries, seen_search_urls)
            self.stats["discovered"] += len(search_results)
            remaining_capacity = self.max_candidates - self.stats["researched"]
            try:
                profiles = self.research_service.research_candidates(search_results[:remaining_capacity])
            except Exception as exc:
                self.stats["failures"] += 1; logger.warning("Iteration %d research failed: %s", iteration, exc); continue
            self._merge_research_diagnostics()
            for company in profiles:
                if len(leads) >= self.target_leads or self.stats["researched"] >= self.max_candidates:
                    break
                company_key = self._lead_key(company)
                if company_key in seen_companies:
                    self.stats["duplicates"] += 1
                    self.stats["duplicate_companies"] += 1
                    continue
                seen_companies.add(company_key); self.stats["researched"] += 1
                try:
                    qualification = self.qualification_service.qualify_profile(company)
                    for name, status in (
                        ("financial", qualification.is_financially_qualified),
                        ("technology", qualification.is_technology_qualified),
                        ("geography", qualification.is_geographically_qualified),
                    ):
                        if status.value in {"UNKNOWN", "FAIL"}:
                            self.stats[f"{name}_{status.value.lower()}"] += 1
                    if not qualification.is_qualified:
                        continue
                    self.stats["qualified"] += 1
                    founder = self.founder_service.discover_founder(company)
                    if (founder.discovery_status != FounderDiscoveryStatus.FOUND
                            or not founder.founder_name
                            or not self._eligible_founder_role(founder.role)):
                        continue
                    self.stats["founders"] += 1
                    email = self.email_service.discover_email(company, founder)
                    if email.discovery_status != EmailDiscoveryStatus.FOUND:
                        continue
                    self.stats["emails"] += 1
                    verification = self.verification_service.verify(email)
                    if verification.status != EmailVerificationStatus.VERIFIED:
                        continue
                    founder_email_key = ((founder.founder_name or "").strip().lower(), (email.email or "").lower())
                    if founder_email_key in seen_founder_emails:
                        self.stats["duplicates"] += 1; continue
                    seen_founder_emails.add(founder_email_key); self.stats["verified"] += 1
                    leads.append(self._to_lead(company, founder, email, verification, qualification))
                except Exception as exc:
                    self.stats["failures"] += 1; logger.warning("Candidate failed | Company: %s | Error: %s", company.company_name, exc)
            logger.info("[Iteration %d] Discovered: %d Researched: %d Qualified: %d Founders: %d Emails: %d Verified: %d Final Leads: %d/%d", iteration, self.stats["discovered"], self.stats["researched"], self.stats["qualified"], self.stats["founders"], self.stats["emails"], self.stats["verified"], len(leads), self.target_leads)
        return leads

    @staticmethod
    def export_csv(leads: List[QualifiedLead], path: Path = Path("data/final_leads.csv")) -> None:
        """Write only final verified leads to a local ignored CSV file."""
        if not leads:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(leads[0].model_dump().keys()))
            writer.writeheader()
            for lead in leads:
                row = lead.model_dump()
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = str(value)
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded TVB final-lead pipeline.")
    parser.add_argument("--target-leads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()
    pipeline = LeadPipeline(args.target_leads, args.max_iterations, args.max_candidates)
    leads = pipeline.run()
    pipeline.export_csv(leads)
    print(f"Final verified leads: {len(leads)}/{args.target_leads}")
    print(f"Pipeline statistics: {pipeline.stats}")


if __name__ == "__main__":
    main()
