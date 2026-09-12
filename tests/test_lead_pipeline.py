"""Bounded pipeline tests with fully mocked services and no API calls."""

import unittest

from agent.lead_pipeline import LeadPipeline
from core.email_models import EmailCandidate, EmailDiscoveryStatus, EmailVerificationResult, EmailVerificationStatus
from core.founder_models import FounderDiscoveryStatus, FounderProfile
from core.models import CompanyProfile, SearchResult


def profile(name: str = "Fixture Platform", qualified: bool = True) -> CompanyProfile:
    return CompanyProfile(company_name=name, description="AI SaaS platform" if qualified else "Consulting company", industry="software", location="Paris, France", funding_amount=2_000_000 if qualified else None, funding_currency="USD" if qualified else None, source_urls=[f"https://{name.lower().replace(' ', '')}.example"], evidence={"funding": f"{name} raised $2M.", "location": f"{name} is Paris, France-based."})


class Planner:
    def generate_queries(self, count=2): return ["query"]

class Search:
    def search(self, query, max_results=5): return [SearchResult(title="result", url="https://result.example", content="x")]

class Research:
    def __init__(self, items): self.items = items
    def research_candidates(self, results): return self.items

class Founder:
    def __init__(self, status=FounderDiscoveryStatus.FOUND): self.status = status
    def discover_founder(self, company): return FounderProfile(company_name=company.company_name, founder_name="Jane Doe" if self.status == FounderDiscoveryStatus.FOUND else None, role="CEO" if self.status == FounderDiscoveryStatus.FOUND else None, discovery_status=self.status, evidence="Jane Doe is CEO.", source_urls=["https://source.example"])

class Email:
    def __init__(self, status=EmailDiscoveryStatus.FOUND): self.status = status
    def discover_email(self, company, founder): return EmailCandidate(company_name=company.company_name, founder_name=founder.founder_name or "", founder_role=founder.role or "", email="jane@example.com" if self.status == EmailDiscoveryStatus.FOUND else None, discovery_status=self.status, evidence="Jane Doe can be reached at jane@example.com.", source_url="https://source.example")

class Verify:
    def __init__(self, status=EmailVerificationStatus.VERIFIED): self.status = status
    def verify(self, email): return EmailVerificationResult(email=email.email, status=self.status, reason="test", verification_method="test")


class LeadPipelineTests(unittest.TestCase):
    def pipeline(self, items, founder=Founder(), email=Email(), verify=Verify(), target=1, iterations=2):
        return LeadPipeline(target, iterations, 10, planner=Planner(), search_service=Search(), research_service=Research(items), founder_service=founder, email_service=email, verification_service=verify)

    def test_only_verified_leads_count_and_deduplicate(self):
        pipeline = self.pipeline([profile(), profile()])
        leads = pipeline.run()
        self.assertEqual(len(leads), 1)
        self.assertTrue({"funding", "technology", "geography", "founder", "email"}.issubset(leads[0].evidence))
        self.assertGreaterEqual(pipeline.stats["duplicates"], 0)

    def test_failure_stages_never_reach_final_leads(self):
        self.assertEqual(len(self.pipeline([profile(qualified=False)]).run()), 0)
        self.assertEqual(len(self.pipeline([profile()], founder=Founder(FounderDiscoveryStatus.NOT_FOUND)).run()), 0)
        self.assertEqual(len(self.pipeline([profile()], email=Email(EmailDiscoveryStatus.NOT_FOUND)).run()), 0)
        self.assertEqual(len(self.pipeline([profile()], verify=Verify(EmailVerificationStatus.UNKNOWN)).run()), 0)
        self.assertEqual(len(self.pipeline([profile()], verify=Verify(EmailVerificationStatus.INVALID)).run()), 0)

    def test_target_and_safety_limits_are_respected(self):
        self.assertEqual(len(self.pipeline([profile("One"), profile("Two")], target=1).run()), 1)
        pipeline = self.pipeline([profile("One"), profile("Two")], target=5, iterations=1)
        pipeline.max_candidates = 1
        self.assertEqual(len(pipeline.run()), 1)

    def test_individual_failure_does_not_stop_pipeline(self):
        pipeline = self.pipeline([profile(qualified=False), profile("Good")])
        self.assertEqual(len(pipeline.run()), 1)

    def test_ineligible_founder_role_never_reaches_final_leads(self):
        pipeline = self.pipeline([profile()], founder=Founder())
        pipeline.founder_service.discover_founder = lambda company: FounderProfile(
            company_name=company.company_name, founder_name="Jane Doe", role="CTO",
            discovery_status=FounderDiscoveryStatus.FOUND, evidence="Jane Doe is CTO.",
        )
        self.assertEqual(len(pipeline.run()), 0)

    def test_founder_email_pair_is_deduplicated_across_companies(self):
        pipeline = self.pipeline([profile("One"), profile("Two")], target=5)
        leads = pipeline.run()
        self.assertEqual(len(leads), 1)
        self.assertGreaterEqual(pipeline.stats["duplicates"], 1)

    def test_repeated_queries_and_urls_are_not_researched_again(self):
        class CountingSearch:
            def __init__(self): self.calls = 0
            def search(self, query, max_results=5):
                self.calls += 1
                return [SearchResult(title="result", url="https://result.example", content="x")]

        class CountingResearch:
            def __init__(self): self.items_seen = 0
            def research_candidates(self, results):
                self.items_seen += len(results)
                return [profile(qualified=False)] if results else []

        search = CountingSearch()
        research = CountingResearch()
        pipeline = LeadPipeline(
            target_leads=1, max_iterations=3, max_candidates=10, planner=Planner(),
            search_service=search, research_service=research, founder_service=Founder(),
            email_service=Email(), verification_service=Verify(),
        )
        self.assertEqual(pipeline.run(), [])
        self.assertEqual(search.calls, 1)
        self.assertEqual(research.items_seen, 1)


if __name__ == "__main__":
    unittest.main()
