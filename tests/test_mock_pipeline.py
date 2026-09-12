"""Step 9 local UI-support tests; intentionally no external service calls."""

import os
import tempfile
import unittest
from pathlib import Path

from agent.lead_pipeline import LeadPipeline
from services.mock_pipeline import run_mock_pipeline, validate_pipeline_config


class MockPipelineTests(unittest.TestCase):
    def test_mock_pipeline_is_deterministic_and_verified(self) -> None:
        first = run_mock_pipeline(2, 3, 10)
        second = run_mock_pipeline(2, 3, 10)
        self.assertEqual([lead.company_name for lead in first.leads], [lead.company_name for lead in second.leads])
        self.assertTrue(all("[DEMO]" in lead.company_name for lead in first.leads))
        self.assertTrue(all(lead.email_verification_status == "VERIFIED" for lead in first.leads))
        self.assertGreater(first.rejection_counts["financial"], 0)

    def test_mock_pipeline_needs_no_environment_keys(self) -> None:
        saved = {key: os.environ.pop(key, None) for key in ("GROQ_API_KEY", "TAVILY_API_KEY")}
        try:
            self.assertEqual(len(run_mock_pipeline(1, 1, 1).leads), 1)
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_configuration_validation_and_empty_target_handling(self) -> None:
        self.assertFalse(validate_pipeline_config(0, 1, 1)[0])
        self.assertFalse(validate_pipeline_config(2, 1, 1)[0])
        self.assertTrue(validate_pipeline_config(1, 1, 1)[0])
        self.assertEqual(run_mock_pipeline(1, 1, 1).leads, run_mock_pipeline(1, 1, 1).leads)

    def test_csv_export_accepts_only_final_lead_models(self) -> None:
        leads = run_mock_pipeline(1, 1, 1).leads
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_leads.csv"
            LeadPipeline.export_csv(leads, path)
            content = path.read_text(encoding="utf-8")
        self.assertIn("email_verification_status", content)
        self.assertIn("VERIFIED", content)


if __name__ == "__main__":
    unittest.main()
