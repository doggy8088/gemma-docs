from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_docs


class AnalyticsBuildTests(unittest.TestCase):
    def test_inject_analytics_property_updates_config(self):
        source = (build_docs.agent_docs.REPO_ROOT / "zensical.toml").read_text(encoding="utf-8")
        injected = build_docs.inject_analytics_property(source, "G-TEST123456")
        self.assertIn('property = "G-TEST123456"', injected)

    def test_build_without_measurement_id_omits_tracking_script(self):
        env = {**os.environ, "GA_MEASUREMENT_ID": ""}
        result = subprocess.run(
            [sys.executable, "scripts/build_docs.py"],
            cwd=build_docs.agent_docs.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        index_html = (build_docs.agent_docs.REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("googletagmanager.com/gtag/js", index_html)
        self.assertNotIn("Change cookie settings", index_html)

    def test_build_with_measurement_id_includes_ga4_and_consent(self):
        env = {**os.environ, "GA_MEASUREMENT_ID": "G-TEST123456"}
        result = subprocess.run(
            [sys.executable, "scripts/build_docs.py"],
            cwd=build_docs.agent_docs.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        index_html = (build_docs.agent_docs.REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
        self.assertIn("https://www.googletagmanager.com/gtag/js?id=G-TEST123456", index_html)
        self.assertIn("G-TEST123456", index_html)
        self.assertIn('id="__consent"', index_html)
        self.assertIn('name="analytics"', index_html)
        self.assertIn("Change cookie settings", index_html)
        # Ensure temporary config file is cleaned up
        self.assertFalse((build_docs.agent_docs.REPO_ROOT / ".zensical.build.toml").exists())


if __name__ == "__main__":
    unittest.main()
