from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import agent_docs


class AgentDocsTests(unittest.TestCase):
    def test_routes_match_rendered_pages(self):
        self.assertEqual(agent_docs.markdown_route_for("index.md"), "/index.md")
        self.assertEqual(agent_docs.markdown_route_for("models/index.md"), "/models.md")
        self.assertEqual(
            agent_docs.markdown_route_for("get-started/quickstart.md"),
            "/get-started/quickstart.md",
        )

    def test_llms_uses_absolute_markdown_links(self):
        rendered = agent_docs.render_llms()
        self.assertTrue(rendered.startswith("# Gemma Documentation\n"))
        self.assertIn("https://gemma.ai/get-started/quickstart.md", rendered)
        self.assertNotIn("llms-full", rendered)

    def test_raw_markdown_renders_model_macros(self):
        pages = agent_docs.source_pages()
        _path, metadata, body = pages["get-started/quickstart.md"]
        rendered = agent_docs.render_raw("get-started/quickstart.md", metadata, body)
        self.assertIn("| **E2B** |", rendered)
        self.assertNotIn("{% for", rendered)


if __name__ == "__main__":
    unittest.main()
