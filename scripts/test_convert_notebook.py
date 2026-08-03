#!/usr/bin/env python3
"""
Golden tests for convert_notebook.py.

Each fixture in scripts/fixtures/ pairs a `.md.txt` source with the `.expected.md` output the
converter must produce. Fixtures are minimised reproductions of real defects found in the port
-- see PLAN.md Step 0.2. Named regression tests sit alongside the golden comparison so a failure
says *which* defect came back rather than just "output differs".

    python scripts/test_convert_notebook.py            # run
    python scripts/test_convert_notebook.py --update   # regenerate .expected.md, then REVIEW the diff

Stdlib only, so CI needs no install step.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_notebook import convert  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def render(source: Path, assets_dir: Path | None = None) -> str:
    """Convert a fixture the way the test harness always does, so results are comparable."""
    return convert(
        source.read_text(encoding="utf-8"),
        assets_dir=assets_dir,
        assets_url="assets",
        assets_stem=source.name.removesuffix(".md.txt"),
    )


def fixtures() -> list[Path]:
    return sorted(FIXTURES.glob("*.md.txt"))


class GoldenTests(unittest.TestCase):
    """Every fixture must convert to exactly its .expected.md."""

    def test_fixtures_exist(self):
        self.assertTrue(fixtures(), f"no fixtures found in {FIXTURES}")

    def test_golden_output(self):
        for source in fixtures():
            with self.subTest(fixture=source.name):
                expected_path = source.with_suffix("").with_suffix(".expected.md")
                self.assertTrue(
                    expected_path.exists(),
                    f"missing {expected_path.name} -- run with --update and review the result",
                )
                with tempfile.TemporaryDirectory() as tmp:
                    actual = render(source, Path(tmp))
                self.assertEqual(
                    expected_path.read_text(encoding="utf-8"),
                    actual,
                    f"\n{source.name} output drifted from {expected_path.name}",
                )


class RegressionTests(unittest.TestCase):
    """Named guards for the specific defects Step 0.2 fixed."""

    def test_wrapped_list_item_is_not_fenced_as_code(self):
        """Defect: models/gemma-4/index.md:117 fenced a wrapped bullet as python."""
        output = render(FIXTURES / "list_continuation.md.txt")
        self.assertNotIn("```", output, "wrapped list continuation was fenced as code")
        self.assertIn(
            "*Available for Gemma 4 E2B, E4B, 12B, 26B A4B, and 31B.*",
            output,
            "wrapped sentence was not rejoined; the italic span is still broken",
        )

    def test_prompt_template_is_tagged_text_not_python(self):
        """Defect: ASR prompt templates were tagged `python` because bash detection failed."""
        output = render(FIXTURES / "prompt_template.md.txt")
        self.assertIn("```text\nTranscribe the following speech segment", output)
        self.assertNotIn("```python\nTranscribe", output)

    def test_prompt_template_keeps_its_bullets_in_one_block(self):
        """Defect: an indented bullet ended the block, splitting one prompt in two."""
        output = render(FIXTURES / "prompt_template.md.txt")
        block = output.split("```text\n", 1)[1].split("```", 1)[0]
        self.assertIn("Only output the transcription", block)
        self.assertIn("When transcribing numbers", block)

    def test_blank_lines_do_not_split_a_code_block(self):
        """Defect: two blank lines mid-cell split one function across two fences."""
        output = render(FIXTURES / "split_code_block.md.txt")
        self.assertEqual(
            output.count("```python"), 1,
            "code block was split -- consecutive blank lines ended it early",
        )
        block = output.split("```python\n", 1)[1].split("```", 1)[0]
        self.assertIn("binary_mask = (mask > 0.5).astype(np.uint8)", block)
        self.assertIn("full_mask[y1:y2, x1:x2] = np.maximum", block)
        self.assertIn("return full_mask", block)

    def test_real_code_still_detected(self):
        """The text fallback must not swallow genuine python and bash blocks."""
        output = render(FIXTURES / "prompt_template.md.txt")
        self.assertIn("```python\nfrom transformers import pipeline", output)
        self.assertIn("```bash\npip install --upgrade transformers", output)

    def test_data_uri_inside_code_is_preserved(self):
        """Only markdown images are extracted -- a data URI in a snippet is code."""
        with tempfile.TemporaryDirectory() as tmp:
            output = render(FIXTURES / "base64_image.md.txt", Path(tmp))
            written = sorted(p.name for p in Path(tmp).iterdir())
        self.assertIn('return f"data:image/jpeg;base64,{image_b64}"', output)
        self.assertNotIn("](data:image", output, "markdown data URI was not extracted")
        self.assertEqual(
            written,
            ["base64_image-01.png", "base64_image-02.png"],
            "expected exactly the two markdown images to be extracted",
        )

    def test_extracted_images_are_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            render(FIXTURES / "base64_image.md.txt", Path(tmp))
            for path in Path(tmp).iterdir():
                self.assertTrue(path.stat().st_size > 0, f"{path.name} is empty")
                self.assertEqual(
                    path.read_bytes()[:8],
                    b"\x89PNG\r\n\x1a\n",
                    f"{path.name} is not a valid PNG",
                )

    def test_images_left_alone_without_assets_dir(self):
        """No assets dir means no silent data loss -- the payload stays put."""
        output = render(FIXTURES / "base64_image.md.txt", None)
        self.assertIn("](data:image/png;base64,", output)

    def test_devsite_admonitions_converted(self):
        """Already working before Step 0.2; pinned so it stays that way."""
        output = render(FIXTURES / "admonitions.md.txt")
        self.assertIn("!!! Note", output)
        self.assertIn("!!! Warning", output)
        self.assertNotIn("> [!", output)

    def test_tables_get_separator_rows(self):
        output = render(FIXTURES / "admonitions.md.txt")
        self.assertIn("|---|---|---|", output)


def update_expected() -> int:
    for source in fixtures():
        expected_path = source.with_suffix("").with_suffix(".expected.md")
        with tempfile.TemporaryDirectory() as tmp:
            expected_path.write_text(render(source, Path(tmp)), encoding="utf-8")
        print(f"  wrote {expected_path.relative_to(FIXTURES.parent.parent)}")
    print("\nReview the diff before committing -- these files are the specification.")
    return 0


if __name__ == "__main__":
    if "--update" in sys.argv:
        sys.exit(update_expected())
    unittest.main(verbosity=2)
