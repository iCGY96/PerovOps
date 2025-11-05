from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perovops.agents.ingestion import DocBundle, DocFigure, DocSection
from perovops.utils.markdown_export import render_docbundle_markdown
from example import write_stage_history


class MarkdownExportTests(unittest.TestCase):
    def test_render_docbundle_markdown_inserts_figures_after_references(self) -> None:
        with TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir)
            image_path = stage_dir / "figure1.png"
            image_path.write_bytes(b"fake")

            docbundle = {
                "sections": [
                    {
                        "identifier": "S1",
                        "text": "Figure 1 illustrates the device layout.",
                        "page_idx": 0,
                        "text_level": 1,
                    },
                    {
                        "identifier": "S2",
                        "text": "Additional context follows here.",
                        "page_idx": 0,
                    },
                ],
                "figures": [
                    {
                        "identifier": "F1",
                        "page_idx": 0,
                        "caption": "Device schematic.",
                        "image_path": image_path,
                        "vlm_description": "Layered stack with contacts.",
                    }
                ],
                "sources": [stage_dir / "source.pdf"],
            }

            markdown, metadata = render_docbundle_markdown(
                docbundle,
                stage_workdir=stage_dir,
            )

            self.assertIn("Figure 1 illustrates the device layout.", markdown)
            self.assertIn("**Automated description:**", markdown)
            self.assertIn("Layered stack with contacts.", markdown)
            self.assertEqual(metadata["placements"]["F1"], 1)

    def test_write_stage_history_creates_json_record(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            stage_dir = run_dir / "01_ingest"
            stage_dir.mkdir(parents=True, exist_ok=True)

            image_path = stage_dir / "figure.png"
            image_path.write_bytes(b"binary")

            bundle = DocBundle(
                sections=[
                    DocSection(
                        text="Figure 1 shows the result.",
                        page_idx=0,
                        text_level=1,
                    )
                ],
                figures=[
                    DocFigure(
                        image_path=image_path,
                        caption="Result figure.",
                        page_idx=0,
                        identifier="F1",
                        vlm_description="Observation summary.",
                    )
                ],
            )
            per_pdf_path = stage_dir / "f1.md"
            per_pdf_path.write_text(
                "![F1](figure.png)\n\n**Automated description:**\n> Observation summary.",
                encoding="utf-8",
            )
            bundle.markdown_paths = [per_pdf_path]
            bundle.combined_markdown = (
                "## Sample Document\n\n"
                "![F1](figure.png)\n\n**Automated description:**\n> Observation summary."
            )
            bundle.markdown_metadata = {
                "per_pdf_documents": [
                    {"pdf": "dummy.pdf", "markdown": per_pdf_path.name}
                ]
            }

            state = {
                "docbundle": bundle,
                "ingestion_complete": True,
            }
            result = {
                "docbundle": bundle,
                "ingestion_complete": True,
            }

            write_stage_history(
                run_dir=run_dir,
                index=1,
                stage="ingest",
                summary="sections=1, tables=0, figures=1",
                result=result,
                state=state,
            )

            history_path = run_dir / "01_ingest.json"
            self.assertTrue(history_path.exists())

            markdown_path = stage_dir / "combined.md"
            self.assertFalse(markdown_path.exists())

            payload = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "ingest")
            self.assertEqual(payload["summary"], "sections=1, tables=0, figures=1")
            self.assertIn("result", payload)
            self.assertIn("state_snapshot", payload)
            self.assertNotIn("markdown_metadata", payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
