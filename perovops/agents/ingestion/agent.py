from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .mineru import load_pdf
from .schema import DocBundle

logger = logging.getLogger(__name__)


class IngestionAgent:
    """LLM/VLM-backed ingestion agent."""

    def ingest(
        self,
        pdf_path: str,
        *,
        use_ocr: bool = False,
        working_dir: Path | str | None = None,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> DocBundle:
        """
        Run the ingestion pipeline for a PDF path or directory of PDFs.

        Args:
            pdf_path: Target PDF file or directory path.
            use_ocr: Whether to enable OCR during MinerU extraction.
            working_dir: Optional directory to store stage artifacts.
            extract_tables: Whether to extract tables.
            extract_images: Whether to extract figures/images.
        """
        logger.info("Starting ingestion agent")
        docbundle = load_pdf(
            pdf_path,
            extract_tables=extract_tables,
            extract_images=extract_images,
            use_ocr=use_ocr,
            working_dir=working_dir,
        )
        logger.info(
            "Ingestion complete: %d sections, %d tables, %d figures",
            len(docbundle.sections),
            len(docbundle.tables),
            len(docbundle.figures),
        )
        return docbundle


__all__ = ["IngestionAgent"]
