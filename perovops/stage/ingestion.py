from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from perovops.agents.ingestion import IngestionAgent
from perovops.utils.config import config

logger = logging.getLogger(__name__)


def _resolve_stage_workdir(state: Dict[str, Any]) -> Optional[Path]:
    """Resolve the ingestion stage work directory from pipeline state."""
    stage_workdirs = state.get("stage_workdirs")
    if isinstance(stage_workdirs, dict):
        raw = stage_workdirs.get("ingest")
        if raw:
            try:
                return Path(raw).expanduser()
            except TypeError:
                pass

    current = state.get("current_stage_workdir")
    if current:
        try:
            return Path(current).expanduser()
        except TypeError:
            pass

    history_run_dir = state.get("history_run_dir")
    if history_run_dir:
        try:
            return Path(history_run_dir).expanduser() / "01_ingest"
        except TypeError:
            pass

    return None


def node(state: Dict[str, Any], *, agent: Optional[IngestionAgent] = None) -> Dict[str, Any]:
    """
    LangGraph ingestion stage.

    Args:
        state: Shared pipeline state.
        agent: Optional pre-configured ingestion agent.

    Returns:
        Updated state payload containing the DocBundle output.
    """
    logger.info("Starting ingestion stage")

    pdf_path = state.get("pdf_path")
    if not pdf_path:
        raise ValueError("pdf_path not provided in state")

    stage_workdir = _resolve_stage_workdir(state)
    ingestion_agent = agent or IngestionAgent()

    docbundle = ingestion_agent.ingest(
        str(pdf_path),
        use_ocr=state.get("use_ocr", False),
        working_dir=stage_workdir,
    )

    markdown_path: Optional[Path] = None
    if docbundle.markdown_paths:
        try:
            markdown_path = Path(docbundle.markdown_paths[0])
        except TypeError:
            markdown_path = None
    elif stage_workdir:
        try:
            candidate = Path(stage_workdir) / config.ingest_markdown_filename
            if candidate.exists():
                markdown_path = candidate
        except TypeError:
            markdown_path = None

    markdown_files = []
    for path in docbundle.markdown_paths:
        try:
            markdown_files.append(str(Path(path).resolve()))
        except TypeError:
            continue

    return {
        "docbundle": docbundle,
        "ingestion_complete": True,
        "ingest_markdown_path": str(markdown_path.resolve()) if markdown_path else None,
        "ingest_markdown_files": markdown_files,
    }


__all__ = ["node"]
