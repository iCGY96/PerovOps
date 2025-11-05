from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from perovops.agents.parser import ParserAgent

logger = logging.getLogger(__name__)


def _resolve_parser_stage_workdir(state: Dict[str, Any]) -> Optional[Path]:
    """Resolve the parser stage work directory from pipeline state."""
    stage_workdirs = state.get("stage_workdirs")
    if isinstance(stage_workdirs, dict):
        raw = stage_workdirs.get("parse")
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
            return Path(history_run_dir).expanduser() / "02_parse"
        except TypeError:
            pass

    return None


def node(state: Dict[str, Any], *, agent: Optional[ParserAgent] = None) -> Dict[str, Any]:
    """
    LangGraph parser stage.

    Args:
        state: Shared pipeline state.
        agent: Optional pre-configured parser agent.

    Returns:
        Updated state payload containing parser outputs.
    """
    logger.info("Starting parser stage")

    docbundle = state.get("docbundle")
    if docbundle is None:
        raise ValueError("docbundle not available in state")

    parser_workdir = _resolve_parser_stage_workdir(state)
    ingest_markdown_path = state.get("ingest_markdown_path")
    pdf_path = state.get("pdf_path")

    parser_agent = agent or ParserAgent()
    payload = parser_agent.parse(
        docbundle=docbundle,
        pdf_path=str(pdf_path) if pdf_path else None,
        ingest_markdown_path=ingest_markdown_path,
        parser_workdir=parser_workdir,
    )

    return payload


__all__ = ["node"]
