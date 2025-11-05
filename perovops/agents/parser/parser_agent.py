from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .orchestrator import LCIOrchestrator

from .core import (
    SharedState,
    build_document_context,
    build_scope_document_context,
    collect_process_details,
    validate_state,
    _docbundle_to_dict,
    _process_to_process_step,
)
from .energy_agent import EnergyAgent
from .material_agent import MaterialAgent
from .process_agent import ProcessAgent
from .scope_agent import ScopeAgent

logger = logging.getLogger(__name__)


class ParserAgent:
    """Agent responsible for running the LangGraph-powered parsing pipeline."""

    def __init__(
        self,
        *,
        scope_agent_factory: Callable[[], ScopeAgent] | None = None,
        process_agent_factory: Callable[[], ProcessAgent] | None = None,
        material_agent_factory: Callable[[], MaterialAgent] | None = None,
        energy_agent_factory: Callable[[], EnergyAgent] | None = None,
        validator: Optional[Callable[[SharedState, str], Dict[str, Any]]] = None,
        enable_energy_agent: bool = False,
    ) -> None:
        self._scope_agent_factory = scope_agent_factory or ScopeAgent
        self._process_agent_factory = process_agent_factory or ProcessAgent
        self._material_agent_factory = material_agent_factory or MaterialAgent
        self._energy_agent_factory = energy_agent_factory or EnergyAgent
        self._validator = validator or validate_state
        self._enable_energy_agent = enable_energy_agent

    def parse(
        self,
        *,
        docbundle: Any,
        pdf_path: Optional[str] = None,
        ingest_markdown_path: Optional[str] = None,
        parser_workdir: Optional[Path | str] = None,
        document_context: Optional[str] = None,
        scope_document_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        docbundle_dict = _docbundle_to_dict(docbundle)

        markdown_context: Optional[str] = None
        if ingest_markdown_path and not document_context:
            try:
                markdown_context = Path(ingest_markdown_path).read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Failed to read ingest markdown at %s: %s",
                    ingest_markdown_path,
                    exc,
                )
            except ValueError as exc:
                logger.warning(
                    "Invalid ingest markdown path %s: %s",
                    ingest_markdown_path,
                    exc,
                )

        if document_context and document_context.strip():
            resolved_document_context = document_context.strip()
        elif markdown_context and markdown_context.strip():
            resolved_document_context = markdown_context.strip()
        else:
            resolved_document_context = build_document_context(docbundle_dict)

        if scope_document_context and scope_document_context.strip():
            resolved_scope_context = scope_document_context.strip()
        elif markdown_context and markdown_context.strip():
            resolved_scope_context = markdown_context.strip()
        else:
            resolved_scope_context = build_scope_document_context(docbundle_dict)
            if not resolved_scope_context.strip():
                resolved_scope_context = resolved_document_context

        metadata = docbundle_dict.get("metadata", {}) if isinstance(docbundle_dict, dict) else {}
        raw_doc_id = (
            metadata.get("doc_id")
            or docbundle_dict.get("path")
            or pdf_path
            or "unknown"
        )
        doc_id = str(raw_doc_id).split("/")[-1]

        parser_dir: Optional[Path] = None
        if parser_workdir:
            try:
                parser_dir = Path(parser_workdir).expanduser().resolve()
            except (TypeError, OSError) as exc:
                logger.warning("Invalid parser workdir %s: %s", parser_workdir, exc)
                parser_dir = None

        scope_output_dir = parser_dir / "scope_attempts" if parser_dir else None
        shared_state = SharedState(doc_id=str(doc_id), scope_output_dir=scope_output_dir)

        orchestrator = LCIOrchestrator(
            scope_agent=self._scope_agent_factory(),
            process_agent=self._process_agent_factory(),
            material_agent=self._material_agent_factory(),
            energy_agent=self._energy_agent_factory(),
            validator=self._validator,
            enable_energy_agent=self._enable_energy_agent,
        )

        result = orchestrator.run(
            document_context=resolved_document_context,
            shared_state=shared_state,
            scope_document_context=resolved_scope_context,
        )
        validation = result.get("validation", {"status": "ok", "issues": []})

        steps: list = []
        for process in shared_state.sorted_processes():
            flows = shared_state.flow_inventories.get(process.id)
            if flows is None:
                continue
            try:
                step = _process_to_process_step(process, flows)
                steps.append(step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to convert process %s to ProcessStep: %s",
                    process.id,
                    exc,
                )

        edge_records = [edge.model_dump() for edge in shared_state.build_edges()]
        process_details = collect_process_details(shared_state)
        coverage = shared_state.coverage_summary()

        logger.info(
            "Parser summary: processes=%s material_edges=%s energy_edges=%s",
            coverage.get("process_count", 0),
            coverage.get("material_flow_count", 0),
            coverage.get("energy_flow_count", 0),
        )

        payload: Dict[str, Any] = {
            "steps": steps,
            "parsing_complete": True,
            "parser_shared_state": shared_state,
            "parser_validation": validation,
            "parser_open_issues": list(shared_state.open_issues),
            "parser_process_details": process_details,
            "parser_edges": edge_records,
            "parser_summary": coverage,
        }

        if shared_state.scope is not None:
            payload["scope"] = shared_state.scope.model_dump()

        return payload


__all__ = ["ParserAgent"]
