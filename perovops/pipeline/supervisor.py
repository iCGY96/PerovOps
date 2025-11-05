"""
Supervisor Agent: Orchestrates the multi-agent LangGraph pipeline.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, TypedDict
from langgraph.graph import END, StateGraph

from perovops.stage import ingestion, parser

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """State schema for LangGraph."""

    # Input
    pdf_path: str
    fu: str
    region: str
    use_ocr: bool

    # Agent outputs
    docbundle: Any
    steps: list
    ingest_markdown_path: str
    ingest_markdown_files: list
    parser_shared_state: Any
    parser_validation: dict
    parser_open_issues: list
    parser_process_details: dict
    parser_edges: list
    parser_summary: dict
    scope: dict

    # Flags
    ingestion_complete: bool
    parsing_complete: bool


def _wrap_stage(stage_name: str, func: Callable[[GraphState], Dict[str, Any]]):
    """Wrap stage nodes to emit consistent supervisor-level logging."""

    @wraps(func)
    def _wrapped(state: GraphState) -> GraphState:
        logger.info("Dispatching stage '%s'", stage_name)
        try:
            result = func(state)
        except Exception:
            logger.exception("Stage '%s' failed", stage_name)
            raise
        logger.info("Stage '%s' completed", stage_name)
        return result

    return _wrapped


def build_graph():
    """
    Build the LangGraph workflow for stage-based execution.

    Returns:
        Compiled graph
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("ingest", _wrap_stage("ingestion", ingestion.node))
    workflow.add_node("parse", _wrap_stage("parser", parser.node))

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "parse")
    workflow.add_edge("parse", END)

    return workflow.compile()


def run_pipeline(
    pdf_path: str,
    fu: str = "1 m^2",
    region: str = "US-avg",
    use_ocr: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete Perovops pipeline.

    Args:
        pdf_path: Path to PDF file
        fu: Functional unit (e.g., "1 m^2")
        region: Grid region code
        use_ocr: Whether to use OCR for PDF parsing

    Returns:
        Final state with LCI model and report
    """
    logger.info("Starting Perovops pipeline for %s", pdf_path)

    # Build graph
    graph = build_graph()

    # Initial state
    initial_state: GraphState = {
        "pdf_path": pdf_path,
        "fu": fu,
        "region": region,
        "use_ocr": use_ocr,
        "ingestion_complete": False,
        "parsing_complete": False,
    }

    try:
        final_state = graph.invoke(initial_state)
        logger.info("Pipeline completed successfully")
        return final_state
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise
