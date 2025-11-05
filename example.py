"""
Standalone agent runner for the Perovops pipeline (DocBundle-ready).

This script allows you to invoke each agent one-by-one, inspect their outputs,
and export intermediate state for further experimentation. It is useful when
you want to iterate on or debug a specific agent without running the LangGraph
supervisor workflow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml

from perovops.stage import ingestion, parser
from perovops.io.schemas import ProcessStep
from perovops.utils.serialization import to_serializable

State = Dict[str, Any]

STAGE_ORDER: List[str] = [
    "ingest",
    "parse",
]

AGENT_NODES: Dict[str, Callable[[State], State]] = {
    "ingest": ingestion.node,
    "parse": parser.node,
}

STAGE_COMPLETION_CHECKS: Dict[str, Callable[[State], bool]] = {
    "ingest": lambda state: bool(state.get("ingestion_complete")),
    "parse": lambda state: bool(state.get("parsing_complete")),
}

HISTORY_ROOT = Path("history")


def build_initial_state(
    pdf_path: Path,
    fu: str,
    region: str,
    use_ocr: bool,
) -> State:
    """Create the baseline state dictionary passed between agents."""
    return {
        # pdf_path can be a single PDF or a directory; ingestion.node handles both
        "pdf_path": str(pdf_path),
        "fu": fu,
        "region": region,
        "use_ocr": use_ocr,
        "docbundle": None,          # <- DocBundle object (dataclass)
        "steps": [],
        "ingestion_complete": False,
        "parsing_complete": False,
    }


def summarize_stage(stage: str, state: State) -> str:
    """Generate a short human-readable summary for a given stage."""
    if stage == "ingest":
        bundle = state.get("docbundle")
        if not bundle:
            return "DocBundle not available"
        sections_count = len(getattr(bundle, "sections", []) or [])
        tables_count = len(getattr(bundle, "tables", []) or [])
        figures_count = len(getattr(bundle, "figures", []) or [])
        markdown_path = state.get("ingest_markdown_path")
        per_pdf_count = 0
        md_metadata = getattr(bundle, "markdown_metadata", {}) or {}
        if isinstance(md_metadata, dict):
            per_pdf_docs = md_metadata.get("per_pdf_documents") or []
            if isinstance(per_pdf_docs, list):
                per_pdf_count = len(per_pdf_docs)

        if markdown_path:
            name = Path(markdown_path).name
            summary = (
                f"sections={sections_count}, tables={tables_count}, "
                f"figures={figures_count}, markdown={name}"
            )
            if per_pdf_count:
                summary += f", documents={per_pdf_count}"
            return summary
        return (
            f"sections={sections_count}, tables={tables_count}, "
            f"figures={figures_count}"
        )

    if stage == "parse":
        steps = state.get("steps") or []
        names = ", ".join(getattr(step, "id", "") for step in steps[:3])
        if len(steps) > 3:
            names += ", …"
        return f"steps={len(steps)} ({names or 'no ids'})"

    return "No summary available"


def run_stage(stage: str, state: State) -> State:
    """Invoke a single agent stage and merge its output into the state."""
    node = AGENT_NODES[stage]
    result = node(state)
    state.update(result)
    return result


def iter_stages(target_stage: str | None) -> Iterable[str]:
    """Yield stages in execution order up to the requested target stage."""
    if target_stage is None:
        yield from STAGE_ORDER
        return

    if target_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage '{target_stage}'. Valid choices: {STAGE_ORDER}")

    limit = STAGE_ORDER.index(target_stage)
    for stage in STAGE_ORDER[: limit + 1]:
        yield stage


def create_history_run_dir() -> Path:
    """Create a unique directory for the current execution history."""
    timestamp = datetime.now().strftime("%Y%m%d")
    run_dir = HISTORY_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _stage_history_path(run_dir: Path, stage: str) -> Path:
    """Return the canonical history file path for a stage."""
    index = STAGE_ORDER.index(stage) + 1
    return run_dir / f"{index:02d}_{stage}.json"


def _stage_workdir_path(run_dir: Path, stage: str) -> Path:
    """Return the working directory path for a stage."""
    index = STAGE_ORDER.index(stage) + 1
    return run_dir / f"{index:02d}_{stage}"


def _rehydrate_state_snapshot(snapshot: Dict[str, Any]) -> State:
    """Convert JSON-serialized state back into runtime structures."""
    hydrated: State = dict(snapshot)

    raw_steps = snapshot.get("steps")
    if raw_steps is not None:
        try:
            hydrated["steps"] = [
                step if isinstance(step, ProcessStep) else ProcessStep(**step)
                for step in raw_steps
            ]
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: could not rehydrate steps from history: {exc}")
            hydrated["steps"] = raw_steps

    return hydrated


def _parse_markdown_front_matter(markdown: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML front matter and body from a Markdown string."""
    if not markdown.startswith("---"):
        return {}, markdown

    lines = markdown.splitlines()
    closing_index: Optional[int] = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_index = idx
            break

    if closing_index is None:
        return {}, markdown

    yaml_str = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])

    try:
        data = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        data = {}

    if not isinstance(data, dict):
        data = {}

    return data, body


def _load_history_snapshot(history_path: Path) -> tuple[State | None, str]:
    """Load and rehydrate a stage snapshot from disk if available."""
    suffix = history_path.suffix.lower()
    text = history_path.read_text(encoding="utf-8")

    if suffix in {".md", ".markdown"}:
        payload, _body = _parse_markdown_front_matter(text)
        snapshot = payload.get("state_snapshot")
        if not isinstance(snapshot, dict):
            print(f"Warning: state snapshot missing in history file {history_path}")
            return None, str(payload.get("summary", ""))
        hydrated_state = _rehydrate_state_snapshot(snapshot)
        summary = str(payload.get("summary", ""))
        return hydrated_state, summary

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"Warning: invalid JSON in history file {history_path}: {exc}")
        return None, ""

    snapshot = payload.get("state_snapshot")
    if not isinstance(snapshot, dict):
        print(f"Warning: state snapshot missing in history file {history_path}")
        return None, payload.get("summary", "")

    hydrated_state = _rehydrate_state_snapshot(snapshot)
    summary = payload.get("summary", "")
    return hydrated_state, summary


def write_stage_history(
    run_dir: Path,
    index: int,
    stage: str,
    summary: str,
    result: State,
    state: State,
) -> None:
    """Persist the output and summary for an executed stage.

    Serializes the current stage result and a snapshot of the full state.
    All stages emit JSON files under ``history/``.
    """
    serializable_result = to_serializable(result)
    state_snapshot = to_serializable(result if stage == "parse" else state)

    payload = {
        "stage": stage,
        "summary": summary,
        "result": serializable_result,
        "state_snapshot": state_snapshot,
    }
    history_path = _stage_history_path(run_dir, stage)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def run_pipeline(
    pdf_path: Path,
    fu: str,
    region: str,
    use_ocr: bool,
    target_stage: str | None,
) -> State:
    """Run agents sequentially up to the requested stage."""
    state = build_initial_state(pdf_path, fu, region, use_ocr)
    history_run_dir = create_history_run_dir()
    print(f"History directory: {history_run_dir}")

    state["history_run_dir"] = str(history_run_dir)

    for stage in iter_stages(target_stage):
        stage_workdir = _stage_workdir_path(history_run_dir, stage)
        stage_workdir.mkdir(parents=True, exist_ok=True)
        stage_workdirs = state.setdefault("stage_workdirs", {})
        if not isinstance(stage_workdirs, dict):
            stage_workdirs = {}
            state["stage_workdirs"] = stage_workdirs
        stage_workdirs[stage] = str(stage_workdir)

        stage_history_path = _stage_history_path(history_run_dir, stage)

        if stage_history_path.exists():
            loaded_state, saved_summary = _load_history_snapshot(stage_history_path)
            if loaded_state is not None:
                print(f"\n=== Skipping {stage} (history record found) ===")
                state.update(loaded_state)
                summary_text = saved_summary or summarize_stage(stage, state)
                if summary_text:
                    print(summary_text)
                continue

        completion_check = STAGE_COMPLETION_CHECKS.get(stage)
        if completion_check and completion_check(state):
            print(f"\n=== Skipping {stage} (already completed) ===")
            summary = summarize_stage(stage, state)
            if summary:
                print(summary)
            continue

        print(f"\n=== Running {stage} ===")
        result = run_stage(stage, state)
        summary = summarize_stage(stage, state)
        print(summary)
        stage_number = STAGE_ORDER.index(stage) + 1
        write_stage_history(history_run_dir, stage_number, stage, summary, result, state)

    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke Perovops agents one stage at a time (DocBundle-ready).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("examples/mat1/"),
        help="Path to the input PDF OR a directory containing PDFs "
             "(default: examples/mat_1.pdf)",
    )
    parser.add_argument(
        "--fu",
        default="1 m^2",
        help="Functional unit string passed to downstream agents (default: '1 m^2')",
    )
    parser.add_argument(
        "--region",
        default="US-avg",
        help="Grid region code for electricity mapping (default: 'US-avg')",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR extraction during ingestion",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        help="Stop after running the specified stage (default: run all stages)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"Input path not found: {args.pdf}")

    # Allow passing a directory; ingestion.node will recurse through all PDFs inside
    run_pipeline(
        pdf_path=args.pdf,
        fu=args.fu,
        region=args.region,
        use_ocr=args.use_ocr,
        target_stage=args.stage,
    )


if __name__ == "__main__":
    main()
