"""Utilities for rendering DocBundle objects into Markdown documents."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

PlacementMap = Dict[str, int]
SlotMap = Dict[int, List[Dict[str, Any]]]


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, Mapping):
        return obj.get(attr, default)
    return default


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_paragraph(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    return _normalize_whitespace(text)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _figure_number(identifier: Optional[str]) -> Optional[int]:
    if not identifier:
        return None
    match = re.search(r"(\d+)", str(identifier))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _resolve_relative_path(path: Optional[Path], base: Path) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    try:
        rel = os.path.relpath(p.resolve(), base.resolve())
    except (OSError, ValueError):
        rel = str(p)
    return rel.replace(os.sep, "/")


def _collect_sections(docbundle: Any) -> List[Dict[str, Any]]:
    sections: Iterable[Any] = _safe_attr(docbundle, "sections", []) or []
    collected: List[Dict[str, Any]] = []

    for idx, section in enumerate(sections):
        text = _safe_attr(section, "text") or _safe_attr(section, "content")
        text = _normalize_paragraph(text or "")
        if not text:
            continue
        section_id = _safe_attr(section, "identifier") or f"S{idx+1}"
        collected.append(
            {
                "identifier": str(section_id),
                "page_idx": _coerce_int(_safe_attr(section, "page_idx")),
                "text_level": _coerce_int(_safe_attr(section, "text_level")),
                "text": text,
            }
        )
    return collected


def _collect_figures(docbundle: Any, stage_workdir: Path) -> List[Dict[str, Any]]:
    figures: Iterable[Any] = _safe_attr(docbundle, "figures", []) or []
    collected: List[Dict[str, Any]] = []

    for idx, figure in enumerate(figures):
        identifier = _safe_attr(figure, "identifier") or f"F{idx+1}"
        page_idx = _coerce_int(_safe_attr(figure, "page_idx"))
        caption = _normalize_whitespace(_safe_attr(figure, "caption") or "")
        footnote = _normalize_whitespace(_safe_attr(figure, "footnote") or "")
        description_raw = _safe_attr(figure, "vlm_description") or ""
        description = str(description_raw).strip()
        image_path = _safe_attr(figure, "image_path")

        rel_path = None
        if image_path:
            rel_path = _resolve_relative_path(Path(image_path), stage_workdir)

        collected.append(
            {
                "identifier": str(identifier),
                "page_idx": page_idx,
                "caption": caption,
                "footnote": footnote,
                "description": description,
                "image_path": rel_path,
                "number": _figure_number(identifier),
            }
        )

    return collected


def _section_mentions_figure(text: str, number: int) -> bool:
    pattern = rf"\b(fig(?:ure|\.)|figs\.?)\s*{number}([a-z0-9\-]*)"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _llm_override_slots(
    sections: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    llm_client: Any,
) -> PlacementMap:
    if not llm_client or not sections or not figures:
        return {}

    # Prepare compact context for the LLM to avoid excessive prompt sizes.
    section_summaries = []
    for idx, section in enumerate(sections, start=1):
        summary = section["text"]
        if len(summary) > 240:
            summary = summary[:237] + "..."
        heading = f"S{idx} ({section['identifier']})"
        section_summaries.append(f"{heading}: {summary}")

    figure_summaries = []
    for figure in figures:
        desc = figure["description"]
        if len(desc) > 320:
            desc = desc[:317] + "..."
        number = figure.get("number")
        label = figure["identifier"]
        figure_summaries.append(
            f"{label} (number={number}): caption='{figure['caption']}' description='{desc}'"
        )

    prompt_lines = [
        "You are organizing figure descriptions into a scientific article outline.",
        "For each figure, choose the best section identifier where its description belongs.",
        "If a figure should appear before all sections, use 'BEFORE_ALL'.",
        "If it should appear after all sections, use 'AFTER_ALL'.",
        "Respond strictly as JSON mapping figure identifiers to section identifiers or the special tokens.",
        "",
        "Sections:",
        *section_summaries,
        "",
        "Figures:",
        *figure_summaries,
    ]

    messages = [
        {"role": "system", "content": "Return only JSON. No commentary."},
        {"role": "user", "content": "\n".join(prompt_lines)},
    ]

    try:
        result = llm_client.generate(messages)
    except Exception:
        return {}

    content = getattr(result, "content", None)
    if not content:
        return {}

    try:
        import json

        parsed = json.loads(content)
    except Exception:
        return {}

    placement: PlacementMap = {}
    index_by_identifier = {section["identifier"]: idx for idx, section in enumerate(sections)}

    for fig in figures:
        target = parsed.get(fig["identifier"])
        if not isinstance(target, str):
            continue
        token = target.strip().upper()
        if token == "BEFORE_ALL":
            placement[fig["identifier"]] = 0
            continue
        if token == "AFTER_ALL":
            placement[fig["identifier"]] = len(sections)
            continue
        section_idx = index_by_identifier.get(target) or index_by_identifier.get(token)
        if section_idx is not None:
            placement[fig["identifier"]] = section_idx + 1
    return placement


def _assign_slots(
    sections: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    override: Optional[PlacementMap] = None,
) -> Tuple[SlotMap, PlacementMap]:
    slot_map: SlotMap = defaultdict(list)
    placements: PlacementMap = {}

    def default_slot(figure: Dict[str, Any]) -> int:
        number = figure.get("number")
        if number is not None:
            for idx, section in enumerate(sections):
                if _section_mentions_figure(section["text"], number):
                    return idx + 1
        page_idx = figure.get("page_idx")
        if page_idx is None:
            return len(sections)
        candidate = len(sections)
        for idx, section in enumerate(sections):
            sec_page = section.get("page_idx")
            if sec_page is None:
                continue
            if sec_page <= page_idx:
                candidate = idx + 1
        return candidate

    override = override or {}

    for figure in figures:
        slot = override.get(figure["identifier"])
        if slot is None:
            slot = default_slot(figure)
        slot = max(0, min(len(sections), slot))
        placements[figure["identifier"]] = slot
        slot_map[slot].append(figure)

    return slot_map, placements


def _render_section(section: Dict[str, Any]) -> str:
    level = section.get("text_level")
    text = section["text"]
    if level is not None and level >= 1:
        hashes = "#" * min(level, 6)
        return f"{hashes} {text}"
    return text


def _render_description(description: str) -> str:
    if not description:
        return ""
    lines = description.splitlines()
    formatted = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append(">")
        else:
            formatted.append(f"> {stripped}")
    return "\n".join(formatted)


def _render_figure(figure: Dict[str, Any]) -> str:
    lines: List[str] = []
    title = figure["identifier"]

    rel_path = figure.get("image_path")
    if rel_path:
        lines.append(f"![{title}]({rel_path})")

    caption_parts = []
    number = figure.get("number")
    if number is not None:
        caption_parts.append(f"Figure {number}")
    else:
        caption_parts.append(f"{title}")
    caption_text = figure.get("caption")
    if caption_text:
        caption_parts.append(caption_text)
    caption_line = ". ".join(part for part in caption_parts if part)
    if caption_line:
        lines.append(f"*{caption_line}.*")

    footnote = figure.get("footnote")
    if footnote:
        lines.append(f"*Footnote:* {footnote}")

    description = figure.get("description")
    if description:
        description_block = _render_description(description)
        if description_block:
            lines.append("")
            lines.append("**Automated description:**")
            lines.append(description_block)

    return "\n".join(lines).strip()


def _collect_sources(docbundle: Any, stage_workdir: Path) -> List[str]:
    sources: Iterable[Any] = _safe_attr(docbundle, "sources", []) or []
    collected: List[str] = []
    for source in sources:
        if isinstance(source, Path):
            collected.append(_resolve_relative_path(source, stage_workdir))
        else:
            collected.append(str(source))
    return [src for src in collected if src]


def render_docbundle_markdown(
    docbundle: Any,
    *,
    stage_workdir: Path,
    llm_client: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    """Render the provided DocBundle-ish object into Markdown.

    Args:
        docbundle: Dataclass-like object returned by ingestion.
        stage_workdir: Directory where the Markdown file will live; used to
            compute relative asset paths.
        llm_client: Optional LiteLLM client that can be used to refine figure
            placement. When omitted or generation fails, heuristic placement is
            used instead.

    Returns:
        A tuple ``(markdown_text, metadata)`` where metadata contains the final
        figure placement slots and counts.
    """
    stage_workdir = Path(stage_workdir)
    sections = _collect_sections(docbundle)
    figures = _collect_figures(docbundle, stage_workdir)

    override_slots: PlacementMap = {}
    if llm_client:
        override_slots = _llm_override_slots(sections, figures, llm_client)

    slot_map, placements = _assign_slots(sections, figures, override_slots)

    sources = _collect_sources(docbundle, stage_workdir)
    body_parts: List[str] = []

    if sources:
        source_line = ", ".join(sources)
        body_parts.append(f"**Sources:** {source_line}")

    # Figures placed before all sections (slot == 0)
    for figure in slot_map.get(0, []):
        rendered = _render_figure(figure)
        if rendered:
            body_parts.append(rendered)

    for idx, section in enumerate(sections):
        rendered_section = _render_section(section)
        if rendered_section:
            body_parts.append(rendered_section)

        for figure in slot_map.get(idx + 1, []):
            rendered_figure = _render_figure(figure)
            if rendered_figure:
                body_parts.append(rendered_figure)

    markdown_text = "\n\n".join(part for part in body_parts if part).strip()

    metadata = {
        "section_count": len(sections),
        "figure_count": len(figures),
        "placements": placements,
    }

    return markdown_text, metadata


def docbundle_to_dict(docbundle: Any) -> Dict[str, Any]:
    """Convert a DocBundle-ish object into a plain dict."""
    if hasattr(docbundle, "model_dump"):
        return docbundle.model_dump()
    if is_dataclass(docbundle):
        return asdict(docbundle)
    if isinstance(docbundle, Mapping):
        return dict(docbundle)
    return dict(asdict(docbundle))  # pragma: no cover - defensive fallback


__all__ = ["render_docbundle_markdown", "docbundle_to_dict"]
