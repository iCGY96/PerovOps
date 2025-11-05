"""Utility helpers for working with DocBundle objects produced by ingestion.

The ingestion agent emits DocBundle instances backed by lightweight
dataclasses rather than the original Pydantic schemas. These helpers provide a
uniform way to extract structured text snippets from those objects so parser
agents can operate without caring about the underlying representation.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any, Iterable, List, Sequence, Tuple

_METHOD_SECTION_KEYWORDS: Tuple[str, ...] = (
    "method",
    "experimental",
    "fabrication",
    "device",
    "procedure",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    """Normalize whitespace to single spaces."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Return attribute or dict key if present, else default."""

    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _parse_level(raw_level: Any) -> int | None:
    """Convert MinerU text_level into an integer if possible."""

    if isinstance(raw_level, (int, float)):
        return int(raw_level)
    if isinstance(raw_level, str):
        try:
            return int(raw_level)
        except ValueError:
            return None
    return None


def html_to_text(value: str | None) -> str:
    """Convert HTML fragments into plain text."""

    if not value:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return _collapse_whitespace(unescape(_TAG_RE.sub(" ", value)))


def extract_methods_text(
    docbundle: Any,
    *,
    keywords: Sequence[str] = _METHOD_SECTION_KEYWORDS,
    min_length: int = 200,
    max_chars: int = 15000,
) -> Tuple[str, List[str], str]:
    """Extract text for methods/experimental sections from a DocBundle.

    Returns a tuple ``(methods_text, headings, fallback_text)`` where:

    - ``methods_text`` contains concatenated paragraphs from sections whose
      heading matches one of the keywords. The string is limited to ``max_chars``.
    - ``headings`` lists the matched section titles/headings (if any).
    - ``fallback_text`` provides a best-effort concatenation of the entire
      document text (also limited to ``max_chars``) for cases where no methods
      section is identified.
    """

    sections: Iterable[Any] = _safe_attr(docbundle, "sections", []) or []
    collected_chunks: List[str] = []
    headings: List[str] = []
    aggregate_chunks: List[str] = []
    in_methods = False

    for section in sections:
        title = _safe_attr(section, "title")
        content = _safe_attr(section, "content")
        text = _safe_attr(section, "text")
        text_level = _parse_level(_safe_attr(section, "text_level"))

        heading_text = None
        body_text = None
        is_heading = False

        if title:
            heading_text = title
            body_text = content or text or ""
            is_heading = True
        elif text_level is not None and text_level <= 2:
            heading_text = text or content or ""
            body_text = ""
            is_heading = True
        else:
            body_text = text or content or ""

        if body_text:
            aggregate_chunks.append(_collapse_whitespace(body_text))

        if is_heading:
            heading_lower = (heading_text or "").lower()
            matches = any(keyword in heading_lower for keyword in keywords)
            if matches:
                in_methods = True
                if heading_text:
                    headings.append(_collapse_whitespace(heading_text))
                if body_text:
                    collected_chunks.append(_collapse_whitespace(body_text))
                continue

            in_methods = False
            continue

        if in_methods and body_text:
            collected_chunks.append(_collapse_whitespace(body_text))

    methods_text = "\n\n".join(collected_chunks).strip()

    raw_text = _safe_attr(docbundle, "raw_text")
    if raw_text:
        fallback_text = _collapse_whitespace(str(raw_text))
    else:
        fallback_text = "\n\n".join(chunk for chunk in aggregate_chunks if chunk).strip()

    methods_text = methods_text[:max_chars]
    fallback_text = fallback_text[:max_chars]

    if len(methods_text) < min_length:
        methods_text = ""

    return methods_text, headings, fallback_text


def collect_table_snippets(
    docbundle: Any,
    *,
    max_tables: int = 5,
) -> List[str]:
    """Return plain-text snippets summarizing tables for heuristic parsing."""

    tables: Iterable[Any] = _safe_attr(docbundle, "tables", []) or []
    snippets: List[str] = []

    for table in tables:
        caption = _safe_attr(table, "caption")
        html = _safe_attr(table, "html")
        footnote = _safe_attr(table, "footnote")
        vlm_description = _safe_attr(table, "vlm_description")
        vlm_numbers = _safe_attr(table, "vlm_numbers") or []
        vlm_expressions = _safe_attr(table, "vlm_expressions") or []

        parts: List[str] = []
        for value in [
            caption,
            html_to_text(html),
            footnote,
            vlm_description,
        ]:
            if isinstance(value, str) and value.strip():
                parts.append(_collapse_whitespace(value))

        if vlm_numbers:
            parts.append(", ".join(str(num) for num in vlm_numbers))
        if vlm_expressions:
            parts.append(", ".join(str(expr) for expr in vlm_expressions))

        snippet = " ".join(parts).strip()
        if snippet:
            snippets.append(snippet)

        if len(snippets) >= max_tables:
            break

    return snippets
