"""Simplified PDF loader utilities used in unit tests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from perovops.io.schemas import DocBundle, Section

_CACHE: dict[str, DocBundle] = {}
_HEADER_PATTERN = re.compile(r"^(?:\d+\.?\s+)?[A-Z][A-Za-z0-9 \-/]{2,}$")


def validate_pdf(path: str) -> Tuple[bool, Optional[str]]:
    file_path = Path(path)
    if not file_path.exists():
        return False, "File not found"
    if file_path.suffix.lower() != ".pdf":
        return False, "Not a PDF file"
    with file_path.open("rb") as handle:
        header = handle.read(4)
    if not header.startswith(b"%PDF"):
        return False, "Invalid PDF signature"
    return True, None


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    normalized_lines: List[str] = []
    for line in lines:
        if not line and normalized_lines and not normalized_lines[-1]:
            continue
        normalized_lines.append(line)
    while normalized_lines and not normalized_lines[0]:
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)


def identify_section_headers(text: str) -> List[Tuple[str, int]]:
    headers: List[Tuple[str, int]] = []
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_PATTERN.match(stripped) or stripped.lower() in {
            "abstract",
            "introduction",
            "methods",
            "experimental methods",
            "results",
            "discussion",
            "device fabrication",
            "characterization",
        }:
            headers.append((stripped, idx))
    return headers


def compute_file_hash(path: str) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_sections(headers: Iterable[Tuple[str, int]], text: str) -> List[Section]:
    normalized_text = clean_text(text)
    header_list = list(headers)
    if not header_list:
        header_list = identify_section_headers(normalized_text)
    if not header_list:
        return [
            Section(
                title="Full Document",
                content=normalized_text,
                page_start=1,
                page_end=1,
            )
        ]

    sections: List[Section] = []
    lines = normalized_text.splitlines()
    resolved_positions: List[Tuple[str, int]] = []
    search_idx = 0
    for title, idx in header_list:
        match_idx = None
        for candidate in range(search_idx, len(lines)):
            if lines[candidate].strip() == title.strip():
                match_idx = candidate
                break
        if match_idx is None:
            match_idx = min(idx, len(lines) - 1)
        resolved_positions.append((title, match_idx))
        search_idx = match_idx + 1

    header_positions = resolved_positions + [("__END__", len(lines))]
    for (title, start_idx), (_, end_idx) in zip(header_positions, header_positions[1:]):
        content_lines = lines[start_idx + 1 : end_idx]
        content = "\n".join(content_lines).strip()
        sections.append(
            Section(
                title=title,
                content=content,
                page_start=1,
                page_end=1,
            )
        )
    return sections


def _extract_text_from_pdf(path: Path) -> str:
    with path.open("rb") as handle:
        data_bytes = handle.read()
    try:
        data = data_bytes.decode("latin-1")
    except UnicodeDecodeError:
        data = data_bytes.decode("utf-8", errors="ignore")
    text_candidates = re.findall(r"\((.*?)\)", data, flags=re.DOTALL)
    if text_candidates:
        return "\n".join(candidate.strip() for candidate in text_candidates)
    return data


def load_pdf(
    path: str,
    *,
    extract_tables: bool = True,
    extract_images: bool = True,
    use_cache: bool = False,
) -> DocBundle:
    valid, error = validate_pdf(path)
    if not valid:
        raise ValueError(error or "Invalid PDF")

    if use_cache and path in _CACHE:
        return _CACHE[path]

    file_path = Path(path)
    raw_text = _extract_text_from_pdf(file_path)
    headers = identify_section_headers(raw_text)
    sections = parse_sections(headers, raw_text)

    bundle = DocBundle(
        path=str(file_path),
        sections=sections,
        tables=[],
        figures=[],
        raw_text=raw_text,
        metadata={"pages": max(1, len(sections))},
    )

    if use_cache:
        _CACHE[path] = bundle
    return bundle
