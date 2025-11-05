from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocSection:
    text: str
    page_idx: int
    text_level: Optional[int] = None
    bbox: Optional[List[float]] = None
    source_pdf: Optional[Path] = None
    identifier: Optional[str] = None


@dataclass
class DocTable:
    image_path: Optional[Path] = None
    html: Optional[str] = None
    caption: str = ""
    footnote: str = ""
    page_idx: Optional[int] = None
    bbox: Optional[List[float]] = None
    source_pdf: Optional[Path] = None
    identifier: Optional[str] = None
    vlm_description: Optional[str] = None
    vlm_numbers: List[str] = field(default_factory=list)
    vlm_expressions: List[str] = field(default_factory=list)


@dataclass
class DocFigure:
    image_path: Optional[Path] = None
    caption: str = ""
    footnote: str = ""
    page_idx: Optional[int] = None
    bbox: Optional[List[float]] = None
    source_pdf: Optional[Path] = None
    identifier: Optional[str] = None
    vlm_description: Optional[str] = None
    vlm_numbers: List[str] = field(default_factory=list)
    vlm_expressions: List[str] = field(default_factory=list)


@dataclass
class DocBundle:
    sections: List[DocSection] = field(default_factory=list)
    tables: List[DocTable] = field(default_factory=list)
    figures: List[DocFigure] = field(default_factory=list)
    sources: List[Path] = field(default_factory=list)
    assets_dir: Optional[Path] = None
    markdown_paths: List[Path] = field(default_factory=list)
    combined_markdown: Optional[str] = None
    markdown_metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DocSection",
    "DocTable",
    "DocFigure",
    "DocBundle",
]
