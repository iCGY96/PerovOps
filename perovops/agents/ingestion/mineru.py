from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from perovops.utils.config import config
from perovops.utils.markdown_export import render_docbundle_markdown

from .schema import DocBundle, DocFigure, DocSection, DocTable
from .vlm import analyze_image

logger = logging.getLogger(__name__)


def _slugify_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _default_out_root(stage_dir: Optional[Path] = None) -> Path:
    if stage_dir is not None:
        stage_dir = stage_dir.resolve()
        stage_dir.mkdir(parents=True, exist_ok=True)
        out = stage_dir / "mineru"
        if out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
        out.mkdir(parents=True, exist_ok=True)
        return out

    root = config.cache_dir / "mineru_runs"
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = root / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ensure_mineru():
    if shutil.which("mineru"):
        return
    if Path(sys.executable).with_name("mineru").exists():
        return
    raise RuntimeError(
        "MinerU CLI not found. Please install it first: pip install -U 'mineru[core]'."
    )


def _run_mineru_single(input_pdf: Path, out_dir: Path, use_ocr: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["mineru", "-p", str(input_pdf), "-o", str(out_dir), "-b", "pipeline"]
    if use_ocr:
        cmd += ["-m", "ocr"]
    logger.info("Running MinerU: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _find_content_list_json(out_dir: Path, base: str) -> Optional[Path]:
    candidates = list(out_dir.rglob("*_content_list.json"))
    if not candidates:
        return None
    exact = [c for c in candidates if c.name == f"{base}_content_list.json"]
    if exact:
        return exact[0]
    prefix = [c for c in candidates if c.name.startswith(base + "_")]
    if prefix:
        return prefix[0]
    return candidates[0]


def _find_middle_json(out_dir: Path, base: str) -> Optional[Path]:
    candidates = list(out_dir.rglob("*_middle.json"))
    if not candidates:
        return None
    exact = [c for c in candidates if c.name == f"{base}_middle.json"]
    if exact:
        return exact[0]
    prefix = [c for c in candidates if c.name.startswith(base + "_")]
    if prefix:
        return prefix[0]
    return candidates[0]


def _resolve_img_path(
    preferred_root: Path,
    fallback_root: Path,
    rel_or_abs: Optional[str],
) -> Optional[Path]:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p if p.exists() else None
    abs1 = (preferred_root / rel_or_abs).resolve()
    if abs1.exists():
        return abs1
    name = p.name
    for cand in fallback_root.rglob(name):
        if cand.is_file():
            return cand.resolve()
    return None


def _content_list_from_middle(middle_path: Path) -> List[Dict[str, Any]]:
    with middle_path.open("r", encoding="utf-8") as handle:
        mid = json.load(handle)

    out: List[Dict[str, Any]] = []
    pdf_info = mid.get("pdf_info") or []
    for page in pdf_info:
        page_idx = page.get("page_idx", -1)

        for blk in page.get("images", []) or []:
            bbox = blk.get("bbox")
            img_path = None
            caption, foot = [], []
            for lv2 in blk.get("blocks", []) or []:
                typ = lv2.get("type")
                for line in lv2.get("lines", []) or []:
                    for span in line.get("spans", []) or []:
                        if typ == "image_body" and span.get("img_path"):
                            img_path = img_path or span.get("img_path")
                        elif typ == "image_caption" and span.get("content"):
                            caption.append(span["content"])
                        elif typ == "image_footnote" and span.get("content"):
                            foot.append(span["content"])
            if img_path:
                out.append(
                    {
                        "type": "image",
                        "img_path": img_path,
                        "img_caption": caption,
                        "img_footnote": foot,
                        "bbox": bbox,
                        "page_idx": page_idx,
                    }
                )

        for blk in page.get("tables", []) or []:
            bbox = blk.get("bbox")
            img_path = None
            table_caption, table_footnote = [], []
            table_body_html = None
            for lv2 in blk.get("blocks", []) or []:
                typ = lv2.get("type")
                for line in lv2.get("lines", []) or []:
                    for span in line.get("spans", []) or []:
                        if typ == "table_body":
                            img_path = img_path or span.get("img_path")
                            table_body_html = table_body_html or span.get("html") or span.get("content")
                        elif typ == "table_caption" and span.get("content"):
                            table_caption.append(span["content"])
                        elif typ == "table_footnote" and span.get("content"):
                            table_footnote.append(span["content"])
            out.append(
                {
                    "type": "table",
                    "img_path": img_path,
                    "table_body": table_body_html,
                    "table_caption": table_caption,
                    "table_footnote": table_footnote,
                    "bbox": bbox,
                    "page_idx": page_idx,
                }
            )

        for blk in page.get("para_blocks", []) or []:
            blk_type = blk.get("type")
            if blk_type not in {"text", "title"}:
                continue
            text_parts = []
            for line in blk.get("lines", []) or []:
                for span in line.get("spans", []) or []:
                    if span.get("type") == "text" and span.get("content"):
                        text_parts.append(span["content"])
            if text_parts:
                item: Dict[str, Any] = {
                    "type": "text",
                    "text": "".join(text_parts),
                    "bbox": blk.get("bbox"),
                    "page_idx": page_idx,
                }
                if blk_type == "title":
                    item["text_level"] = 1
                out.append(item)
    return out


def _analyze_images(
    images_to_analyze: List[Tuple[str, Path]],
    pdf_figures: List[DocFigure],
    pdf_tables: List[DocTable],
) -> None:
    if not images_to_analyze:
        return

    model_name = config.vlm_model
    provider_name = config.vlm_provider
    logger.info(
        "Analyzing %d images via %s vision model %s...",
        len(images_to_analyze),
        provider_name,
        model_name,
    )

    fig_by_path = {f.image_path.resolve(): f for f in pdf_figures if f.image_path}
    tbl_by_path = {t.image_path.resolve(): t for t in pdf_tables if t.image_path}

    for kind, img_path in images_to_analyze:
        figure = fig_by_path.get(img_path.resolve())
        table = tbl_by_path.get(img_path.resolve())
        identifier = None
        caption = None
        footnote = None

        if figure and kind == "figure":
            identifier = figure.identifier
            caption = figure.caption
            footnote = figure.footnote
        elif table and kind == "table":
            identifier = table.identifier
            caption = table.caption
            footnote = table.footnote

        try:
            description = analyze_image(
                img_path,
                kind=kind,
                identifier=identifier,
                caption=caption,
                footnote=footnote,
                model=model_name,
            )
        except Exception:
            description = None

        desc = description.strip() if isinstance(description, str) else None

        if figure and kind == "figure":
            figure.vlm_description = desc
            figure.vlm_numbers = []
            figure.vlm_expressions = []
        if table and kind == "table":
            table.vlm_description = desc
            table.vlm_numbers = []
            table.vlm_expressions = []


def load_pdf(
    pdf_path: str,
    extract_tables: bool = True,
    extract_images: bool = True,
    use_ocr: bool = False,
    working_dir: Path | str | None = None,
) -> DocBundle:
    _ensure_mineru()

    input_path = Path(pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    resolved_workdir: Optional[Path] = None
    if working_dir:
        resolved_workdir = Path(working_dir)

    out_root = _default_out_root(resolved_workdir)
    bundle = DocBundle(assets_dir=out_root)
    section_counter = len(bundle.sections) + 1
    table_counter = len(bundle.tables) + 1
    figure_counter = len(bundle.figures) + 1

    markdown_root = resolved_workdir if resolved_workdir else out_root
    markdown_root.mkdir(parents=True, exist_ok=True)
    combined_markdown_parts: List[str] = []
    per_pdf_metadata: List[Dict[str, Any]] = []

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError("Provided file is not a .pdf")
        pdfs = [input_path]
    else:
        pdfs = sorted(input_path.rglob("*.pdf"))
        if not pdfs:
            raise ValueError("No .pdf files found under the provided directory")

    for pdf in pdfs:
        base = _slugify_name(pdf.stem)
        per_file_out = out_root / base

        try:
            _run_mineru_single(pdf, per_file_out, use_ocr=use_ocr)
        except subprocess.CalledProcessError:
            logger.exception("MinerU failed: %s", pdf)
            continue

        content_list = _find_content_list_json(per_file_out, base)
        if not content_list:
            middle_json = _find_middle_json(per_file_out, base)
            if middle_json:
                logger.warning("content_list.json missing. Falling back to middle.json: %s", middle_json)
                items = _content_list_from_middle(middle_json)
                content_list_dir = middle_json.parent
            else:
                logger.warning("No content_list.json or middle.json found in: %s", per_file_out)
                items = []
                content_list_dir = per_file_out
        else:
            with content_list.open("r", encoding="utf-8") as handle:
                items = json.load(handle)
            content_list_dir = content_list.parent

        pdf_sections: List[DocSection] = []
        pdf_tables: List[DocTable] = []
        pdf_figures: List[DocFigure] = []
        images_to_analyze: List[Tuple[str, Path]] = []

        for item in items:
            typ = item.get("type")

            if typ == "text":
                text = (item.get("text") or "").strip()
                if text:
                    section = DocSection(
                        text=text,
                        page_idx=item.get("page_idx", -1),
                        text_level=item.get("text_level"),
                        bbox=item.get("bbox"),
                        source_pdf=pdf,
                        identifier=f"S{section_counter}",
                    )
                    bundle.sections.append(section)
                    pdf_sections.append(section)
                    section_counter += 1

            elif typ == "image" and extract_images:
                img_abs = _resolve_img_path(
                    preferred_root=content_list_dir,
                    fallback_root=per_file_out,
                    rel_or_abs=item.get("img_path"),
                )
                caption = " ".join(item.get("img_caption", []) or [])
                foot = " ".join(item.get("img_footnote", []) or [])
                figure = DocFigure(
                    image_path=img_abs,
                    caption=caption,
                    footnote=foot,
                    page_idx=item.get("page_idx", -1),
                    bbox=item.get("bbox"),
                    source_pdf=pdf,
                    identifier=f"F{figure_counter}",
                )
                bundle.figures.append(figure)
                pdf_figures.append(figure)
                figure_counter += 1
                if img_abs and img_abs.exists():
                    images_to_analyze.append(("figure", img_abs))

            elif typ == "table" and extract_tables:
                img_abs = _resolve_img_path(
                    preferred_root=content_list_dir,
                    fallback_root=per_file_out,
                    rel_or_abs=item.get("img_path"),
                )
                caption = " ".join(item.get("table_caption", []) or [])
                foot = " ".join(item.get("table_footnote", []) or [])
                html = item.get("table_body")
                table = DocTable(
                    image_path=img_abs,
                    html=html,
                    caption=caption,
                    footnote=foot,
                    page_idx=item.get("page_idx", -1),
                    bbox=item.get("bbox"),
                    source_pdf=pdf,
                    identifier=f"T{table_counter}",
                )
                bundle.tables.append(table)
                pdf_tables.append(table)
                table_counter += 1
                if img_abs and img_abs.exists():
                    images_to_analyze.append(("table", img_abs))

        _analyze_images(images_to_analyze, pdf_figures, pdf_tables)

        bundle.sources.append(pdf)

        pdf_bundle = DocBundle(
            sections=list(pdf_sections),
            tables=list(pdf_tables),
            figures=list(pdf_figures),
            sources=[pdf],
            assets_dir=bundle.assets_dir,
        )
        pdf_markdown, pdf_meta = render_docbundle_markdown(
            pdf_bundle,
            stage_workdir=markdown_root,
            llm_client=None,
        )

        combined_part = f"## {pdf.stem}\n\n{pdf_markdown}".strip()
        combined_markdown_parts.append(combined_part)
        per_pdf_metadata.append(
            {
                "pdf": str(pdf),
                "title": pdf.stem,
                "markdown_content": pdf_markdown,
                "metadata": pdf_meta,
            }
        )

    if combined_markdown_parts:
        bundle.combined_markdown = "\n\n---\n\n".join(
            part.strip() for part in combined_markdown_parts if part.strip()
        )
    else:
        bundle.combined_markdown = ""

    combined_markdown_path: Optional[Path] = None
    if resolved_workdir and bundle.combined_markdown:
        combined_filename = config.ingest_markdown_filename
        combined_markdown_path = (resolved_workdir / combined_filename).resolve()
        combined_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        combined_markdown_path.write_text(bundle.combined_markdown, encoding="utf-8")

    bundle.markdown_paths = [combined_markdown_path] if combined_markdown_path else []
    bundle.markdown_metadata = {
        "per_pdf_documents": per_pdf_metadata,
        "combined_delimiter": "---",
        "combined_markdown_filename": config.ingest_markdown_filename,
        "combined_markdown_path": str(combined_markdown_path) if combined_markdown_path else None,
    }

    return bundle


__all__ = [
    "load_pdf",
    "DocBundle",
    "DocSection",
    "DocTable",
    "DocFigure",
]
