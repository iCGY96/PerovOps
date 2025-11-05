from __future__ import annotations

import base64
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from perovops.models import LiteLLMClient, LiteLLMError, get_vision_llm_client
from perovops.utils.config import config

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "ingestion_vlm_prompt.md"
_VISION_CLIENT: Optional[LiteLLMClient] = None


def _load_base_prompt() -> List[str]:
    text = _prompt_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive fallback
        logger.warning("Falling back to baked-in ingestion VLM prompt: %s", exc)
        return (
            "You are an expert in photovoltaic materials. Provide a rigorous, evidence-based interpretation "
            "of the scientific figure.\n"
            "Follow the structure depicted in the image and report information strictly in the order it appears "
            "(e.g., from bottom to top in layer stacks, left to right in plots).\n"
            "For layered device schematics, enumerate every layer sequentially, giving material names, functional "
            "roles, and notable properties (carrier selectivity, transparency, thickness cues, interfaces).\n"
            "For experimental results (spectra, IV curves, microscopy), describe axes, labeled series, key trends, "
            "and quantitative readouts in the order presented.\n"
            "Highlight any annotations, callouts, or inset panels and explain how they support the main finding.\n"
            "Avoid speculation; ground statements in visible labels or patterns."
        )


def _build_prompt(
    *,
    kind: str,
    identifier: Optional[str],
    caption: Optional[str],
    footnote: Optional[str],
) -> str:
    parts: List[str] = _load_base_prompt()
    caption_clean = " ".join(str(caption).split()) if caption else ""
    footnote_clean = " ".join(str(footnote).split()) if footnote else ""
    if identifier:
        parts.append(f"Identifier: {identifier} ({kind})")
    if caption_clean:
        parts.append(f"Caption: {caption_clean}")
    if footnote_clean:
        parts.append(f"Footnote: {footnote_clean}")
    return " ".join(parts)


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"


def _encode_image_as_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    mime = _guess_mime(path)
    return f"data:{mime};base64,{b64}"


def _get_vision_client() -> LiteLLMClient:
    global _VISION_CLIENT
    if _VISION_CLIENT is None:
        try:
            _VISION_CLIENT = get_vision_llm_client(
                mode=config.get_agent_mode("ingestion", family="vision")
            )
        except LiteLLMError as exc:  # pragma: no cover - dependent on external service
            raise RuntimeError(f"LiteLLM vision client unavailable: {exc}") from exc
    return _VISION_CLIENT


def analyze_image(
    image_path: Path,
    *,
    kind: str,
    identifier: Optional[str] = None,
    caption: Optional[str] = None,
    footnote: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    if not config.vlm_api_key:
        raise RuntimeError(
            f"No API key configured for vision provider '{config.vlm_provider}'"
        )

    resolved_model = model or config.vlm_model
    if not resolved_model:
        raise RuntimeError("No vision model configured. Check llm.vision_model in defaults.yaml")

    data_url = _encode_image_as_data_url(image_path)
    prompt = _build_prompt(
        kind=kind,
        identifier=identifier,
        caption=caption,
        footnote=footnote,
    )

    try:
        vision_request = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        messages = [
            {"role": "system", "content": "You respond in English only."},
            {"role": "user", "content": vision_request},
        ]
        client = _get_vision_client()
        result = client.generate(messages, model=resolved_model)
        content = (result.content or "").strip()
        logger.debug("LiteLLM completion (model=%s) output: %s", resolved_model, content)
        return content or None
    except Exception as exc:  # pragma: no cover - depends on remote service availability
        logger.debug("LiteLLM VLM call failed for %s: %s", image_path, exc, exc_info=True)
        return None


__all__ = ["analyze_image"]
