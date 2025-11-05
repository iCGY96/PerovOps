from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from perovops.io.schemas import Citation, EnergyFlow, MaterialFlow, ProcessStep, Quantity
from perovops.models import LiteLLMError, get_text_llm_client
from perovops.utils.config import config

try:  # pragma: no cover - optional dependency
    import litellm  # type: ignore
except Exception:  # pragma: no cover - fallback when LiteLLM unavailable
    litellm = None  # type: ignore

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Base parser error."""


class LLMError(ParserError):
    """Raised when LLM interaction fails."""


class ParseError(ParserError):
    """Raised when structured payloads cannot be parsed."""


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PLACEHOLDER_PATTERN = re.compile(r"{{\s*([A-Z0-9_]+)\s*}}")
_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_BASE_SYSTEM_PROMPT = (
    "You are Perovops's structured extraction agent. "
    "Always emit valid JSON that matches the provided schema. "
    "All factual statements must be grounded in the supplied document context and include citations."
)


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------


def _load_prompt(prompt_name: str) -> str:
    prompt_path = _PROMPT_DIR / prompt_name
    if not prompt_path.exists():
        raise ParserError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


@lru_cache(maxsize=16)
def _cached_prompt(prompt_name: str) -> str:
    return _load_prompt(prompt_name)


def _render_prompt(prompt_name: str, variables: Dict[str, str]) -> str:
    template = _cached_prompt(prompt_name)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, "")

    return _PLACEHOLDER_PATTERN.sub(_replace, template)


def extract_json(response: str) -> Any:
    """Extract a JSON payload from an LLM response string."""
    if not isinstance(response, str):
        raise ParseError("LLM response must be a string")
    text = response.strip()
    if not text:
        raise ParseError("Empty LLM response")

    fenced_blocks = re.findall(r"```(?:json)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        candidate = block.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    brace_idx = text.find("{")
    bracket_idx = text.find("[")
    start = -1
    if brace_idx != -1 and (bracket_idx == -1 or brace_idx < bracket_idx):
        start = brace_idx
    elif bracket_idx != -1:
        start = bracket_idx

    if start == -1:
        raise ParseError("No JSON object or array found in response")

    substring = text[start:]
    for end in range(len(substring), 0, -1):
        candidate = substring[:end].strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ParseError("Failed to parse JSON from response")


def extract_json_from_response(response: str) -> str:
    """Return JSON payload embedded in an LLM response as a string."""
    payload = extract_json(response)
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _collect_vlm_context_lines(item: Dict[str, Any], indent: str = "    ") -> List[str]:
    """Return human-readable lines summarising vision-language model signals."""
    if not isinstance(item, dict):
        return []

    lines: List[str] = []

    desc_value = item.get("vlm_description")
    if isinstance(desc_value, str):
        desc = desc_value.strip()
        if desc:
            lines.append(f"{indent}Auto summary: {desc}")

    number_values = [value for value in _as_list(item.get("vlm_numbers")) if value is not None]
    formatted_numbers = [str(value).strip() for value in number_values if str(value).strip()]
    if formatted_numbers:
        lines.append(f"{indent}Numbers: {'; '.join(formatted_numbers)}")

    expr_values = [value for value in _as_list(item.get("vlm_expressions")) if value is not None]
    formatted_exprs = [str(value).strip() for value in expr_values if str(value).strip()]
    if formatted_exprs:
        lines.append(f"{indent}Expressions: {'; '.join(formatted_exprs)}")

    return lines


def build_document_context(docbundle: Any) -> str:
    """Create a structured prompt context from a docbundle payload."""
    if docbundle is None:
        return ""
    if isinstance(docbundle, Path):
        docbundle = json.loads(docbundle.read_text(encoding="utf-8"))
    if not isinstance(docbundle, dict):
        try:
            docbundle = json.loads(json.dumps(docbundle))
        except TypeError as exc:  # pragma: no cover - defensive
            raise ParserError("Docbundle must be JSON serialisable") from exc

    lines: List[str] = []
    metadata = docbundle.get("metadata", {})
    doc_id = metadata.get("doc_id") or docbundle.get("path")
    title = metadata.get("title")
    if doc_id:
        lines.append(f"# Document: {doc_id}")
    if title:
        lines.append(f"Title: {title}")

    sections = _as_list(docbundle.get("sections"))

    for idx, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            continue
        label = section.get("identifier") or f"S{idx}"
        heading = section.get("title") or "Untitled Section"
        span = []
        if section.get("page_start") is not None:
            span.append(str(section.get("page_start")))
        if section.get("page_end") not in (None, section.get("page_start")):
            span.append(str(section.get("page_end")))
        page_span = "-".join(span) if span else ""
        header = f"[{label}] {heading}"
        if page_span:
            header += f" (pp. {page_span})"
        lines.append(header)
        content = section.get("content") or section.get("text") or ""
        if content:
            lines.append(content.strip())
        lines.extend(_collect_vlm_context_lines(section))

    tables = _as_list(docbundle.get("tables"))
    for idx, table in enumerate(tables, start=1):
        if not isinstance(table, dict):
            continue
        caption = table.get("caption") or f"Table {idx}"
        identifier = table.get("identifier")
        label = str(identifier).strip() if isinstance(identifier, str) and identifier.strip() else f"T{idx}"
        lines.append(f"[{label}] {caption}")
        dataframe = table.get("dataframe")
        if isinstance(dataframe, dict):
            cols = dataframe.get("columns") or []
            data = dataframe.get("data") or []
            lines.append("Columns: " + ", ".join(map(str, cols)))
            for row in data:
                if isinstance(row, list):
                    lines.append(" | ".join(map(str, row)))
        lines.extend(_collect_vlm_context_lines(table))

    figures = _as_list(docbundle.get("figures"))
    for idx, figure in enumerate(figures, start=1):
        if not isinstance(figure, dict):
            continue
        caption = figure.get("caption") or "Figure"
        identifier = figure.get("identifier")
        label = str(identifier).strip() if isinstance(identifier, str) and identifier.strip() else f"F{idx}"
        lines.append(f"[{label}] {caption}")
        lines.extend(_collect_vlm_context_lines(figure))

    raw_text = docbundle.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        lines.append("# Raw Snippets")
        lines.append(raw_text.strip())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pydantic models for agent payloads
# ---------------------------------------------------------------------------

_SCOPE_EMBEDDING_QUERIES: tuple[str, ...] = (
    "Detailed descriptions of halide perovskite device structures, layer stacks, "
    "compositions, synthesis and fabrication steps, deposition methods, precursors, "
    "and processing conditions relevant to perovskite materials.",
    "Schematic view of perovskite solar cell architecture, including each layer in the device stack.",
    "Cell design and microstructure of the perovskite top cell on a textured silicon heterojunction bottom cell.",
    "Cross-sectional illustration or schematic of halide perovskite devices showing materials and fabrication steps.",
    "Layer-by-layer device design, stack schematic, or fabrication flow for perovskite solar cells and tandem structures.",
    "Layered device schematic highlighting each functional layer in a perovskite solar cell stack.",
    "Process descriptions detailing synthesis routes, deposition parameters, and annealing treatments for perovskite films.",
)


@lru_cache(maxsize=1)
def _scope_embedding_model() -> str:
    model_name = config.llm_embedding_model
    if not model_name:
        raise ParserError("Embedding model is not configured for scope retrieval scoring")
    return model_name


def _ensure_embedding_runtime() -> None:
    if litellm is None:
        raise LLMError("LiteLLM is not installed; embedding-based scoring is unavailable")


@lru_cache(maxsize=1)
def _scope_query_embeddings() -> tuple[tuple[float, ...], ...]:
    _ensure_embedding_runtime()
    model_name = _scope_embedding_model()
    try:
        response = litellm.embedding(model=model_name, input=list(_SCOPE_EMBEDDING_QUERIES))
    except Exception as exc:  # pragma: no cover - network/SDK failure
        raise LLMError(f"Failed to generate scope query embedding: {exc}") from exc
    vectors = _extract_embedding_vector(response, expected_length=len(_SCOPE_EMBEDDING_QUERIES))
    return tuple(vectors)


def _extract_embedding_vector(
    response: Dict[str, Any],
    *,
    expected_length: Optional[int] = None,
) -> List[tuple[float, ...]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise LLMError("Embedding response missing data array")
    if expected_length is not None and len(data) != expected_length:
        raise LLMError(
            f"Embedding response length mismatch: expected {expected_length}, got {len(data)}"
        )
    vectors: List[tuple[float, ...]] = []
    for item in data:
        embedding = item.get("embedding")
        if not isinstance(embedding, Sequence):
            raise LLMError("Embedding entry missing numeric vector")
        try:
            vector = tuple(float(value) for value in embedding)
        except (TypeError, ValueError) as exc:
            raise LLMError("Embedding vector contains non-numeric values") from exc
        vectors.append(vector)
    return vectors


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _batch_snippet_embeddings(snippets: Sequence[str]) -> List[tuple[float, ...]]:
    cleaned = [snippet.strip() or "perovskite" for snippet in snippets]
    if not cleaned:
        return []
    _ensure_embedding_runtime()
    model_name = _scope_embedding_model()
    try:
        response = litellm.embedding(model=model_name, input=cleaned)
    except Exception as exc:  # pragma: no cover - network/SDK failure
        raise LLMError(f"Failed to generate snippet embeddings: {exc}") from exc
    # LiteLLM preserves ordering via the index field; ensure alignment
    vectors = _extract_embedding_vector(response)
    if len(vectors) != len(cleaned):
        raise LLMError("Embedding response count does not match the number of snippets")
    return vectors


def _score_scope_candidates(candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return
    try:
        query_embeddings = _scope_query_embeddings()
        if not query_embeddings:
            raise LLMError("No query embeddings available for scope scoring")
        snippet_embeddings = _batch_snippet_embeddings([item["raw_text"] for item in candidates])
        for item, embedding in zip(candidates, snippet_embeddings):
            item["score"] = max(_cosine_similarity(query_embedding, embedding) for query_embedding in query_embeddings)
    except LLMError as exc:
        logger.warning("Scope embedding scoring failed; defaulting scores to zero: %s", exc)
        for item in candidates:
            item["score"] = 0.0


def _scope_candidate_lines(
    *,
    entry_type: str,
    label: str,
    header: str,
    body_lines: List[str],
    raw_text: str,
    order: int,
) -> Dict[str, Any]:
    return {
        "type": entry_type,
        "label": label,
        "lines": [header] + body_lines,
        "raw_text": raw_text,
        "score": 0.0,
        "order": order,
    }


def build_scope_document_context(docbundle: Dict[str, Any], *, max_entries: int = 10) -> str:
    """Construct a RAG-filtered context focusing on perovskite structure and synthesis."""
    if not isinstance(docbundle, dict):
        return ""  

    metadata = docbundle.get("metadata", {})
    doc_id = metadata.get("doc_id") or docbundle.get("path")
    title = metadata.get("title")

    candidates: List[Dict[str, Any]] = []
    order_counter = 0

    sections = _as_list(docbundle.get("sections"))
    for idx, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            continue
        label = section.get("identifier") or f"S{idx}"
        heading = section.get("title") or "Untitled Section"
        span = []
        if section.get("page_start") is not None:
            span.append(str(section.get("page_start")))
        if section.get("page_end") not in (None, section.get("page_start")):
            span.append(str(section.get("page_end")))
        page_span = "-".join(span) if span else ""
        header = f"[{label}] {heading}"
        if page_span:
            header += f" (pp. {page_span})"

        content = section.get("content") or section.get("text") or ""
        body_lines: List[str] = []
        if content:
            body_lines.append(content.strip())
        body_lines.extend(_collect_vlm_context_lines(section))

        raw_text_fragments = [heading, content]
        raw_text_fragments.extend(body_lines)
        raw_text = " ".join(fragment for fragment in raw_text_fragments if fragment)
        candidates.append(
            _scope_candidate_lines(
                entry_type="section",
                label=label,
                header=header,
                body_lines=body_lines,
                raw_text=raw_text,
                order=order_counter,
            )
        )
        order_counter += 1

    tables = _as_list(docbundle.get("tables"))
    for idx, table in enumerate(tables, start=1):
        if not isinstance(table, dict):
            continue
        caption = table.get("caption") or f"Table {idx}"
        identifier = table.get("identifier")
        label = str(identifier).strip() if isinstance(identifier, str) and identifier.strip() else f"T{idx}"
        header = f"[{label}] {caption}"
        body_lines: List[str] = []
        dataframe = table.get("dataframe")
        raw_chunks: List[str] = [caption]
        if isinstance(dataframe, dict):
            cols = dataframe.get("columns") or []
            data = dataframe.get("data") or []
            if cols:
                columns_line = "Columns: " + ", ".join(map(str, cols))
                body_lines.append(columns_line)
                raw_chunks.append(columns_line)
            for row in data:
                if isinstance(row, list):
                    row_line = " | ".join(map(str, row))
                    body_lines.append(row_line)
                    raw_chunks.append(row_line)
        body_lines.extend(_collect_vlm_context_lines(table))
        raw_chunks.extend(body_lines)
        candidates.append(
            _scope_candidate_lines(
                entry_type="table",
                label=label,
                header=header,
                body_lines=body_lines,
                raw_text=" ".join(raw_chunks),
                order=order_counter,
            )
        )
        order_counter += 1

    figures = _as_list(docbundle.get("figures"))
    for idx, figure in enumerate(figures, start=1):
        if not isinstance(figure, dict):
            continue
        caption = figure.get("caption") or "Figure"
        identifier = figure.get("identifier")
        label = str(identifier).strip() if isinstance(identifier, str) and identifier.strip() else f"F{idx}"
        header = f"[{label}] {caption}"
        vlm_lines = _collect_vlm_context_lines(figure)
        raw_chunks = [caption] + vlm_lines
        candidates.append(
            _scope_candidate_lines(
                entry_type="figure",
                label=label,
                header=header,
                body_lines=vlm_lines,
                raw_text=" ".join(raw_chunks),
                order=order_counter,
            )
        )
        order_counter += 1

    _score_scope_candidates(candidates)

    candidates_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        candidates_by_type.setdefault(item["type"], []).append(item)

    selected: List[Dict[str, Any]] = []
    for entry_type in ("section", "table", "figure"):
        typed_candidates = candidates_by_type.get(entry_type, [])
        if not typed_candidates:
            continue
        typed_candidates.sort(key=lambda item: (-item["score"], item["order"]))
        selected.extend(typed_candidates[:max_entries])

    if not selected:
        selected = candidates[: max_entries * 3]

    relevant = sorted(selected, key=lambda item: item["order"])
    for item in relevant:
        item.pop("raw_text", None)

    lines: List[str] = []
    if doc_id:
        lines.append(f"# Document: {doc_id}")
    if title:
        lines.append(f"Title: {title}")
    lines.append("# Retrieved Context (perovskite structure & synthesis)")

    if relevant:
        lines.append("# Selected Entries")
        for idx, item in enumerate(relevant, start=1):
            entry_type = item["type"].capitalize()
            label = item["label"]
            score = item["score"]
            lines.append(f"{idx}. {entry_type} {label} (score={score:.2f})")

    for item in relevant:
        lines.extend(item["lines"])

    return "\n".join(lines)


class CitationInfo(BaseModel):
    identifier: str
    snippet: str
    unit: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class ScopeLayer(BaseModel):
    role: str
    material: str
    formula: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "ScopeLayer":
        role = (self.role or "").strip()
        material = (self.material or "").strip()
        formula = (self.formula or "").strip() or None
        if not role:
            raise ValueError("Layer role cannot be empty")
        if not material:
            raise ValueError("Layer material cannot be empty")
        self.role = role
        self.material = material
        self.formula = formula
        return self


class ScopeCitation(BaseModel):
    identifier: str
    snippet: str

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize(self) -> "ScopeCitation":
        identifier = (self.identifier or "").strip()
        snippet = (self.snippet or "").strip()
        if not identifier:
            raise ValueError("Citation identifier cannot be empty")
        if not snippet:
            raise ValueError("Citation snippet cannot be empty")
        self.identifier = identifier
        self.snippet = snippet
        return self


class AgentDialogueTurn(BaseModel):
    speaker: str
    message: str

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _ensure_message(self) -> "AgentDialogueTurn":
        if not self.speaker or not self.speaker.strip():
            raise ValueError("Dialogue speaker cannot be empty")
        if not self.message or not self.message.strip():
            raise ValueError("Dialogue message cannot be empty")
        self.speaker = self.speaker.strip()
        self.message = self.message.strip()
        return self


class ScopeInit(BaseModel):
    device_name: str
    architecture: Optional[str] = None
    expression: str
    layers: List[ScopeLayer] = Field(default_factory=list)
    route_overview: List[str] = Field(default_factory=list)
    candidate_processes: List[str] = Field(default_factory=list)
    functional_unit: str = "1 m^2"
    notes: List[str] = Field(default_factory=list)
    citations: List[ScopeCitation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _ensure_route_overview(self) -> "ScopeInit":
        self.device_name = (self.device_name or "").strip()
        self.expression = (self.expression or "").strip()
        if not self.device_name:
            raise ValueError("Device name is required")
        if not self.expression:
            raise ValueError("Device expression is required")

        if self.architecture is not None:
            architecture = self.architecture.strip().lower()
            if architecture not in {"p-i-n", "n-i-p", "tandem"}:
                raise ValueError("Architecture must be one of 'p-i-n', 'n-i-p', or 'tandem'")
            self.architecture = architecture

        if not self.layers:
            raise ValueError("At least one device layer is required")

        normalized_candidates: Dict[str, str] = {}
        for item in self.candidate_processes:
            if not isinstance(item, str):
                continue
            token = item.strip().lower()
            if token:
                normalized_candidates[token] = token
        self.candidate_processes = list(normalized_candidates.values())

        if not self.route_overview:
            self.route_overview = list(self.candidate_processes)
        if not self.route_overview:
            self.route_overview = [layer.role for layer in self.layers[:3]]

        fu = (self.functional_unit or "").strip()
        self.functional_unit = fu or "1 m^2"

        return self


class ScopeHistoryEntry(BaseModel):
    attempt: int
    status: Literal["ok", "error"]
    prompt_digest: Optional[str] = None
    scope: Optional[ScopeInit] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _ensure_payload(self) -> "ScopeHistoryEntry":
        if self.status == "ok" and self.scope is None:
            raise ValueError("Successful scope attempt must include scope data")
        if self.status == "error" and not self.error:
            raise ValueError("Failed scope attempt must include an error message")
        return self


class ProcessEvidence(BaseModel):
    statement: str
    citation: CitationInfo

    model_config = ConfigDict(extra="forbid")


class ProcessNode(BaseModel):
    id: str
    name: str
    stage: str
    description: Optional[str] = None
    parameters: Dict[str, str] = Field(default_factory=dict)
    equipment: List[str] = Field(default_factory=list)
    evidence: List[ProcessEvidence]
    citations: List[CitationInfo]
    confidence: float = Field(ge=0.0, le=1.0)
    chemical_expression: Optional[str] = None
    expected_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    energy_expressions: List[str] = Field(default_factory=list)
    reaction_conditions: List[str] = Field(default_factory=list)
    normalized_expression: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class ProcessAgentResult(BaseModel):
    stop_candidate: bool = Field(alias="STOP_CANDIDATE", default=False)
    process: Optional[ProcessNode] = None
    issues: List[str] = Field(default_factory=list)
    conversation: List[AgentDialogueTurn] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FlowQuantity(BaseModel):
    value: str
    unit: str
    basis: Optional[str] = None
    method: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class MaterialRecord(BaseModel):
    material_id: str
    name: str
    role: str
    quantity: FlowQuantity
    normalized_name: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    citations: List[CitationInfo]
    confidence: float = Field(ge=0.0, le=1.0)
    chemical_expression: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class EnergyRecord(BaseModel):
    energy_type: str
    quantity: FlowQuantity
    estimation_method: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    citations: List[CitationInfo]
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class FlowInventory(BaseModel):
    process_id: str
    inputs: List[MaterialRecord] = Field(default_factory=list)
    outputs: List[MaterialRecord] = Field(default_factory=list)
    energy: List[EnergyRecord] = Field(default_factory=list)
    normalization_actions: List[str] = Field(default_factory=list)
    deduplication_actions: List[str] = Field(default_factory=list)
    citations: List[CitationInfo]
    confidence: float = Field(ge=0.0, le=1.0)
    reaction_conditions: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MaterialAgentResult(BaseModel):
    process_id: str
    inputs: List[MaterialRecord] = Field(default_factory=list)
    outputs: List[MaterialRecord] = Field(default_factory=list)
    normalization_actions: List[str] = Field(default_factory=list)
    deduplication_actions: List[str] = Field(default_factory=list)
    citations: List[CitationInfo]
    confidence: float = Field(ge=0.0, le=1.0)
    open_issues: List[str] = Field(default_factory=list)
    inventory_expression: Optional[str] = None
    conversation: List[AgentDialogueTurn] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class EnergyAgentResult(BaseModel):
    process_id: str
    energy: List[EnergyRecord] = Field(default_factory=list)
    citations: List[CitationInfo]
    confidence: float = Field(ge=0.0, le=1.0)
    open_issues: List[str] = Field(default_factory=list)
    reaction_conditions: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class FlowEdge(BaseModel):
    process_id: str
    name: str
    kind: str
    quantity: FlowQuantity
    role: Optional[str] = None
    energy_type: Optional[str] = None
    normalized_name: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    citations: List[CitationInfo] = Field(default_factory=list)
    material_id: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class SharedState(BaseModel):
    doc_id: str
    scope: Optional[ScopeInit] = None
    scope_history: List[ScopeHistoryEntry] = Field(default_factory=list)
    scope_output_dir: Optional[Path] = None
    processes: Dict[str, ProcessNode] = Field(default_factory=dict)
    flow_inventories: Dict[str, FlowInventory] = Field(default_factory=dict)
    material_nodes: Dict[str, MaterialRecord] = Field(default_factory=dict)
    process_material_map: Dict[str, List[str]] = Field(default_factory=dict)
    open_issues: List[str] = Field(default_factory=list)
    process_inventory: Dict[str, str] = Field(default_factory=dict)
    material_inventory: Dict[str, str] = Field(default_factory=dict)
    conversation_history: List[AgentDialogueTurn] = Field(default_factory=list)

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, extra="forbid")

    def register_scope(self, scope: ScopeInit) -> None:
        self.scope = scope
        expression = scope.expression.strip()
        if expression:
            self.process_inventory["scope_device"] = expression
            self.material_inventory["scope_device"] = expression

    def record_scope_attempt(self, entry: ScopeHistoryEntry | Dict[str, Any]) -> None:
        if isinstance(entry, ScopeHistoryEntry):
            validated = entry
        else:
            validated = ScopeHistoryEntry.model_validate(entry)
        self.scope_history.append(validated)
        if len(self.scope_history) > 20:
            self.scope_history = self.scope_history[-20:]

    def upsert_process(self, process: ProcessNode) -> None:
        self.processes[process.id] = process
        expression = (process.normalized_expression or process.chemical_expression or "").strip()
        if not expression:
            expression = self._default_process_expression(process)
        if expression:
            self.process_inventory[process.id] = expression

    def upsert_flows(self, flows: FlowInventory) -> None:
        self.flow_inventories[flows.process_id] = flows
        for record in list(flows.inputs) + list(flows.outputs):
            self._register_material_node(flows.process_id, record)

    def ensure_flow_inventory(self, process_id: str) -> FlowInventory:
        inventory = self.flow_inventories.get(process_id)
        if inventory is None:
            inventory = FlowInventory(
                process_id=process_id,
                citations=[],
                confidence=0.0,
            )
            self.flow_inventories[process_id] = inventory
        return inventory

    def _register_material_node(self, process_id: str, record: MaterialRecord) -> bool:
        added = record.material_id not in self.material_nodes
        self.material_nodes[record.material_id] = record
        mapping = self.process_material_map.setdefault(process_id, [])
        if record.material_id not in mapping:
            mapping.append(record.material_id)
        return added

    @staticmethod
    def _merge_citations(existing: Sequence[CitationInfo], incoming: Sequence[CitationInfo]) -> List[CitationInfo]:
        combined: Dict[tuple[str, str, Optional[str]], CitationInfo] = {}
        for item in list(existing) + list(incoming):
            key = (item.identifier, item.snippet, item.unit)
            if key not in combined or item.confidence > combined[key].confidence:
                combined[key] = item
        return list(combined.values())

    @staticmethod
    def _material_token(record: MaterialRecord) -> str:
        name = (record.name or "unknown").strip() or "unknown"
        formula = (record.chemical_expression or "").strip()
        if formula:
            return f"{name} ({formula})"
        return name

    @classmethod
    def _default_process_expression(cls, process: ProcessNode) -> str:
        inputs = process.expected_inputs or []
        outputs = process.expected_outputs or []
        left = " + ".join(item.strip() for item in inputs if item and str(item).strip()) or "∅"
        right = " + ".join(item.strip() for item in outputs if item and str(item).strip()) or "∅"
        return f"{process.id}: {left} -> {right}" if process.id else f"{left} -> {right}"

    @classmethod
    def _default_inventory_expression(
        cls,
        process_id: str,
        inputs: Sequence[MaterialRecord],
        outputs: Sequence[MaterialRecord],
    ) -> str:
        left = " + ".join(cls._material_token(record) for record in inputs) or "∅"
        right = " + ".join(cls._material_token(record) for record in outputs) or "∅"
        prefix = f"{process_id}: " if process_id else ""
        return f"{prefix}{left} -> {right}"

    def update_material_inventory(
        self,
        *,
        process_id: str,
        inputs: Sequence[MaterialRecord],
        outputs: Sequence[MaterialRecord],
        normalization_actions: Sequence[str],
        deduplication_actions: Sequence[str],
        citations: Sequence[CitationInfo],
        confidence: float,
        inventory_expression: Optional[str] = None,
    ) -> List[str]:
        inventory = self.ensure_flow_inventory(process_id)
        inventory.inputs = list(inputs)
        inventory.outputs = list(outputs)
        inventory.normalization_actions = list(normalization_actions)
        inventory.deduplication_actions = list(deduplication_actions)
        inventory.citations = self._merge_citations(inventory.citations, citations)
        inventory.confidence = max(confidence, inventory.confidence)
        self.flow_inventories[process_id] = inventory
        expression = (inventory_expression or "").strip()
        added_ids: List[str] = []
        for record in list(inputs) + list(outputs):
            if self._register_material_node(process_id, record):
                added_ids.append(record.material_id)
        if not expression:
            expression = self._default_inventory_expression(process_id, inputs, outputs)
        if expression:
            self.material_inventory[process_id] = expression
        return list(dict.fromkeys(added_ids))

    def update_energy_inventory(
        self,
        *,
        process_id: str,
        energy: Sequence[EnergyRecord],
        reaction_conditions: Sequence[str],
        citations: Sequence[CitationInfo],
        confidence: float,
    ) -> FlowInventory:
        inventory = self.ensure_flow_inventory(process_id)
        inventory.energy = list(energy)
        inventory.reaction_conditions = list(reaction_conditions)
        inventory.citations = self._merge_citations(inventory.citations, citations)
        inventory.confidence = max(confidence, inventory.confidence)
        self.flow_inventories[process_id] = inventory
        process = self.processes.get(process_id)
        if process is not None:
            energy_expressions: List[str] = []
            for record in energy:
                raw_value = record.quantity.value
                if isinstance(raw_value, str):
                    value = raw_value.strip() or "unknown"
                else:
                    value = str(raw_value)
                unit = (record.quantity.unit or "").strip()
                expression = f"{(record.energy_type or 'energy').strip() or 'energy'} = {value}"
                if unit:
                    expression += f" {unit}"
                if record.quantity.basis:
                    expression += f" (basis={record.quantity.basis})"
                if record.quantity.method:
                    expression += f" [method={record.quantity.method}]"
                energy_expressions.append(expression)
            process.energy_expressions = energy_expressions
            if reaction_conditions:
                process.reaction_conditions = list(reaction_conditions)
            self.processes[process_id] = process
        return inventory

    def add_issues(self, issues: Sequence[str]) -> None:
        for item in issues:
            if not item:
                continue
            item_str = str(item).strip()
            if item_str and item_str not in self.open_issues:
                self.open_issues.append(item_str)

    def material_inventory_summary(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for material_id, node in self.material_nodes.items():
            summary.append(
                {
                    "material_id": material_id,
                    "name": node.name,
                    "role": node.role,
                    "chemical_expression": node.chemical_expression,
                    "confidence": node.confidence,
                    "linked_processes": [
                        pid
                        for pid, ids in self.process_material_map.items()
                        if material_id in ids
                    ],
                }
            )
        return summary

    def process_inventory_summary(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        if self.scope is not None:
            summary.append(
                {
                    "process_id": "scope_device",
                    "name": self.scope.device_name,
                    "stage": "device_scope",
                    "normalized_expression": self.process_inventory.get(
                        "scope_device",
                        self.scope.expression,
                    ),
                }
            )
        for process_id, process in self.processes.items():
            expression = self.process_inventory.get(process_id) or (process.chemical_expression or None)
            summary.append(
                {
                    "process_id": process_id,
                    "name": process.name,
                    "stage": process.stage,
                    "normalized_expression": expression,
                }
            )
        return summary

    def material_inventory_overview(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for process_id, expression in self.material_inventory.items():
            summary.append(
                {
                    "process_id": process_id,
                    "inventory_expression": expression,
                }
            )
        return summary

    def record_conversation(self, turns: Sequence[AgentDialogueTurn]) -> None:
        if not turns:
            return
        for turn in turns:
            if isinstance(turn, AgentDialogueTurn):
                validated = turn
            else:
                validated = AgentDialogueTurn.model_validate(turn)
            self.conversation_history.append(validated)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def conversation_summary(self, limit: int = 10) -> List[Dict[str, str]]:
        if limit <= 0:
            limit = 10
        recent = self.conversation_history[-limit:]
        return [
            {"speaker": turn.speaker, "message": turn.message}
            for turn in recent
        ]

    def energy_inventory_summary(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for process in self.sorted_processes():
            summary.append(
                {
                    "process_id": process.id,
                    "name": process.name,
                    "chemical_expression": process.chemical_expression,
                    "energy_expressions": list(process.energy_expressions),
                    "reaction_conditions": list(process.reaction_conditions),
                }
            )
        return summary

    def prompt_snapshot(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.model_dump() if self.scope else None,
            "process_ids": list(self.processes.keys()),
            "issues": list(self.open_issues),
            "flow_counts": {
                "materials": len(self.material_nodes),
                "energy": sum(1 for process in self.processes.values() if process.energy_expressions),
            },
            "process_inventory": self.process_inventory_summary(),
            "material_inventory": self.material_inventory_overview(),
            "conversation": self.conversation_summary(limit=5),
        }

    def sorted_processes(self) -> List[ProcessNode]:
        return [self.processes[key] for key in sorted(self.processes.keys())]

    def build_edges(self) -> List[FlowEdge]:
        edges: List[FlowEdge] = []
        for process_id in sorted(self.flow_inventories.keys()):
            inventory = self.flow_inventories[process_id]
            for record in inventory.inputs:
                edges.append(
                    FlowEdge(
                        process_id=process_id,
                        name=record.name,
                        kind="material_input",
                        quantity=record.quantity,
                        role=record.role,
                        normalized_name=record.normalized_name,
                        notes=record.notes,
                        citations=record.citations,
                        material_id=record.material_id,
                    )
                )
            for record in inventory.outputs:
                edges.append(
                    FlowEdge(
                        process_id=process_id,
                        name=record.name,
                        kind="material_output",
                        quantity=record.quantity,
                        role=record.role,
                        normalized_name=record.normalized_name,
                        notes=record.notes,
                        citations=record.citations,
                        material_id=record.material_id,
                    )
                )
            for record in inventory.energy:
                edges.append(
                    FlowEdge(
                        process_id=process_id,
                        name=record.energy_type,
                        kind="energy",
                        energy_type=record.energy_type,
                        quantity=record.quantity,
                        notes=record.notes,
                        citations=record.citations,
                    )
                )
        return edges

    def coverage_summary(self) -> Dict[str, Any]:
        process_count = len(self.processes)
        material_edges = sum(len(inv.inputs) + len(inv.outputs) for inv in self.flow_inventories.values())
        energy_edges = sum(len(inv.energy) for inv in self.flow_inventories.values())
        return {
            "doc_id": self.doc_id,
            "device_name": self.scope.device_name if self.scope else None,
            "functional_unit": self.scope.functional_unit if self.scope else None,
            "route_overview": self.scope.route_overview if self.scope else [],
            "process_count": process_count,
            "material_flow_count": material_edges,
            "energy_flow_count": energy_edges,
        }

    def log_progress(self, process: ProcessNode) -> None:
        slug = process.name.lower().replace(" ", "_")
        edges = len(self.build_edges())
        message = (
            f"[{self.doc_id}] round={len(self.processes)} added={slug} "
            f"edges={edges} open_issues={len(self.open_issues)}"
        )
        logger.info(message)
        print(message)


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------


def collect_process_details(shared_state: SharedState) -> List[Dict[str, Any]]:
    """Return a structured summary for each process with inventory metrics."""

    details: List[Dict[str, Any]] = []
    edges_by_process: Dict[str, List[FlowEdge]] = {}
    for edge in shared_state.build_edges():
        edges_by_process.setdefault(edge.process_id, []).append(edge)

    for process in shared_state.sorted_processes():
        flows = shared_state.flow_inventories.get(process.id)
        material_in = len(flows.inputs) if flows else 0
        material_out = len(flows.outputs) if flows else 0
        energy_count = len(flows.energy) if flows else 0
        detail = {
            "process": process.model_dump(),
            "flows": flows.model_dump() if flows else None,
            "material_input_count": material_in,
            "material_output_count": material_out,
            "energy_flow_count": energy_count,
            "edge_count": len(edges_by_process.get(process.id, [])),
        }
        details.append(detail)
    return details


# ---------------------------------------------------------------------------
# Agent implementations
# ---------------------------------------------------------------------------


class _BaseAgent:
    def __init__(
        self,
        *,
        prompt_name: str,
        output_model: type[BaseModel],
        agent_key: str,
        dry_handler: Optional[Any] = None,
        mode: Optional[str] = None,
    ) -> None:
        self._prompt_name = prompt_name
        self._output_model = output_model
        self._dry_handler = dry_handler
        self._agent_key = agent_key
        self._mode_override = self._normalize_mode(mode)
        self._llm = None

    @staticmethod
    def _normalize_mode(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"direct", "react"}:
            return normalized
        return None

    def _ensure_llm(self) -> None:
        if self._llm is None:
            resolved_mode = self._mode_override or config.get_agent_mode(self._agent_key, family="text")
            self._llm = get_text_llm_client(mode=resolved_mode)

    def _call_llm(self, user_prompt: str) -> str:
        self._ensure_llm()
        messages = [
            {"role": "system", "content": _BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = self._llm.generate(
                messages,
                temperature=config.llm_text_temperature,
                max_tokens=config.llm_text_max_tokens,
            )
        except LiteLLMError as exc:  # pragma: no cover - network failure
            raise LLMError(str(exc)) from exc
        return result.content

    def _parse_payload(self, payload: Any) -> BaseModel:
        try:
            return self._output_model.model_validate(payload)
        except ValidationError as exc:
            raise ParseError(str(exc)) from exc


def validate_state(shared_state: SharedState, _document_context: str) -> Dict[str, Any]:
    issues: List[str] = []
    if shared_state.scope is None:
        issues.append("Scope missing")
    else:
        if not shared_state.scope.citations:
            issues.append("Scope missing citations")
        if len(shared_state.scope.route_overview) < 1:
            issues.append("Scope lacks route overview")

    if not shared_state.processes:
        issues.append("No processes extracted")
    else:
        for process in shared_state.processes.values():
            if not process.citations:
                issues.append(f"Process {process.id} missing citations")
            if not process.evidence:
                issues.append(f"Process {process.id} missing evidence")

    for inventory in shared_state.flow_inventories.values():
        if not inventory.citations:
            issues.append(f"Inventory {inventory.process_id} missing citations")
        for record in inventory.inputs + inventory.outputs:
            if not record.citations:
                issues.append(f"Material {record.name} missing citations")
        for energy in inventory.energy:
            if not energy.citations:
                issues.append(f"Energy {energy.energy_type} missing citations")

    status = "ok" if not issues else "warnings"
    return {"status": status, "issues": issues}


# ---------------------------------------------------------------------------
# Agent facade
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Legacy compatibility layer for example.py / supervisor graph
# ---------------------------------------------------------------------------


_CATEGORY_SYNONYMS = {
    "spin": "spin_coat",
    "spin-coat": "spin_coat",
    "spin_coating": "spin_coat",
    "spin coat": "spin_coat",
    "spin-coater": "spin_coat",
    "annealing": "anneal",
    "thermal": "anneal",
    "heat": "anneal",
    "clean": "cleaning",
    "cleaning": "cleaning",
    "evap": "evaporation",
    "sputter": "sputter",
    "ald": "ALD",
    "lamination": "lamination",
    "print": "screen_print",
    "pecvd": "PECVD",
}

_CANONICAL_CATEGORIES = {
    "evaporation": "evaporation",
    "sputter": "sputter",
    "ald": "ALD",
    "spin_coat": "spin_coat",
    "anneal": "anneal",
    "lamination": "lamination",
    "cleaning": "cleaning",
    "screen_print": "screen_print",
    "pecvd": "PECVD",
    "other": "other",
}


def _legacy_category(stage: str) -> str:
    if not stage:
        return "other"
    token = stage.strip().lower()
    for key, value in _CATEGORY_SYNONYMS.items():
        if key in token:
            return value if value in {"ALD", "PECVD"} else _CANONICAL_CATEGORIES.get(value, value)
    normalized = token.replace(" ", "_")
    canonical = _CANONICAL_CATEGORIES.get(normalized)
    if canonical:
        return canonical
    canonical = _CANONICAL_CATEGORIES.get(normalized.lower())
    if canonical:
        return canonical
    return "other"


def _docbundle_to_dict(docbundle: Any) -> Dict[str, Any]:
    if docbundle is None:
        raise ParserError("Docbundle is required for parser node")
    if isinstance(docbundle, dict):
        return docbundle
    if hasattr(docbundle, "model_dump"):
        return docbundle.model_dump()
    if is_dataclass(docbundle):
        return asdict(docbundle)
    try:
        return json.loads(json.dumps(docbundle))
    except TypeError as exc:
        raise ParserError("Docbundle must be JSON serialisable") from exc


def _to_quantity(quantity: FlowQuantity) -> Quantity:
    raw_value = quantity.value
    numeric: float = 0.0
    if isinstance(raw_value, (int, float)):
        numeric = float(raw_value)
    elif isinstance(raw_value, str):
        match = _NUMBER_PATTERN.search(raw_value.replace(",", ""))
        if match:
            try:
                numeric = float(match.group())
            except ValueError:
                numeric = 0.0
    unit = quantity.unit or "unitless"
    return Quantity(value=numeric, unit=unit)


def _to_citations(citations: Sequence[CitationInfo]) -> List[Citation]:
    result: List[Citation] = []
    for item in citations:
        reference = item.snippet or None
        identifier = item.identifier or "[unknown]"
        result.append(Citation(source=identifier, reference=reference, confidence=item.confidence))
    return result


def _material_record_to_flow(record: MaterialRecord) -> MaterialFlow:
    return MaterialFlow(
        material=record.name,
        quantity=_to_quantity(record.quantity),
        citations=_to_citations(record.citations),
    )


def _energy_record_to_flow(record: EnergyRecord) -> EnergyFlow:
    energy_type = (record.energy_type or "").lower()
    if any(token in energy_type for token in ("heat", "thermal", "anneal", "hot")):
        canonical = "thermal"
    else:
        canonical = "electricity"
    return EnergyFlow(
        type=canonical,  # type: ignore[arg-type]
        quantity=_to_quantity(record.quantity),
        region=getattr(config, "default_region", "US-avg"),
        factor=None,
        source_mix=None,
        citations=_to_citations(record.citations),
    )


def _process_to_process_step(process: ProcessNode, flows: FlowInventory) -> ProcessStep:
    inputs = [_material_record_to_flow(record) for record in flows.inputs]
    outputs = [_material_record_to_flow(record) for record in flows.outputs]
    energy = [_energy_record_to_flow(record) for record in flows.energy]
    citations = _to_citations(process.citations)
    conditions = {key: value for key, value in process.parameters.items()}
    equipment = ", ".join(process.equipment) if process.equipment else None

    return ProcessStep(
        id=process.id,
        name=process.name,
        description=process.description,
        category=_legacy_category(process.stage),
        inputs=inputs,
        outputs=outputs,
        energy=energy,
        equipment=equipment,
        conditions=conditions,
        citations=citations,
        uncertainty=None,
        energy_summary=None,
        carbon_emissions=None,
    )


__all__ = [
    "ScopeInit",
    "ScopeHistoryEntry",
    "ScopeLayer",
    "ScopeCitation",
    "AgentDialogueTurn",
    "ProcessNode",
    "ProcessAgentResult",
    "MaterialAgentResult",
    "EnergyAgentResult",
    "FlowInventory",
    "FlowEdge",
    "SharedState",
    "collect_process_details",
    "build_document_context",
    "build_scope_document_context",
    "extract_json",
    "extract_json_from_response",
    "validate_state",
    "ParserError",
    "LLMError",
    "ParseError",
]
