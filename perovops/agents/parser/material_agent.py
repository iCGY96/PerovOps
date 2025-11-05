from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .core import (
    MaterialAgentResult,
    ProcessNode,
    SharedState,
    _BaseAgent,
    extract_json,
    _render_prompt,
)

logger = logging.getLogger(__name__)


class MaterialAgent(_BaseAgent):
    """Agent responsible for material inventory extraction."""

    def __init__(self, *, dry_handler: Optional[Any] = None) -> None:
        super().__init__(
            prompt_name="parser_material_agent.md",
            output_model=MaterialAgentResult,
            agent_key="material",
            dry_handler=dry_handler,
        )

    def run(self, document_context: str, state: SharedState, process: ProcessNode) -> MaterialAgentResult:
        if self._dry_handler is not None:
            payload = json.loads(json.dumps(self._dry_handler(process.id)))
            return self._parse_payload(payload)

        prompt = _render_prompt(
            self._prompt_name,
            {
                "PROCESS": json.dumps(process.model_dump(), ensure_ascii=False, indent=2),
                "MATERIAL_INVENTORY": json.dumps(state.material_inventory_summary(), ensure_ascii=False, indent=2),
                "PROCESS_INVENTORY": json.dumps(state.process_inventory_summary(), ensure_ascii=False, indent=2),
                "MATERIAL_INVENTORY_OVERVIEW": json.dumps(state.material_inventory_overview(), ensure_ascii=False, indent=2),
                "EXPECTED_INPUTS": json.dumps(process.expected_inputs, ensure_ascii=False, indent=2),
                "EXPECTED_OUTPUTS": json.dumps(process.expected_outputs, ensure_ascii=False, indent=2),
                "OPEN_ISSUES": json.dumps(state.open_issues, ensure_ascii=False, indent=2),
                "CONVERSATION_LOG": json.dumps(state.conversation_summary(), ensure_ascii=False, indent=2),
                "DOCUMENT": document_context,
            },
        )
        response = self._call_llm(prompt)
        payload = extract_json(response)
        result = self._parse_payload(payload)
        return result


__all__ = ["MaterialAgent"]
