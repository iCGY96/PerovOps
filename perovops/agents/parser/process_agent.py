from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .core import (
    ProcessAgentResult,
    SharedState,
    _BaseAgent,
    extract_json,
    _render_prompt,
)

logger = logging.getLogger(__name__)


class ProcessAgent(_BaseAgent):
    """Agent responsible for expanding process nodes."""

    def __init__(self, *, dry_handler: Optional[Any] = None) -> None:
        super().__init__(
            prompt_name="parser_process_agent.md",
            output_model=ProcessAgentResult,
            agent_key="process",
            dry_handler=dry_handler,
        )

    def run(self, document_context: str, state: SharedState) -> ProcessAgentResult:
        if self._dry_handler is not None:
            payload = json.loads(json.dumps(self._dry_handler()))
            return self._parse_payload(payload)

        prompt = _render_prompt(
            self._prompt_name,
            {
                "SCOPE_SUMMARY": json.dumps(state.scope.model_dump() if state.scope else None, ensure_ascii=False, indent=2),
                "KNOWN_PROCESSES": json.dumps([proc.model_dump() for proc in state.sorted_processes()], ensure_ascii=False, indent=2),
                "MATERIAL_INVENTORY": json.dumps(state.material_inventory_summary(), ensure_ascii=False, indent=2),
                "PROCESS_INVENTORY": json.dumps(state.process_inventory_summary(), ensure_ascii=False, indent=2),
                "MATERIAL_INVENTORY_OVERVIEW": json.dumps(state.material_inventory_overview(), ensure_ascii=False, indent=2),
                "ENERGY_INVENTORY": json.dumps(state.energy_inventory_summary(), ensure_ascii=False, indent=2),
                "OPEN_ISSUES": json.dumps(state.open_issues, ensure_ascii=False, indent=2),
                "CONVERSATION_LOG": json.dumps(state.conversation_summary(), ensure_ascii=False, indent=2),
                "DOCUMENT": document_context,
            },
        )
        response = self._call_llm(prompt)
        payload = extract_json(response)
        result = self._parse_payload(payload)
        return result


__all__ = ["ProcessAgent"]
