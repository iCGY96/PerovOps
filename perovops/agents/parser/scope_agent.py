from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

from .core import (
    ParserError,
    ScopeHistoryEntry,
    ScopeInit,
    SharedState,
    _BaseAgent,
    extract_json,
    _render_prompt,
    config,
)

logger = logging.getLogger(__name__)


class ScopeAgent(_BaseAgent):
    """Agent responsible for extracting scope information."""

    def __init__(
        self,
        *,
        dry_handler: Optional[Any] = None,
        runs: Optional[int] = None,
        parallel: Optional[bool] = None,
    ) -> None:
        super().__init__(
            prompt_name="parser_scope_agent.md",
            output_model=ScopeInit,
            agent_key="scope",
            dry_handler=dry_handler,
        )
        configured_runs = runs if runs is not None else config.scope_agent_runs
        try:
            self._runs = max(1, int(configured_runs))
        except (TypeError, ValueError):
            self._runs = 1
        self._parallel = parallel if parallel is not None else config.scope_agent_parallel

    def run(self, document_context: str, state: SharedState) -> ScopeInit:
        total_runs = max(1, self._runs)

        if self._dry_handler is not None:
            payload = json.loads(json.dumps(self._dry_handler()))
            scope = self._parse_payload(payload)
            entry = ScopeHistoryEntry(
                attempt=1,
                status="ok",
                prompt_digest=None,
                scope=scope,
                raw_response=json.dumps(payload, ensure_ascii=False),
            )
            state.record_scope_attempt(entry)
            self._persist_scope_history(state, [entry])
            return scope

        prompt = _render_prompt(
            self._prompt_name,
            {
                "SHARED_STATE": json.dumps(state.prompt_snapshot(), ensure_ascii=False, indent=2),
                "DOCUMENT": document_context,
            },
        )
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        attempt_results: list[tuple[int, ScopeHistoryEntry, Optional[ParserError]]] = []

        def _execute_attempt(attempt: int) -> tuple[int, ScopeHistoryEntry, Optional[ParserError]]:
            response: Optional[str] = None
            try:
                response = self._call_llm(prompt)
                payload = extract_json(response)
                scope_obj = self._parse_payload(payload)
                entry = ScopeHistoryEntry(
                    attempt=attempt,
                    status="ok",
                    prompt_digest=prompt_digest,
                    scope=scope_obj,
                    raw_response=response,
                )
                return attempt, entry, None
            except ParserError as exc:
                entry = ScopeHistoryEntry(
                    attempt=attempt,
                    status="error",
                    prompt_digest=prompt_digest,
                    scope=None,
                    raw_response=response,
                    error=str(exc),
                )
                return attempt, entry, exc

        if total_runs == 1 or not self._parallel:
            for attempt in range(1, total_runs + 1):
                attempt_results.append(_execute_attempt(attempt))
        else:
            with ThreadPoolExecutor(max_workers=total_runs) as executor:
                futures = [executor.submit(_execute_attempt, attempt) for attempt in range(1, total_runs + 1)]
                for future in as_completed(futures):
                    attempt_results.append(future.result())

        attempt_results.sort(key=lambda item: item[0])

        recorded_entries: list[ScopeHistoryEntry] = []
        first_success: Optional[ScopeInit] = None
        last_error: Optional[ParserError] = None

        for _attempt, entry, error in attempt_results:
            state.record_scope_attempt(entry)
            recorded_entries.append(entry)
            if entry.status == "ok" and entry.scope is not None and first_success is None:
                first_success = entry.scope
            if error is not None:
                last_error = error

        self._persist_scope_history(state, recorded_entries)

        if first_success is None:
            raise last_error if last_error is not None else ParserError("Scope agent failed for all attempts")

        return first_success

    def _persist_scope_history(
        self,
        state: SharedState,
        entries: Sequence[ScopeHistoryEntry],
    ) -> None:
        output_dir = state.scope_output_dir
        if output_dir is None:
            return

        try:
            path = Path(output_dir).expanduser()
        except TypeError:
            return

        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem failure
            logger.warning("Failed to create scope output directory %s: %s", path, exc)
            return

        for existing in path.glob("scope_attempt_*.json"):
            try:
                existing.unlink()
            except OSError:
                continue

        for entry in entries:
            filename = f"scope_attempt_{entry.attempt:02d}_{entry.status}.json"
            data = entry.model_dump(mode="json")
            file_path = path / filename
            try:
                file_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                logger.warning("Failed to write scope attempt file %s: %s", file_path, exc)


__all__ = ["ScopeAgent"]
