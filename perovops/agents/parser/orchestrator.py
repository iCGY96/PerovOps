from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:  # pragma: no cover
    from perovops.agents.parser import (
        EnergyAgent,
        EnergyAgentResult,
        MaterialAgent,
        MaterialAgentResult,
        MaterialRecord,
        ProcessAgent,
        ProcessAgentResult,
        ProcessNode,
        ScopeAgent,
        SharedState,
    )


class LCIOrchestrator:
    """LangGraph-based orchestrator coordinating the parser agents."""

    def __init__(
        self,
        *,
        scope_agent: "ScopeAgent",
        process_agent: "ProcessAgent",
        material_agent: "MaterialAgent",
        energy_agent: "EnergyAgent",
        validator: Optional[Callable[["SharedState", str], Dict[str, Any]]] = None,
        show_agent_outputs: bool = True,
        enable_energy_agent: bool = True,
    ) -> None:
        self._scope_agent = scope_agent
        self._process_agent = process_agent
        self._material_agent = material_agent
        self._energy_agent = energy_agent
        self._validator = validator
        self._show_agent_outputs = show_agent_outputs
        self._enable_energy_agent = enable_energy_agent
        self._max_refinement_rounds = 3
        self._graph = self._build_graph()
        self._last_displayed_round = 0

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _print_expressions(self, label: str, expressions: Sequence[Optional[str]]) -> None:
        if not self._show_agent_outputs:
            return
        unique: list[str] = []
        seen: set[str] = set()
        for item in expressions:
            if not isinstance(item, str):
                continue
            expression = item.strip()
            if not expression or expression in seen:
                continue
            seen.add(expression)
            unique.append(expression)
        for expression in unique:
            print(f"[{label}] {expression}")

    def _print_lines(self, label: str, lines: Sequence[str]) -> None:
        if not self._show_agent_outputs:
            return
        header = f"=== {label} ==="
        print(header)
        for line in lines:
            text = (line or "").strip()
            if text:
                print(f" - {text}")
        print("=" * len(header))

    def _print_round_banner(self, big_round: int) -> None:
        if not self._show_agent_outputs:
            return
        if big_round <= 0 or big_round == self._last_displayed_round:
            return
        header = f"=== Big Round {big_round} ==="
        print(header)
        print("=" * len(header))
        self._last_displayed_round = big_round

    @staticmethod
    def _extract_material_tokens(items: Sequence[str]) -> list[str]:
        tokens: list[str] = []
        for raw in items:
            if not raw:
                continue
            text = str(raw)
            if "(" in text and ")" in text:
                inside = text[text.rfind("(") + 1 : text.find(")", text.rfind("(") + 1)]
                if inside.strip():
                    tokens.append(inside.strip().lower())
            tokens.append(text.split("(")[0].strip().lower())
        return [token for token in tokens if token]

    @staticmethod
    def _normalize_species(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        return re.sub(r"\s+", "", cleaned).upper()

    @classmethod
    def _species_from_expression(cls, expression: Optional[str]) -> set[str]:
        if not expression:
            return set()
        species: set[str] = set()
        fragments = re.split(r"->|⇌|=>", expression)
        for fragment in fragments:
            for token in fragment.split("+"):
                cleaned = token.strip()
                if not cleaned:
                    continue
                cleaned = re.sub(r"^[\d\.\s/]+", "", cleaned)
                normalized = cls._normalize_species(cleaned)
                if normalized:
                    species.add(normalized)
        return species

    def _check_process_expression(self, process: Optional["ProcessNode"]) -> list[str]:
        if process is None:
            return []
        expression = (process.chemical_expression or "").lower()
        issues: list[str] = []
        expression_species = self._species_from_expression(process.chemical_expression)
        expected_tokens = self._extract_material_tokens(process.expected_inputs + process.expected_outputs)
        expected_species = {
            token
            for token in (self._normalize_species(token) for token in expected_tokens)
            if token
        }
        if not expression:
            issues.append(f"Process {process.id} lacks chemical expression covering materials")
            return issues
        missing = sorted(expected_species - expression_species)
        if missing:
            issues.append(
                f"Process {process.id} expression missing materials: {', '.join(missing)}"
            )
        return issues

    def _check_material_coverage(
        self,
        process: Optional["ProcessNode"],
        materials: Sequence["MaterialRecord"],
    ) -> list[str]:
        if process is None:
            return []
        issues: list[str] = []
        records = list(materials)
        process_species = self._species_from_expression(process.chemical_expression)
        material_species: set[str] = set()
        material_species_list: list[str] = []

        expected_tokens = self._extract_material_tokens(process.expected_inputs + process.expected_outputs)
        expected_species = {
            token
            for token in (self._normalize_species(token) for token in expected_tokens)
            if token
        }

        for record in records:
            quantity_value = record.quantity.value
            if isinstance(quantity_value, str):
                norm_value = quantity_value.strip().lower()
            else:
                norm_value = str(quantity_value).strip().lower()
            if not norm_value or norm_value in {"unknown", "", "n/a"}:
                issues.append(f"Material {record.material_id} missing quantified amount")

            material_expr = self._normalize_species(record.chemical_expression)
            if material_expr is None:
                issues.append(f"Material {record.material_id} missing chemical expression")
            else:
                material_species.add(material_expr)
                material_species_list.append(material_expr)

        missing_from_materials = sorted(process_species - material_species)
        if missing_from_materials:
            issues.append(
                f"Process {process.id} expression species missing in material inventory: {', '.join(missing_from_materials)}"
            )

        extra_materials = sorted(material_species - process_species)
        if extra_materials:
            issues.append(
                f"Material inventory contains species not in process expression: {', '.join(extra_materials)}"
            )

        missing_expected = sorted(expected_species - material_species)
        if missing_expected:
            issues.append(
                f"Expected materials not captured: {', '.join(missing_expected)}"
            )

        if len(material_species) != len(process_species):
            issues.append(
                f"Process {process.id} material set size mismatch: process has {len(process_species)} species, materials provide {len(material_species)}"
            )

        if len(material_species_list) != len(process_species):
            issues.append(
                f"Process {process.id} material record count mismatch: expected {len(process_species)} records, found {len(material_species_list)}"
            )

        return issues

    def _display_scope_result(
        self,
        scope: "ScopeInit",
        *,
        attempt: Optional[int] = None,
        reused: bool = False,
    ) -> None:
        lines: list[str] = []
        if attempt is not None:
            lines.append(f"Attempt: {attempt}")
        lines.append(f"Device: {scope.device_name}")
        if scope.architecture:
            lines.append(f"Architecture: {scope.architecture}")
        if scope.expression:
            expression_display = scope.expression
            materails = []
            for layer in scope.layers:
                if layer.formula:
                    materails.append(layer.formula)
                else:
                    materails.append(layer.material)
            lines.append(f"Expression: {expression_display}")
        if scope.functional_unit:
            lines.append(f"Functional unit: {scope.functional_unit}")
        if scope.route_overview:
            lines.append(f"Route: {scope.route_overview}")
        if scope.notes:
            lines.append(scope.notes[0])
        self._print_lines("Scope Agent", lines[:5])

    def _display_process_result(
        self,
        result: "ProcessAgentResult",
        big_round: int,
    ) -> None:
        self._print_round_banner(big_round)
        lines: list[str] = []
        if result.process is not None:
            display_name = result.process.name or "unnamed"
            lines.append(f"{result.process.id}: {display_name}")
            if result.process.stage:
                lines.append(f"Stage: {result.process.stage}")
            confidence = result.process.confidence
            if isinstance(confidence, (int, float)):
                lines.append(f"Confidence: {confidence:.2f}")
        if not lines:
            lines.append("No process extracted")
        self._print_lines("Process Agent", lines)
        expressions: list[Optional[str]] = []
        if result.process is not None:
            expressions.append(result.process.chemical_expression)
        self._print_expressions("Process Agent", expressions)

    def _display_material_result(
        self,
        result: "MaterialAgentResult",
        big_round: int,
    ) -> None:
        inputs = len(result.inputs)
        outputs = len(result.outputs)
        lines = [f"{result.process_id}: {inputs} inputs, {outputs} outputs"]
        if result.open_issues:
            lines.append(f"Notable issue: {result.open_issues[0]}")
        self._print_lines("Material Agent", lines)
        expressions = [record.chemical_expression for record in result.inputs + result.outputs]
        self._print_expressions("Material Agent", expressions)

    def _display_material_skip(self, process: Optional["ProcessNode"]) -> None:
        _ = process

    def _display_energy_result(self, result: "EnergyAgentResult") -> None:
        expressions: list[str] = []
        for record in result.energy:
            raw_value = record.quantity.value
            if isinstance(raw_value, str):
                value = raw_value.strip() or "unknown"
            else:
                value = str(raw_value)
            unit = (record.quantity.unit or "").strip()
            basis = (record.quantity.basis or "").strip()
            method = (record.quantity.method or "").strip()
            energy_label = (record.energy_type or "energy").strip() or "energy"
            expression = f"{energy_label} = {value}"
            if unit:
                expression += f" {unit}"
            if basis:
                expression += f" (basis={basis})"
            if method:
                expression += f" [method={method}]"
            expressions.append(expression)

        if expressions:
            self._print_lines("Energy Agent", expressions)
        elif result.reaction_conditions:
            self._print_lines(
                "Energy Agent",
                [f"Condition: {text}" for text in result.reaction_conditions[:3]],
            )
        else:
            self._print_lines("Energy Agent", ["No energy data captured"])

    def _display_round_summary(
        self,
        big_round: int,
        process_nodes: Sequence[tuple[str, Optional[str]]],
        material_nodes: Sequence[tuple[str, Optional[str]]],
    ) -> None:
        if not self._show_agent_outputs:
            return
        entries: list[str] = []
        if process_nodes:
            unique_processes = list(dict.fromkeys(process_nodes))
            formatted = ", ".join(
                f"{pid} ({expr or 'n/a'})" for pid, expr in unique_processes
            )
            entries.append("Processes: " + formatted)
        if material_nodes:
            unique_materials = list(dict.fromkeys(material_nodes))
            formatted = ", ".join(
                f"{mid} ({expr or 'n/a'})" for mid, expr in unique_materials
            )
            entries.append("Materials: " + formatted)
        if not entries:
            entries.append("No new nodes added this round")
        self._print_lines("Round Summary", entries)

    @staticmethod
    def _normalize_material_ids(
        process_id: str,
        records: Sequence["MaterialRecord"],
        suffix: str,
    ) -> List["MaterialRecord"]:
        normalized: List["MaterialRecord"] = []
        for index, record in enumerate(records, start=1):
            expected_prefix = f"{process_id}_{suffix}_"
            target_id = record.material_id or f"{suffix}_{index:02d}"
            if not target_id.startswith(expected_prefix):
                target_id = f"{expected_prefix}{index:02d}"
            normalized.append(record.model_copy(update={"material_id": target_id}))
        return normalized

    def _finalize_material_round(
        self,
        *,
        shared_state: "SharedState",
        process: "ProcessNode",
        material_result: "MaterialAgentResult",
        state: Dict[str, Any],
    ) -> tuple[
        list[tuple[str, Optional[str]]],
        list[tuple[str, Optional[str]]],
        list[tuple[str, Optional[str]]],
        list[tuple[str, Optional[str]]],
    ]:
        shared_state.upsert_process(process)
        added_material_ids = shared_state.update_material_inventory(
            process_id=material_result.process_id,
            inputs=material_result.inputs,
            outputs=material_result.outputs,
            normalization_actions=material_result.normalization_actions,
            deduplication_actions=material_result.deduplication_actions,
            citations=material_result.citations,
            confidence=material_result.confidence,
            inventory_expression=material_result.inventory_expression,
        )
        shared_state.log_progress(process)

        normalized_expression = process.normalized_expression or process.chemical_expression
        new_process_entry = (process.id, normalized_expression)
        new_process_nodes = list(state.get("new_process_nodes", []))
        if new_process_entry not in new_process_nodes:
            new_process_nodes.append(new_process_entry)

        expression_lookup = {
            record.material_id: record.chemical_expression
            for record in list(material_result.inputs) + list(material_result.outputs)
        }
        recent_material_nodes = [
            (material_id, expression_lookup.get(material_id)) for material_id in added_material_ids
        ]
        new_material_nodes = list(state.get("new_material_nodes", []))
        if recent_material_nodes:
            new_material_nodes.extend(recent_material_nodes)

        return new_process_nodes, new_material_nodes, [new_process_entry], recent_material_nodes

    def _display_energy_skip(self, process: Optional["ProcessNode"], stop_requested: bool) -> None:
        _ = (process, stop_requested)

    def _display_validation(self, report: Optional[Dict[str, Any]]) -> None:
        _ = report

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(dict)

        graph.add_node("scope", self._scope_node)
        graph.add_node("process", self._process_node)
        graph.add_node("material", self._material_node)
        if self._enable_energy_agent:
            graph.add_node("energy", self._energy_node)
        graph.add_node("validate", self._validate_node)

        graph.add_edge(START, "scope")
        graph.add_edge("scope", "process")
        graph.add_conditional_edges(
            "process",
            self._process_condition,
            {"continue": "material", "stop": "validate"},
        )
        if self._enable_energy_agent:
            graph.add_edge("material", "energy")
            graph.add_edge("energy", "process")
        else:
            graph.add_edge("material", "process")
        graph.add_edge("validate", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------
    def _scope_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_state = state["shared_state"]
        document_context = state["document_context"]
        scope_document_context = state.get("scope_document_context") or document_context
        if shared_state.scope is None:
            previous_attempts = len(shared_state.scope_history)
            scope = self._scope_agent.run(scope_document_context, shared_state)
            shared_state.register_scope(scope)
            new_entries = shared_state.scope_history[previous_attempts:]
            if new_entries:
                success_entries = [
                    entry
                    for entry in new_entries
                    if getattr(entry, "status", None) == "ok" and getattr(entry, "scope", None) is not None
                ]
                error_entries = [entry for entry in new_entries if getattr(entry, "status", None) == "error"]
                for entry in error_entries:
                    message = entry.error or "Unknown error"
                    self._print_lines(
                        "Scope Agent Error",
                        [f"Attempt {entry.attempt}: {message}"],
                    )
                if success_entries:
                    for idx, entry in enumerate(success_entries):
                        exit_after = idx == len(success_entries) - 1
                        self._display_scope_result(
                            entry.scope,  # type: ignore[arg-type]
                            attempt=entry.attempt,
                        )
                else:
                    self._display_scope_result(scope)
            else:
                self._display_scope_result(scope)
            scope_obj = scope
        else:
            scope_obj = shared_state.scope
            self._display_scope_result(scope_obj, reused=True)
        
        return {
            "shared_state": shared_state,
            "document_context": document_context,
            "scope_document_context": scope_document_context,
            "process_round": state.get("process_round", 0),
            "new_process_nodes": list(state.get("new_process_nodes", [])),
            "new_material_nodes": list(state.get("new_material_nodes", [])),
            "material_revision_required": state.get("material_revision_required", False),
            "repeat_cycle": state.get("repeat_cycle", False),
            "conversation_attempts": state.get("conversation_attempts", 0),
            "pending_material_result": state.get("pending_material_result"),
            "current_round_process_id": state.get("current_round_process_id"),
        }

    def _process_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_state = state["shared_state"]
        document_context = state["document_context"]
        scope_document_context = state.get("scope_document_context") or document_context

        pending_process: Optional["ProcessNode"] = state.get("pending_process")
        pending_material_result: Optional["MaterialAgentResult"] = state.get("pending_material_result")
        new_process_nodes_state = list(state.get("new_process_nodes", []))
        new_material_nodes_state = list(state.get("new_material_nodes", []))
        current_round_process_id = state.get("current_round_process_id")
        repeat_cycle = bool(state.get("repeat_cycle", False))
        big_round_completed = int(state.get("big_round_completed", 0))
        current_round_index = int(state.get("current_round_index", 0))
        small_round = int(state.get("small_round", 0))
        conversation_attempts = int(state.get("conversation_attempts", 0))

        starting_new_round = pending_process is None or not repeat_cycle

        if starting_new_round:
            current_round_index = big_round_completed + 1
            small_round = 1
            conversation_attempts = 0
            new_process_nodes_state = []
            new_material_nodes_state = []
            current_round_process_id = None
        else:
            small_round += 1

        result = self._process_agent.run(document_context, shared_state)
        shared_state.record_conversation(result.conversation)
        stop_requested = bool(result.stop_candidate or result.process is None)

        if result.process is None:
            shared_state.add_issues(result.issues)
            summary_process_nodes: list[tuple[str, Optional[str]]] = []
            summary_material_nodes: list[tuple[str, Optional[str]]] = []
            if pending_process is not None and pending_material_result is not None:
                (
                    new_process_nodes_state,
                    new_material_nodes_state,
                    summary_process_nodes,
                    summary_material_nodes,
                ) = self._finalize_material_round(
                    shared_state=shared_state,
                    process=pending_process,
                    material_result=pending_material_result,
                    state=state,
                )
                big_round_completed = max(big_round_completed, current_round_index)
                current_round_process_id = None
            if summary_process_nodes or summary_material_nodes:
                self._display_round_summary(
                    current_round_index,
                    summary_process_nodes,
                    summary_material_nodes,
                )
            return {
                "shared_state": shared_state,
                "last_process": None,
                "stop_requested": True,
                "document_context": document_context,
                "scope_document_context": scope_document_context,
                "process_round": current_round_index,
                "big_round_completed": big_round_completed,
                "current_round_index": current_round_index,
                "small_round": small_round,
                "pending_process": None,
                "pending_material_result": None,
                "new_process_nodes": new_process_nodes_state,
                "new_material_nodes": new_material_nodes_state,
                "material_revision_required": state.get("material_revision_required", False),
                "repeat_cycle": False,
                "conversation_attempts": conversation_attempts,
                "current_round_process_id": current_round_process_id,
            }

        shared_state.add_issues(result.issues)
        shared_state.add_issues(self._check_process_expression(result.process))
        if result.process is not None:
            shared_state.upsert_process(result.process)
            current_round_process_id = result.process.id
        self._display_process_result(result, current_round_index)

        return {
            "shared_state": shared_state,
            "last_process": result.process,
            "stop_requested": stop_requested,
            "document_context": document_context,
            "scope_document_context": scope_document_context,
            "process_round": current_round_index,
            "big_round_completed": big_round_completed,
            "current_round_index": current_round_index,
            "small_round": small_round,
            "pending_process": result.process,
            "pending_material_result": None,
            "new_process_nodes": new_process_nodes_state,
            "new_material_nodes": new_material_nodes_state,
            "material_revision_required": state.get("material_revision_required", False),
            "repeat_cycle": False,
            "conversation_attempts": conversation_attempts,
            "current_round_process_id": current_round_process_id,
        }

    def _material_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_state = state["shared_state"]
        document_context = state["document_context"]
        scope_document_context = state.get("scope_document_context") or document_context
        pending_process: Optional["ProcessNode"] = state.get("pending_process")
        if pending_process is None:
            self._display_material_skip(pending_process)
            return {
                "shared_state": shared_state,
                "stop_requested": True,
                "document_context": document_context,
                "scope_document_context": scope_document_context,
                "process_round": state.get("process_round", 0),
                "big_round_completed": int(state.get("big_round_completed", 0)),
                "current_round_index": int(state.get("current_round_index", 0)),
                "small_round": int(state.get("small_round", 0)),
                "pending_process": None,
                "pending_material_result": state.get("pending_material_result"),
                "new_process_nodes": list(state.get("new_process_nodes", [])),
                "new_material_nodes": list(state.get("new_material_nodes", [])),
                "material_revision_required": state.get("material_revision_required", False),
                "repeat_cycle": state.get("repeat_cycle", False),
                "conversation_attempts": int(state.get("conversation_attempts", 0)),
                "current_round_process_id": state.get("current_round_process_id"),
            }

        current_round_index = int(state.get("current_round_index", int(state.get("big_round_completed", 0)) + 1))
        small_round = int(state.get("small_round", 1))
        conversation_attempts = int(state.get("conversation_attempts", 0))

        agent_result = self._material_agent.run(document_context, shared_state, pending_process)
        normalized_inputs = self._normalize_material_ids(pending_process.id, agent_result.inputs, "in")
        normalized_outputs = self._normalize_material_ids(pending_process.id, agent_result.outputs, "out")
        material_result = agent_result.model_copy(update={"inputs": normalized_inputs, "outputs": normalized_outputs})
        shared_state.add_issues(material_result.open_issues)
        shared_state.record_conversation(material_result.conversation)
        self._display_material_result(material_result, current_round_index)

        material_issues = self._check_material_coverage(
            pending_process, list(material_result.inputs) + list(material_result.outputs)
        )
        if material_issues:
            shared_state.add_issues(material_issues)

        needs_revision = bool(material_issues)

        if needs_revision:
            conversation_attempts += 1
            if conversation_attempts >= self._max_refinement_rounds:
                shared_state.add_issues(
                    [
                        f"Process {pending_process.id} reached refinement limit; manual review recommended",
                    ]
                )
                needs_revision = False
                conversation_attempts = 0

        if needs_revision:
            return {
                "shared_state": shared_state,
                "last_process": pending_process,
                "stop_requested": False,
                "document_context": document_context,
                "scope_document_context": scope_document_context,
                "process_round": current_round_index,
                "big_round_completed": int(state.get("big_round_completed", 0)),
                "current_round_index": current_round_index,
                "small_round": small_round,
                "pending_process": pending_process,
                "pending_material_result": material_result,
                "new_process_nodes": list(state.get("new_process_nodes", [])),
                "new_material_nodes": list(state.get("new_material_nodes", [])),
                "material_revision_required": True,
                "repeat_cycle": True,
                "conversation_attempts": conversation_attempts,
                "current_round_process_id": state.get("current_round_process_id", pending_process.id),
            }

        (
            new_process_nodes,
            new_material_nodes,
            summary_process_nodes,
            summary_material_nodes,
        ) = self._finalize_material_round(
            shared_state=shared_state,
            process=pending_process,
            material_result=material_result,
            state=state,
        )

        self._display_round_summary(
            current_round_index,
            summary_process_nodes,
            summary_material_nodes,
        )

        return {
            "shared_state": shared_state,
            "last_process": pending_process,
            "stop_requested": False,
            "document_context": document_context,
            "scope_document_context": scope_document_context,
            "process_round": current_round_index,
            "big_round_completed": current_round_index,
            "current_round_index": current_round_index,
            "small_round": 0,
            "pending_process": None,
            "pending_material_result": None,
            "new_process_nodes": new_process_nodes,
            "new_material_nodes": new_material_nodes,
            "material_revision_required": False,
            "repeat_cycle": False,
            "conversation_attempts": 0,
            "current_round_process_id": pending_process.id,
        }

    def _energy_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self._enable_energy_agent:
            return state
        shared_state = state["shared_state"]
        document_context = state["document_context"]
        scope_document_context = state.get("scope_document_context") or document_context
        process = state.get("last_process")
        stop_requested = state.get("stop_requested", False)

        new_process_nodes_state = state.get("new_process_nodes") or []
        new_material_nodes_state = state.get("new_material_nodes") or []
        pending_revision = bool(state.get("material_revision_required"))
        attempts = int(state.get("conversation_attempts", 0))

        if process is None or stop_requested:
            self._display_energy_skip(process, stop_requested)
            big_round = int(state.get("process_round", state.get("current_round_index", 0)))
            self._display_round_summary(big_round, new_process_nodes_state, new_material_nodes_state)
            return {
                "shared_state": shared_state,
                "last_process": process,
                "stop_requested": True,
                "document_context": document_context,
                "scope_document_context": scope_document_context,
                "process_round": state.get("process_round", 0),
                "pending_material_result": state.get("pending_material_result"),
                "new_process_nodes": [],
                "new_material_nodes": [],
                "material_revision_required": pending_revision,
                "repeat_cycle": False,
                "conversation_attempts": attempts,
                "current_round_process_id": state.get("current_round_process_id"),
            }

        result = self._energy_agent.run(document_context, shared_state, process)
        shared_state.update_energy_inventory(
            process_id=result.process_id,
            energy=result.energy,
            reaction_conditions=result.reaction_conditions,
            citations=result.citations,
            confidence=result.confidence,
        )
        shared_state.add_issues(result.open_issues)
        self._display_energy_result(result)
        shared_state.log_progress(process)

        inventory = shared_state.flow_inventories.get(process.id)
        materials: Sequence[MaterialRecord] = []
        if inventory is not None:
            materials = list(inventory.inputs) + list(inventory.outputs)
        consistency_issues = self._check_process_expression(process)
        consistency_issues += self._check_material_coverage(process, materials)
        needs_revision = bool(consistency_issues) or pending_revision
        if needs_revision:
            attempts += 1
            if attempts >= self._max_refinement_rounds:
                shared_state.add_issues(
                    [
                        f"Process {process.id} reached refinement limit; manual review recommended",
                    ]
                )
                needs_revision = False
                attempts = 0
        else:
            attempts = 0
        if consistency_issues:
            shared_state.add_issues(consistency_issues)

        big_round = int(state.get("process_round", state.get("current_round_index", 0)))
        self._display_round_summary(big_round, new_process_nodes_state, new_material_nodes_state)

        return {
            "shared_state": shared_state,
            "last_process": process,
            "stop_requested": False,
            "document_context": document_context,
            "scope_document_context": scope_document_context,
            "process_round": state.get("process_round", 0),
            "pending_material_result": state.get("pending_material_result"),
            "new_process_nodes": [],
            "new_material_nodes": [],
            "material_revision_required": needs_revision,
            "repeat_cycle": needs_revision,
            "conversation_attempts": attempts,
            "current_round_process_id": state.get("current_round_process_id"),
        }

    def _validate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_state = state["shared_state"]
        document_context = state["document_context"]
        scope_document_context = state.get("scope_document_context") or document_context
        if self._validator is None:
            self._display_validation(None)
            return {
                "shared_state": shared_state,
                "document_context": document_context,
                "scope_document_context": scope_document_context,
                "pending_material_result": state.get("pending_material_result"),
                "current_round_process_id": state.get("current_round_process_id"),
            }
        report = self._validator(shared_state, document_context)
        self._display_validation(report)
        return {
            "shared_state": shared_state,
            "validation": report,
            "document_context": document_context,
            "scope_document_context": scope_document_context,
            "pending_material_result": state.get("pending_material_result"),
            "current_round_process_id": state.get("current_round_process_id"),
        }

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------
    @staticmethod
    def _process_condition(state: Dict[str, Any]) -> str:
        stop_requested = state.get("stop_requested", False)
        last_process = state.get("last_process")
        if stop_requested or last_process is None:
            return "stop"
        return "continue"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        document_context: str,
        shared_state: "SharedState",
        *,
        scope_document_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._last_displayed_round = 0
        initial_state = {
            "document_context": document_context,
            "scope_document_context": scope_document_context or document_context,
            "shared_state": shared_state,
            "last_process": None,
            "stop_requested": False,
            "process_round": 0,
            "big_round_completed": 0,
            "current_round_index": 0,
            "small_round": 0,
            "pending_process": None,
            "pending_material_result": None,
            "new_process_nodes": [],
            "new_material_nodes": [],
            "material_revision_required": False,
            "repeat_cycle": False,
            "conversation_attempts": 0,
            "current_round_process_id": None,
        }
        return self._graph.invoke(initial_state)
