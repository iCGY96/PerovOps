from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency during some tests
    import litellm  # type: ignore
except Exception:  # pragma: no cover - allow graceful failure when LiteLLM missing
    litellm = None  # type: ignore

from perovops.utils.config import config

logger = logging.getLogger(__name__)


class LiteLLMError(RuntimeError):
    """Raised when LiteLLM is unavailable or returns an unexpected payload."""


@dataclass
class LiteLLMResult:
    """Container for LiteLLM responses."""

    content: str
    raw: Dict[str, Any]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    iterations: List[str] = field(default_factory=list)
    validation_feedback: List[str] = field(default_factory=list)
    validated: Optional[bool] = None


@dataclass
class ReActConfig:
    """Configuration for self-reflective LiteLLM iterations."""

    max_iterations: int = 3
    stop_signal: Optional[str] = "FINAL_ANSWER:"
    feedback_template: str = (
        "Reflect on your previous response and improve it if needed.\n"
        "Previous response:\n{previous_response}\n\n"
        "If the answer is complete, reply with 'FINAL_ANSWER:' followed by the final output. "
        "Otherwise, explain your next reasoning steps before giving an updated answer."
    )

    def normalized_stop_signal(self) -> Optional[str]:
        if self.stop_signal:
            return self.stop_signal.lower()
        return None


class LiteLLMClient:
    """Shared LiteLLM client supporting direct calls and ReAct-style refinement."""

    def __init__(
        self,
        *,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        default_mode: str = "direct",
        react_config: Optional[ReActConfig] = None,
        extra_completion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if litellm is None:  # pragma: no cover - dependency not installed
            raise LiteLLMError("LiteLLM is not installed or failed to import")

        self.default_model = default_model or config.llm_text_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_mode = default_mode.lower() if default_mode else "direct"
        if self.default_mode not in {"direct", "react"}:
            self.default_mode = "direct"
        self.react_config = react_config
        self.extra_completion_kwargs = extra_completion_kwargs or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        mode: Optional[str] = None,
        react_config: Optional[ReActConfig] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        validator: Optional[Callable[[LiteLLMResult], Any]] = None,
        **kwargs: Any,
    ) -> LiteLLMResult:
        """Generate a response using the configured strategy."""

        strategy = (mode or self.default_mode or "direct").lower()
        if strategy == "react":
            config_obj = react_config or self.react_config or ReActConfig()
            return self._run_react(
                messages,
                config_obj,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                validator=validator,
                **kwargs,
            )

        result = self._run_direct(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if validator is not None:
            valid, feedback = self._apply_validator(validator, result)
            if not valid:
                details = "; ".join(feedback) if feedback else "Validator rejected the response"
                raise LiteLLMError(f"Direct generation failed validation: {details}")
        else:
            result.validated = None
            result.validation_feedback = []
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(messages, Iterable):
            raise LiteLLMError("messages must be an iterable of role/content dicts")
        prepared: List[Dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                raise LiteLLMError("Each message must be a dict with at least role/content")
            prepared.append(copy.deepcopy(item))
        return prepared

    def _apply_validator(
        self,
        validator: Callable[[LiteLLMResult], Any],
        result: LiteLLMResult,
    ) -> tuple[bool, list[str]]:
        if validator is None:
            result.validated = None
            result.validation_feedback = []
            return True, []

        try:
            verdict = validator(result)
        except Exception as exc:  # pragma: no cover - defensive validator handling
            raise LiteLLMError(f"Validator raised an exception: {exc}") from exc

        success: bool
        extra: Any = None

        if isinstance(verdict, dict):
            success = bool(
                verdict.get("ok")
                or verdict.get("success")
                or verdict.get("valid")
                or verdict.get("passed")
            )
            extra = (
                verdict.get("messages")
                or verdict.get("errors")
                or verdict.get("reason")
                or verdict.get("message")
                or verdict.get("details")
            )
        elif isinstance(verdict, tuple):
            success = bool(verdict[0])
            if len(verdict) > 1:
                extra = verdict[1]
        else:
            success = bool(verdict)

        feedback: list[str] = []
        if extra is not None:
            if isinstance(extra, str):
                text = extra.strip()
                if text:
                    feedback.append(text)
            elif isinstance(extra, Sequence) and not isinstance(extra, (str, bytes)):
                for item in extra:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if text:
                        feedback.append(text)
            else:
                text = str(extra).strip()
                if text:
                    feedback.append(text)

        result.validated = success
        result.validation_feedback = feedback
        return success, feedback

    def _extract_content(self, response: Dict[str, Any]) -> str:
        try:
            choice = response["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
        except Exception as exc:  # pragma: no cover - unexpected provider payload
            raise LiteLLMError("Malformed LiteLLM response") from exc

        if isinstance(content, list):
            parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if text:
                        parts.append(str(text))
                elif isinstance(chunk, str):
                    parts.append(chunk)
            content = "".join(parts)

        if isinstance(content, str):
            logger.debug("LiteLLM content: %s", content)
            return content

        logger.debug("LiteLLM returned non-text content: %r", content)
        return ""

    def _call_litellm(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LiteLLMResult:
        prepared = self._prepare_messages(messages)
        call_kwargs: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": prepared,
        }

        resolved_temperature = (
            temperature
            if temperature is not None
            else self.default_temperature
            if self.default_temperature is not None
            else config.llm_text_temperature
        )
        call_kwargs["temperature"] = resolved_temperature

        resolved_max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.default_max_tokens
            if self.default_max_tokens is not None
            else config.llm_text_max_tokens
        )
        if resolved_max_tokens is not None:
            call_kwargs["max_tokens"] = resolved_max_tokens

        merged = dict(self.extra_completion_kwargs)
        merged.update(kwargs)
        call_kwargs.update(merged)

        try:
            raw_response = litellm.completion(**call_kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - network/provider failures
            logger.debug("LiteLLM call failed: %s", exc)
            raise LiteLLMError(str(exc)) from exc

        content = self._extract_content(raw_response)
        conversation = prepared + [{"role": "assistant", "content": content}]
        return LiteLLMResult(content=content, raw=raw_response, messages=conversation)

    def _run_direct(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LiteLLMResult:
        return self._call_litellm(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _run_react(
        self,
        messages: Sequence[Dict[str, Any]],
        react_config: ReActConfig,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        validator: Optional[Callable[[LiteLLMResult], Any]] = None,
        **kwargs: Any,
    ) -> LiteLLMResult:
        conversation = self._prepare_messages(messages)
        iterations: List[str] = []
        stop_token = react_config.normalized_stop_signal()
        last_result: Optional[LiteLLMResult] = None

        for step in range(max(1, react_config.max_iterations)):
            result = self._call_litellm(
                conversation,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            iterations.append(result.content)
            conversation = [copy.deepcopy(m) for m in result.messages]
            last_result = result

            if validator is not None:
                validation_passed, validation_feedback = self._apply_validator(validator, result)
            else:
                result.validated = None
                result.validation_feedback = []
                validation_passed = True
                validation_feedback = []

            stop_triggered = bool(stop_token and stop_token in result.content.lower())

            if validator is not None:
                if validation_passed:
                    break

                if step == react_config.max_iterations - 1:
                    detail = "; ".join(validation_feedback) if validation_feedback else "validation failed"
                    raise LiteLLMError(
                        f"Validation failed after {react_config.max_iterations} ReAct iterations: {detail}"
                    )

                feedback_parts: list[str] = []
                if validation_feedback:
                    formatted = "\n".join(f"- {item}" for item in validation_feedback)
                    feedback_parts.append(f"Validation feedback:\n{formatted}")
                else:
                    feedback_parts.append(
                        "Validation feedback: Output did not pass validation checks. Address these issues."
                    )

                if stop_triggered:
                    feedback_parts.append(
                        "You used the stop signal, but validation failed. Provide a corrected answer without the stop signal until validation passes."
                    )

                feedback_parts.append(
                    react_config.feedback_template.format(
                        iteration=step + 1,
                        previous_response=result.content,
                    )
                )
                conversation.append({"role": "user", "content": "\n\n".join(feedback_parts)})
                continue

            if stop_triggered:
                break
            if step == react_config.max_iterations - 1:
                break

            feedback = react_config.feedback_template.format(
                iteration=step + 1,
                previous_response=result.content,
            )
            conversation.append({"role": "user", "content": feedback})

        if last_result is None:
            raise LiteLLMError("LiteLLM did not return a result")

        last_result.iterations = iterations
        last_result.messages = conversation
        return last_result


def _resolve_feedback(default_text: Optional[str]) -> str:
    if default_text:
        return str(default_text)
    return ReActConfig().feedback_template


def get_text_llm_client(**overrides: Any) -> LiteLLMClient:
    """Factory for a text-focused LiteLLM client."""

    react_cfg = ReActConfig(
        max_iterations=overrides.pop("react_max_iterations", config.llm_text_react_iterations),
        stop_signal=overrides.pop("react_stop_signal", config.llm_text_react_stop_signal),
        feedback_template=_resolve_feedback(
            overrides.pop("react_feedback", config.llm_text_react_feedback)
        ),
    )

    return LiteLLMClient(
        default_model=overrides.pop("model", config.llm_text_model),
        default_temperature=overrides.pop("temperature", config.llm_text_temperature),
        default_max_tokens=overrides.pop("max_tokens", config.llm_text_max_tokens),
        default_mode=overrides.pop("mode", config.llm_text_mode),
        react_config=react_cfg,
        extra_completion_kwargs=overrides.pop("extra_kwargs", None),
    )


def get_vision_llm_client(**overrides: Any) -> LiteLLMClient:
    """Factory for a vision-language LiteLLM client."""

    react_cfg = ReActConfig(
        max_iterations=overrides.pop("react_max_iterations", config.vlm_react_iterations),
        stop_signal=overrides.pop("react_stop_signal", config.vlm_react_stop_signal),
        feedback_template=_resolve_feedback(
            overrides.pop("react_feedback", config.vlm_react_feedback)
        ),
    )

    return LiteLLMClient(
        default_model=overrides.pop("model", config.vlm_model),
        default_temperature=overrides.pop("temperature", config.vlm_temperature),
        default_max_tokens=overrides.pop("max_tokens", config.vlm_max_tokens),
        default_mode=overrides.pop("mode", config.vlm_mode),
        react_config=react_cfg,
        extra_completion_kwargs=overrides.pop("extra_kwargs", None),
    )


__all__ = [
    "LiteLLMClient",
    "LiteLLMError",
    "LiteLLMResult",
    "ReActConfig",
    "get_text_llm_client",
    "get_vision_llm_client",
]
