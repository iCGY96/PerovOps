"""Configuration management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Global configuration for Perovops."""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self._defaults = self._load_defaults()
        self._mappings = self._load_mappings()

    def _load_defaults(self) -> Dict[str, Any]:
        """Load defaults.yaml"""
        defaults_path = self.data_dir / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path) as f:
                return yaml.safe_load(f)
        return {}

    def _load_mappings(self) -> Dict[str, Any]:
        """Load mappings.yaml"""
        mappings_path = self.data_dir / "mappings.yaml"
        if mappings_path.exists():
            with open(mappings_path) as f:
                return yaml.safe_load(f)
        return {}

    def _llm_defaults(self) -> Dict[str, Any]:
        """Return LLM configuration defaults."""
        return self._defaults.get("llm", {})

    def _path_defaults(self) -> Dict[str, Any]:
        return self._defaults.get("paths", {})

    def _ingest_defaults(self) -> Dict[str, Any]:
        return self._defaults.get("ingest", {})

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._defaults

    @property
    def mappings(self) -> Dict[str, Any]:
        return self._mappings

    def get_density(self, material: str) -> Optional[float]:
        """Get density for a material (g/cm^3)."""
        densities = self.defaults.get("densities", {})
        # Try exact match first
        if material in densities:
            return densities[material]
        # Try with underscores replaced
        material_key = material.replace("-", "_").replace(" ", "_")
        return densities.get(material_key)

    def get_energy_default(self, process_type: str) -> Optional[Dict[str, Any]]:
        """Get default energy intensity for a process type."""
        return self.defaults.get("energy_defaults", {}).get(process_type)

    def get_chemical_synonym(self, abbreviation: str) -> Optional[Dict[str, Any]]:
        """Get canonical chemical name and info from abbreviation."""
        return self.mappings.get("chemical_synonyms", {}).get(abbreviation)

    def get_lci_mapping(self, material: str) -> Optional[Dict[str, Any]]:
        """Get LCI database mapping for a material."""
        return self.mappings.get("lci_mappings", {}).get(material)

    def get_fedefl_mapping(self, material: str) -> Optional[Dict[str, Any]]:
        """Get FEDEFL mapping for elementary flows."""
        return self.mappings.get("fedefl_mappings", {}).get(material)

    def get_pedigree_default(self, source_type: str) -> Optional[Dict[str, int]]:
        """Get default pedigree scores for a source type."""
        return self.defaults.get("pedigree_defaults", {}).get(source_type)

    # Environment variables
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def tavily_api_key(self) -> Optional[str]:
        return os.getenv("TAVILY_API_KEY")

    @property
    def azure_openai_api_key(self) -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

    @property
    def llm_provider(self) -> str:
        defaults = self._llm_defaults()
        return os.getenv("LLM_PROVIDER", defaults.get("provider", "openai"))

    @property
    def vlm_provider(self) -> str:
        defaults = self._llm_defaults()
        return os.getenv(
            "VLM_PROVIDER",
            os.getenv(
                "LLM_PROVIDER",
                defaults.get("vision_provider", defaults.get("provider", "openai")),
            ),
        )

    @property
    def llm_text_model(self) -> str:
        defaults = self._llm_defaults()
        return (
            os.getenv("LITELLM_TEXT_MODEL")
            or os.getenv("LLM_TEXT_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or os.getenv("OPENAI_MODEL")
            or defaults.get("text_model", "gpt-4-turbo-preview")
        )

    @property
    def llm_text_temperature(self) -> float:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_TEXT_TEMPERATURE")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return float(defaults.get("text_temperature", 0.0))

    @property
    def llm_text_max_tokens(self) -> Optional[int]:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_TEXT_MAX_TOKENS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        value = defaults.get("text_max_tokens")
        return int(value) if value is not None else None

    @property
    def llm_text_mode(self) -> str:
        defaults = self._llm_defaults()
        value = os.getenv("LLM_TEXT_MODE") or defaults.get("text_mode", "direct")
        value_str = str(value).lower() if value is not None else "direct"
        return value_str if value_str in {"direct", "react"} else "direct"

    @property
    def llm_embedding_model(self) -> str:
        defaults = self._llm_defaults()
        return (
            os.getenv("LLM_EMBEDDING_MODEL")
            or defaults.get("embedding_model", "text-embedding-3-small")
        )

    @property
    def llm_text_react_iterations(self) -> int:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_TEXT_REACT_ITERATIONS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        try:
            return max(1, int(defaults.get("text_react_iterations", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def llm_text_react_stop_signal(self) -> Optional[str]:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_TEXT_REACT_STOP_SIGNAL")
        if override is not None:
            return override or None
        value = defaults.get("text_react_stop_signal")
        return value if value else None

    @property
    def llm_text_react_feedback(self) -> Optional[str]:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_TEXT_REACT_FEEDBACK")
        if override is not None:
            return override or None
        value = defaults.get("text_react_feedback")
        return value if value else None

    @property
    def llm_context_chars_max(self) -> int:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_MAX_CONTEXT_CHARS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        value = defaults.get("max_context_chars")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 16000

    @property
    def llm_segment_overlap(self) -> int:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_SEGMENT_OVERLAP")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        value = defaults.get("segment_overlap")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 400

    @property
    def llm_max_segments(self) -> int:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_MAX_SEGMENTS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        value = defaults.get("max_segments")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 20

    @property
    def vlm_model(self) -> str:
        defaults = self._llm_defaults()
        return (
            os.getenv("LITELLM_VLM_MODEL")
            or os.getenv("LLM_VISION_MODEL")
            or os.getenv("OPENAI_VLM_MODEL")
            or defaults.get("vision_model", "gpt-4o-mini")
        )

    @property
    def vlm_mode(self) -> str:
        defaults = self._llm_defaults()
        value = os.getenv("LLM_VISION_MODE") or defaults.get("vision_mode", "direct")
        value_str = str(value).lower() if value is not None else "direct"
        return value_str if value_str in {"direct", "react"} else "direct"

    @property
    def vlm_react_iterations(self) -> int:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_VISION_REACT_ITERATIONS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        try:
            return max(1, int(defaults.get("vision_react_iterations", 2)))
        except (TypeError, ValueError):
            return 2

    @property
    def vlm_react_stop_signal(self) -> Optional[str]:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_VISION_REACT_STOP_SIGNAL")
        if override is not None:
            return override or None
        value = defaults.get("vision_react_stop_signal")
        return value if value else None

    @property
    def vlm_react_feedback(self) -> Optional[str]:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_VISION_REACT_FEEDBACK")
        if override is not None:
            return override or None
        value = defaults.get("vision_react_feedback")
        return value if value else None

    @property
    def vlm_temperature(self) -> float:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_VISION_TEMPERATURE")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return float(defaults.get("vision_temperature", 0.0))

    @property
    def vlm_max_tokens(self) -> Optional[int]:
        defaults = self._llm_defaults()
        override = os.getenv("LLM_VISION_MAX_TOKENS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        value = defaults.get("vision_max_tokens")
        return int(value) if value is not None else None

    @property
    def llm_api_key(self) -> Optional[str]:
        override = os.getenv("LLM_API_KEY")
        if override:
            return override

        provider = self.llm_provider.lower()
        if provider == "anthropic":
            return self.anthropic_api_key
        if provider in {"openai", "azure"}:
            return self.openai_api_key or self.azure_openai_api_key
        # Fallback to any available key
        return self.openai_api_key or self.azure_openai_api_key or self.anthropic_api_key

    @property
    def vlm_api_key(self) -> Optional[str]:
        override = os.getenv("VLM_API_KEY")
        if override:
            return override

        provider = self.vlm_provider.lower()
        if provider == "anthropic":
            return self.anthropic_api_key
        if provider in {"openai", "azure"}:
            return self.openai_api_key or self.azure_openai_api_key
        return self.llm_api_key

    @property
    def cache_dir(self) -> Path:
        override = os.getenv("PEROVOPS_CACHE_DIR")
        if override:
            return Path(os.path.expanduser(override)).resolve()

        defaults = self._path_defaults()
        default_path = defaults.get("cache_dir")
        if default_path:
            return Path(os.path.expanduser(default_path)).resolve()

        return (Path.home() / ".cache" / "perovops").resolve()

    @property
    def ingest_markdown_filename(self) -> str:
        defaults = self._ingest_defaults()
        return os.getenv(
            "INGEST_MARKDOWN_FILENAME",
            defaults.get("markdown_filename", "combined.md"),
        )

    @property
    def ingest_markdown_use_llm(self) -> bool:
        defaults = self._ingest_defaults()
        override = os.getenv("INGEST_MARKDOWN_USE_LLM")
        if override is not None:
            value = override.strip().lower()
            return value not in {"", "0", "false", "no"}
        raw = defaults.get("markdown_use_llm", False)
        if isinstance(raw, str):
            return raw.strip().lower() not in {"", "0", "false", "no"}
        return bool(raw)

    @property
    def scope_agent_runs(self) -> int:
        """Number of times the scope agent should call the LLM for a single prompt."""

        override = os.getenv("SCOPE_AGENT_RUNS")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass

        agent_defaults = self.defaults.get("agents", {}).get("scope", {})
        value = agent_defaults.get("runs")
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    @property
    def scope_agent_parallel(self) -> bool:
        """Whether the scope agent should run attempts in parallel."""

        override = os.getenv("SCOPE_AGENT_PARALLEL")
        if override is not None:
            normalized = str(override).strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False

        agent_defaults = self.defaults.get("agents", {}).get("scope", {})
        value = agent_defaults.get("parallel")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return False

    def get_agent_mode(self, agent_name: str, *, family: str = "text") -> str:
        """
        Return the configured LLM interaction mode for a given agent.

        Args:
            agent_name: Agent identifier used in defaults.yaml under `agents`.
            family: Either "text" or "vision" to determine the fallback mode.

        Returns:
            Normalised mode string ("direct" or "react").
        """

        agents_cfg = self.defaults.get("agents", {}) or {}
        entry = agents_cfg.get(agent_name)
        mode_value = None

        if isinstance(entry, dict):
            mode_value = entry.get("mode")
        elif isinstance(entry, str):
            mode_value = entry

        if isinstance(mode_value, str):
            normalized = mode_value.strip().lower()
            if normalized in {"direct", "react"}:
                return normalized

        if family == "vision":
            return self.vlm_mode
        return self.llm_text_mode

    @property
    def default_region(self) -> str:
        return os.getenv("DEFAULT_REGION", "US-avg")

    @property
    def default_fu(self) -> str:
        return os.getenv("DEFAULT_FU", "1 m^2")

    @property
    def solvent_recovery_rate(self) -> float:
        return float(os.getenv("SOLVENT_RECOVERY_RATE", "0.90"))

    @property
    def brightway_project(self) -> str:
        return os.getenv("BRIGHTWAY_PROJECT_NAME", "perovops")

    @property
    def ecoinvent_path(self) -> Optional[str]:
        return os.getenv("ECOINVENT_PATH")

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")


# Global config instance
config = Config()
