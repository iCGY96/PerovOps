"""Unit conversion utilities with optional pint integration."""

import logging
from typing import Optional, Tuple, Dict, TYPE_CHECKING, Any, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pint import Quantity as PintQuantity  # type: ignore
else:  # Fallback for runtime without pint
    PintQuantity = Any  # type: ignore

try:  # Optional dependency: pint
    import pint
except ImportError:  # pragma: no cover - import guard
    pint = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

if pint is not None:
    # Create unit registry with custom units
    ureg = pint.UnitRegistry()

    # Define custom units for materials science
    ureg.define("rpm = revolution / minute = RPM")
    ureg.define("angstrom = 1e-10 * meter = Å = A")
    ureg.define("nm = nanometer")
    ureg.define("m2 = meter ** 2")
    ureg.define("cm2 = centimeter ** 2")
    ureg.define("g_cm3 = gram / centimeter ** 3")
else:
    ureg = None


class UnitConverter:
    """Handle unit conversions and parsing."""

    _LENGTH_UNITS: Dict[str, Tuple[str, float]] = {
        "nm": ("m", 1e-9),
        "m": ("m", 1.0),
        "cm": ("m", 1e-2),
    }
    _MASS_UNITS: Dict[str, Tuple[str, float]] = {
        "g": ("kg", 1e-3),
        "kg": ("kg", 1.0),
    }
    _ENERGY_UNITS: Dict[str, Tuple[str, float]] = {
        "kWh": ("J", 3.6e6),
        "Wh": ("J", 3600.0),
        "MJ": ("J", 1e6),
        "J": ("J", 1.0),
    }

    class _SimpleQuantity:
        """Fallback quantity implementation when pint is unavailable."""

        def __init__(self, converter: "UnitConverter", value: float, unit: str):
            self._converter = converter
            self.magnitude = float(value)
            self._unit = unit

        def to(self, unit: str) -> "UnitConverter._SimpleQuantity":
            value = self._converter._convert_fallback(self.magnitude, self._unit, unit)
            if value is None:
                raise ValueError(f"Cannot convert {self._unit} -> {unit}")
            return UnitConverter._SimpleQuantity(self._converter, value, unit)

        @property
        def units(self) -> str:
            return self._unit

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"Quantity({self.magnitude!r}, '{self._unit}')"

    def __init__(self):
        self.ureg = ureg

    QuantityLike = Union["UnitConverter._SimpleQuantity", PintQuantity]

    def parse_quantity(self, value: float, unit: str) -> Optional[QuantityLike]:
        """
        Parse a quantity with units.

        Args:
            value: Numeric value
            unit: Unit string

        Returns:
            pint.Quantity or None if parsing fails
        """
        if self.ureg is not None:
            try:
                unit = unit.strip().replace("^", "**")
                return self.ureg.Quantity(value, unit)
            except Exception as e:
                logger.warning(f"Could not parse unit '{unit}': {e}")
                return None

        # Fallback quantity
        unit = unit.strip()
        if not unit:
            logger.warning("No unit provided for quantity parsing")
            return None
        return self._SimpleQuantity(self, value, unit)

    def convert(
        self, value: float, from_unit: str, to_unit: str
    ) -> Optional[float]:
        """
        Convert a value from one unit to another.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value, or None if conversion fails
        """
        try:
            q = self.parse_quantity(value, from_unit)
            if q is None:
                return None
            return q.to(to_unit).magnitude
        except Exception as e:
            logger.warning(
                f"Unit conversion failed: {value} {from_unit} -> {to_unit}: {e}"
            )
            return None

    def normalize_unit(self, unit: str) -> str:
        """
        Normalize unit string to standard form.

        Args:
            unit: Unit string

        Returns:
            Normalized unit string
        """
        if self.ureg is not None:
            try:
                q = self.ureg.Quantity(1, unit)
                return f"{q.units:~}"  # Compact form
            except Exception:
                return unit

        # Fallback: normalize based on known units
        unit = unit.strip()
        if unit in self._LENGTH_UNITS:
            return unit
        if unit in self._MASS_UNITS:
            return unit
        if unit in self._ENERGY_UNITS:
            return unit
        return unit

    def extract_from_text(self, text: str) -> list[Tuple[float, str]]:
        """
        Extract quantities with units from text using quantulum3.

        Args:
            text: Input text

        Returns:
            List of (value, unit) tuples
        """
        try:
            from quantulum3 import parser

            quantities = parser.parse(text)
            results = []
            for q in quantities:
                if q.value and q.unit:
                    unit_str = q.unit.name
                    results.append((q.value, unit_str))
            return results
        except ImportError:
            logger.warning("quantulum3 not installed")
            return []
        except Exception as e:
            logger.warning(f"Quantity extraction failed: {e}")
            return []

    def _convert_fallback(
        self, value: float, from_unit: str, to_unit: str
    ) -> Optional[float]:
        """Fallback converter when pint is unavailable."""
        if from_unit == to_unit:
            return value

        if (
            from_unit in self._LENGTH_UNITS
            and to_unit in self._LENGTH_UNITS
        ):
            base_value = value * self._LENGTH_UNITS[from_unit][1]
            return base_value / self._LENGTH_UNITS[to_unit][1]

        if from_unit in self._MASS_UNITS and to_unit in self._MASS_UNITS:
            base_value = value * self._MASS_UNITS[from_unit][1]
            return base_value / self._MASS_UNITS[to_unit][1]

        if from_unit in self._ENERGY_UNITS and to_unit in self._ENERGY_UNITS:
            base_value = value * self._ENERGY_UNITS[from_unit][1]
            return base_value / self._ENERGY_UNITS[to_unit][1]

        return None


# Global converter instance
converter = UnitConverter()
