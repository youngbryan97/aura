"""Physical quantities that carry their dimensions and refuse to lose them.

A design is a graph of numbers, and the failure mode of a language model
asked for one is a number with the right shape and the wrong meaning: 48
where volts were wanted and amps were meant, 2900 metres of depth rating on
a hull sized for 290. Prose cannot catch that. Arithmetic can, if every
number knows what it measures.

So a :class:`Quantity` is a magnitude in SI base units plus a
:class:`Dimension` — the seven exponents of the SI base quantities. Adding
metres to seconds raises :class:`DimensionError` at the moment it is tried,
not three panels later on a drawing somebody trusts. Multiplication and
division derive the new dimension rather than assuming one, so a force
divided by an area is a pressure whether or not anybody said so.

Exponents are :class:`~fractions.Fraction`, because real engineering
quantities carry half powers: the speed of sound in a solid goes as the
square root of stiffness over density, and a fatigue stress intensity is
measured in pascal root-metre.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

__all__ = [
    "DimensionError",
    "Dimension",
    "Quantity",
    "Q",
    "parse_quantity",
    "dimension_of",
    "named_dimension",
    "UNITS",
    "DIMENSIONLESS",
    "LENGTH",
    "MASS",
    "TIME",
    "CURRENT",
    "TEMPERATURE",
    "AMOUNT",
    "LUMINOUS",
    "PREFIXES",
    "convert",
    "si_symbol",
]


class DimensionError(ValueError):
    """Two quantities were combined in a way physics does not allow."""


#: The order of the seven SI base quantities in every exponent tuple.
BASE_SYMBOLS: tuple[str, ...] = ("m", "kg", "s", "A", "K", "mol", "cd")

#: What each base quantity is called when the message is for a person.
BASE_NAMES: tuple[str, ...] = (
    "length",
    "mass",
    "time",
    "electric current",
    "temperature",
    "amount of substance",
    "luminous intensity",
)


def _frac(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(360)
    return Fraction(str(value))


@dataclass(frozen=True, slots=True)
class Dimension:
    """The seven SI exponents that say what a number measures."""

    exponents: tuple[Fraction, ...] = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )

    @staticmethod
    def of(**kwargs: Any) -> Dimension:
        """Build a dimension from named base quantities.

        ``Dimension.of(length=1, time=-2)`` is an acceleration.
        """
        slot = {
            "length": 0,
            "mass": 1,
            "time": 2,
            "current": 3,
            "temperature": 4,
            "amount": 5,
            "luminous": 6,
        }
        exps = [Fraction(0)] * 7
        for key, value in kwargs.items():
            index = slot.get(key)
            if index is None:
                raise KeyError(f"{key} is not one of the seven SI base quantities")
            exps[index] = _frac(value)
        return Dimension(tuple(exps))

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: Any) -> Dimension:
        factor = _frac(power)
        return Dimension(tuple(e * factor for e in self.exponents))

    @property
    def dimensionless(self) -> bool:
        return all(e == 0 for e in self.exponents)

    def symbol(self) -> str:
        """The dimension written in SI base units, ``kg m^2 s^-3`` style."""
        parts: list[str] = []
        for base, exponent in zip(BASE_SYMBOLS, self.exponents):
            if exponent == 0:
                continue
            if exponent == 1:
                parts.append(base)
            else:
                parts.append(f"{base}^{_exp_text(exponent)}")
        return " ".join(parts) or "1"

    def spoken(self) -> str:
        """The dimension named for a reader, ``mass x length^2 / time^3``."""
        top: list[str] = []
        bottom: list[str] = []
        for name, exponent in zip(BASE_NAMES, self.exponents):
            if exponent == 0:
                continue
            magnitude = abs(exponent)
            text = name if magnitude == 1 else f"{name}^{_exp_text(magnitude)}"
            (top if exponent > 0 else bottom).append(text)
        head = " x ".join(top) or "1"
        if not bottom:
            return head
        return f"{head} / {' x '.join(bottom)}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.symbol()


def _exp_text(exponent: Fraction) -> str:
    if exponent.denominator == 1:
        return str(exponent.numerator)
    return f"{exponent.numerator}/{exponent.denominator}"


DIMENSIONLESS = Dimension()
LENGTH = Dimension.of(length=1)
MASS = Dimension.of(mass=1)
TIME = Dimension.of(time=1)
CURRENT = Dimension.of(current=1)
TEMPERATURE = Dimension.of(temperature=1)
AMOUNT = Dimension.of(amount=1)
LUMINOUS = Dimension.of(luminous=1)

AREA = LENGTH**2
VOLUME = LENGTH**3
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME**2
FORCE = MASS * ACCELERATION
PRESSURE = FORCE / AREA
ENERGY = FORCE * LENGTH
POWER = ENERGY / TIME
CHARGE = CURRENT * TIME
VOLTAGE = POWER / CURRENT
RESISTANCE = VOLTAGE / CURRENT
CAPACITANCE = CHARGE / VOLTAGE
INDUCTANCE = VOLTAGE * TIME / CURRENT
FREQUENCY = DIMENSIONLESS / TIME
DENSITY = MASS / VOLUME
FLOW = VOLUME / TIME
MASS_FLOW = MASS / TIME
TORQUE = FORCE * LENGTH
VISCOSITY = PRESSURE * TIME
THERMAL_CONDUCTIVITY = POWER / (LENGTH * TEMPERATURE)
HEAT_CAPACITY = ENERGY / (MASS * TEMPERATURE)
ANGULAR_VELOCITY = DIMENSIONLESS / TIME
MOMENT_OF_INERTIA = MASS * AREA
SECOND_MOMENT_OF_AREA = LENGTH**4
RESISTIVITY = RESISTANCE * LENGTH
MAGNETIC_FLUX = VOLTAGE * TIME
MAGNETIC_FLUX_DENSITY = MAGNETIC_FLUX / AREA


#: The dimensions worth naming when a message has to say what went wrong.
#: Ordered most specific first so that a lookup prefers "pressure" over the
#: base-unit spelling of the same exponents.
NAMED_DIMENSIONS: tuple[tuple[str, Dimension], ...] = (
    ("dimensionless", DIMENSIONLESS),
    ("length", LENGTH),
    ("area", AREA),
    ("volume", VOLUME),
    ("mass", MASS),
    ("time", TIME),
    ("velocity", VELOCITY),
    ("acceleration", ACCELERATION),
    ("force", FORCE),
    ("pressure", PRESSURE),
    ("energy", ENERGY),
    ("power", POWER),
    ("electric current", CURRENT),
    ("charge", CHARGE),
    ("voltage", VOLTAGE),
    ("resistance", RESISTANCE),
    ("capacitance", CAPACITANCE),
    ("inductance", INDUCTANCE),
    ("frequency", FREQUENCY),
    ("density", DENSITY),
    ("volumetric flow", FLOW),
    ("mass flow", MASS_FLOW),
    ("torque", TORQUE),
    ("dynamic viscosity", VISCOSITY),
    ("thermal conductivity", THERMAL_CONDUCTIVITY),
    ("specific heat capacity", HEAT_CAPACITY),
    ("angular velocity", ANGULAR_VELOCITY),
    ("moment of inertia", MOMENT_OF_INERTIA),
    ("second moment of area", SECOND_MOMENT_OF_AREA),
    ("resistivity", RESISTIVITY),
    ("magnetic flux", MAGNETIC_FLUX),
    ("magnetic flux density", MAGNETIC_FLUX_DENSITY),
    ("temperature", TEMPERATURE),
    ("amount of substance", AMOUNT),
    ("luminous intensity", LUMINOUS),
)

_BY_DIMENSION: dict[Dimension, str] = {}
for _name, _dim in NAMED_DIMENSIONS:
    _BY_DIMENSION.setdefault(_dim, _name)


def named_dimension(dimension: Dimension) -> str:
    """What this dimension is called, or its base-unit spelling."""
    return _BY_DIMENSION.get(dimension) or dimension.spoken()


#: Every unit this package accepts, as (SI factor, dimension, offset).
#: The offset is zero for all but the two temperature scales that do not
#: share an origin with kelvin, which is why a temperature is converted
#: rather than merely scaled.
UNITS: dict[str, tuple[float, Dimension, float]] = {
    # Length
    "m": (1.0, LENGTH, 0.0),
    "metre": (1.0, LENGTH, 0.0),
    "meter": (1.0, LENGTH, 0.0),
    "in": (0.0254, LENGTH, 0.0),
    "inch": (0.0254, LENGTH, 0.0),
    "ft": (0.3048, LENGTH, 0.0),
    "foot": (0.3048, LENGTH, 0.0),
    "yd": (0.9144, LENGTH, 0.0),
    "mile": (1609.344, LENGTH, 0.0),
    "nmi": (1852.0, LENGTH, 0.0),
    "thou": (2.54e-5, LENGTH, 0.0),
    "angstrom": (1e-10, LENGTH, 0.0),
    # Mass
    "g": (1e-3, MASS, 0.0),
    "gram": (1e-3, MASS, 0.0),
    "t": (1000.0, MASS, 0.0),
    "tonne": (1000.0, MASS, 0.0),
    "lb": (0.45359237, MASS, 0.0),
    "lbm": (0.45359237, MASS, 0.0),
    "oz": (0.028349523125, MASS, 0.0),
    "slug": (14.593903, MASS, 0.0),
    # Time
    "s": (1.0, TIME, 0.0),
    "sec": (1.0, TIME, 0.0),
    "second": (1.0, TIME, 0.0),
    "min": (60.0, TIME, 0.0),
    "h": (3600.0, TIME, 0.0),
    "hr": (3600.0, TIME, 0.0),
    "hour": (3600.0, TIME, 0.0),
    "day": (86400.0, TIME, 0.0),
    "year": (31557600.0, TIME, 0.0),
    # Current, charge, potential
    "A": (1.0, CURRENT, 0.0),
    "amp": (1.0, CURRENT, 0.0),
    "C": (1.0, CHARGE, 0.0),
    "Ah": (3600.0, CHARGE, 0.0),
    "V": (1.0, VOLTAGE, 0.0),
    "volt": (1.0, VOLTAGE, 0.0),
    "ohm": (1.0, RESISTANCE, 0.0),
    "F": (1.0, CAPACITANCE, 0.0),
    "H": (1.0, INDUCTANCE, 0.0),
    "Wb": (1.0, MAGNETIC_FLUX, 0.0),
    "T": (1.0, MAGNETIC_FLUX_DENSITY, 0.0),
    "S": (1.0, CURRENT / VOLTAGE, 0.0),
    # Temperature
    "K": (1.0, TEMPERATURE, 0.0),
    "degC": (1.0, TEMPERATURE, 273.15),
    "C_deg": (1.0, TEMPERATURE, 273.15),
    "degF": (5.0 / 9.0, TEMPERATURE, 255.372222222222),
    # Amount and light
    "mol": (1.0, AMOUNT, 0.0),
    "cd": (1.0, LUMINOUS, 0.0),
    "lm": (1.0, LUMINOUS, 0.0),
    # Angle and count, dimensionless by definition
    "rad": (1.0, DIMENSIONLESS, 0.0),
    "deg": (math.pi / 180.0, DIMENSIONLESS, 0.0),
    "turn": (2.0 * math.pi, DIMENSIONLESS, 0.0),
    "%": (0.01, DIMENSIONLESS, 0.0),
    "ppm": (1e-6, DIMENSIONLESS, 0.0),
    "": (1.0, DIMENSIONLESS, 0.0),
    "count": (1.0, DIMENSIONLESS, 0.0),
    # Force, pressure, energy, power
    "N": (1.0, FORCE, 0.0),
    "kgf": (9.80665, FORCE, 0.0),
    "lbf": (4.4482216152605, FORCE, 0.0),
    "Pa": (1.0, PRESSURE, 0.0),
    "bar": (1e5, PRESSURE, 0.0),
    "atm": (101325.0, PRESSURE, 0.0),
    "psi": (6894.757293168, PRESSURE, 0.0),
    "torr": (133.322368421, PRESSURE, 0.0),
    "mmHg": (133.322368421, PRESSURE, 0.0),
    "J": (1.0, ENERGY, 0.0),
    "cal": (4.184, ENERGY, 0.0),
    "eV": (1.602176634e-19, ENERGY, 0.0),
    "Wh": (3600.0, ENERGY, 0.0),
    "BTU": (1055.05585262, ENERGY, 0.0),
    "W": (1.0, POWER, 0.0),
    "hp": (745.6998715823, POWER, 0.0),
    "Hz": (1.0, FREQUENCY, 0.0),
    "rpm": (2.0 * math.pi / 60.0, ANGULAR_VELOCITY, 0.0),
    "Nm": (1.0, TORQUE, 0.0),
    # Volume and flow
    "L": (1e-3, VOLUME, 0.0),
    "litre": (1e-3, VOLUME, 0.0),
    "liter": (1e-3, VOLUME, 0.0),
    "gal": (3.785411784e-3, VOLUME, 0.0),
    "lpm": (1e-3 / 60.0, FLOW, 0.0),
    "gpm": (3.785411784e-3 / 60.0, FLOW, 0.0),
    "cfm": (4.719474432e-4, FLOW, 0.0),
    # Viscosity
    "P": (0.1, VISCOSITY, 0.0),
    "cP": (1e-3, VISCOSITY, 0.0),
    "St": (1e-4, AREA / TIME, 0.0),
}

#: SI prefixes. A prefix binds to a unit symbol only when the remainder is a
#: unit in its own right, so ``min`` stays minutes rather than becoming
#: milli-inches.
PREFIXES: dict[str, float] = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1e2,
    "da": 1e1,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,
    "μ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}

#: Units whose symbol collides with a prefixed spelling of another unit.
#: Each is resolved in favour of the whole symbol.
_WHOLE_SYMBOLS = frozenset(
    {"min", "mol", "Pa", "Ah", "Wh", "cal", "cd", "cP", "cfm", "day", "deg", "degC", "degF"}
)


def _resolve_symbol(symbol: str) -> tuple[float, Dimension, float]:
    """Turn one unit symbol, prefix and all, into its SI conversion."""
    text = symbol.strip()
    if text in UNITS:
        return UNITS[text]
    if text in _WHOLE_SYMBOLS:
        raise KeyError(text)
    for prefix, scale in PREFIXES.items():
        if not text.startswith(prefix) or len(text) <= len(prefix):
            continue
        stem = text[len(prefix) :]
        entry = UNITS.get(stem)
        if entry is None:
            continue
        factor, dimension, offset = entry
        if offset:
            # A prefixed degree Celsius has no meaning; refuse rather than
            # silently scale an offset scale.
            continue
        return (factor * scale, dimension, 0.0)
    raise KeyError(symbol)


_TERM_RE = re.compile(r"([A-Za-zµμ%][A-Za-z_0-9µμ%]*)\s*(?:\^|\*\*)?\s*(-?\d+(?:/\d+)?)?")


def _parse_unit_expression(text: str) -> tuple[float, Dimension, float]:
    """Read ``kg m^2 / s^3`` or ``N*m`` or ``W/(m K)`` into one conversion."""
    body = text.strip()
    if not body:
        return (1.0, DIMENSIONLESS, 0.0)
    if body in UNITS:
        return UNITS[body]

    numerator, _, denominator = body.partition("/")
    factor = 1.0
    dimension = DIMENSIONLESS
    offset = 0.0
    for side, sign in ((numerator, 1), (denominator, -1)):
        cleaned = side.replace("(", " ").replace(")", " ").replace("*", " ").replace("·", " ")
        for match in _TERM_RE.finditer(cleaned):
            symbol = match.group(1)
            power = Fraction(match.group(2)) if match.group(2) else Fraction(1)
            unit_factor, unit_dim, unit_offset = _resolve_symbol(symbol)
            if unit_offset and (power != 1 or sign != 1 or body != symbol):
                raise DimensionError(
                    f"{symbol} is an offset scale and cannot be combined; use K"
                )
            offset = unit_offset
            factor *= unit_factor ** (float(power) * sign)
            dimension = dimension * (unit_dim ** (power * sign))
    return (factor, dimension, offset)


def convert(value: float, unit: str) -> tuple[float, Dimension]:
    """Convert ``value`` expressed in ``unit`` into SI base units."""
    factor, dimension, offset = _parse_unit_expression(unit)
    return (value * factor + offset, dimension)


def si_symbol(dimension: Dimension) -> str:
    """The coherent SI unit symbol for a dimension, ``W`` before ``kg m^2 s^-3``."""
    for symbol in ("N", "Pa", "J", "W", "V", "ohm", "F", "H", "Hz", "C", "T", "Wb"):
        entry = UNITS.get(symbol)
        if entry and entry[1] == dimension and entry[0] == 1.0:
            return symbol
    return dimension.symbol()


@dataclass(frozen=True, slots=True)
class Quantity:
    """A magnitude in SI base units, and what it measures.

    ``value`` is always SI. ``display_unit`` is only a preference for how the
    number is written down; it never changes what the number is.
    """

    value: float
    dimension: Dimension = DIMENSIONLESS
    display_unit: str = ""

    # -- construction ----------------------------------------------------
    @staticmethod
    def of(value: float, unit: str = "") -> Quantity:
        si_value, dimension = convert(float(value), unit)
        return Quantity(si_value, dimension, unit)

    # -- arithmetic ------------------------------------------------------
    def _check(self, other: Quantity, operation: str) -> None:
        if self.dimension != other.dimension:
            raise DimensionError(
                f"cannot {operation} {named_dimension(self.dimension)} "
                f"and {named_dimension(other.dimension)} "
                f"({self.dimension.symbol()} vs {other.dimension.symbol()})"
            )

    def __add__(self, other: Quantity) -> Quantity:
        self._check(other, "add")
        return Quantity(self.value + other.value, self.dimension, self.display_unit)

    def __sub__(self, other: Quantity) -> Quantity:
        self._check(other, "subtract")
        return Quantity(self.value - other.value, self.dimension, self.display_unit)

    def __mul__(self, other: Any) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.dimension * other.dimension)
        return Quantity(self.value * float(other), self.dimension, self.display_unit)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Quantity:
        if isinstance(other, Quantity):
            if other.value == 0.0:
                raise ZeroDivisionError("division by a zero quantity")
            return Quantity(self.value / other.value, self.dimension / other.dimension)
        return Quantity(self.value / float(other), self.dimension, self.display_unit)

    def __rtruediv__(self, other: Any) -> Quantity:
        if self.value == 0.0:
            raise ZeroDivisionError("division by a zero quantity")
        return Quantity(float(other) / self.value, DIMENSIONLESS / self.dimension)

    def __pow__(self, power: Any) -> Quantity:
        exponent = _frac(power)
        return Quantity(self.value ** float(exponent), self.dimension**exponent)

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.dimension, self.display_unit)

    def __abs__(self) -> Quantity:
        return Quantity(abs(self.value), self.dimension, self.display_unit)

    def sqrt(self) -> Quantity:
        if self.value < 0:
            raise ValueError("square root of a negative quantity")
        return Quantity(math.sqrt(self.value), self.dimension ** Fraction(1, 2))

    # -- comparison ------------------------------------------------------
    def __lt__(self, other: Quantity) -> bool:
        self._check(other, "compare")
        return self.value < other.value

    def __le__(self, other: Quantity) -> bool:
        self._check(other, "compare")
        return self.value <= other.value

    def __gt__(self, other: Quantity) -> bool:
        self._check(other, "compare")
        return self.value > other.value

    def __ge__(self, other: Quantity) -> bool:
        self._check(other, "compare")
        return self.value >= other.value

    # -- reading out -----------------------------------------------------
    def to(self, unit: str) -> float:
        """This quantity's magnitude expressed in ``unit``."""
        factor, dimension, offset = _parse_unit_expression(unit)
        if dimension != self.dimension:
            raise DimensionError(
                f"cannot express {named_dimension(self.dimension)} in {unit}, "
                f"which measures {named_dimension(dimension)}"
            )
        return (self.value - offset) / factor

    @property
    def unit(self) -> str:
        return self.display_unit or si_symbol(self.dimension)

    def as_(self, unit: str) -> Quantity:
        """The same quantity, written in a different unit."""
        self.to(unit)
        return Quantity(self.value, self.dimension, unit)

    def text(self, *, digits: int = 3) -> str:
        """The quantity as it belongs on a drawing.

        The unit somebody wrote down is the unit it is read back in, so a
        2900 m depth rating stays in metres. A magnitude that has drifted far
        outside reading range is re-prefixed instead — 0.000047 F is written
        47 µF, which is what the part is called.
        """
        if self.display_unit:
            magnitude = self.to(self.display_unit)
            if _READABLE_LOW <= abs(magnitude) < _READABLE_HIGH or magnitude == 0:
                return f"{_round_text(magnitude, digits)} {self.display_unit}".strip()
            stem = _prefixable_stem(self.display_unit)
            if stem is not None:
                return engineering_text(self.value, stem, digits=digits)
            return f"{_round_text(magnitude, digits)} {self.display_unit}".strip()
        return engineering_text(self.value, si_symbol(self.dimension), digits=digits)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text()

    def __format__(self, spec: str) -> str:  # pragma: no cover - trivial
        return self.text() if not spec else format(self.value, spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "si": self.dimension.symbol(),
            "measures": named_dimension(self.dimension),
            "text": self.text(),
        }


#: A magnitude inside this window is written plainly. Outside it, the number
#: is re-prefixed if the unit allows and written in exponent form if not.
_READABLE_LOW = 0.1
_READABLE_HIGH = 1e5


def _round_text(value: float, digits: int) -> str:
    if value == 0:
        return "0"
    if not math.isfinite(value):
        return str(value)
    magnitude = abs(value)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{value:.{max(digits - 1, 1)}e}"
    decimals = max(0, digits - 1 - int(math.floor(math.log10(magnitude))))
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _prefixable_stem(unit: str) -> str | None:
    """The unprefixed SI symbol behind a display unit, when there is one.

    ``kN`` gives ``N`` and ``F`` gives ``F``; ``psi`` and ``m/s^2`` give
    nothing, because there is no prefix ladder to walk on a customary unit
    or a compound expression.
    """
    text = unit.strip()
    if text in UNITS and UNITS[text][0] == 1.0 and UNITS[text][2] == 0.0:
        return text
    for prefix in PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            stem = text[len(prefix) :]
            entry = UNITS.get(stem)
            if entry and entry[0] == 1.0 and entry[2] == 0.0:
                return stem
    return None


#: Prefixes offered when a bare SI value is written for a person, largest
#: first, so that 2_060 newtons is read back as 2.06 kN.
_ENGINEERING_STEPS: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
)


def engineering_text(value: float, symbol: str, *, digits: int = 3) -> str:
    """Write an SI value with the prefix an engineer would say out loud."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g} {symbol}".strip()
    if symbol in {"", "1"}:
        return _round_text(value, digits)
    magnitude = abs(value)
    for step, prefix in _ENGINEERING_STEPS:
        if magnitude >= step:
            return f"{_round_text(value / step, digits)} {prefix}{symbol}".strip()
    return f"{_round_text(value, digits)} {symbol}".strip()


_QUANTITY_RE = re.compile(
    r"^\s*(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>[^\s].*?)?\s*$"
)


def parse_quantity(text: str) -> Quantity:
    """Read ``"2.06 kN"`` or ``"48V"`` or ``"2900 m"`` into a quantity.

    Raises :class:`ValueError` when the text carries no number, and
    :class:`KeyError` when it carries a unit nothing here recognises. Both
    are better than the alternative, which is a drawing with a number on it
    that nobody can convert.
    """
    match = _QUANTITY_RE.match(str(text or ""))
    if not match:
        raise ValueError(f"no quantity in {text!r}")
    unit = (match.group("unit") or "").strip()
    unit = unit.replace("°C", "degC").replace("°F", "degF").replace("°", "deg")
    unit = unit.replace("Ω", "ohm").replace("Ω", "ohm")
    return Quantity.of(float(match.group("value")), unit)


def Q(value: Any, unit: str = "") -> Quantity:  # noqa: N802 - a unit constructor
    """Make a quantity from a number and a unit, or from ``"2.06 kN"``."""
    if isinstance(value, Quantity):
        return value.as_(unit) if unit else value
    if isinstance(value, str):
        parsed = parse_quantity(value)
        return parsed.as_(unit) if unit else parsed
    return Quantity.of(float(value), unit)


def dimension_of(unit: str) -> Dimension:
    """What a unit measures, without needing a value to go with it."""
    return _parse_unit_expression(unit)[1]
