"""The physical domains a design connects things through, and their laws.

Modelica settled this question decades ago and every physical modelling tool
since has kept the answer: a connection in any energy domain carries exactly
two variables. One is an ACROSS variable, measured between two points, and
it is equal for everything joined at a node — voltage, pressure,
temperature, velocity. The other is a THROUGH variable, measured by cutting
the connection, and it sums to zero at a node — current, mass flow, heat
flow, force.

That pair is what makes a design graph checkable instead of decorative.
Kirchhoff's current law, a mass balance around a tee, and the requirement
that the forces on a bracket add up are the same statement in three domains,
so one checker covers all of them. A wiring diagram whose currents do not
sum to zero has a fault in it, and the fault is found before anything is
drawn.

Signals are the exception and are marked as such. A sensor reading has a
direction and no conservation law; nothing is used up by being measured.

References: Modelica Language Specification chapter 9, connectors and
connections; Modelica.UsersGuide.Connectors on the choice of variables per
domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.engineering.units import Dimension, dimension_of

__all__ = [
    "Domain",
    "DOMAINS",
    "domain",
    "domain_names",
    "conserved_domains",
    "DomainError",
]


class DomainError(ValueError):
    """A port or connection named a domain that does not exist."""


@dataclass(frozen=True, slots=True)
class Domain:
    """One physical domain, and the two variables its connections carry."""

    key: str
    name: str
    across_name: str
    across_unit: str
    through_name: str
    through_unit: str
    conserved: bool = True
    #: How a person who is not an engineer should read the two variables.
    across_plain: str = ""
    through_plain: str = ""
    #: The colour family a schematic uses for this domain's lines.
    line_style: str = "solid"
    #: What flows, in one word, for a legend.
    carries: str = ""

    @property
    def across_dimension(self) -> Dimension:
        return dimension_of(self.across_unit)

    @property
    def through_dimension(self) -> Dimension:
        return dimension_of(self.through_unit)

    def power_of(self, across: float, through: float) -> float:
        """The power crossing this connection, in watts.

        Across times through is power in every conserved domain here, which
        is what makes an energy balance a single sum rather than a special
        case per domain.
        """
        return float(across) * float(through)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "across": {
                "name": self.across_name,
                "unit": self.across_unit,
                "plain": self.across_plain,
                "law": "equal for everything joined at one point",
            },
            "through": {
                "name": self.through_name,
                "unit": self.through_unit,
                "plain": self.through_plain,
                "law": "sums to zero at every junction" if self.conserved else "not conserved",
            },
            "conserved": self.conserved,
            "carries": self.carries,
            "line_style": self.line_style,
        }


_DOMAIN_LIST: tuple[Domain, ...] = (
    Domain(
        "electrical", "Electrical", "voltage", "V", "current", "A",
        across_plain="how hard the electricity is pushed",
        through_plain="how much electricity is flowing",
        line_style="solid", carries="electricity",
    ),
    Domain(
        "thermal", "Thermal", "temperature", "K", "heat flow", "W",
        across_plain="how hot it is",
        through_plain="how fast heat is moving",
        line_style="wavy", carries="heat",
    ),
    Domain(
        "fluid", "Fluid", "pressure", "Pa", "mass flow", "kg/s",
        across_plain="how hard the fluid is pushed",
        through_plain="how much fluid passes per second",
        line_style="solid-heavy", carries="liquid or gas",
    ),
    Domain(
        "hydraulic", "Hydraulic power", "pressure", "Pa", "mass flow", "kg/s",
        across_plain="oil pressure",
        through_plain="oil flow",
        line_style="solid-heavy", carries="pressurised oil",
    ),
    Domain(
        "pneumatic", "Pneumatic", "pressure", "Pa", "mass flow", "kg/s",
        across_plain="air pressure",
        through_plain="air flow",
        line_style="dashed-double", carries="compressed air",
    ),
    Domain(
        "mechanical_linear", "Mechanical, sliding", "velocity", "m/s", "force", "N",
        across_plain="how fast it moves",
        through_plain="how hard it pushes",
        line_style="solid-heavy", carries="push and pull",
    ),
    Domain(
        "mechanical_rotary", "Mechanical, turning", "angular velocity", "rad/s",
        "torque", "N m",
        across_plain="how fast it spins",
        through_plain="how hard it twists",
        line_style="solid-heavy", carries="rotation",
    ),
    Domain(
        "structural", "Structural", "displacement", "m", "force", "N",
        across_plain="how far it deflects",
        through_plain="the load it carries",
        line_style="solid-heavy", carries="load",
    ),
    Domain(
        "magnetic", "Magnetic", "magnetomotive force", "A", "magnetic flux", "Wb",
        across_plain="magnetic drive",
        through_plain="magnetic field passing through",
        line_style="dash-dot", carries="magnetic field",
    ),
    Domain(
        "chemical", "Chemical", "chemical potential", "J/mol", "molar flow", "mol/s",
        across_plain="how strongly the reaction wants to go",
        through_plain="how much substance moves per second",
        line_style="solid", carries="reagent",
    ),
    Domain(
        "optical", "Optical", "irradiance", "W/m^2", "radiant power", "W",
        across_plain="how bright it is",
        through_plain="how much light energy arrives",
        line_style="dotted", carries="light",
    ),
    Domain(
        "acoustic", "Acoustic", "sound pressure", "Pa", "volume velocity", "m^3/s",
        across_plain="how loud it is",
        through_plain="how much air the sound moves",
        line_style="wavy", carries="sound",
    ),
    # Not conserved: an information channel has a direction and no balance.
    Domain(
        "signal", "Signal", "signal value", "", "", "",
        conserved=False,
        across_plain="the reading or command being sent",
        through_plain="nothing is used up by being measured",
        line_style="dashed", carries="a measurement or a command",
    ),
    Domain(
        "data", "Data", "message", "", "", "",
        conserved=False,
        across_plain="the data being carried",
        through_plain="nothing is used up by being read",
        line_style="dashed-long", carries="data",
    ),
    Domain(
        "biological", "Biological", "concentration", "mol/m^3", "molar flow", "mol/s",
        across_plain="how concentrated it is",
        through_plain="how much moves per second",
        line_style="solid", carries="a molecule or a cell population",
    ),
)

DOMAINS: dict[str, Domain] = {entry.key: entry for entry in _DOMAIN_LIST}

#: What people call these domains when they are not reading a standard.
_ALIASES: dict[str, str] = {
    "electric": "electrical",
    "power": "electrical",
    "elec": "electrical",
    "wire": "electrical",
    "heat": "thermal",
    "cooling": "thermal",
    "flow": "fluid",
    "liquid": "fluid",
    "water": "fluid",
    "process": "fluid",
    "gas": "pneumatic",
    "air": "pneumatic",
    "oil": "hydraulic",
    "mechanical": "mechanical_linear",
    "linear": "mechanical_linear",
    "translation": "mechanical_linear",
    "rotation": "mechanical_rotary",
    "rotary": "mechanical_rotary",
    "shaft": "mechanical_rotary",
    "torque": "mechanical_rotary",
    "load": "structural",
    "mount": "structural",
    "mechanical_mount": "structural",
    "light": "optical",
    "sound": "acoustic",
    "audio": "acoustic",
    "control": "signal",
    "sensor": "signal",
    "measurement": "signal",
    "bus": "data",
    "network": "data",
    "comms": "data",
    "bio": "biological",
    "metabolic": "biological",
}


def domain_names() -> tuple[str, ...]:
    return tuple(DOMAINS)


def conserved_domains() -> tuple[str, ...]:
    return tuple(key for key, entry in DOMAINS.items() if entry.conserved)


def domain(key: str) -> Domain:
    """Look a domain up by key or by what somebody would call it."""
    text = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in DOMAINS:
        return DOMAINS[text]
    if text in _ALIASES:
        return DOMAINS[_ALIASES[text]]
    plain = text.replace("_", " ")
    if plain in _ALIASES:
        return DOMAINS[_ALIASES[plain]]
    raise DomainError(
        f"{key!r} is not a domain this connects; the domains are "
        f"{', '.join(sorted(DOMAINS))}"
    )
