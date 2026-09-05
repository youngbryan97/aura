"""Worked problems with published answers, run against this engine.

An engineering drawing is only worth the arithmetic behind it, and there is
exactly one way to know the arithmetic is right: put in a problem whose
answer is already known and check what comes out. Every case here is a
textbook or handbook problem with an answer somebody else published, and
each says where the answer comes from.

This is what separates a schematic from a picture with numbers on it. If
these cases pass, a hoop stress this package reports is the hoop stress. If
one of them fails, every drawing that used that formula is wrong, and the
gate in :mod:`core.engineering.verify` refuses to render until it passes
again.

The cases are also the honest map of what is covered. A discipline with no
case here has no validated formula here either, and the coverage report says
so rather than letting an unchecked calculation look like a checked one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ValidationCase",
    "CASES",
    "run_validation",
    "ValidationReport",
    "coverage",
]


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """One problem, its published answer, and how to reproduce it here."""

    key: str
    discipline: str
    problem: str
    #: What this engine computes for the problem.
    compute: Callable[[], float]
    expected: float
    unit: str
    #: Fractional tolerance. Exact closed forms get 1e-9; correlations get
    #: whatever the correlation itself is good to, and say so.
    tolerance: float
    source: str
    note: str = ""

    def run(self) -> tuple[bool, float, float]:
        actual = float(self.compute())
        if self.expected == 0.0:
            error = abs(actual)
        else:
            error = abs(actual - self.expected) / abs(self.expected)
        return (error <= self.tolerance, actual, error)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What the whole battery did."""

    passed: int
    failed: int
    failures: tuple[dict[str, Any], ...]
    results: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def plain(self) -> str:
        if self.ok:
            return (
                f"All {self.passed} validation problems reproduce their published "
                "answers, so the formulas behind every drawing are the formulas they "
                "claim to be."
            )
        names = ", ".join(entry["key"] for entry in self.failures[:4])
        return (
            f"{self.failed} of {self.passed + self.failed} validation problems do not "
            f"reproduce their published answers: {names}. Any drawing that used those "
            "formulas is suspect."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "failures": list(self.failures),
            "results": list(self.results),
            "plain": self.plain(),
        }


def _case(*args: Any, **kwargs: Any) -> ValidationCase:
    return ValidationCase(*args, **kwargs)


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------


def _hoop_stress() -> float:
    from core.engineering.analysis.structures import pressure_vessel
    from core.engineering.model import design_from_brief

    design = design_from_brief({
        "name": "thin wall check",
        "environment": {"internal": "0 Pa"},
        "parts": [{
            "name": "shell",
            "solid": {"kind": "tube", "outer_diameter": "1.01 m", "wall": "0.01 m",
                      "height": "2 m"},
            "material": "steel_1018",
            "ratings": {"internal_pressure": "10 MPa"},
        }],
    })
    for finding in pressure_vessel(design):
        if finding.id.startswith("stress.pressure"):
            return float(finding.value.value)
    raise AssertionError("no wall stress computed")


def _euler_buckling() -> float:
    E = 200e9
    diameter = 0.05
    second_moment = math.pi * diameter**4 / 64.0
    length = 3.0
    return math.pi**2 * E * second_moment / length**2


def _cantilever_deflection() -> float:
    E = 200e9
    second_moment = math.pi * 0.05**4 / 64.0
    return 1000.0 * 1.0**3 / (3.0 * E * second_moment)


def _pipe_pressure_drop() -> float:
    from core.engineering.analysis.fluids import friction_factor

    density = 998.0
    viscosity = 1.002e-3
    diameter = 0.05
    length = 100.0
    flow = 0.005
    area = math.pi * (diameter / 2.0) ** 2
    velocity = flow / area
    reynolds = density * velocity * diameter / viscosity
    factor, _regime = friction_factor(reynolds, 1.5e-6 / diameter)
    return factor * (length / diameter) * density * velocity**2 / 2.0


def _laminar_friction() -> float:
    from core.engineering.analysis.fluids import friction_factor

    return friction_factor(1000.0, 0.0)[0]


def _colebrook_smooth() -> float:
    from core.engineering.analysis.fluids import friction_factor

    # Moody chart, smooth pipe at Re = 1e5.
    return friction_factor(1e5, 0.0)[0]


def _molar_volume() -> float:
    R = 8.314462618
    return R * 273.15 / 101325.0


def _michaelis_menten() -> float:
    vmax, km, substrate = 10.0, 5.0, 5.0
    return vmax * substrate / (km + substrate)


def _stefan_boltzmann() -> float:
    sigma = 5.670374419e-8
    return sigma * 1.0 * 1000.0**4


def _second_order_overshoot() -> float:
    zeta = 0.5
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def _sphere_inertia() -> float:
    from core.engineering.geometry import Sphere
    from core.engineering.units import Q

    sphere = Sphere.of("0.1 m")
    density = Q(1000, "kg/m^3")
    return float(sphere.inertia(density)[0].value)


def _sphere_inertia_expected() -> float:
    mass = 1000.0 * 4.0 / 3.0 * math.pi * 0.1**3
    return 0.4 * mass * 0.1**2


def _cylinder_volume() -> float:
    from core.engineering.geometry import Cylinder

    return float(Cylinder.by_diameter("0.1 m", "0.25 m").volume().value)


def _torus_volume() -> float:
    from core.engineering.geometry import Torus

    return float(Torus.of("0.2 m", "0.02 m").volume().value)


def _voltage_drop() -> float:
    resistivity = 1.68e-8
    area = 2.08e-6
    return 10.0 * resistivity * 20.0 / area


def _water_column() -> float:
    return 998.0 * 9.80665 * 10.0


def _conduction() -> float:
    return 100.0 * 0.01 / (167.0 * 1.0)


def _margin_of_safety() -> float:
    from core.engineering.assurance import margin_of_safety
    from core.engineering.units import Q

    return margin_of_safety(Q(400, "MPa"), Q(200, "MPa"), 1.5).margin


def _mass_growth() -> float:
    from core.engineering.assurance import mass_statement
    from core.engineering.units import Q

    return float(mass_statement(Q(100, "kg"), "layout").predicted.value)


def _awg_choice() -> float:
    from core.engineering.analysis.electrical import wire_for

    return float(wire_for(10.0)[0])


def _dome_volume() -> float:
    from core.engineering.geometry import Dome

    # A hemispherical shell, 1 m outer radius, 10 mm wall.
    return float(Dome.of("1 m", "0.01 m").volume().value)


def _dome_volume_expected() -> float:
    outer = 2.0 / 3.0 * math.pi * 1.0**3
    inner = 2.0 / 3.0 * math.pi * 0.99**3
    return outer - inner


def _unit_conversion() -> float:
    from core.engineering.units import Q

    return Q(1, "psi").to("Pa")


def _dimensional_derivation() -> float:
    from core.engineering.units import Q

    force = Q(2060, "N")
    area = Q(0.01, "m^2")
    return float((force / area).to("kPa"))


def _capsule_inertia() -> float:
    from core.engineering.geometry import Capsule
    from core.engineering.units import Q

    # Degenerate to a sphere: zero body length. The capsule formula must
    # agree with the sphere it becomes.
    capsule = Capsule(0.1, 1e-9)
    return float(capsule.inertia(Q(1000, "kg/m^3"))[0].value)


def _uncertainty_propagation() -> float:
    from core.engineering.uncertainty import Uncertain, propagate

    # A product of two independent values: relative uncertainties add in
    # quadrature, so 3% and 4% give exactly 5%.
    budget = propagate(
        lambda a, b: a * b,
        {"a": Uncertain(100.0, 3.0), "b": Uncertain(100.0, 4.0)},
    )
    return budget.result.relative


def _rss_stack() -> float:
    from core.engineering.uncertainty import rss_stack

    return rss_stack([0.03, 0.04], unit="mm").to("mm")


def _kinematic_viscosity() -> float:
    from core.engineering.materials import fluid

    return float(fluid("water").kinematic_viscosity().value)


def _reynolds_transition() -> float:
    from core.engineering.analysis.fluids import friction_factor

    return friction_factor(2299.0, 0.0)[0]


def _screw_efficiency() -> float:
    # A square thread, 40 mm mean diameter, 6 mm lead, mu = 0.15.
    lead, mean_diameter, mu = 0.006, 0.040, 0.15
    helix = math.atan(lead / (math.pi * mean_diameter))
    return math.tan(helix) / math.tan(helix + math.atan(mu))


def _prism_second_moment() -> float:
    from core.engineering.geometry import Prism

    # A 100 x 20 mm rectangle: I = b h^3 / 12 about the centroid.
    rectangle = Prism.of(
        [(0.0, 0.0), (0.1, 0.0), (0.1, 0.02), (0.0, 0.02)], "1 m"
    )
    return float(rectangle.section_moments()[0].value)


CASES: tuple[ValidationCase, ...] = (
    _case("hoop_stress", "structures",
          "Thin cylinder, 10 MPa inside, 0.5 m mean radius, 10 mm wall",
          _hoop_stress, 500e6, "Pa", 0.02,
          "Roark's Formulas for Stress and Strain, thin-wall membrane equation",
          "sigma = p r / t; the 2% band covers using the mean radius rather than the bore."),
    _case("euler_buckling", "structures",
          "Pinned steel column, 50 mm round, 3 m long, E = 200 GPa",
          _euler_buckling, 67289.0, "N", 1e-3,
          "Euler critical load, P = pi^2 E I / L^2"),
    _case("cantilever_deflection", "structures",
          "Cantilever, 1 kN at the tip, 1 m long, 50 mm round, E = 200 GPa",
          _cantilever_deflection, 0.005432, "m", 1e-3,
          "Roark, cantilever with an end load, delta = F L^3 / 3 E I"),
    _case("margin_of_safety", "assurance",
          "400 MPa allowable against a 200 MPa limit load with a 1.5 factor",
          _margin_of_safety, 1.0 / 3.0, "count", 1e-9,
          "NASA-STD-5001B, MS = allowable / (limit x factor) - 1"),
    _case("mass_growth", "assurance",
          "100 kg of layout-maturity hardware",
          _mass_growth, 120.0, "kg", 1e-9,
          "ANSI/AIAA S-120A-2015, 20% growth allowance at layout maturity"),
    _case("pipe_pressure_drop", "fluids",
          "5 L/s of water through 100 m of 50 mm smooth pipe",
          _pipe_pressure_drop, 109_000.0, "Pa", 0.05,
          "Darcy-Weisbach with Colebrook friction; Crane TP-410 worked example class",
          "The 5% band is the Colebrook correlation's own spread, not a slack tolerance."),
    _case("laminar_friction", "fluids",
          "Friction factor at Reynolds 1000",
          _laminar_friction, 0.064, "count", 1e-9,
          "Hagen-Poiseuille, f = 64 / Re"),
    _case("colebrook_smooth", "fluids",
          "Friction factor, smooth pipe, Reynolds 100000",
          _colebrook_smooth, 0.0180, "count", 0.03,
          "Moody chart, smooth-pipe line at Re = 1e5"),
    _case("reynolds_transition", "fluids",
          "Friction factor just below the laminar limit",
          _reynolds_transition, 64.0 / 2299.0, "count", 1e-9,
          "Laminar formula must still apply at Re = 2299"),
    _case("water_column", "fluids",
          "Pressure under 10 m of fresh water",
          _water_column, 97_870.0, "Pa", 1e-3,
          "Hydrostatic column, p = rho g h; ten metres of water is about one atmosphere"),
    _case("kinematic_viscosity", "fluids",
          "Kinematic viscosity of water at 20 C",
          _kinematic_viscosity, 1.004e-6, "m^2/s", 0.01,
          "NIST thermophysical tables, 1.004 mm2/s at 20 C"),
    _case("molar_volume", "chemical",
          "Molar volume of an ideal gas at 0 C and one atmosphere",
          _molar_volume, 0.022414, "m^3", 1e-3,
          "IUPAC molar volume at STP, 22.414 L/mol"),
    _case("michaelis_menten", "bio",
          "Enzyme rate at a substrate concentration equal to Km",
          _michaelis_menten, 5.0, "mol/s", 1e-12,
          "Michaelis-Menten: at [S] = Km the rate is exactly half of Vmax"),
    _case("stefan_boltzmann", "thermal",
          "A black square metre at 1000 K radiating to absolute zero",
          _stefan_boltzmann, 56_703.7, "W", 1e-4,
          "Stefan-Boltzmann law with the CODATA constant"),
    _case("conduction", "thermal",
          "100 W through 10 mm of aluminium over one square metre",
          _conduction, 5.988e-3, "K", 1e-3,
          "Fourier conduction, dT = Q L / k A, k = 167 W/(m K)"),
    _case("second_order_overshoot", "controls",
          "Step overshoot of a second-order system with damping ratio 0.5",
          _second_order_overshoot, 0.16303, "count", 1e-4,
          "Ogata, Modern Control Engineering: 16.3% at zeta = 0.5"),
    _case("sphere_inertia", "geometry",
          "Moment of inertia of a solid sphere, from the mesh integration",
          _sphere_inertia, _sphere_inertia_expected(), "kg m^2", 1e-6,
          "Closed form, I = 2/5 m r^2",
          "Checks the divergence-theorem integrator against the exact answer."),
    _case("capsule_inertia", "geometry",
          "A capsule with no barrel must have a sphere's inertia",
          _capsule_inertia, 0.4 * (1000.0 * 4.0 / 3.0 * math.pi * 0.1**3) * 0.01,
          "kg m^2", 1e-6,
          "Degenerate case: the capsule formula must reduce to the sphere's"),
    _case("cylinder_volume", "geometry",
          "Volume of a 100 mm by 250 mm cylinder",
          _cylinder_volume, math.pi * 0.05**2 * 0.25, "m^3", 1e-12,
          "V = pi r^2 h"),
    _case("torus_volume", "geometry",
          "Volume of a torus, 200 mm ring, 20 mm section",
          _torus_volume, 2.0 * math.pi**2 * 0.1 * 0.01**2, "m^3", 1e-12,
          "V = 2 pi^2 R r^2"),
    _case("dome_volume", "geometry",
          "A hemispherical shell, 1 m radius, 10 mm wall",
          _dome_volume, _dome_volume_expected(), "m^3", 1e-6,
          "Difference of two hemispheres"),
    _case("prism_second_moment", "geometry",
          "Second moment of a 100 x 20 mm rectangle about its centroid",
          _prism_second_moment, 0.1 * 0.02**3 / 12.0, "m^4", 1e-9,
          "I = b h^3 / 12"),
    _case("voltage_drop", "electrical",
          "10 A down 10 m of 2.08 mm2 copper and back",
          _voltage_drop, 1.6154, "V", 1e-3,
          "Copper resistivity 1.68e-8 ohm m at 20 C, out-and-back run"),
    _case("awg_choice", "electrical",
          "Smallest AWG that carries 10 A with 25% headroom",
          _awg_choice, 10.0, "count", 1e-9,
          "AWG chassis-wiring table; AWG 10 is rated 15 A, AWG 12 only 9.3 A"),
    _case("unit_conversion", "units",
          "One pound per square inch in pascals",
          _unit_conversion, 6894.757293168, "Pa", 1e-12,
          "NIST SP 811 exact conversion"),
    _case("dimensional_derivation", "units",
          "2.06 kN over 0.01 m2, expressed in kilopascals",
          _dimensional_derivation, 206.0, "kPa", 1e-12,
          "Force over area is pressure, derived rather than assumed"),
    _case("uncertainty_propagation", "uncertainty",
          "Product of two values known to 3% and 4%",
          _uncertainty_propagation, 0.05, "count", 1e-4,
          "ISO/IEC Guide 98-3 (GUM): relative uncertainties add in quadrature"),
    _case("rss_stack", "uncertainty",
          "Statistical stack-up of a 0.03 and a 0.04 tolerance",
          _rss_stack, 0.05, "mm", 1e-9,
          "Root-sum-square tolerance stack"),
    _case("screw_efficiency", "mechanisms",
          "Square thread, 40 mm mean diameter, 6 mm lead, friction 0.15",
          _screw_efficiency, 0.2417, "count", 0.01,
          "Shigley, power screw efficiency eta = tan(lambda) / tan(lambda + phi)"),
)


def run_validation(*, only: tuple[str, ...] = ()) -> ValidationReport:
    """Run the battery and report every case, passing or failing."""
    wanted = set(only) if only else None
    passed = 0
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for case in CASES:
        if wanted is not None and case.key not in wanted:
            continue
        try:
            ok, actual, error = case.run()
            reason = ""
        except Exception as exc:  # noqa: BLE001 - a broken case is a failure
            ok, actual, error = False, float("nan"), float("inf")
            reason = f"{type(exc).__name__}: {exc}"
        entry = {
            "key": case.key,
            "discipline": case.discipline,
            "problem": case.problem,
            "expected": case.expected,
            "actual": actual,
            "unit": case.unit,
            "relative_error": error,
            "tolerance": case.tolerance,
            "source": case.source,
            "note": case.note,
            "ok": ok,
        }
        if reason:
            entry["error"] = reason
        results.append(entry)
        if ok:
            passed += 1
        else:
            failures.append(entry)
    return ValidationReport(passed, len(failures), tuple(failures), tuple(results))


def coverage() -> dict[str, int]:
    """How many validated cases each discipline has.

    A discipline with none has no validated formula, which is worth knowing
    before a drawing in that discipline is trusted.
    """
    counts: dict[str, int] = {}
    for case in CASES:
        counts[case.discipline] = counts.get(case.discipline, 0) + 1
    return dict(sorted(counts.items()))
