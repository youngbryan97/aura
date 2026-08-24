"""What things are made of, with the numbers that decide whether they work.

A design is only as honest as its material properties. Asking a language
model for the yield strength of 6061-T6 gets a plausible number roughly as
often as it gets the right one, and the difference shows up as a part that
snaps. So the properties live here as data, each entry carrying the room
temperature value, the units it is in, and the reference class it came
from.

Values are the conventional room-temperature design figures used in
handbook practice (ASM Handbook properties for metals, ISO 527 tensile data
for polymers, published supplier datasheets for composites). They are
design starting points, and every one of them is reported alongside the
analysis that used it so the number can be argued with.

Every material also carries ``feels_like`` and ``used_for``, because a
reader who does not know what Ti-6Al-4V is still deserves to know that it
is as strong as steel, half the weight, and expensive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.engineering.units import Q, Quantity

__all__ = [
    "Material",
    "Fluid",
    "MATERIALS",
    "FLUIDS",
    "material",
    "fluid",
    "material_names",
    "closest_material",
    "STANDARD_GRAVITY",
]

#: Standard gravity, CGPM 1901. Every weight in this package uses it.
STANDARD_GRAVITY = Q(9.80665, "m/s^2")


@dataclass(frozen=True, slots=True)
class Material:
    """One solid material and the properties a design decision needs."""

    key: str
    name: str
    family: str
    density: Quantity
    youngs_modulus: Quantity | None = None
    yield_strength: Quantity | None = None
    ultimate_strength: Quantity | None = None
    poisson_ratio: float | None = None
    thermal_conductivity: Quantity | None = None
    specific_heat: Quantity | None = None
    thermal_expansion: Quantity | None = None
    electrical_resistivity: Quantity | None = None
    melting_point: Quantity | None = None
    max_service_temperature: Quantity | None = None
    cost_per_kg: Quantity | None = None
    feels_like: str = ""
    used_for: str = ""
    cautions: str = ""
    source: str = ""

    def property(self, name: str) -> Quantity | None:
        value = getattr(self, name, None)
        return value if isinstance(value, Quantity) else None

    def specific_strength(self) -> Quantity | None:
        """Yield strength divided by density: strength per kilogram carried."""
        if self.yield_strength is None:
            return None
        return self.yield_strength / self.density

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "feels_like": self.feels_like,
            "used_for": self.used_for,
            "source": self.source,
        }
        if self.cautions:
            out["cautions"] = self.cautions
        if self.poisson_ratio is not None:
            out["poisson_ratio"] = self.poisson_ratio
        for field_name in (
            "density",
            "youngs_modulus",
            "yield_strength",
            "ultimate_strength",
            "thermal_conductivity",
            "specific_heat",
            "thermal_expansion",
            "electrical_resistivity",
            "melting_point",
            "max_service_temperature",
            "cost_per_kg",
        ):
            value = self.property(field_name)
            if value is not None:
                out[field_name] = value.to_dict()
        return out


@dataclass(frozen=True, slots=True)
class Fluid:
    """One working fluid, for the flow and thermal analyses."""

    key: str
    name: str
    density: Quantity
    dynamic_viscosity: Quantity
    thermal_conductivity: Quantity | None = None
    specific_heat: Quantity | None = None
    vapour_pressure: Quantity | None = None
    at_temperature: Quantity = field(default_factory=lambda: Q(20, "degC"))
    feels_like: str = ""
    source: str = ""

    def kinematic_viscosity(self) -> Quantity:
        return self.dynamic_viscosity / self.density

    def to_dict(self) -> dict[str, Any]:
        out = {
            "key": self.key,
            "name": self.name,
            "density": self.density.to_dict(),
            "dynamic_viscosity": self.dynamic_viscosity.to_dict(),
            "at_temperature": self.at_temperature.to_dict(),
            "feels_like": self.feels_like,
            "source": self.source,
        }
        for name in ("thermal_conductivity", "specific_heat", "vapour_pressure"):
            value = getattr(self, name, None)
            if isinstance(value, Quantity):
                out[name] = value.to_dict()
        return out


def _m(
    key: str,
    name: str,
    family: str,
    density: str,
    *,
    E: str | None = None,
    sy: str | None = None,
    su: str | None = None,
    nu: float | None = None,
    k: str | None = None,
    cp: str | None = None,
    alpha: str | None = None,
    rho_e: str | None = None,
    tm: str | None = None,
    tmax: str | None = None,
    cost: str | None = None,
    feels_like: str = "",
    used_for: str = "",
    cautions: str = "",
    source: str = "",
) -> Material:
    """Build one material entry, keeping the table below readable."""
    return Material(
        key=key,
        name=name,
        family=family,
        density=Q(density),
        youngs_modulus=Q(E) if E else None,
        yield_strength=Q(sy) if sy else None,
        ultimate_strength=Q(su) if su else None,
        poisson_ratio=nu,
        thermal_conductivity=Q(k) if k else None,
        specific_heat=Q(cp) if cp else None,
        thermal_expansion=Q(alpha) if alpha else None,
        electrical_resistivity=Q(rho_e) if rho_e else None,
        melting_point=Q(tm) if tm else None,
        max_service_temperature=Q(tmax) if tmax else None,
        cost_per_kg=Q(cost) if cost else None,
        feels_like=feels_like,
        used_for=used_for,
        cautions=cautions,
        source=source,
    )


_HANDBOOK = "ASM Handbook room-temperature design values"
_POLYMER = "ISO 527 tensile data, supplier datasheets"
_CERAMIC = "Published technical-ceramic datasheets"

_ENTRIES: tuple[Material, ...] = (
    # -- steels ----------------------------------------------------------
    _m(
        "steel_1018", "AISI 1018 mild steel", "steel", "7870 kg/m^3",
        E="205 GPa", sy="370 MPa", su="440 MPa", nu=0.29,
        k="51.9 W/(m K)", cp="486 J/(kg K)", alpha="11.5e-6 1/K",
        rho_e="1.59e-7 ohm m", tm="1420 degC", tmax="400 degC", cost="1.1 count",
        feels_like="Ordinary structural steel: heavy, cheap, easy to weld and machine.",
        used_for="Frames, brackets, shafts, anything that has to be strong before it has to be light.",
        cautions="Rusts without paint, plating or oil.",
        source=_HANDBOOK,
    ),
    _m(
        "steel_4140", "AISI 4140 alloy steel, quenched and tempered", "steel", "7850 kg/m^3",
        E="205 GPa", sy="655 MPa", su="1020 MPa", nu=0.29,
        k="42.6 W/(m K)", cp="473 J/(kg K)", alpha="12.3e-6 1/K",
        rho_e="2.2e-7 ohm m", tm="1416 degC", tmax="425 degC", cost="2.4 count",
        feels_like="Mild steel's tougher cousin, near twice the strength for the same weight.",
        used_for="Drive shafts, gears, high-load pins, tooling.",
        cautions="Heat treatment sets the properties; welding after it undoes them.",
        source=_HANDBOOK,
    ),
    _m(
        "steel_304", "304 stainless steel", "stainless steel", "8000 kg/m^3",
        E="193 GPa", sy="215 MPa", su="505 MPa", nu=0.29,
        k="16.2 W/(m K)", cp="500 J/(kg K)", alpha="17.3e-6 1/K",
        rho_e="7.2e-7 ohm m", tm="1400 degC", tmax="870 degC", cost="4.5 count",
        feels_like="The stainless of kitchen sinks: corrosion-proof, springy, a poor conductor of heat.",
        used_for="Anything wet, food-contact or outdoors; tanks, fasteners, enclosures.",
        cautions="Work-hardens while machining; pits in salt water where 316 would not.",
        source=_HANDBOOK,
    ),
    _m(
        "steel_316", "316L stainless steel", "stainless steel", "8000 kg/m^3",
        E="193 GPa", sy="205 MPa", su="515 MPa", nu=0.29,
        k="16.3 W/(m K)", cp="500 J/(kg K)", alpha="16.0e-6 1/K",
        rho_e="7.4e-7 ohm m", tm="1390 degC", tmax="870 degC", cost="6.0 count",
        feels_like="304 with molybdenum added, which is what makes it survive seawater.",
        used_for="Marine hardware, chemical plant, implants, pressure vessels.",
        source=_HANDBOOK,
    ),
    # -- light alloys ----------------------------------------------------
    _m(
        "al_6061_t6", "6061-T6 aluminium", "aluminium", "2700 kg/m^3",
        E="68.9 GPa", sy="276 MPa", su="310 MPa", nu=0.33,
        k="167 W/(m K)", cp="896 J/(kg K)", alpha="23.6e-6 1/K",
        rho_e="3.99e-8 ohm m", tm="582 degC", tmax="170 degC", cost="4.0 count",
        feels_like="A third the weight of steel, strong enough for most structures, easy to cut.",
        used_for="Chassis, heat sinks, enclosures, robot arms, bicycle frames.",
        cautions="Loses its temper above about 170 C, including in the heat of a weld.",
        source=_HANDBOOK,
    ),
    _m(
        "al_7075_t6", "7075-T6 aluminium", "aluminium", "2810 kg/m^3",
        E="71.7 GPa", sy="503 MPa", su="572 MPa", nu=0.33,
        k="130 W/(m K)", cp="960 J/(kg K)", alpha="23.6e-6 1/K",
        rho_e="5.15e-8 ohm m", tm="477 degC", tmax="120 degC", cost="9.0 count",
        feels_like="Aircraft aluminium: as strong as mild steel at a third of the weight.",
        used_for="Airframes, high-load brackets, competition parts.",
        cautions="Practically unweldable, and corrodes faster than 6061.",
        source=_HANDBOOK,
    ),
    _m(
        "ti_6al4v", "Ti-6Al-4V titanium", "titanium", "4430 kg/m^3",
        E="113.8 GPa", sy="880 MPa", su="950 MPa", nu=0.342,
        k="6.7 W/(m K)", cp="526 J/(kg K)", alpha="8.6e-6 1/K",
        rho_e="1.7e-6 ohm m", tm="1604 degC", tmax="400 degC", cost="35 count",
        feels_like="Stronger than most steel, 45% lighter, and it does not corrode at all.",
        used_for="Implants, aerospace joints, deep-sea housings, anywhere weight costs more than money.",
        cautions="Expensive, slow to machine, and a poor conductor of heat.",
        source=_HANDBOOK,
    ),
    _m(
        "mg_az31b", "AZ31B magnesium", "magnesium", "1770 kg/m^3",
        E="45 GPa", sy="200 MPa", su="260 MPa", nu=0.35,
        k="96 W/(m K)", cp="1000 J/(kg K)", alpha="26e-6 1/K",
        rho_e="9.2e-8 ohm m", tm="630 degC", tmax="150 degC", cost="6.5 count",
        feels_like="The lightest structural metal, noticeably lighter in the hand than aluminium.",
        used_for="Camera bodies, laptop shells, drone frames.",
        cautions="Machining swarf burns; corrodes badly unless coated.",
        source=_HANDBOOK,
    ),
    # -- conductors ------------------------------------------------------
    _m(
        "copper_c101", "C101 oxygen-free copper", "copper", "8940 kg/m^3",
        E="117 GPa", sy="70 MPa", su="220 MPa", nu=0.34,
        k="391 W/(m K)", cp="385 J/(kg K)", alpha="17e-6 1/K",
        rho_e="1.68e-8 ohm m", tm="1085 degC", tmax="200 degC", cost="9.5 count",
        feels_like="The best everyday conductor of both electricity and heat, and heavy with it.",
        used_for="Wiring, busbars, heat spreaders, water pipe.",
        source=_HANDBOOK,
    ),
    _m(
        "brass_360", "C360 free-machining brass", "copper alloy", "8500 kg/m^3",
        E="97 GPa", sy="310 MPa", su="400 MPa", nu=0.34,
        k="115 W/(m K)", cp="380 J/(kg K)", alpha="20.5e-6 1/K",
        rho_e="6.6e-8 ohm m", tm="900 degC", tmax="200 degC", cost="8.0 count",
        feels_like="Cuts like butter on a lathe, does not spark, goes dull gold with age.",
        used_for="Fittings, valve bodies, bushings, decorative hardware.",
        source=_HANDBOOK,
    ),
    # -- polymers --------------------------------------------------------
    _m(
        "abs", "ABS", "thermoplastic", "1040 kg/m^3",
        E="2.3 GPa", sy="43 MPa", su="43 MPa", nu=0.35,
        k="0.17 W/(m K)", cp="1400 J/(kg K)", alpha="90e-6 1/K",
        rho_e="1e13 ohm m", tmax="80 degC", cost="3.0 count",
        feels_like="Lego plastic: tough, matte, forgiving to drop.",
        used_for="Housings, printed prototypes, consumer casings.",
        cautions="Softens in a hot car; sunlight makes it chalky.",
        source=_POLYMER,
    ),
    _m(
        "pla", "PLA", "thermoplastic", "1240 kg/m^3",
        E="3.5 GPa", sy="50 MPa", su="50 MPa", nu=0.36,
        k="0.13 W/(m K)", cp="1800 J/(kg K)", alpha="68e-6 1/K",
        rho_e="1e14 ohm m", tmax="55 degC", cost="2.5 count",
        feels_like="The default 3D-printing plastic: stiff, glossy, brittle when it fails.",
        used_for="Prototypes, jigs, models, anything that stays indoors and cool.",
        cautions="Deforms above 55 C, which a closed car exceeds.",
        source=_POLYMER,
    ),
    _m(
        "nylon_66", "Nylon 6,6", "thermoplastic", "1140 kg/m^3",
        E="2.8 GPa", sy="82 MPa", su="82 MPa", nu=0.39,
        k="0.25 W/(m K)", cp="1670 J/(kg K)", alpha="80e-6 1/K",
        rho_e="1e12 ohm m", tmax="105 degC", cost="5.0 count",
        feels_like="Slippery and tough, the plastic gears and cable ties are made of.",
        used_for="Gears, bearings, bushings, living hinges, cable management.",
        cautions="Absorbs water and swells, which moves the dimensions.",
        source=_POLYMER,
    ),
    _m(
        "peek", "PEEK", "thermoplastic", "1320 kg/m^3",
        E="3.6 GPa", sy="97 MPa", su="97 MPa", nu=0.4,
        k="0.25 W/(m K)", cp="1340 J/(kg K)", alpha="47e-6 1/K",
        rho_e="1e14 ohm m", tmax="250 degC", cost="90 count",
        feels_like="A plastic that behaves like a metal, holds up at 250 C, costs like silver.",
        used_for="Pump parts, implants, seals in hot chemistry, aerospace bushings.",
        source=_POLYMER,
    ),
    _m(
        "polycarbonate", "Polycarbonate", "thermoplastic", "1200 kg/m^3",
        E="2.4 GPa", sy="62 MPa", su="65 MPa", nu=0.37,
        k="0.2 W/(m K)", cp="1200 J/(kg K)", alpha="65e-6 1/K",
        rho_e="1e14 ohm m", tmax="115 degC", cost="4.5 count",
        feels_like="Clear, and hard to break: riot shields and safety glazing.",
        used_for="Windows, guards, lenses, light pipes, enclosures.",
        cautions="Scratches easily and cracks in contact with some solvents.",
        source=_POLYMER,
    ),
    _m(
        "ptfe", "PTFE", "thermoplastic", "2200 kg/m^3",
        E="0.5 GPa", sy="23 MPa", su="31 MPa", nu=0.46,
        k="0.25 W/(m K)", cp="1000 J/(kg K)", alpha="135e-6 1/K",
        rho_e="1e18 ohm m", tmax="260 degC", cost="20 count",
        feels_like="The non-stick pan coating: waxy, slippery, chemically inert.",
        used_for="Seals, low-friction bearings, chemical tubing, wire insulation.",
        cautions="Creeps under steady load, so it will not hold a clamped dimension.",
        source=_POLYMER,
    ),
    _m(
        "hdpe", "HDPE", "thermoplastic", "960 kg/m^3",
        E="1.1 GPa", sy="26 MPa", su="31 MPa", nu=0.42,
        k="0.48 W/(m K)", cp="1900 J/(kg K)", alpha="120e-6 1/K",
        rho_e="1e15 ohm m", tmax="80 degC", cost="1.8 count",
        feels_like="Milk-jug plastic: waxy, floats, almost impossible to break.",
        used_for="Tanks, chopping boards, pipe, chemical containers.",
        cautions="Nothing sticks to it, glue included.",
        source=_POLYMER,
    ),
    _m(
        "silicone_rubber", "Silicone rubber, 60 Shore A", "elastomer", "1200 kg/m^3",
        E="0.005 GPa", sy="7 MPa", su="9 MPa", nu=0.49,
        k="0.2 W/(m K)", cp="1300 J/(kg K)", alpha="300e-6 1/K",
        rho_e="1e13 ohm m", tmax="200 degC", cost="12 count",
        feels_like="Bakeware rubber: soft, stretchy, indifferent to heat and cold.",
        used_for="Seals, gaskets, soft grippers, membranes, wearables.",
        cautions="Tears easily once a cut starts.",
        source=_POLYMER,
    ),
    _m(
        "nitrile_rubber", "Nitrile rubber, 70 Shore A", "elastomer", "1250 kg/m^3",
        E="0.008 GPa", sy="14 MPa", su="17 MPa", nu=0.49,
        k="0.25 W/(m K)", cp="1400 J/(kg K)", alpha="230e-6 1/K",
        tmax="120 degC", cost="6 count",
        feels_like="The black O-ring rubber, the one that shrugs off oil and fuel.",
        used_for="O-rings, hydraulic seals, fuel hose, gloves.",
        cautions="Ozone and sunlight crack it; it is not for outdoor exposure.",
        source=_POLYMER,
    ),
    # -- composites and ceramics -----------------------------------------
    _m(
        "cfrp_ud", "Carbon fibre / epoxy, unidirectional", "composite", "1600 kg/m^3",
        E="135 GPa", sy="1500 MPa", su="1500 MPa", nu=0.3,
        k="7 W/(m K)", cp="900 J/(kg K)", alpha="0.5e-6 1/K",
        tmax="120 degC", cost="45 count",
        feels_like="Stiffer than steel along the fibres and a fifth of the weight.",
        used_for="Spars, drone arms, pressure vessel overwrap, tubing.",
        cautions="Strong one way only; across the fibres it is weak epoxy.",
        source="Supplier prepreg datasheets, fibre direction",
    ),
    _m(
        "gfrp", "Glass fibre / epoxy laminate", "composite", "1900 kg/m^3",
        E="25 GPa", sy="300 MPa", su="350 MPa", nu=0.28,
        k="0.3 W/(m K)", cp="1000 J/(kg K)", alpha="9e-6 1/K",
        tmax="130 degC", cost="8 count",
        feels_like="Boat-hull fibreglass: strong, cheap, and electrically insulating.",
        used_for="Hulls, panels, circuit board substrate, radomes, tanks.",
        source="Supplier laminate datasheets",
    ),
    _m(
        "alumina_96", "96% alumina ceramic", "ceramic", "3720 kg/m^3",
        E="303 GPa", su="358 MPa", nu=0.21,
        k="25 W/(m K)", cp="880 J/(kg K)", alpha="8.2e-6 1/K",
        rho_e="1e12 ohm m", tm="2050 degC", tmax="1500 degC", cost="20 count",
        feels_like="Spark-plug ceramic: very hard, very stiff, and it shatters rather than bends.",
        used_for="Insulators, wear plates, seal faces, furnace parts.",
        cautions="No yield: it goes from intact to broken with no warning.",
        source=_CERAMIC,
    ),
    _m(
        "borosilicate_glass", "Borosilicate glass", "ceramic", "2230 kg/m^3",
        E="64 GPa", su="70 MPa", nu=0.2,
        k="1.2 W/(m K)", cp="830 J/(kg K)", alpha="3.3e-6 1/K",
        rho_e="1e12 ohm m", tmax="500 degC", cost="6 count",
        feels_like="Laboratory glass: clear, and it survives a temperature change that cracks window glass.",
        used_for="Sight glasses, vessels, optics, lab ware.",
        source=_CERAMIC,
    ),
    _m(
        "concrete_c30", "C30/37 structural concrete", "mineral", "2400 kg/m^3",
        E="33 GPa", su="30 MPa", nu=0.2,
        k="1.7 W/(m K)", cp="880 J/(kg K)", alpha="10e-6 1/K",
        tmax="300 degC", cost="0.12 count",
        feels_like="Strong when squashed, nearly useless when pulled, which is why it has steel in it.",
        used_for="Foundations, slabs, walls, ballast.",
        cautions="Tensile strength is about a tenth of the compressive figure.",
        source="Eurocode 2 characteristic strengths",
    ),
    _m(
        "wood_pine", "Softwood pine, along the grain", "natural", "500 kg/m^3",
        E="9 GPa", sy="40 MPa", su="40 MPa", nu=0.3,
        k="0.12 W/(m K)", cp="1700 J/(kg K)", alpha="4e-6 1/K",
        tmax="100 degC", cost="0.8 count",
        feels_like="Light, warm, and much weaker across the grain than along it.",
        used_for="Framing, jigs, patterns, furniture, formwork.",
        cautions="Moves with humidity and burns.",
        source="Wood Handbook, USDA Forest Products Laboratory",
    ),
    # -- biological and soft matter --------------------------------------
    _m(
        "cortical_bone", "Cortical bone", "biological", "1900 kg/m^3",
        E="17 GPa", sy="115 MPa", su="133 MPa", nu=0.3,
        k="0.32 W/(m K)", cp="1300 J/(kg K)",
        feels_like="About as stiff as concrete and far tougher, and it repairs itself.",
        used_for="The reference case for implants, prosthetics and orthopaedic hardware.",
        cautions="Properties vary with site, age and loading direction.",
        source="Currey, Bones: Structure and Mechanics",
    ),
    _m(
        "soft_tissue", "Soft tissue, generic", "biological", "1050 kg/m^3",
        E="0.00005 GPa", su="0.5 MPa", nu=0.49,
        k="0.5 W/(m K)", cp="3600 J/(kg K)",
        feels_like="Nearly incompressible and a hundred thousand times floppier than bone.",
        used_for="Contact modelling for wearables, grippers, surgical tools.",
        cautions="Strongly nonlinear: a single stiffness figure only holds at small strain.",
        source="Fung, Biomechanics: Mechanical Properties of Living Tissues",
    ),
    _m(
        "pdms", "PDMS, 10:1", "elastomer", "970 kg/m^3",
        E="0.0018 GPa", su="7 MPa", nu=0.49,
        k="0.15 W/(m K)", cp="1460 J/(kg K)",
        tmax="200 degC", cost="60 count",
        feels_like="Clear, soft and gas-permeable, which is why cells live in it.",
        used_for="Microfluidic chips, organ-on-chip devices, soft actuators.",
        source="Published PDMS characterisation data",
    ),
)

MATERIALS: dict[str, Material] = {entry.key: entry for entry in _ENTRIES}


_FLUID_ENTRIES: tuple[Fluid, ...] = (
    Fluid(
        "water", "Fresh water", Q(998, "kg/m^3"), Q(1.002, "cP"),
        thermal_conductivity=Q(0.598, "W/(m K)"), specific_heat=Q(4182, "J/(kg K)"),
        vapour_pressure=Q(2.34, "kPa"), at_temperature=Q(20, "degC"),
        feels_like="The reference fluid; everything else is compared to it.",
        source="NIST thermophysical property tables",
    ),
    Fluid(
        "seawater", "Seawater, 35 PSU", Q(1025, "kg/m^3"), Q(1.08, "cP"),
        thermal_conductivity=Q(0.596, "W/(m K)"), specific_heat=Q(3993, "J/(kg K)"),
        at_temperature=Q(20, "degC"),
        feels_like="Denser than fresh water, and it conducts electricity and eats metal.",
        source="UNESCO equation of state for seawater",
    ),
    Fluid(
        "air", "Air at one atmosphere", Q(1.204, "kg/m^3"), Q(0.01813, "cP"),
        thermal_conductivity=Q(0.02514, "W/(m K)"), specific_heat=Q(1006, "J/(kg K)"),
        at_temperature=Q(20, "degC"),
        feels_like="Light, and a poor conductor of heat, which is what makes insulation work.",
        source="NIST thermophysical property tables",
    ),
    Fluid(
        "hydraulic_oil", "ISO VG 46 hydraulic oil", Q(875, "kg/m^3"), Q(41.4, "cP"),
        thermal_conductivity=Q(0.13, "W/(m K)"), specific_heat=Q(1900, "J/(kg K)"),
        at_temperature=Q(40, "degC"),
        feels_like="Forty times thicker than water, which is what lets it seal and lubricate.",
        source="ISO 3448 viscosity grade definition",
    ),
    Fluid(
        "ethylene_glycol_50", "50% ethylene glycol coolant", Q(1071, "kg/m^3"), Q(3.8, "cP"),
        thermal_conductivity=Q(0.38, "W/(m K)"), specific_heat=Q(3300, "J/(kg K)"),
        at_temperature=Q(20, "degC"),
        feels_like="Antifreeze: it carries less heat than water and will not freeze solid.",
        source="ASHRAE Handbook, Fundamentals",
    ),
    Fluid(
        "blood", "Whole blood at body temperature", Q(1060, "kg/m^3"), Q(3.5, "cP"),
        thermal_conductivity=Q(0.52, "W/(m K)"), specific_heat=Q(3600, "J/(kg K)"),
        at_temperature=Q(37, "degC"),
        feels_like="Three times thicker than water, and it thins as it is pushed faster.",
        source="Chien, shear-dependent blood viscosity data",
    ),
)

FLUIDS: dict[str, Fluid] = {entry.key: entry for entry in _FLUID_ENTRIES}


#: Everyday names for the materials above, so a brief that says "aluminium"
#: resolves without the grade suffix nobody outside a machine shop uses.
_ALIASES: dict[str, str] = {
    "aluminium": "al_6061_t6",
    "aluminum": "al_6061_t6",
    "6061": "al_6061_t6",
    "7075": "al_7075_t6",
    "steel": "steel_1018",
    "mild steel": "steel_1018",
    "carbon steel": "steel_1018",
    "alloy steel": "steel_4140",
    "stainless": "steel_316",
    "stainless steel": "steel_316",
    "304": "steel_304",
    "316": "steel_316",
    "titanium": "ti_6al4v",
    "magnesium": "mg_az31b",
    "copper": "copper_c101",
    "brass": "brass_360",
    "plastic": "abs",
    "nylon": "nylon_66",
    "rubber": "nitrile_rubber",
    "silicone": "silicone_rubber",
    "carbon fibre": "cfrp_ud",
    "carbon fiber": "cfrp_ud",
    "fibreglass": "gfrp",
    "fiberglass": "gfrp",
    "glass": "borosilicate_glass",
    "ceramic": "alumina_96",
    "alumina": "alumina_96",
    "concrete": "concrete_c30",
    "wood": "wood_pine",
    "timber": "wood_pine",
    "bone": "cortical_bone",
    "tissue": "soft_tissue",
    "polyethylene": "hdpe",
    "teflon": "ptfe",
    "lexan": "polycarbonate",
}


def material_names() -> tuple[str, ...]:
    return tuple(MATERIALS)


def material(key: str) -> Material:
    """Look a material up by key or by the name a person would use."""
    text = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in MATERIALS:
        return MATERIALS[text]
    plain = str(key or "").strip().lower()
    if plain in _ALIASES:
        return MATERIALS[_ALIASES[plain]]
    if text.replace("_", " ") in _ALIASES:
        return MATERIALS[_ALIASES[text.replace("_", " ")]]
    found = closest_material(str(key or ""))
    if found is not None:
        return found
    raise KeyError(
        f"no material named {key!r}; known materials are {', '.join(sorted(MATERIALS))}"
    )


def closest_material(text: str) -> Material | None:
    """The best match for a loosely written material name, or nothing."""
    needle = str(text or "").strip().lower()
    if not needle:
        return None
    for alias, key in _ALIASES.items():
        if alias in needle:
            return MATERIALS[key]
    for key, entry in MATERIALS.items():
        if key in needle.replace(" ", "_") or entry.name.lower() in needle:
            return entry
    return None


def fluid(key: str) -> Fluid:
    text = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in FLUIDS:
        return FLUIDS[text]
    for name, entry in FLUIDS.items():
        if name in text or text in entry.name.lower():
            return entry
    raise KeyError(f"no fluid named {key!r}; known fluids are {', '.join(sorted(FLUIDS))}")
