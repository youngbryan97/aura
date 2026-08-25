"""A drawing may only carry numbers the runtime computed.

The failure this suite exists to prevent is a schematic that looks like
engineering and is not: technical words and plausible figures arranged
around a picture, with nothing behind any of them. Every test here pins one
link in the chain that makes that impossible.

The published-answer battery is the load-bearing one. If those stop
reproducing their textbook answers, every number this package has ever put
on a drawing is suspect, and the suite says so rather than letting the
drawings keep looking authoritative.
"""

from __future__ import annotations

import math

import pytest

from core.engineering.units import DimensionError, Q, parse_quantity


# ── the arithmetic reproduces published answers ───────────────────────────


def test_every_validation_case_reproduces_its_published_answer():
    """The whole point. Without this, nothing else here means anything."""
    from core.engineering.validation import run_validation

    report = run_validation()
    assert report.ok, report.plain()
    assert report.passed >= 29


def test_every_validation_case_names_where_its_answer_came_from():
    from core.engineering.validation import CASES

    for case in CASES:
        assert case.source.strip(), f"{case.key} does not say where its answer comes from"
        assert case.tolerance > 0, f"{case.key} has no tolerance"


def test_validation_coverage_is_reported_per_discipline():
    """A discipline with no validated case has no validated formula."""
    from core.engineering.validation import coverage

    counts = coverage()
    for discipline in ("structures", "fluids", "electrical", "thermal", "geometry"):
        assert counts.get(discipline, 0) >= 1


# ── quantities refuse to lose their dimensions ────────────────────────────


def test_adding_unlike_quantities_raises_rather_than_guessing():
    with pytest.raises(DimensionError):
        Q(1, "m") + Q(1, "s")
    with pytest.raises(DimensionError):
        Q(48, "V") + Q(12, "A")


def test_derived_dimensions_are_computed_not_assumed():
    stress = Q(2060, "N") / Q(0.01, "m^2")
    assert stress.dimension == Q(1, "Pa").dimension
    assert stress.to("kPa") == pytest.approx(206.0)


def test_a_unit_survives_being_written_and_read_back():
    for text in ("2.06 kN", "48 V", "2900 m", "60 bar", "1.4 L/s", "210 GPa"):
        assert parse_quantity(text).text() == text


def test_a_mass_is_never_written_with_a_doubled_prefix():
    """kg already carries a prefix; prefixing it again produced "mkg"."""
    assert Q(0.0292, "kg").text() == "29.2 g"
    assert Q(2_500_000, "kg").text() == "2500 t"
    # A figure already in reading range keeps the unit it was written in.
    assert Q(2500, "kg").text() == "2500 kg"
    assert "mkg" not in " ".join(
        Q(value, "kg").text() for value in (1e-6, 1e-3, 1.0, 1e3, 1e6)
    )


# ── geometry is exact ─────────────────────────────────────────────────────


def test_volumes_are_exact_not_tessellated():
    from core.engineering.geometry import Cylinder, Sphere, Torus

    assert float(Cylinder.by_diameter("100 mm", "200 mm").volume().value) == pytest.approx(
        math.pi * 0.05**2 * 0.2, rel=1e-12
    )
    assert float(Sphere.of("50 mm").volume().value) == pytest.approx(
        4.0 / 3.0 * math.pi * 0.05**3, rel=1e-12
    )
    assert float(Torus.of("120 mm", "6 mm").volume().value) == pytest.approx(
        2.0 * math.pi**2 * 0.06 * 0.003**2, rel=1e-12
    )


def test_closed_form_inertia_is_used_where_one_exists():
    """A 32-sided polygon is about 1% light on a circle's second moment."""
    from core.engineering.geometry import Cylinder
    from core.engineering.units import Q as Quantity

    cylinder = Cylinder.by_diameter("100 mm", "200 mm")
    density = Quantity(2700, "kg/m^3")
    mass = float(cylinder.mass(density).value)
    # Ascending order, so the spin axis is the smaller one for a long bar.
    axial, first, second = (float(m.value) for m in cylinder.inertia(density))
    assert axial == pytest.approx(0.5 * mass * 0.05**2, rel=1e-9)
    transverse = mass * (3 * 0.05**2 + 0.2**2) / 12.0
    assert first == pytest.approx(transverse, rel=1e-9)
    assert second == pytest.approx(transverse, rel=1e-9)


def test_a_wall_thicker_than_its_radius_is_refused():
    from core.engineering.geometry import Tube

    with pytest.raises(ValueError):
        Tube.of("50 mm", "40 mm", "100 mm")


# ── nothing reaches a drawing without its working ─────────────────────────


def _minimal_design():
    from core.engineering.model import design_from_brief

    return design_from_brief({
        "name": "Bench bracket",
        "purpose": "Hold a 20 kg shelf off a wall.",
        "parts": [{
            "name": "Bracket",
            "function": "Carries the shelf load into the wall",
            "solid": {"kind": "plate", "width": "120 mm", "depth": "80 mm",
                      "thickness": "6 mm"},
            "material": "al_6061_t6",
            "sourcing": {"method": "cut", "specification": "6 mm 6061 plate, laser cut"},
        }],
    })


def test_a_finding_without_a_formula_is_not_drawn():
    from core.engineering.analysis import Finding
    from core.engineering.verify import grounded_findings, ungrounded

    good = Finding(
        id="x.good", name="Good", value=Q(1, "kg"),
        formula="m = V rho", method="handbook", plain="It weighs a kilogram.",
    )
    bare = Finding(id="x.bare", name="Bare", value=Q(1, "kg"), plain="It weighs a kilogram.")
    assert ungrounded(good) == ""
    assert ungrounded(bare)
    kept, dropped = grounded_findings([good, bare])
    assert [f.id for f in kept] == ["x.good"]
    assert [f.id for f in dropped] == ["x.bare"]


def test_a_number_that_is_not_finite_is_not_drawn():
    from core.engineering.analysis import Finding
    from core.engineering.verify import ungrounded

    broken = Finding(
        id="x.nan", name="Broken", value=Q(float("inf"), "kg"),
        formula="m = V rho", method="handbook", plain="Something divided by zero.",
    )
    assert ungrounded(broken)


def test_the_verifier_reports_which_checks_it_ran():
    from core.engineering.analysis import run_analyses
    from core.engineering.verify import verify_design

    design = _minimal_design()
    verdict = verify_design(design, run_analyses(design), check_validation=False)
    for expected in ("provenance", "physical plausibility", "unit and domain consistency"):
        assert expected in verdict.checks_run


# ── physics that cannot happen is blocked ─────────────────────────────────


def test_an_efficiency_above_one_blocks_the_drawing():
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    design = design_from_brief({
        "name": "Impossible motor",
        "parts": [{
            "name": "Motor",
            "function": "Turns the shaft",
            "solid": {"kind": "cylinder", "diameter": "40 mm", "height": "60 mm"},
            "material": "steel_1018",
            "ratings": {"efficiency": "1.4", "power": "100 W"},
        }],
    })
    verdict = verify_design(design, (), check_validation=False)
    assert any(p.code == "impossible_ratio" for p in verdict.blocking)


def test_wiring_two_different_voltages_together_blocks_the_drawing():
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    design = design_from_brief({
        "name": "Bad rail",
        "parts": [
            {"name": "Supply", "function": "Supplies power",
             "solid": {"kind": "box", "width": "50 mm", "height": "20 mm", "depth": "30 mm"},
             "material": "abs",
             "ports": [{"name": "out", "domain": "electrical", "role": "source",
                        "across": "48 V", "through": "2 A"}]},
            {"name": "Load", "function": "Draws power",
             "solid": {"kind": "box", "width": "30 mm", "height": "20 mm", "depth": "20 mm"},
             "material": "abs",
             "ports": [{"name": "in", "domain": "electrical", "role": "sink",
                        "across": "5 V", "through": "2 A"}]},
        ],
        "connections": [{"from": "supply.out", "to": "load.in", "domain": "electrical"}],
    })
    verdict = verify_design(design, (), check_validation=False)
    assert any(p.code == "potential_mismatch" for p in verdict.blocking)


def test_joining_two_different_domains_blocks_the_drawing():
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    design = design_from_brief({
        "name": "Wrong domain",
        "parts": [
            {"name": "Heater", "function": "Makes heat",
             "solid": {"kind": "box", "width": "50 mm", "height": "20 mm", "depth": "30 mm"},
             "material": "abs",
             "ports": [{"name": "hot", "domain": "thermal", "role": "source",
                        "through": "10 W"}]},
            {"name": "Board", "function": "Runs the logic",
             "solid": {"kind": "plate", "width": "50 mm", "depth": "30 mm",
                       "thickness": "1.6 mm"},
             "material": "gfrp",
             "ports": [{"name": "power", "domain": "electrical", "role": "sink",
                        "across": "5 V", "through": "1 A"}]},
        ],
        "connections": [{"from": "heater.hot", "to": "board.power", "domain": "thermal"}],
    })
    verdict = verify_design(design, (), check_validation=False)
    assert any(p.code == "domain_mismatch" for p in verdict.blocking)


def test_a_unit_that_does_not_measure_what_the_domain_carries_blocks():
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    design = design_from_brief({
        "name": "Wrong unit",
        "parts": [{
            "name": "Cell", "function": "Supplies power",
            "solid": {"kind": "box", "width": "50 mm", "height": "20 mm", "depth": "30 mm"},
            "material": "abs",
            "ports": [{"name": "out", "domain": "electrical", "role": "source",
                       "across": "48 m", "through": "2 A"}],
        }],
    })
    verdict = verify_design(design, (), check_validation=False)
    assert any(p.code == "wrong_unit" for p in verdict.blocking)


# ── one conservation law, several domains ─────────────────────────────────


def test_currents_that_do_not_sum_to_zero_are_reported():
    from core.engineering.analysis import run_analyses
    from core.engineering.model import design_from_brief

    design = design_from_brief({
        "name": "Unbalanced junction",
        "parts": [
            {"name": "Cell", "function": "Supplies power",
             "solid": {"kind": "box", "width": "50 mm", "height": "20 mm", "depth": "30 mm"},
             "material": "abs",
             "ports": [{"name": "out", "domain": "electrical", "role": "source",
                        "across": "12 V", "through": "1 A"}]},
            {"name": "Motor", "function": "Turns the shaft",
             "solid": {"kind": "cylinder", "diameter": "30 mm", "height": "40 mm"},
             "material": "steel_1018",
             "ports": [{"name": "feed", "domain": "electrical", "role": "sink",
                        "across": "12 V", "through": "5 A"}]},
        ],
        "connections": [{"from": "cell.out", "to": "motor.feed", "domain": "electrical",
                         "across": "12 V", "through": "1 A"}],
    })
    findings = run_analyses(design)
    balance = [f for f in findings if f.id.startswith("conservation.")]
    assert balance, "no conservation check ran"
    assert any(f.verdict == "fail" for f in balance)


def test_a_junction_nobody_declared_a_direction_for_is_unchecked_not_passed():
    from core.engineering.analysis import run_analyses
    from core.engineering.model import design_from_brief

    design = design_from_brief({
        "name": "Undeclared junction",
        "parts": [
            {"name": "Cell", "function": "Supplies power",
             "solid": {"kind": "box", "width": "50 mm", "height": "20 mm", "depth": "30 mm"},
             "material": "abs",
             "ports": [{"name": "out", "domain": "electrical",
                        "across": "12 V", "through": "1 A"}]},
            {"name": "Motor", "function": "Turns the shaft",
             "solid": {"kind": "cylinder", "diameter": "30 mm", "height": "40 mm"},
             "material": "steel_1018",
             "ports": [{"name": "feed", "domain": "electrical",
                        "across": "12 V", "through": "1 A"}]},
        ],
        "connections": [{"from": "cell.out", "to": "motor.feed", "domain": "electrical",
                         "across": "12 V", "through": "1 A"}],
    })
    balance = [f for f in run_analyses(design) if f.id.startswith("conservation.")]
    assert balance
    assert all(f.verdict != "pass" for f in balance)


# ── nobody is asked for a coordinate ──────────────────────────────────────


def test_parts_are_placed_without_the_brief_giving_positions():
    from core.engineering.layout import arrange, interference
    from core.engineering.model import design_from_brief

    design = arrange(design_from_brief({
        "name": "Packed box",
        "parts": [
            {"name": "Case", "function": "Holds everything",
             "solid": {"kind": "tube", "outer_diameter": "200 mm", "wall": "5 mm",
                       "height": "300 mm"},
             "material": "al_6061_t6", "subsystem": "body", "tags": ["enclosure"]},
            {"name": "Battery", "function": "Stores the energy",
             "solid": {"kind": "box", "width": "80 mm", "height": "40 mm", "depth": "60 mm"},
             "material": "abs", "subsystem": "body"},
            {"name": "Board", "function": "Runs the logic",
             "solid": {"kind": "plate", "width": "80 mm", "depth": "50 mm",
                       "thickness": "1.6 mm"},
             "material": "gfrp", "subsystem": "body"},
        ],
    }))
    positions = {p.id: p.placement.position for p in design.parts}
    assert len({tuple(v) for v in positions.values()}) == len(positions), (
        "every part was left at the origin"
    )
    assert interference(design) == (), "the auto-placed parts overlap each other"


def test_every_part_gets_an_explode_direction():
    from core.engineering.layout import arrange
    from core.engineering.model import design_from_brief

    design = arrange(design_from_brief({
        "name": "Two parts",
        "parts": [
            {"name": "Base", "function": "Sits underneath",
             "solid": {"kind": "box", "width": "100 mm", "height": "10 mm", "depth": "100 mm"},
             "material": "steel_1018"},
            {"name": "Lid", "function": "Covers it",
             "solid": {"kind": "box", "width": "100 mm", "height": "10 mm", "depth": "100 mm"},
             "material": "steel_1018"},
        ],
    }))
    for part in design.parts:
        assert any(abs(v) > 1e-9 for v in part.explode), f"{part.id} explodes nowhere"


# ── what she can say about it ─────────────────────────────────────────────


def test_she_does_not_claim_a_track_record_she_has_not_earned():
    from core.engineering import faculty

    faculty._RECORDS.clear()
    statement = faculty.capability_statement()
    assert "have not designed anything yet" in statement
    assert "%" not in statement.split("have not designed")[0].split("reproduce")[-1]


def test_the_capability_statement_names_what_it_cannot_do():
    from core.engineering.faculty import capability_statement

    statement = capability_statement()
    assert "cannot do" in statement
    assert "finite-element" in statement


def test_the_faculty_declares_metrics_metacognition_can_read():
    from core.engineering.faculty import declare_engineering_faculty
    from core.metacognition.faculty_model import FacultyRegistry

    registry = FacultyRegistry()
    faculty = declare_engineering_faculty(registry)
    assert faculty.faculty_id == "engineering_design"
    metric_ids = {m.metric_id for m in faculty.metrics}
    assert "validation_pass_rate" in metric_ids
    assert "grounded_result_rate" in metric_ids


def test_an_unexercised_metric_reads_as_unmeasured_rather_than_healthy():
    from core.engineering import faculty

    faculty._RECORDS.clear()
    assert faculty._grounded_result_rate() is None
    assert faculty._buildable_rate() is None


# ── what a design teaches generalises ─────────────────────────────────────


def test_a_finished_design_yields_principles_with_the_case_attached():
    from core.engineering.analysis import run_analyses
    from core.engineering.knowledge import lessons_from
    from core.engineering.model import design_from_brief

    design = design_from_brief({
        "name": "Deep housing",
        "purpose": "Keep electronics dry at depth.",
        "environment": {"depth": "500 m", "fluid": "seawater"},
        "parts": [{
            "name": "Hull", "function": "Holds the pressure out",
            "solid": {"kind": "tube", "outer_diameter": "200 mm", "wall": "8 mm",
                      "height": "400 mm"},
            "material": "al_6061_t6", "tags": ["enclosure"],
        }],
    })
    lessons = lessons_from(design, run_analyses(design))
    assert lessons, "a design that computed collapse pressure taught nothing"
    for lesson in lessons:
        assert lesson.finding_id, "a lesson with no finding behind it"
        assert lesson.principle.strip()
        assert design.name in lesson.evidence


def test_no_principle_is_taught_that_no_analysis_produced():
    from core.engineering.knowledge import lessons_from
    from core.engineering.model import design_from_brief

    design = design_from_brief({"name": "Nothing", "parts": []})
    assert lessons_from(design, ()) == ()
