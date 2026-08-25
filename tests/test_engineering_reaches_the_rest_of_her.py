"""The engineering has to be wired into the cognition, not sat beside it.

Four connections, one test each.

A figure she WRITES about a design is checked against the figure the runtime
COMPUTED, because the gap between the two is where a right answer becomes a
wrong sentence: 33.7 kg in the model and "about forty kilos" in the
paragraph, both plausible.

An imagined subject that is a physical thing externalises as a computed
schematic rather than as a generated picture, because a picture of a machine
is a machine that does not work.

Metacognition can see engineering design as a faculty and read real numbers
off it.

And what a design taught is written where the next design can find it.
"""

from __future__ import annotations

import asyncio

import pytest

FINDINGS = [
    {"id": "mass.total", "name": "Total mass",
     "value": {"value": 33.7, "unit": "kg", "si": "kg", "text": "33.7 kg"}},
    {"id": "electrical.total_draw", "name": "Total power draw",
     "value": {"value": 163.0, "unit": "W", "si": "m^2 kg s^-3", "text": "163 W"}},
    {"id": "buckle.external.hull", "name": "Collapse pressure",
     "value": {"value": 45.8e6, "unit": "Pa", "si": "m^-1 kg s^-2", "text": "45.8 MPa"}},
]


def _verify(text: str, findings=FINDINGS):
    from core.brain.verifiers.engineering_engine import EngineeringTruthEngine

    context = {"engineering_findings": findings} if findings is not None else {}
    return asyncio.run(EngineeringTruthEngine().verify(text, context=context))


# ── a written figure is checked against the computed one ──────────────────


@pytest.mark.parametrize(
    "text",
    [
        "The hull weighs 33.7 kg and the whole thing draws 163 W.",
        "The hull weighs about 34 kg.",
        "It collapses at 458 bar.",
    ],
)
def test_a_figure_that_agrees_with_the_model_passes(text):
    result = _verify(text)
    assert result.verdict == "PASSED", result.issues


@pytest.mark.parametrize(
    "text",
    [
        "The hull weighs 40 kg.",
        "It draws 500 W continuously.",
        "It collapses at 200 bar.",
    ],
)
def test_a_figure_that_contradicts_the_model_fails(text):
    result = _verify(text)
    assert result.verdict == "FAILED"
    assert result.issues


def test_a_contradiction_says_what_the_computed_figure_was_in_readable_units():
    """Quoting a power back as 163 m^2 kg s^-3 is correct and unreadable."""
    result = _verify("It draws 500 W continuously.")
    assert "163 W" in result.issues[0]


@pytest.mark.parametrize(
    "text",
    [
        "I spent three hours on it and read six papers.",
        "It is a good design.",
        "There are four of them.",
    ],
)
def test_a_number_that_measures_nothing_computed_is_not_checked(text):
    """Most numbers in a sentence are not claims about the design."""
    assert _verify(text).verdict == "UNCHECKED"


def test_with_nothing_computed_it_reports_unchecked_rather_than_passed():
    """Reporting 'verified' with nothing to verify against is the whole failure."""
    result = _verify("The hull weighs 40 kg.", findings=None)
    assert result.verdict == "UNCHECKED"
    assert result.ok is True


def test_the_engineering_engine_is_on_the_registry():
    from core.brain.verifiers.registry import VerifierRegistry

    registry = VerifierRegistry()
    for task in ("engineering", "design", "schematic", "mechanical", "electrical"):
        names = [getattr(v, "name", "") for v in registry.select(registry.normalize_task(task))]
        assert "engineering" in names, task


# ── imagination externalises a physical thing as a drawing ────────────────


@pytest.mark.parametrize(
    "focus",
    [
        ["thruster", "propeller", "housing"],
        ["pressure", "vessel", "titanium"],
        ["gearbox", "winch"],
        ["circuit", "battery", "resistor"],
        ["bracket", "shelf"],
    ],
)
def test_a_physical_subject_externalises_as_a_computed_schematic(focus):
    from core.brain.imagination import _externalization_path

    assert "schematic" in _externalization_path(focus, " ".join(focus))


@pytest.mark.parametrize(
    "focus",
    [
        ["loneliness", "memory", "evening"],
        ["poem", "metaphor", "light"],
        ["friendship", "trust"],
        ["sunset", "ocean", "colour"],
    ],
)
def test_everything_else_still_externalises_as_an_image(focus):
    """A poem about light is a poem, not a lamp."""
    from core.brain.imagination import _externalization_path

    assert "schematic" not in _externalization_path(focus, " ".join(focus))


def test_the_subject_test_reads_the_live_registries():
    """Not a word list kept in the imagination module."""
    import inspect

    from core.brain import imagination

    source = inspect.getsource(imagination._engineerable_score)
    assert "SOLID_KINDS" in source
    assert "closest_material" in source
    assert "object_class_of" in source


def test_imagination_still_works_with_no_engineering_package():
    """A runtime without it must still imagine, not raise."""
    import inspect

    from core.brain import imagination

    assert "except ImportError" in inspect.getsource(imagination._engineerable_score)


# ── metacognition can see it ──────────────────────────────────────────────


def test_the_self_model_includes_engineering_design():
    from core.metacognition.default_faculties import (
        ensure_default_faculties,
        reset_default_faculties_for_test,
    )

    reset_default_faculties_for_test()
    model = ensure_default_faculties().assess()
    assert model.by_id("engineering_design") is not None


def test_the_faculty_reads_a_real_number_not_a_claim():
    from core.engineering.faculty import _validation_pass_rate

    rate = _validation_pass_rate()
    assert rate == pytest.approx(1.0), "the published-answer battery is not green"


def test_a_faculty_that_cannot_be_declared_does_not_take_the_self_model_down():
    import ast
    import inspect
    import textwrap

    from core.metacognition import default_faculties

    source = inspect.getsource(default_faculties._declare_owned_faculties)
    assert "record_degradation" in source
    # Read the code, not the prose: the docstring says "rather than raised".
    tree = ast.parse(textwrap.dedent(source))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]


# ── what it taught is written down ────────────────────────────────────────


def test_a_lesson_carries_the_principle_and_the_case_that_showed_it():
    from core.engineering.knowledge import PRINCIPLES, Lesson

    for prefix, (principle, discipline, source) in PRINCIPLES.items():
        assert principle.strip(), prefix
        assert discipline.strip(), prefix
        assert source.strip(), f"{prefix} states a principle with no source"

    lesson = Lesson("A tube collapses before it yields.", "Seen in X: 458 bar.",
                    "buckle.external.hull", "structures", "Windenburg and Trilling")
    assert lesson.statement().startswith("A tube collapses")
    assert "458 bar" in lesson.statement()


def test_recording_works_with_no_memory_service_up():
    """A design produced from a script has no container behind it."""
    from core.engineering.knowledge import record_design_knowledge
    from core.engineering.model import design_from_brief

    design = design_from_brief({
        "name": "Deep housing",
        "environment": {"depth": "500 m", "fluid": "seawater"},
        "parts": [{
            "name": "Hull", "function": "Holds the pressure out",
            "solid": {"kind": "tube", "outer_diameter": "200 mm", "wall": "8 mm",
                      "height": "400 mm"},
            "material": "al_6061_t6", "tags": ["enclosure"],
        }],
    })
    from core.engineering.analysis import run_analyses

    lessons = record_design_knowledge(design, run_analyses(design))
    assert lessons, "nothing was learned and nothing was raised either"
