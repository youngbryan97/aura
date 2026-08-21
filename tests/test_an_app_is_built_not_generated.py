"""The runtime builds apps; the language model only plans them.

Asked to build a web app, the old path sent the request to a code model and
wrote out whatever came back. That put the whole job on the one part of the
system that cannot check its own work, and it needed a 21.5GB model this host
cannot admit, so the capability was dead in practice.

These tests hold the replacement: a typed spec, one definition of what each
operation means, a compiler, and a check that the page and the model agree.
"""

from __future__ import annotations

import pytest

from core.construction.app_compiler import compile_app
from core.construction.app_model import (
    Action,
    AppSpec,
    Control,
    Field,
    Op,
    View,
    apply,
    initial_state,
)
from core.construction.app_planner import plan_from_json, spec_from_plan
from core.construction.app_verifier import dom_driver_available, node_available, verify_app


def counter() -> AppSpec:
    return AppSpec(
        title="Counter",
        fields=(Field("count", "number"),),
        actions=(
            Action("inc", (Op("add", "count", value=1),)),
            Action("reset", (Op("set", "count", value=0),)),
        ),
        controls=(Control("button", "inc", "Add one"), Control("button", "reset", "Reset")),
        views=(View("value", "count", "Count"),),
    )


def reading_list() -> AppSpec:
    return AppSpec(
        title="Reading list",
        fields=(Field("books", "list"), Field("total", "number")),
        actions=(
            Action("add", (Op("append", "books", source="entry"), Op("count", "total", source="books"))),
            Action(
                "remove_item",
                (Op("remove", "books", source="index"), Op("count", "total", source="books")),
            ),
        ),
        controls=(
            Control("text_input", input_name="entry", label="Title"),
            Control("button", "add", "Add"),
        ),
        views=(
            View("list", "books", "Books", row_action="remove_item"),
            View("value", "total", "Count"),
        ),
    )


def test_the_model_runs_the_operations():
    spec = counter()
    state = initial_state(spec)
    for _ in range(4):
        state = apply(spec, state, "inc")
    assert state["count"] == 4
    assert apply(spec, state, "reset")["count"] == 0


def test_a_spec_reports_what_would_not_compile():
    broken = AppSpec(
        title="Broken",
        fields=(Field("count", "number"),),
        actions=(Action("inc", (Op("add", "missing", value=1),)),),
        controls=(Control("button", "nope", "Press"),),
        views=(View("value", "gone"),),
    )
    problems = broken.problems()
    assert any("missing" in problem for problem in problems)
    assert any("nope" in problem for problem in problems)
    assert any("gone" in problem for problem in problems)
    with pytest.raises(ValueError):
        compile_app(broken)


def test_the_page_is_self_contained():
    html = compile_app(reading_list())
    assert "<!doctype html>" in html.lower()
    for outside in ("http://", "https://", "cdn.", "<link"):
        assert outside not in html.lower(), f"the page reaches for {outside}"


def test_every_control_and_view_binds():
    spec = reading_list()
    report = verify_app(spec, compile_app(spec))
    assert report.ok, report.problems
    assert any("bound to declared actions" in check for check in report.checks)
    assert any("bound to declared state" in check for check in report.checks)


@pytest.mark.skipif(not node_available(), reason="node is not installed here")
def test_the_page_and_the_model_agree():
    """The page's state machine and the Python one come from the same ops."""
    for spec in (counter(), reading_list()):
        report = verify_app(spec, compile_app(spec))
        assert report.semantics_checked
        assert report.ok, report.problems
        assert report.sequences_run >= len(spec.actions)


@pytest.mark.skipif(not node_available(), reason="node is not installed here")
def test_a_compiler_that_disagrees_is_caught():
    """The equivalence check has to be able to fail, or it proves nothing."""
    import core.construction.app_verifier as verifier

    spec = counter()
    html = compile_app(spec)
    original = verifier._run_in_node

    def wrong(_spec, runs, _inputs):
        return [{"count": 999} for _ in runs]

    verifier._run_in_node = wrong
    try:
        report = verify_app(spec, html)
    finally:
        verifier._run_in_node = original
    assert not report.ok
    assert any("in the page" in problem for problem in report.problems)


def test_a_thin_plan_is_repaired_into_a_working_app():
    """A plan naming no state, an invented operation and an undeclared box."""
    planned = spec_from_plan(
        {
            "title": "Water tracker",
            "actions": [
                {"name": "drink", "ops": [{"op": "add", "target": "glasses", "source": "amount"}]},
                {"name": "bogus", "ops": [{"op": "teleport", "target": "glasses"}]},
            ],
        },
        "track how much water I drink",
    )
    assert planned.spec.problems() == ()
    assert planned.spec.field_named("glasses") is not None
    assert any(control.input_name == "amount" for control in planned.spec.controls)
    assert planned.spec.action_named("bogus") is None
    assert verify_app(planned.spec, compile_app(planned.spec)).ok


def test_a_plan_is_read_out_of_surrounding_text():
    planned = plan_from_json(
        'Here is the plan:\n```json\n{"title": "Tally", "fields": [{"name": "hits", '
        '"kind": "number"}], "actions": [{"name": "hit", "ops": [{"op": "add", '
        '"target": "hits", "value": 1}]}]}\n```\nHope that helps.',
        "a tally counter",
    )
    assert planned is not None
    assert planned.spec.title == "Tally"
    assert verify_app(planned.spec, compile_app(planned.spec)).ok


def test_an_untitled_plan_takes_its_name_from_the_request():
    planned = spec_from_plan(
        {"actions": [{"name": "tap", "ops": [{"op": "add", "target": "taps", "value": 1}]}]},
        "build me a tap counter for the kitchen",
    )
    assert planned.spec.title.strip()
    assert verify_app(planned.spec, compile_app(planned.spec)).ok


@pytest.mark.skipif(not dom_driver_available(), reason="jsdom is not installed here")
def test_the_page_is_clicked_through_in_a_real_dom():
    spec = reading_list()
    report = verify_app(spec, compile_app(spec))
    assert report.driven_in_dom
    assert report.ok, report.problems
    assert any("real DOM" in check for check in report.checks)


@pytest.mark.skipif(not dom_driver_available(), reason="jsdom is not installed here")
def test_a_page_that_renders_the_wrong_thing_is_caught():
    """The reducer can be right while the page shows nothing of it.

    This breaks rendering only, which the state-machine comparison cannot
    see — that is the whole reason the page is opened and clicked.
    """
    spec = reading_list()
    html = compile_app(spec).replace(
        "node.textContent = typeof value === \"boolean\"",
        "node.textContent = \"0\"; void (typeof value === \"boolean\"",
    ).replace(
        "      : (typeof value === \"number\" ? String(Math.round(value * 100) / 100) : String(value ?? \"\"));",
        "      : (typeof value === \"number\" ? String(Math.round(value * 100) / 100) : String(value ?? \"\")));",
    )
    report = verify_app(spec, html)
    assert not report.ok
    assert any("the page shows" in problem for problem in report.problems)


@pytest.mark.skipif(not dom_driver_available(), reason="jsdom is not installed here")
def test_a_row_action_is_planned_and_driven():
    """A per-row button used to be written into the page with no name on it,
    so nothing could bind, drive or check it — and the DOM check reported a
    control that was missing from a page that was otherwise correct."""
    planned = spec_from_plan(
        {
            "title": "Errands",
            "fields": [{"name": "jobs", "kind": "list"}],
            "actions": [
                {"name": "add", "ops": [{"op": "append", "target": "jobs", "source": "job"}]},
                {"name": "drop", "ops": [{"op": "remove", "target": "jobs", "source": "index"}]},
            ],
            "inputs": [{"name": "job", "kind": "text"}],
        },
        "a list of errands",
    )
    view = next(item for item in planned.spec.views if item.field == "jobs")
    assert view.row_action == "drop"
    # A row action does not also get a page button: there is no row to name.
    assert not any(control.action == "drop" for control in planned.spec.controls)
    report = verify_app(planned.spec, compile_app(planned.spec))
    assert report.ok, report.problems
    assert report.driven_in_dom
