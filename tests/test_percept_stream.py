"""The percept stream has to be legible to the things that consume it.

A percept is written by nine places and read by four. Before this contract
existed the readers and the writers disagreed on every field: the workspace
priced its perception bid from a `salience` no producer wrote, the affect phase
keyed on a `type` half the producers omitted, and the state read a `timestamp`
that was often absent, so what she had just seen read as infinitely old.

These are the tests that would have failed then.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from core.state.aura_state import AuraState
from core.state.percepts import DEFAULT_INTENSITY, emit_percept, read_percept

REPO = Path(__file__).resolve().parents[1]


def test_salience_falls_back_to_intensity() -> None:
    """No producer in the tree writes salience. Every one writes intensity."""
    seen = read_percept({"type": "goal_achieved", "intensity": 0.7})
    assert seen.salience == pytest.approx(0.7)


def test_explicit_salience_wins_over_intensity() -> None:
    seen = read_percept({"type": "x", "intensity": 0.2, "salience": 0.9})
    assert seen.salience == pytest.approx(0.9)
    assert seen.intensity == pytest.approx(0.2)


def test_an_unstamped_percept_is_now_not_the_epoch() -> None:
    seen = read_percept({"type": "vision", "content": "a window"})
    assert seen.timestamp == pytest.approx(time.time(), abs=5.0)
    assert seen.age < 5.0


def test_role_and_source_are_read_as_the_kind() -> None:
    assert read_percept({"role": "vision"}).kind == "vision"
    assert read_percept({"source": "chat"}).kind == "chat"
    assert read_percept({"type": "error", "role": "vision"}).kind == "error"


def test_a_percept_with_no_strength_is_worth_half_not_nothing() -> None:
    """Zero would silently delete the event in every consumer that multiplies."""
    assert read_percept({"type": "vision"}).intensity == DEFAULT_INTENSITY


def test_emit_writes_every_field_a_consumer_reads() -> None:
    state = AuraState.default()
    record = emit_percept(state.world, "novel_stimulus", content="a probe", intensity=0.4)
    assert record is not None
    assert set(record) >= {"type", "content", "intensity", "salience", "timestamp"}
    seen = read_percept(state.world.recent_percepts[-1])
    assert seen.kind == "novel_stimulus"
    assert seen.salience == pytest.approx(0.4)


def test_the_workspace_bids_on_a_real_percept() -> None:
    """The bid that could never fire: every real percept priced at zero."""
    from core.consciousness.workspace_feed import build_candidates

    state = AuraState.default()
    emit_percept(state.world, "goal_achieved", content="a file was written", intensity=0.8)
    sources = {getattr(bid, "source", "") for bid in build_candidates(state)}
    assert "perception" in sources


def test_every_percept_type_the_tree_emits_can_move_affect() -> None:
    """A type with no entry in the event map arrives and does nothing.

    Three had none: a phase crash, her own apology, and every stimulus injected
    through the compatibility bridge. Two more are deliberately unmapped and
    named here, because a frame of the screen arriving is not by itself an
    emotional event and mapping it would make her react to wallpaper.
    """
    from core.phases.affect_update import AffectUpdatePhase

    unmapped_on_purpose = {"vision", "ambient_observation", "legacy_update", "unknown"}
    source = (REPO / "core" / "phases" / "affect_update.py").read_text()
    body = source[source.index("emotion_map = {") : source.index("command_impacts = {")]
    mapped = set(re.findall(r'"([a-z_]+)":\s*\[', body))
    assert {"internal_error", "self_correction", "error", "goal_achieved"} <= mapped

    emitted: set[str] = set()
    for path in (REPO / "core").rglob("*.py"):
        if "subject" in path.parts or "test" in path.name:
            continue
        text = path.read_text(errors="ignore")
        if "emit_percept(" not in text and "recent_percepts.append" not in text:
            continue
        for match in re.finditer(r'emit_percept\(\s*[^,]+,\s*"([a-z_]+)"', text):
            emitted.add(match.group(1))
        for match in re.finditer(r'"type":\s*"([a-z_]+)"', text):
            if "recent_percepts" in text[max(0, match.start() - 400) : match.start()]:
                emitted.add(match.group(1))
        # `"type": p_type` hides the literal one assignment up. Three of the
        # types the tree emits are only visible this way, and one of them —
        # her own apology — was among the three the event map could not feel.
        if "recent_percepts" in text:
            for match in re.finditer(r'\b\w*_?type\s*=\s*"([a-z_]+)"', text):
                emitted.add(match.group(1))

    orphans = emitted - mapped - unmapped_on_purpose
    assert not orphans, f"percept types nothing can feel: {sorted(orphans)}"
    assert AffectUpdatePhase is not None


def test_the_subject_probe_displaces_perception_in_the_real_shape() -> None:
    from core.subject.state import perturb, read_core_state

    state = AuraState.default()
    before = read_core_state(state).domain("P").copy()
    assert perturb(state, "P", 0.15)
    latest = read_percept(state.world.recent_percepts[-1])
    assert latest.kind == "novel_stimulus"
    assert latest.salience > 0.0
    after = read_core_state(state).domain("P")
    assert (after != before).any()


def test_a_percept_survives_the_phase_that_feels_it() -> None:
    """Affect used to clear the stream, so every later phase saw nothing."""
    from core.state.percepts import drop_consumed, fresh_for, mark_consumed

    state = AuraState.default()
    emit_percept(state.world, "goal_achieved", content="done", intensity=0.6)
    felt = fresh_for(state.world.recent_percepts, "affect")
    assert len(felt) == 1
    for item in felt:
        mark_consumed(item, "affect")
    # Still there for the workspace, the world model and the state's reading.
    assert len(state.world.recent_percepts) == 1
    assert not fresh_for(state.world.recent_percepts, "affect")


def test_a_percept_lives_exactly_one_turn() -> None:
    from core.state.percepts import drop_consumed, fresh_for, mark_consumed

    state = AuraState.default()
    for turn in range(4):
        assert drop_consumed(state.world, "affect") == (1 if turn else 0)
        emit_percept(state.world, "interaction", content=f"turn {turn}", intensity=0.5)
        for item in fresh_for(state.world.recent_percepts, "affect"):
            mark_consumed(item, "affect")
        assert len(state.world.recent_percepts) == 1


def test_the_body_under_strain_becomes_a_percept_something_can_feel() -> None:
    """`resource_pressure` was in the threat list with no producer anywhere."""
    from core.phases.affect_update import AffectUpdatePhase

    source = (REPO / "core" / "phases" / "proprioceptive_loop.py").read_text()
    assert '"resource_pressure"' in source, "the body still has no way to report strain"
    affect = (REPO / "core" / "phases" / "affect_update.py").read_text()
    body = affect[affect.index("emotion_map = {") : affect.index("command_impacts = {")]
    assert '"resource_pressure":' in body
    assert AffectUpdatePhase is not None
