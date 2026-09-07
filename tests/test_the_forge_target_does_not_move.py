"""A redraft is verified against the probes the first draft set."""
from __future__ import annotations

import asyncio
import logging

import pytest

from core.skill_management import hephaestus as forge
from core.skill_management.skill_verification import Probe, VerificationReport
from core.skill_management.what_the_probes_were_held_to import (
    how_the_target_has_held,
    reset_how_the_target_has_held,
)


@pytest.fixture(autouse=True)
def _a_clean_count():
    reset_how_the_target_has_held()
    yield
    reset_how_the_target_has_held()


def _an_engine() -> forge.HephaestusEngine:
    """The forge without its constructor, which makes directories and logs."""
    engine = forge.HephaestusEngine.__new__(forge.HephaestusEngine)
    engine.logger = logging.getLogger("test.forge")
    return engine


def _a_draft(expected_value: int) -> dict:
    return {
        "ok": True,
        "description": "returns a number",
        "code": "return {'ok': True, 'value': 42}",
        "imports": [],
        "probes": (Probe.of({}, expect={"ok": True, "value": expected_value}),),
        "deterministic": True,
        "gap": "a gap",
    }


def test_the_second_attempt_is_judged_by_the_first_attempts_probes(monkeypatch):
    """The defect: answer a failing expectation by rewriting the expectation."""
    engine = _an_engine()
    offered = [_a_draft(42), _a_draft(0)]
    judged: list[tuple[Probe, ...]] = []

    async def draft(name, objective, *, feedback=""):
        return offered.pop(0)

    def verify(draft_obj):
        judged.append(draft_obj.probes)
        return VerificationReport(
            draft=draft_obj,
            passed=len(judged) > 1,
            stage="probe",
            reason="value mismatch" if len(judged) == 1 else "",
        )

    monkeypatch.setattr(engine, "_draft_logic", draft)
    monkeypatch.setattr(forge, "verify_draft", verify)

    report = asyncio.run(
        engine._forge_verified_draft("a_skill", "do a thing", max_attempts=2)
    )

    assert report is not None and report.passed
    assert len(judged) == 2
    # The second attempt offered `value: 0`. It was still asked `value: 42`.
    first = judged[0][0]
    assert first in judged[1], "the redraft moved the target it was judged by"


def test_moving_the_target_is_counted_where_the_inspector_can_read_it(monkeypatch):
    engine = _an_engine()
    offered = [_a_draft(42), _a_draft(0)]

    async def draft(name, objective, *, feedback=""):
        return offered.pop(0)

    calls = {"n": 0}

    def verify(draft_obj):
        calls["n"] += 1
        return VerificationReport(
            draft=draft_obj,
            passed=calls["n"] > 1,
            stage="probe",
            reason="value mismatch" if calls["n"] == 1 else "",
        )

    monkeypatch.setattr(engine, "_draft_logic", draft)
    monkeypatch.setattr(forge, "verify_draft", verify)
    asyncio.run(engine._forge_verified_draft("a_skill", "do a thing", max_attempts=2))

    read = how_the_target_has_held()
    assert read["redrafts"] == 1
    assert read["moved_the_target"] == 1


def test_a_forge_that_passes_first_time_never_redrafts(monkeypatch):
    engine = _an_engine()

    async def draft(name, objective, *, feedback=""):
        return _a_draft(42)

    def verify(draft_obj):
        return VerificationReport(draft=draft_obj, passed=True, stage="probe", reason="")

    monkeypatch.setattr(engine, "_draft_logic", draft)
    monkeypatch.setattr(forge, "verify_draft", verify)
    asyncio.run(engine._forge_verified_draft("a_skill", "do a thing", max_attempts=2))

    assert how_the_target_has_held()["redrafts"] == 0
