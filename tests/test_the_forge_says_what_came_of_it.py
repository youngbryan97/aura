"""What a forged skill did after it was installed — the arrow nothing wrote.

`SynthesizedSkill.use_count` was declared, serialised and loaded, and
incremented by nothing anywhere in the tree. It read 0 for every forged skill
forever, so "did the thing she built help?" had a field and no answer, and a
forge that produced skills nobody ever took reported exactly like one whose
skills were working.

The outcome is read from the skill library instead, which already counts
successes and failures per skill because that is what running one produces.
A second counter is a second thing to forget to write.
"""

from __future__ import annotations

import pytest

from core.agi.skill_synthesizer import SkillSynthesizer, SynthesizedSkill


class _Library:
    def __init__(self, skills):
        self.skills = skills


class _Held:
    def __init__(self, successes, failures):
        self.successes = successes
        self.failures = failures

    @property
    def reliability(self):
        total = self.successes + self.failures
        return self.successes / total if total else 0.5


@pytest.fixture
def a_forge_with_two_skills(monkeypatch):
    forge = SkillSynthesizer.__new__(SkillSynthesizer)
    forge._gap_counts = {}
    forge._synthesized = [
        SynthesizedSkill(name="taken", description="", gap="a", verified=True),
        SynthesizedSkill(name="never_taken", description="", gap="b", verified=True),
        SynthesizedSkill(name="not_installed", description="", gap="c", verified=False),
    ]
    from core.container import ServiceContainer

    real = ServiceContainer.get
    library = _Library({"taken": _Held(4, 1), "never_taken": _Held(0, 0)})
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda key, default=None, **kw: library
            if key == "skill_library"
            else real(key, default=default, **kw)
        ),
    )
    return forge


def test_a_skill_that_was_taken_reports_what_it_did(a_forge_with_two_skills):
    said = a_forge_with_two_skills.what_the_forge_has_produced()
    row = next(one for one in said["skills"] if one["name"] == "taken")
    assert row["since"]["taken"] == 5
    assert row["since"]["successes"] == 4
    assert row["since"]["reliability"] == pytest.approx(0.8)


def test_installed_and_never_taken_is_distinguishable_from_working(
    a_forge_with_two_skills,
):
    """The reading the whole thing is for."""
    said = a_forge_with_two_skills.what_the_forge_has_produced()
    assert said["installed"] == 2
    assert said["taken_at_least_once"] == 1, (
        "a forge whose skills are never taken must not report like one whose are"
    )


def test_a_skill_the_library_never_heard_of_says_so(a_forge_with_two_skills):
    """Not an error. A skill nobody installed is itself the answer."""
    said = a_forge_with_two_skills.what_the_forge_has_produced()
    row = next(one for one in said["skills"] if one["name"] == "not_installed")
    assert row["since"]["known"] is False
    assert "does not hold it" in row["since"]["why"]


def test_the_status_carries_the_outcome_not_only_the_attempt(a_forge_with_two_skills):
    said = a_forge_with_two_skills.get_status()
    assert said["attempted"] == 3
    assert said["verified"] == 2
    assert said["installed"] == 2
    assert said["taken_at_least_once"] == 1


def test_no_library_is_reported_rather_than_raising():
    """The forge's report must survive a runtime that has no skill library."""
    forge = SkillSynthesizer.__new__(SkillSynthesizer)
    forge._gap_counts = {}
    forge._synthesized = [
        SynthesizedSkill(name="x", description="", gap="a", verified=True)
    ]
    said = forge.what_the_forge_has_produced()
    assert said["installed"] == 0
    assert said["skills"][0]["since"]["known"] is False
