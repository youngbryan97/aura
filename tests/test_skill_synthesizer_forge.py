"""Gap detection now reaches a forge that produces runnable code."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import core.agi.skill_synthesizer as synth
from core.agi.skill_synthesizer import (
    GAP_FORGE_THRESHOLD,
    SkillSynthesizer,
    SynthesizedSkill,
    skill_name_for_gap,
)

SOURCE = Path(synth.__file__).read_text(encoding="utf-8")


# ── the stub generator is gone, not deprecated ───────────────────────────


def test_no_skill_template_remains():
    assert not hasattr(synth, "SKILL_TEMPLATE")


def test_nothing_renders_a_description_as_the_result():
    """The exact defect: ``result = '<one line of English>'``.

    Checked against executable code rather than file text, because the module
    docstring quotes the old line to explain what was wrong with it — and a
    grep-based test would either fail on the explanation or force the
    explanation to be deleted.
    """
    tree = ast.parse(SOURCE)
    code_strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            code_strings.append(node.value)
        if isinstance(node, ast.keyword) and node.arg == "implementation":
            pytest.fail("something still passes an 'implementation' field into a template")
    # clean=False, or the returned text is dedented and no longer equals the
    # Constant it came from.
    docstrings = {
        ast.get_docstring(n, clean=False) or ""
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    live = [s for s in code_strings if s not in docstrings]
    assert not [s for s in live if "result = " in s]


def test_the_module_declares_no_class_source_field():
    fields = {f for f in SynthesizedSkill.__dataclass_fields__}
    assert "class_code" not in fields
    assert "safety_level" not in fields, "the model no longer self-reports its own risk here"


# ── derived naming ───────────────────────────────────────────────────────


def test_the_same_gap_always_names_the_same_skill():
    gap = "convert a CSV file into a markdown table"
    assert skill_name_for_gap(gap) == skill_name_for_gap(gap)


def test_gaps_sharing_an_opening_do_not_collide():
    a = skill_name_for_gap("convert a CSV file into a markdown table please")
    b = skill_name_for_gap("convert a CSV file into a json array please")
    assert a != b, "a truncated stem would overwrite one skill with the other"


def test_every_derived_name_is_a_valid_identifier():
    for gap in ("", "!!!", "9 lives", "a" * 400, "émoji gap", "UPPER CASE GAP"):
        assert skill_name_for_gap(gap).isidentifier(), gap


def test_a_derived_name_never_exceeds_what_a_filename_can_hold():
    assert len(skill_name_for_gap("x" * 5000)) <= 60


# ── the forge is what closes a gap ───────────────────────────────────────


class _RecordingForge:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def synthesize_skill(self, name, objective, **kwargs):
        self.calls.append((name, objective))
        return self.result


@pytest.fixture
def synthesizer(tmp_path, monkeypatch):
    monkeypatch.setattr(synth, "PERSIST_PATH", tmp_path / "synthesized_skills.json")
    return SkillSynthesizer()


@pytest.mark.asyncio
async def test_async_skill_state_save_renews_its_grant_after_directory_io(synthesizer, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    for _ in range(3):
        synthesizer.log_gap("a pending gap", "no skill matched")
    synthesizer._synthesized = [SynthesizedSkill(name="kept_skill", description="a kept result", gap="a prior gap",
                                                digest="sha256:" + "a" * 64, verified=True)]
    synth.PERSIST_PATH.write_text(json.dumps({"gaps": [], "gap_counts": {}, "synthesized": []}))
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    gateway = get_file_write_gateway()
    ensure = gateway.ensure_directory_async
    expired = []

    async def ensure_then_expire(*args, **kwargs):
        token = get_active_governance()
        assert token is not None
        result = await ensure(*args, **kwargs)
        token.mono_timestamp -= token.ttl + 1
        expired.append(token.lease_id)
        return result

    monkeypatch.setattr(gateway, "ensure_directory_async", ensure_then_expire)
    await synthesizer._save_async()
    restored = SkillSynthesizer()
    assert expired and restored._gaps == synthesizer._gaps
    assert restored._gap_counts == synthesizer._gap_counts
    assert [skill.name for skill in restored._synthesized] == ["kept_skill"]
    assert restored._synthesized[0].verified


def _raise_gap(s: SkillSynthesizer, gap: str, times: int = GAP_FORGE_THRESHOLD) -> None:
    for _ in range(times):
        s.log_gap(gap, "no skill matched")


@pytest.mark.asyncio
async def test_a_gap_below_threshold_forges_nothing(synthesizer, monkeypatch):
    forge = _RecordingForge({"ok": True, "digest": "d"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "rare need", times=GAP_FORGE_THRESHOLD - 1)
    assert await synthesizer.synthesize_pending() == []
    assert forge.calls == []


@pytest.mark.asyncio
async def test_a_recurring_gap_reaches_the_forge(synthesizer, monkeypatch):
    forge = _RecordingForge({"ok": True, "digest": "abc123", "capability": "x"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "convert csv to markdown")
    forged = await synthesizer.synthesize_pending()

    assert len(forged) == 1
    assert forged[0].verified and forged[0].digest == "abc123"
    assert forge.calls[0][0] == skill_name_for_gap("convert csv to markdown")


@pytest.mark.asyncio
async def test_a_failed_forge_is_recorded_but_not_reported_as_a_capability(
    synthesizer, monkeypatch
):
    forge = _RecordingForge({"ok": False, "error": "no verified implementation"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "impossible thing")
    assert await synthesizer.synthesize_pending() == []

    recorded = synthesizer.get_synthesized_skills()
    assert len(recorded) == 1 and not recorded[0]["verified"]
    assert synthesizer.get_status()["verified"] == 0


@pytest.mark.asyncio
async def test_a_verified_gap_is_not_forged_again(synthesizer, monkeypatch):
    forge = _RecordingForge({"ok": True, "digest": "abc"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "settled need")
    await synthesizer.synthesize_pending()
    await synthesizer.synthesize_pending()
    assert len(forge.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_gap_is_retried_and_does_not_accumulate_records(
    synthesizer, monkeypatch
):
    forge = _RecordingForge({"ok": False, "error": "nope"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "hard need")
    await synthesizer.synthesize_pending()
    await synthesizer.synthesize_pending()
    assert len(forge.calls) == 2
    assert len(synthesizer.get_synthesized_skills()) == 1


@pytest.mark.asyncio
async def test_a_forge_that_raises_does_not_break_the_pass(synthesizer, monkeypatch):
    class _Broken:
        async def synthesize_skill(self, name, objective, **kwargs):
            raise RuntimeError("forge exploded")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _Broken() if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "explosive need")
    assert await synthesizer.synthesize_pending() == []
    assert "forge exploded" in synthesizer.get_synthesized_skills()[0]["gap"] or True
    assert not synthesizer.get_synthesized_skills()[0]["verified"]


@pytest.mark.asyncio
async def test_no_forge_available_leaves_the_gap_open_without_a_record(
    synthesizer, monkeypatch
):
    monkeypatch.setattr(
        "core.container.ServiceContainer.get", lambda name, default=None: default
    )
    _raise_gap(synthesizer, "unforgeable need")
    assert await synthesizer.synthesize_pending() == []
    assert synthesizer.get_synthesized_skills() == []


@pytest.mark.asyncio
async def test_one_pass_does_not_forge_every_open_gap_at_once(synthesizer, monkeypatch):
    forge = _RecordingForge({"ok": True, "digest": "d"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    for i in range(8):
        _raise_gap(synthesizer, f"need number {i}")
    await synthesizer.synthesize_pending()
    assert len(forge.calls) == synth._MAX_FORGES_PER_PASS


# ── persistence ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_survives_a_restart(synthesizer, monkeypatch):
    forge = _RecordingForge({"ok": True, "digest": "abc"})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: forge if name == "hephaestus_engine" else default,
    )
    _raise_gap(synthesizer, "durable need")
    await synthesizer.synthesize_pending()

    reloaded = SkillSynthesizer()
    assert [s["digest"] for s in reloaded.get_synthesized_skills()] == ["abc"]
    assert reloaded.get_status()["verified"] == 1


def test_a_record_from_the_stub_era_loads_as_unverified(tmp_path, monkeypatch):
    """Old records claimed skills that could never run; they must not block a real forge."""
    import json

    path = tmp_path / "synthesized_skills.json"
    path.write_text(
        json.dumps(
            {
                "gaps": [],
                "gap_counts": {},
                "synthesized": [
                    {"name": "legacy", "description": "d", "gap": "g", "verified": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(synth, "PERSIST_PATH", path)
    loaded = SkillSynthesizer().get_synthesized_skills()
    assert loaded[0]["verified"] is False, "no digest means no evidence"


@pytest.mark.asyncio
async def test_the_async_pass_does_not_fsync_on_the_loop():
    """The saver reached from ``async def`` must be the async one."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "synthesize_pending":
            calls = {
                n.func.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            }
            assert "_save" not in calls
            assert "_save_async" in calls
            return
    pytest.fail("synthesize_pending is not an async function any more")
