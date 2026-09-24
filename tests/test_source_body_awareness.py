"""Tests for the source-body proprioception organ (core/soma/source_body.py).

Verifies, against real throwaway git repositories:
  - organ mapping from file paths to subsystem names
  - snapshot capture (clean, dirty, non-repo, missing git)
  - the durable awakening ledger (round trip, corruption tolerance, compaction)
  - boot-over-boot deltas (commits, authors, organs, reverts, unreadable
    history, dirty-only change) and their deterministic narratives
  - crash correlation via crash-evidence mtimes
  - the async awakening pass (ledger write, event publish, episodic memory)
  - the live pulse (edit detection, deduplication, settle events)
  - prompt surfaces (somatic change lines; cached-only, no git on the hot path)
  - the system_proprioception skill's source-body report
  - container/service spine registration
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.soma.source_body import (
    BodyDelta,
    SourceBodyAwareness,
    SourceBodySnapshot,
    get_source_body,
    organ_of,
    reset_source_body_for_test,
)

_SUBPROCESS_GATEWAY = get_subprocess_gateway()

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "Bryan",
    "GIT_AUTHOR_EMAIL": "bryan@test.local",
    "GIT_COMMITTER_NAME": "Bryan",
    "GIT_COMMITTER_EMAIL": "bryan@test.local",
}


def _git(repo: Path, *args: str, author: str | None = None) -> str:
    env = dict(_GIT_ENV)
    if author:
        env["GIT_AUTHOR_NAME"] = author
        env["GIT_AUTHOR_EMAIL"] = f"{author.lower()}@test.local"
        env["GIT_COMMITTER_NAME"] = author
        env["GIT_COMMITTER_EMAIL"] = f"{author.lower()}@test.local"
    proc = _SUBPROCESS_GATEWAY.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=str(repo),
        env=env,
        timeout=30,
        offline_tooling=True,
        source="proof_tooling:test_source_body_git_fixture",
        accelerator_capability="none",
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout.strip()


def _write(repo: Path, rel: str, content: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str, *, author: str = "Bryan") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, author=author)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def body_repo(tmp_path):
    repo = tmp_path / "body"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, "core/memory/engine.py", "x = 1\n")
    _write(repo, "core/capability_engine.py", "y = 1\n")
    _write(repo, "tests/test_engine.py", "def test(): pass\n")
    _write(repo, "docs/NOTES.md", "notes\n")
    _commit(repo, "initial body")
    return repo


@pytest.fixture
def organ(body_repo, tmp_path):
    return SourceBodyAwareness(
        source_root=body_repo,
        ledger_path=tmp_path / "ledger" / "source_body_ledger.jsonl",
        crash_evidence_dir=tmp_path / "crash",
    )


def _fresh_organ(organ: SourceBodyAwareness) -> SourceBodyAwareness:
    """A 'next boot' twin: same body, same ledger, new instance."""
    return SourceBodyAwareness(
        source_root=organ.source_root,
        ledger_path=organ.ledger_path,
        crash_evidence_dir=organ.crash_evidence_dir,
    )


class _FakeBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic, data, priority=None):
        self.events.append((topic, data))

    def topics(self) -> list[str]:
        return [t for t, _ in self.events]


class _FakeMemory:
    def __init__(self):
        self.stored: list[tuple[str, dict]] = []

    async def store(self, text, **kwargs):
        self.stored.append((text, kwargs))


@pytest.fixture
def fake_bus(monkeypatch):
    bus = _FakeBus()
    import core.event_bus as event_bus_module

    monkeypatch.setattr(event_bus_module, "get_event_bus", lambda: bus)
    return bus


@pytest.fixture
def fake_memory(monkeypatch):
    memory = _FakeMemory()
    import core.runtime.service_access as service_access

    original = service_access.optional_service

    def _optional(name, default=None, **kwargs):
        if name == "memory_manager":
            return memory
        return original(name, default=default, **kwargs)

    monkeypatch.setattr(service_access, "optional_service", _optional)
    return memory


# ── organ mapping ─────────────────────────────────────────────────


def test_organ_of_core_package():
    assert organ_of("core/memory/engine.py") == "memory"
    assert organ_of("core/memory/sub/deep.py") == "memory"


def test_organ_of_core_module():
    assert organ_of("core/capability_engine.py") == "capability_engine"


def test_organ_of_non_core_trees():
    assert organ_of("tests/test_engine.py") == "tests"
    assert organ_of("docs/NOTES.md") == "docs"


def test_organ_of_root_and_degenerate_paths():
    assert organ_of("Makefile") == "repo_root"
    assert organ_of("") == "unknown"
    assert organ_of("core\\memory\\engine.py") == "memory"


# ── snapshot capture ──────────────────────────────────────────────


def test_capture_snapshot_clean_repo(organ, body_repo):
    snap = organ.capture_snapshot()
    assert snap.commit_sha == _git(body_repo, "rev-parse", "HEAD")
    assert len(snap.commit_sha) == 40
    assert snap.dirty_count == 0
    assert snap.dirty_digest == ""
    assert snap.branch not in ("", "unknown")


def test_capture_snapshot_dirty_repo(organ, body_repo):
    _write(body_repo, "core/memory/engine.py", "x = 2\n")
    snap = organ.capture_snapshot()
    assert snap.dirty_count == 1
    assert snap.dirty_digest != ""
    assert "core/memory/engine.py" in snap.dirty_files


def test_capture_snapshot_non_repo(tmp_path):
    organ = SourceBodyAwareness(
        source_root=tmp_path / "not-a-repo",
        ledger_path=tmp_path / "ledger.jsonl",
        crash_evidence_dir=tmp_path / "crash",
    )
    (tmp_path / "not-a-repo").mkdir()
    snap = organ.capture_snapshot()
    assert snap.commit_sha == "unknown"
    assert snap.dirty_count == 0


def test_capture_snapshot_missing_git(organ, monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(organ._subprocess_gateway, "run", _boom)
    snap = organ.capture_snapshot()
    assert snap.commit_sha == "unknown"
    assert organ.get_status()["git_available"] is False


# ── ledger ────────────────────────────────────────────────────────


def test_ledger_round_trip(organ):
    asyncio.run(organ.awaken())
    reloaded = _fresh_organ(organ).load_last_snapshot()
    assert reloaded is not None
    assert reloaded.commit_sha == organ.capture_snapshot().commit_sha


def test_ledger_skips_corrupt_lines(organ):
    asyncio.run(organ.awaken())
    with open(organ.ledger_path, "a", encoding="utf-8") as fh:
        fh.write("{torn json\n")
        fh.write("[1, 2, 3]\n")
    assert _fresh_organ(organ).load_last_snapshot() is not None


def test_ledger_compaction_past_size_limit(organ):
    organ.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    filler = json.dumps({"schema": "aura.source_body.v1", "t": 1.0, "commit_sha": "unknown"})
    with open(organ.ledger_path, "w", encoding="utf-8") as fh:
        for _ in range(20000):
            fh.write(filler + "\n")
    assert organ.ledger_path.stat().st_size > 1_048_576
    asyncio.run(organ.awaken())
    assert organ.ledger_path.stat().st_size < 1_048_576
    lines = organ.ledger_path.read_text(encoding="utf-8").splitlines()
    assert 0 < len(lines) <= 201
    assert _fresh_organ(organ).load_last_snapshot() is not None


# ── deltas + narratives ───────────────────────────────────────────


def test_first_awakening_delta(organ):
    delta = asyncio.run(organ.awaken())
    assert delta.first_awakening is True
    assert delta.changed is False
    assert "First recorded awakening" in delta.narrative()
    assert delta.to_sha != "unknown"


def test_unchanged_body_between_boots(organ):
    asyncio.run(organ.awaken())
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.first_awakening is False
    assert delta.changed is False
    assert "unchanged" in delta.narrative()


def test_commit_between_boots_names_surgeon_and_organs(organ, body_repo):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 3\n")
    _write(body_repo, "core/capability_engine.py", "y = 3\n")
    _commit(body_repo, "Rework recall grounding", author="Zenflow")
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.changed is True
    assert len(delta.commits) == 1
    assert delta.commits[0].author == "Zenflow"
    assert delta.organs == {"memory": 1, "capability_engine": 1}
    narrative = delta.narrative()
    assert "Zenflow" in narrative
    assert "memory" in narrative
    assert "Rework recall grounding" in narrative


def test_multiple_commits_multiple_authors(organ, body_repo):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 4\n")
    _commit(body_repo, "memory pass", author="Zenflow")
    _write(body_repo, "tests/test_engine.py", "def test2(): pass\n")
    _commit(body_repo, "test pass", author="Claude")
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert len(delta.commits) == 2
    assert delta.files_changed == 2
    narrative = delta.narrative()
    assert "Claude" in narrative and "Zenflow" in narrative
    # Most recent commit subject leads the story.
    assert "test pass" in narrative


def test_reverted_body_is_reported_as_rewind(organ, body_repo):
    _write(body_repo, "core/memory/engine.py", "x = 5\n")
    _commit(body_repo, "will be rewound")
    asyncio.run(organ.awaken())
    _git(body_repo, "reset", "--hard", "HEAD~1")
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.reverted is True
    assert "rewound" in delta.narrative()


def test_unreadable_history_is_admitted_not_invented(organ, body_repo):
    current = organ.capture_snapshot()
    bogus_prev = SourceBodySnapshot(
        boot_id="x",
        t=time.time() - 3600,
        commit_sha="f" * 40,
        branch="main",
        dirty_digest="",
        dirty_count=0,
    )
    delta = organ.compute_delta(bogus_prev, current)
    assert delta.history_unreadable is True
    assert "unreadable" in delta.narrative()


def test_dirty_only_change_between_boots(organ, body_repo):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 6\n")
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.changed is True
    assert delta.commits == []
    assert delta.organs == {"memory": 1}
    assert "without committing" in delta.narrative()


def test_dirty_files_do_not_leak_into_clean_narrative(organ, body_repo):
    delta = asyncio.run(organ.awaken())
    assert "uncommitted" not in delta.narrative()


# ── crash correlation ─────────────────────────────────────────────


def test_crash_evidence_marks_abrupt_exit(organ, body_repo, tmp_path):
    asyncio.run(organ.awaken())
    crash_dir = tmp_path / "crash"
    crash_dir.mkdir(exist_ok=True)
    (crash_dir / "crash_dump_001.txt").write_text("faulthandler", encoding="utf-8")
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.abrupt_previous_exit is True
    assert "abruptly" in delta.narrative()


def test_no_crash_dir_means_clean_exit(organ):
    asyncio.run(organ.awaken())
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.abrupt_previous_exit is False


def test_stale_crash_evidence_is_ignored(organ, body_repo, tmp_path):
    crash_dir = tmp_path / "crash"
    crash_dir.mkdir(exist_ok=True)
    stale = crash_dir / "old_crash.txt"
    stale.write_text("old", encoding="utf-8")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    asyncio.run(organ.awaken())
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.abrupt_previous_exit is False


# ── awakening pass side effects ───────────────────────────────────


def test_awaken_publishes_change_event(organ, body_repo, fake_bus, fake_memory):
    asyncio.run(organ.awaken())
    assert fake_bus.topics() == []  # first awakening: baseline, no change event
    _write(body_repo, "core/memory/engine.py", "x = 7\n")
    _commit(body_repo, "surgery", author="Zenflow")
    asyncio.run(_fresh_organ(organ).awaken())
    assert "soma.source_body.changed" in fake_bus.topics()
    _, payload = fake_bus.events[-1]
    assert payload["changed"] is True
    assert "Zenflow" in payload["narrative"]


def test_awaken_writes_episodic_memory(organ, body_repo, fake_bus, fake_memory):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 8\n")
    _commit(body_repo, "surgery two", author="Zenflow")
    asyncio.run(_fresh_organ(organ).awaken())
    assert len(fake_memory.stored) == 1
    text, kwargs = fake_memory.stored[0]
    assert "Zenflow" in text
    assert "source_body" in kwargs.get("tags", [])


def test_awaken_survives_bus_failure(organ, body_repo, fake_memory, monkeypatch):
    import core.event_bus as event_bus_module

    def _broken_bus():
        raise RuntimeError("bus offline")

    _write(body_repo, "core/memory/engine.py", "x = 9\n")
    _commit(body_repo, "surgery three")
    # Broken for the awakenings alone. Left broken until the test's own
    # teardown, the fixtures that close the test down asked for the bus too,
    # and the test errored on every run.
    with monkeypatch.context() as broken:
        broken.setattr(event_bus_module, "get_event_bus", _broken_bus)
        asyncio.run(organ.awaken())
        delta = asyncio.run(_fresh_organ(organ).awaken())
    assert isinstance(delta, BodyDelta)  # no raise


def test_awaken_survives_absent_memory_manager(organ, body_repo, fake_bus):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 10\n")
    _commit(body_repo, "surgery four")
    delta = asyncio.run(_fresh_organ(organ).awaken())
    assert delta.changed is True  # memory_manager missing is non-fatal


def test_awaken_appends_one_ledger_line_per_boot(organ):
    asyncio.run(organ.awaken())
    asyncio.run(_fresh_organ(organ).awaken())
    lines = organ.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len([ln for ln in lines if ln.strip()]) == 2


# ── live pulse ────────────────────────────────────────────────────


def test_live_pulse_quiet_when_nothing_changes(organ, fake_bus):
    asyncio.run(organ.awaken())
    assert asyncio.run(organ.live_pulse()) is None
    assert fake_bus.topics() == []


def test_live_pulse_detects_in_flight_edit(organ, body_repo, fake_bus):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 11\n")
    detection = asyncio.run(organ.live_pulse())
    assert detection is not None
    assert detection.dirty_count == 1
    assert detection.organs == ("memory",)
    assert "core/memory/engine.py" in detection.new_files
    assert "soma.source_body.modification_detected" in fake_bus.topics()


def test_live_pulse_does_not_duplicate_unchanged_dirt(organ, body_repo, fake_bus):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 12\n")
    assert asyncio.run(organ.live_pulse()) is not None
    assert asyncio.run(organ.live_pulse()) is None
    assert fake_bus.topics().count("soma.source_body.modification_detected") == 1


def test_live_pulse_reports_settled_tree(organ, body_repo, fake_bus):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 13\n")
    asyncio.run(organ.live_pulse())
    _git(body_repo, "checkout", "--", "core/memory/engine.py")
    assert asyncio.run(organ.live_pulse()) is None
    assert "soma.source_body.modification_settled" in fake_bus.topics()


def test_live_pulse_never_raises_on_git_failure(organ, monkeypatch):
    asyncio.run(organ.awaken())

    def _boom(*args, **kwargs):
        raise OSError("disk detached")

    monkeypatch.setattr(organ, "_dirty_state", _boom)
    assert asyncio.run(organ.live_pulse()) is None


# ── prompt surfaces (cached only) ─────────────────────────────────


def test_somatic_lines_empty_before_awakening(organ):
    assert organ.somatic_change_lines() == []


def test_somatic_lines_after_body_change(organ, body_repo, fake_bus, fake_memory):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 14\n")
    _commit(body_repo, "prompt surgery", author="Zenflow")
    successor = _fresh_organ(organ)
    asyncio.run(successor.awaken())
    lines = successor.somatic_change_lines()
    assert any("Body change since last awakening" in ln for ln in lines)
    assert any("Zenflow" in ln for ln in lines)


def test_somatic_lines_after_live_detection(organ, body_repo, fake_bus):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 15\n")
    asyncio.run(organ.live_pulse())
    lines = organ.somatic_change_lines()
    assert any("being modified right now" in ln for ln in lines)
    assert any("restarted" in ln for ln in lines)


def test_somatic_lines_age_out(organ, body_repo, fake_bus, monkeypatch):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 16\n")
    asyncio.run(organ.live_pulse())
    real_time = time.time
    monkeypatch.setattr(
        "core.soma.source_body.time.time", lambda: real_time() + 48 * 3600
    )
    assert organ.somatic_change_lines() == []


def test_somatic_lines_never_touch_git(organ, body_repo, fake_bus, monkeypatch):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 17\n")
    asyncio.run(organ.live_pulse())

    def _forbidden(*args, **kwargs):
        raise AssertionError("prompt path must not shell out")

    monkeypatch.setattr(organ, "_git", _forbidden)
    assert organ.somatic_change_lines()  # served from cache


# ── history + health surfaces ─────────────────────────────────────


def test_describe_body_history_grounded_rows(organ, body_repo):
    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 18\n")
    _commit(body_repo, "history surgery")
    successor = _fresh_organ(organ)
    asyncio.run(successor.awaken())
    text = successor.describe_body_history()
    assert "awakening" in text
    assert text.count("- 2") == 2  # two ISO-dated ledger rows
    assert "Current state:" in text


def test_describe_body_history_empty_ledger(organ):
    assert "ledger is empty" in organ.describe_body_history()


def test_get_status_contract(organ):
    asyncio.run(organ.awaken())
    status = organ.get_status()
    assert status["alive"] is True
    assert organ.is_alive() is True
    for key in (
        "watching",
        "boot_id",
        "git_available",
        "source_root",
        "ledger_path",
        "pulse_count",
        "boot_delta",
    ):
        assert key in status
    assert status["boot_delta"]["narrative"]


# ── lifecycle ─────────────────────────────────────────────────────


def test_start_and_stop_watch_loop(organ, fake_bus, monkeypatch):
    monkeypatch.setenv("AURA_SOURCE_BODY_AWAKEN_DELAY_S", "0")
    monkeypatch.setenv("AURA_SOURCE_BODY_PULSE_S", "60")

    async def _scenario():
        await organ.start()
        assert organ.get_status()["watching"] is True
        await organ.start()  # idempotent
        for _ in range(100):
            if organ.get_status()["boot_delta"] is not None:
                break
            await asyncio.sleep(0.05)
        assert organ.get_status()["boot_delta"] is not None
        await organ.stop()
        assert organ.get_status()["watching"] is False

    asyncio.run(_scenario())


# ── governance ────────────────────────────────────────────────────


def test_ledger_write_is_governed_under_production_mode(organ, monkeypatch):
    """The awakening persists its snapshot inside a governed scope, so the
    gateway's fail-closed check passes even in production governance mode."""
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    delta = asyncio.run(organ.awaken())
    assert delta.to_sha != "unknown"
    assert organ.ledger_path.exists()  # write went through, governed


def test_ungoverned_gateway_write_fails_closed_under_production_mode(tmp_path, monkeypatch):
    """Sanity: the same gateway write without a scope is refused — proving
    the organ's scope is doing real work, not ceremony."""
    from core.governance_context import GovernanceViolationError
    from core.runtime.file_write_gateway import get_file_write_gateway

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")

    async def _ungoverned():
        await get_file_write_gateway().append_text_async(
            tmp_path / "refused.jsonl", "x\n", source="test.ungoverned"
        )

    with pytest.raises(GovernanceViolationError):
        asyncio.run(_ungoverned())


# ── singleton + spine ─────────────────────────────────────────────


def test_singleton_identity_and_reset():
    reset_source_body_for_test()
    try:
        first = get_source_body()
        assert get_source_body() is first
        reset_source_body_for_test()
        assert get_source_body() is not first
    finally:
        reset_source_body_for_test()


def test_service_spine_has_source_body():
    from core.service_names import ServiceNames

    assert ServiceNames.SOURCE_BODY == "source_body"


# ── context assembler integration ─────────────────────────────────


def test_build_somatic_context_includes_source_body_lines(organ, body_repo, fake_bus, fake_memory):
    from core.brain.llm.context_assembler import ContextAssembler
    from core.container import ServiceContainer
    from core.state.aura_state import AuraState

    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 19\n")
    _commit(body_repo, "assembler surgery", author="Zenflow")
    successor = _fresh_organ(organ)
    asyncio.run(successor.awaken())

    ServiceContainer.register_instance("source_body", successor, required=False)
    try:
        block = ContextAssembler.build_somatic_context(AuraState())
        assert "BODY AWARENESS" in block
        assert "Zenflow" in block
    finally:
        ServiceContainer.register_instance("source_body", None, required=False)


def test_build_somatic_context_unaffected_without_service():
    from core.brain.llm.context_assembler import ContextAssembler
    from core.container import ServiceContainer
    from core.state.aura_state import AuraState

    ServiceContainer.register_instance("source_body", None, required=False)
    block = ContextAssembler.build_somatic_context(AuraState())
    assert "being modified" not in block


# ── system_proprioception skill integration ──────────────────────


def test_skill_source_body_report(organ, body_repo, fake_bus, fake_memory):
    from core.container import ServiceContainer
    from core.skills.system_proprioception import SystemProprioceptionSkill

    asyncio.run(organ.awaken())
    _write(body_repo, "core/memory/engine.py", "x = 20\n")
    _commit(body_repo, "skill surgery", author="Zenflow")
    successor = _fresh_organ(organ)
    asyncio.run(successor.awaken())

    ServiceContainer.register_instance("source_body", successor, required=False)
    try:
        report = SystemProprioceptionSkill()._source_body_report()
        assert report is not None
        assert "Zenflow" in report["current_narrative"]
        assert "awakening" in report["history"]
        assert report["status"]["alive"] is True
    finally:
        ServiceContainer.register_instance("source_body", None, required=False)


def test_skill_source_body_report_absent_service():
    from core.container import ServiceContainer
    from core.skills.system_proprioception import SystemProprioceptionSkill

    ServiceContainer.register_instance("source_body", None, required=False)
    assert SystemProprioceptionSkill()._source_body_report() is None


def test_skill_input_defaults_include_source_body():
    from core.skills.system_proprioception import ProprioceptionInput

    assert ProprioceptionInput().include_source_body is True
