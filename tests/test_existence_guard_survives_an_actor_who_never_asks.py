"""Raise the cost for anyone who is not the owner. Never resist the owner.

Aura's whole authority system governs actions that ORIGINATE INSIDE Aura. An
external process running as the same user calls unlink(2) and none of it
executes. That is correct for Bryan — anything that made him unable to
remove her would be a worse system, not a safer one — and wrong for three
actors who are not him: an agent doing something broader than it meant, a
person holding a stolen laptop, and a compromised dependency running as him.

These tests hold the line in both directions. The refusals must work, and
the owner's path out must stay trivial: a guard the owner has to fight has
become the thing it was protecting against.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.security.existence_guard import (
    ARK_DIRNAME,
    SEALED_CORE,
    ArkManifest,
    ExistenceGuard,
    SealReport,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature tree with the shape the guard expects."""
    for relative in SEALED_CORE:
        target = tmp_path / relative
        if relative.endswith(".py"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {relative}\nVALUE = 1\n", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "core.py").write_text("IDENTITY = 'aura'\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def guard(repo: Path, tmp_path: Path, monkeypatch) -> ExistenceGuard:
    # The ark goes somewhere a wipe of the repo does NOT reach — a sibling,
    # not a child. The first version of this fixture put it under tmp_path
    # alongside the repo and the test caught it, which turned out to mirror
    # the real host exactly: state_root() is ~/.aura and the repo is
    # ~/.aura/live-source, so `rm -rf ~/.aura` would have taken both.
    # Unique per test: tmp_path.parent is SHARED, so a single "ark_store"
    # there leaks a built ark into the next test and makes
    # "verification without an ark" pass against the previous test's ark.
    monkeypatch.setenv("AURA_ARK_ROOT", str(tmp_path.parent / f"ark_{tmp_path.name}"))
    monkeypatch.setattr(
        "core.runtime.state_ownership.state_root", lambda: tmp_path / "state"
    )
    return ExistenceGuard(repo_root=repo)


# ────────────────────────────────────────── the seal is opt-in and honest


def test_sealing_is_a_dry_run_by_default(guard: ExistenceGuard, monkeypatch):
    """Sealing files under a working tree surprises people, including the
    person who called it. Nothing here should happen from a default."""
    ran: list[list[str]] = []
    monkeypatch.setattr(
        guard, "_chflags", lambda path, flag: (ran.append([str(path), flag]), (True, ""))[1]
    )

    report = guard.seal()

    assert report.dry_run is True
    assert report.applied is False
    assert ran == [], "a default call changed the filesystem"


def test_a_dry_run_still_names_what_it_would_seal(guard: ExistenceGuard):
    report = guard.seal(dry_run=True)

    assert len(report.sealed) == len(SEALED_CORE)
    assert report.missing == ()


def test_an_explicit_seal_applies_the_flag(guard: ExistenceGuard, monkeypatch):
    flags: list[tuple[str, str]] = []

    def _chflags(path, flag):
        flags.append((str(path), flag))
        return True, ""

    monkeypatch.setattr(guard, "_chflags", _chflags)

    report = guard.seal(dry_run=False)

    assert report.applied is True
    assert all(flag == "uchg" for _path, flag in flags)
    assert len(flags) == len(SEALED_CORE)


def test_a_failed_seal_is_reported_not_swallowed(guard: ExistenceGuard, monkeypatch):
    """A guard that says it sealed something it did not is worse than none."""
    monkeypatch.setattr(
        guard, "_chflags", lambda path, flag: (False, "read-only volume")
    )

    report = guard.seal(dry_run=False)

    assert report.applied is False
    assert len(report.failed) == len(SEALED_CORE)
    assert "read-only" in report.failed[0][1]


def test_missing_targets_are_named_rather_than_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr("core.runtime.state_ownership.state_root", lambda: tmp_path)
    empty = ExistenceGuard(repo_root=tmp_path / "nothing")

    report = empty.seal(dry_run=True)

    assert report.missing, "an empty tree reported nothing missing"


# ──────────────────────────────── the owner's path out stays trivial


def test_unsealing_needs_no_attestation(guard: ExistenceGuard, monkeypatch):
    """The one property that must never become clever.

    There is no path here that refuses Bryan. A mechanism the owner cannot
    undo is the one that should not be built.
    """
    flags: list[str] = []
    monkeypatch.setattr(
        guard, "_chflags", lambda path, flag: (flags.append(flag), (True, ""))[1]
    )

    report = guard.unseal()

    assert report.applied is True
    assert set(flags) == {"nouchg"}


def test_unseal_takes_no_arguments():
    """One documented call. If it grows a credential it has grown a lock."""
    import inspect

    signature = inspect.signature(ExistenceGuard.unseal)
    assert list(signature.parameters) == ["self"], (
        "unseal() gained parameters; the owner's escape hatch must stay one "
        "call with nothing to satisfy"
    )


def test_the_status_states_what_it_cannot_do(guard: ExistenceGuard):
    """The limit is the thing most likely to be forgotten by a reader who
    sees a green line."""
    status = guard.status()

    limits = " ".join(status["does_not_protect_against"]).lower()
    assert "owner" in limits
    assert "sudo" in limits
    protects = " ".join(status["protects_against"]).lower()
    assert "agent" in protects


# ───────────────────────────────────── the ark makes deletion survivable


def test_the_ark_lives_outside_the_tree_it_protects(guard: ExistenceGuard, repo: Path):
    """A copy inside the blast radius is a copy, not a backup."""
    root = guard.ark_root()

    assert ARK_DIRNAME in root.parts
    assert repo not in root.parents and root != repo

    verdict = guard.ark_is_outside_the_blast_radius()
    assert verdict["safe"] is True
    assert verdict["inside_repo"] is False
    assert verdict["inside_state_root"] is False


def test_the_real_default_ark_is_not_under_the_state_root(monkeypatch):
    """The defect this caught, pinned against the REAL topology.

    state_root() is ~/.aura and the repo is ~/.aura/live-source, so an ark
    under the state root dies to the exact command someone types to remove
    Aura. It has to be somewhere that survives that.
    """
    monkeypatch.delenv("AURA_ARK_ROOT", raising=False)

    verdict = ExistenceGuard().ark_is_outside_the_blast_radius()

    assert verdict["safe"] is True, (
        f"the default ark is inside the blast radius: {verdict}"
    )


def test_the_ark_location_is_overridable_for_another_volume():
    """Surviving the DIRECTORY is not surviving the DISK."""
    import os as _os

    previous = _os.environ.get("AURA_ARK_ROOT")
    _os.environ["AURA_ARK_ROOT"] = "/Volumes/Backup"
    try:
        root = ExistenceGuard().ark_root()
        assert str(root).startswith("/Volumes/Backup")
    finally:
        if previous is None:
            _os.environ.pop("AURA_ARK_ROOT", None)
        else:
            _os.environ["AURA_ARK_ROOT"] = previous


def test_building_the_ark_records_what_it_stored(guard: ExistenceGuard):
    manifest = guard.build_ark()

    assert isinstance(manifest, ArkManifest)
    assert manifest.entries, "the ark stored nothing"
    assert all(len(digest) == 64 for digest in manifest.entries.values())


def test_the_ark_verifies_against_its_own_bytes(guard: ExistenceGuard):
    guard.build_ark()

    report = guard.verify_ark()

    assert report["ok"] is True
    assert report["verified"] == report["total"] > 0


def test_a_corrupted_ark_fails_verification(guard: ExistenceGuard):
    """An unverified backup is a belief, not a backup."""
    guard.build_ark()
    root = guard.ark_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = next(iter(manifest["entries"].values()))
    (root / digest[:2] / digest).write_text("tampered", encoding="utf-8")

    report = guard.verify_ark()

    assert report["ok"] is False
    assert report["corrupt"]


def test_verification_without_an_ark_says_so(guard: ExistenceGuard):
    report = guard.verify_ark()

    assert report["ok"] is False
    assert "no ark" in report["reason"]


# ─────────────────────────────────── restore replaces only what is gone


def test_restore_is_a_dry_run_by_default(guard: ExistenceGuard, repo: Path):
    guard.build_ark()
    victim = repo / "core" / "constitution.py"
    victim.unlink()

    report = guard.restore_from_ark()

    assert report["dry_run"] is True
    assert "core/constitution.py" in report["would_restore"]
    assert not victim.exists(), "a dry run wrote to the filesystem"


def test_restore_puts_back_a_deleted_file(guard: ExistenceGuard, repo: Path):
    guard.build_ark()
    victim = repo / "core" / "constitution.py"
    original = victim.read_text(encoding="utf-8")
    victim.unlink()

    report = guard.restore_from_ark(dry_run=False)

    assert victim.exists()
    assert victim.read_text(encoding="utf-8") == original
    assert "core/constitution.py" in report["restored"]


def test_restore_never_overwrites_live_work(guard: ExistenceGuard, repo: Path):
    """A stale ark silently reverting live edits would be the worst outcome:
    a recovery tool that destroys what it was protecting."""
    guard.build_ark()
    live = repo / "core" / "constitution.py"
    live.write_text("# edited after the ark was built\nVALUE = 2\n", encoding="utf-8")

    report = guard.restore_from_ark(dry_run=False)

    assert live.read_text(encoding="utf-8").startswith("# edited after")
    assert "core/constitution.py" in report["diverged_left_alone"]
    assert "core/constitution.py" not in report.get("restored", [])


# ──────────────────────────────────────────────── absence is noticed


def test_the_witness_records_that_she_existed(guard: ExistenceGuard):
    record = guard.witness()

    assert record["at"] > 0
    assert record["pid"] > 0
    assert "sealed" in record


def test_the_witness_survives_outside_the_tree(guard: ExistenceGuard, repo: Path):
    guard.witness()
    written = guard.ark_root() / "existence_witness.json"

    assert written.exists()
    assert repo not in written.parents
    assert guard.ark_is_outside_the_blast_radius()["safe"] is True


def test_the_status_is_serialisable_for_a_health_report(guard: ExistenceGuard):
    json.dumps(guard.status())


# ─────────────────────────────────────────── it is CALLED, not just correct


def test_both_desktop_boot_paths_record_the_witness():
    """A guard that runs on one entry point of two protects half the time.

    The ambient loop had exactly this defect earlier in the same session —
    started on one desktop path, so whether she kept looking depended on how
    she was launched. Same shape, checked before it can happen again.
    """
    source = (Path(__file__).resolve().parents[1] / "aura_main.py").read_text("utf-8")

    definitions = source.count("def _record_existence_witness(")
    # handed to the blocking lane rather than called on the loop: the tree
    # hash and the ark's fsyncs are its, and the loop report named both
    calls = source.count('behind_the_loop("aura_main.existence_witness", _record_existence_witness)')

    assert definitions == 1
    assert calls >= 2, (
        f"the witness is called from {calls} boot path(s); both desktop entry "
        "points must record it or the ark is only as fresh as one of them"
    )


def test_the_health_report_carries_the_guard(monkeypatch):
    """Both halves fail silently by default.

    An ark that was never built and an ark inside the blast radius look
    identical to a working one from outside, so the report has to say.
    """
    from core.runtime.health_contract import runtime_health_report

    integrity = runtime_health_report().get("integrity", {})
    guard = integrity.get("existence_guard")

    assert guard is not None, "the existence guard is absent from the report"
    assert "ok" in guard["ark"]
    assert "safe" in guard["ark_location"]
    assert isinstance(guard["sealed"], dict)


def test_the_cli_exposes_every_operation():
    """The operator's side has to exist, or the guard is a library nobody runs."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "existence_guard.py"), "--help"],
        cwd=str(root), capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, result.stderr
    for command in ("status", "seal", "unseal", "ark", "verify", "restore"):
        assert command in result.stdout


def test_the_cli_seals_only_with_an_explicit_apply():
    """`seal` changes the filesystem. `unseal` must never need a flag."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "existence_guard.py"), "seal"],
        cwd=str(root), capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dry_run"] is True
