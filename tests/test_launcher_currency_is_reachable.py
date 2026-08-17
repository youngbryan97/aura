"""The repair existed, was tested, and could not fire for the user.

LIVE DEFECT, 2026-08-10. Bryan reported that companion mode did not work —
he closed the window and no bubble ever appeared. Every Python organ behind it
was correct. The installed ``/Applications/Aura.app`` launcher binary was
built 2026-08-03; ``scripts/AuraLauncher.swift`` gained the entire companion
surface on 2026-08-09. ``strings`` on the resident binary found zero
occurrences of ``/api/ambient/visibility``. The feature was not broken, it was
absent from the executable he was running.

``core.runtime.app_bundle_sync`` detects and repairs precisely this, and had
exactly one caller: ``launch_aura.sh``. The ordinary launch does not run that
script — ``AuraLauncher.swift``'s ``spawnAuraProcess`` execs ``aura_main.py``
directly and reaches the shell only via ``requiresProtectedFolderFallback()``.
So the detector was unreachable on the path every user takes, and the
condition it detects is invisible by nature: a stale launcher does not error,
it silently lacks features.

Two properties are pinned here, and neither is "the sync works" — that was
already true and already tested while the bug shipped:

  * the repair is reachable from the RUNTIME, which every launch path ends in,
    not only from a shell script that the common path skips;
  * the condition is REPORTED, so a stale launcher is something a person can
    see rather than something they have to deduce from a missing feature.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from core.runtime.app_bundle_sync import keep_launcher_current, launcher_currency

_AURA_MAIN = Path("aura_main.py")
_LAUNCH_SCRIPT = Path("launch_aura.sh")


def test_the_runtime_itself_keeps_the_launcher_current():
    """The bypass-proof half: aura_main.py runs on every launch path."""
    source = _AURA_MAIN.read_text(encoding="utf-8")

    assert "keep_launcher_current" in source, (
        "aura_main.py must check the launcher itself. Wiring this only into "
        "launch_aura.sh is what let a six-day-stale binary ship: the normal "
        "Aura.app launch execs aura_main.py directly and never sources it."
    )


def test_aura_main_does_not_block_boot_on_a_compile():
    """A rebuild may take minutes; a boot may not wait for one."""
    tree = ast.parse(_AURA_MAIN.read_text(encoding="utf-8"))
    awaited_directly = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "keep_launcher_current"
    ]

    assert not awaited_directly, (
        "keep_launcher_current must be scheduled, not awaited on the boot path"
    )


def test_launch_script_still_syncs_for_the_paths_that_use_it():
    """Adding the runtime hook must not remove the shell one."""
    assert "app_bundle_sync" in _LAUNCH_SCRIPT.read_text(encoding="utf-8")


def test_staleness_is_reported_not_merely_detectable():
    """A condition nothing surfaces is one nobody can act on."""
    report = launcher_currency()

    assert report["schema"] == "aura.launcher_currency.v1"
    assert "stale" in report


def test_unknown_never_reads_as_current(monkeypatch):
    """"Cannot tell" must not be reported as a clean bill of health."""
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no such bundle")),
    )

    report = launcher_currency()

    assert report["stale"] is None
    assert report["error"]


def test_a_stale_launcher_states_its_consequence(monkeypatch):
    """"stale: true" does not tell a person what they will experience."""
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: {
            "bundle": "/Applications/Aura.app",
            "bundle_present": True,
            "launcher_binary_present": True,
            "launcher_source_sha256": "a" * 64,
            "built_from_sha256": "b" * 64,
            "stale": True,
        },
    )

    report = launcher_currency()

    assert report["stale"] is True
    assert report["consequence"]


def test_a_current_launcher_claims_no_consequence(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: {
            "bundle": "/Applications/Aura.app",
            "bundle_present": True,
            "launcher_binary_present": True,
            "launcher_source_sha256": "a" * 64,
            "built_from_sha256": "a" * 64,
            "stale": False,
        },
    )

    assert launcher_currency()["consequence"] == ""


def test_the_task_never_compiles_under_test(monkeypatch):
    """A test run must not build or replace anything on the host."""
    monkeypatch.setenv("AURA_TESTING", "1")
    called: list[str] = []
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.sync_app_bundle",
        lambda *a, **k: called.append("sync"),
    )

    receipt = asyncio.run(keep_launcher_current())

    assert receipt["action"] == "skipped"
    assert not called


def test_a_missing_bundle_is_skipped_not_failed(monkeypatch):
    """Running from a checkout with no installed app is normal."""
    monkeypatch.delenv("AURA_TESTING", raising=False)
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: {"bundle_present": False, "stale": False},
    )

    receipt = asyncio.run(keep_launcher_current())

    assert receipt["action"] == "skipped"


def test_the_health_surface_carries_launcher_currency():
    """The block a person or an agent actually reads.

    Asserts the VALUE, not the source text. Grepping health_contract.py for
    the string would pass while the reading never reached the block — which is
    the exact shape of every defect this file is about, and it would have
    passed for a call that raised on every invocation.
    """
    from core.runtime.health_contract import _runtime_integrity_block

    block = _runtime_integrity_block()

    assert "launcher_currency" in block, block.get("launcher_currency_error")
    reading = block["launcher_currency"]
    assert reading["schema"] == "aura.launcher_currency.v1"
    # Present AND decided: None means "could not tell", which is a legitimate
    # answer, but the key has to exist either way.
    assert "stale" in reading


@pytest.mark.parametrize("flag", ["/api/ambient/visibility", "/api/ambient/state"])
def test_the_companion_surface_is_still_in_the_launcher_source(flag: str):
    """Guards the thing that was missing from the binary.

    Not a substitute for the currency check — this asserts the source has the
    feature, while `launcher_currency` asserts the binary was built from it.
    Both were needed: the source was correct the whole time.
    """
    assert flag in Path("scripts/AuraLauncher.swift").read_text(encoding="utf-8")


def test_a_staged_build_that_cannot_install_itself_says_so(monkeypatch, tmp_path):
    """Staging without installing is the same silence one step later.

    install_staged_bundle refuses while the resident app runs — correctly,
    because replacing a bundle underneath the process executing from it is how
    a signed app loses its TCC identity. But the runtime only exists WHILE it
    runs, so on the double-click path the staged build waits for a
    launch_aura.sh invocation that never comes.
    """
    live = "a" * 64
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: {
            "bundle": "/Applications/Aura.app",
            "bundle_present": True,
            "launcher_binary_present": True,
            "launcher_source_sha256": live,
            "built_from_sha256": "b" * 64,
            "stale": True,
        },
    )
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync._read_manifest",
        lambda bundle: {"launcher_source_sha256": live},
    )

    report = launcher_currency()

    assert report["staged_install_pending"] is True
    assert report["clears_by"]


def test_the_remedy_is_not_a_bare_relaunch(monkeypatch):
    """"Relaunch" would be false: double-click starts the OLD launcher.

    A remedy that does not work is worse than none — it converts a visible
    problem into a person believing they already fixed it.
    """
    live = "a" * 64
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: {
            "bundle": "/Applications/Aura.app",
            "bundle_present": True,
            "launcher_binary_present": True,
            "launcher_source_sha256": live,
            "built_from_sha256": "b" * 64,
            "stale": True,
        },
    )
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync._read_manifest",
        lambda bundle: {"launcher_source_sha256": live},
    )

    remedy = launcher_currency()["clears_by"]

    assert "--install-staged" in remedy


def test_no_staged_build_means_no_pending_claim(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync.launcher_drift",
        lambda *a, **k: {
            "bundle": "/Applications/Aura.app",
            "bundle_present": True,
            "launcher_binary_present": True,
            "launcher_source_sha256": "a" * 64,
            "built_from_sha256": "b" * 64,
            "stale": True,
        },
    )
    monkeypatch.setattr(
        "core.runtime.app_bundle_sync._read_manifest", lambda bundle: {}
    )

    report = launcher_currency()

    assert report["staged_install_pending"] is False
    assert report["clears_by"] == ""
