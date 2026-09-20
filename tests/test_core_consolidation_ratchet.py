"""Ratchet: core/'s top level is a spine, not a junk drawer.

July critique: 'too many files still live under core/.' The consolidation
pass (July 8) moved 129 loose modules into their subsystem packages,
bringing the top level from 176 files to the set below: the genuine spine
(container, config, will, event_bus, runtime, governance_context, ...) plus
heavily-integrated organs and a handful of legacy shims queued for the
kill-shim pass. This allowlist ONLY shrinks — a new top-level core module is
an architectural decision made here, not a default.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

# The sanctioned top level as of the July 8 consolidation. Entries leave this
# set when they are moved/killed; nothing enters without deliberate review.
ALLOWED_TOP_LEVEL = frozenset({
    "__init__",
    # contract-pinned package shims (tests/test_forensic_audit_regressions.py
    # requires these exact top-level re-export files; each is <25 lines and
    # only forwards to its package — they are doors, not residents)
    # orchestrator_boot removed: the shim is GONE, and
    # test_forensic_audit_regressions now asserts it must not come back.
    # Leaving it here would silently re-permit its return.
    "goals",
    # spine
    "container", "config", "service_names", "exceptions", "runtime",
    "event_bus", "will", "governance_context", "constitution",
    "service_registration", "schemas",
    # heavily-integrated organs (>15 importers or dynamic references)
    "orchestrator", "orchestrator_types", "capability_engine", "agency_core",
    "world_state", "thought_stream", "mind_tick", "scheduler", "mycelium",
    "identity", "self_model", "soul", "volition", "continuity", "synthesis",
    "curiosity_engine", "drive_engine", "reaper", "terminal_monitor",
    "initiative_synthesis", "fictional_ai_synthesis", "final_engines",
    "continuous_cognition", "conversation_reflection", "reliability_engine",
    # self_modification_engine removed: it no longer exists at core/ top level,
    # and the ratchet only tightens when a departed module leaves the allowlist
    # with it. Left in place, it would silently re-permit the module's return.
    "local_chat_brain", "global_workspace",
    # the one seam that has to reach every package. The fourteen dispositions
    # are homed next to what each is a kind of, in nine packages, and nine
    # packages cannot import each other. Registration is by import, so one
    # module has to do the importing, and a module that imports nine packages
    # cannot live inside any of them. It holds no state.
    "phenomena_wiring",
    # legacy shims shadowed by or redirecting to packaged implementations —
    # queued for the kill-shim pass; do NOT add new shims
    # drives removed: the shim no longer exists at core/ top level.
    "agency_bus", "circuit_breaker", "dual_memory",
    "long_term_memory_engine", "memory_compaction_patch", "memory_synthesizer",
})


def _lifted_out_of_an_allowed_module(stem: str) -> bool:
    """A sibling the size ratchet lifted out of a sanctioned module.

    ``container_seal`` beside ``container``, ``mycelium_vault`` beside
    ``mycelium``, ``mind_tick_loop_steps`` beside ``mind_tick``: the parent
    imports each straight back, so it is a piece of a module already on the
    list, and one that cannot move into a package without importing its
    parent across the package boundary. The file says so in its first
    lines; a new module that merely shares a prefix does not.
    """

    parent = next(
        (name for name in ALLOWED_TOP_LEVEL if stem.startswith(f"{name}_")), None
    )
    if parent is None:
        return False
    head = (REPO_ROOT / "core" / f"{stem}.py").read_text(encoding="utf-8")[:600]
    return f"out of `{parent}`" in head


def test_core_top_level_is_allowlisted():
    actual = {p.stem for p in (REPO_ROOT / "core").glob("*.py")}
    strays = {s for s in actual - ALLOWED_TOP_LEVEL if not _lifted_out_of_an_allowed_module(s)}
    assert not strays, (
        f"new loose module(s) under core/: {sorted(strays)} — put new modules "
        "in their subsystem package; the top level is reserved for the spine"
    )


def test_ratchet_only_shrinks():
    """When a module is moved or killed, remove it here — the count is the score."""
    actual = {p.stem for p in (REPO_ROOT / "core").glob("*.py")}
    vanished = ALLOWED_TOP_LEVEL - actual - {"__init__"}
    assert not vanished, (
        f"{sorted(vanished)} no longer exist at core/ top level — congratulations, "
        "now remove them from ALLOWED_TOP_LEVEL so the ratchet tightens"
    )
