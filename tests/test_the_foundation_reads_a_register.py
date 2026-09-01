"""Subsystems publish health fragments; the foundation never reaches down.

`core/runtime` is the foundation and its DEPS file bans imports of cognition,
agency and learning, for a stated reason: an import from an upper module into
the foundation makes the foundation un-bootable without the thing it is meant
to be able to run WITHOUT — which is how a health surface ends up unable to
report on a mind that failed to start.

Wiring a memory register and a sealed-artifact probe into `health_contract`
broke that ban on the first attempt and the layering gate caught it. The fix is
not a narrower import; it is the other direction. A subsystem registers a
callable returning its own fragment, and the surface asks the register. Doing
that also moved `_external_reach_snapshot` out of the foundation and into the
module that owns the MCP transport, where it always belonged.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.runtime.health_fragments import (
    EXPECTED_FRAGMENTS,
    collect_health_fragments,
    register_health_fragment,
    health_fragments_reset,
)

ROOT = Path(__file__).resolve().parents[1]
HEALTH_CONTRACT = ROOT / "core" / "runtime" / "health_contract.py"


@pytest.fixture(autouse=True)
def _clean_register():
    """An empty register for these tests, and everything back afterwards.

    Clearing without restoring was permanent: registration happens at module
    import, once per process, so every later test file found an empty register
    and every expected fragment reported absent.
    """
    with health_fragments_reset():
        yield


def test_an_unpublished_fragment_is_reported_absent_not_omitted():
    """A channel with no writer is the defect this repository keeps
    rediscovering. An omitted fragment is exactly that shape."""
    fragments = collect_health_fragments()

    for name in EXPECTED_FRAGMENTS:
        assert name in fragments, name
        assert fragments[name]["registered"] is False


def test_a_published_fragment_is_returned():
    register_health_fragment("memory_inventory", lambda: {"declared": 9})

    fragment = collect_health_fragments()["memory_inventory"]

    assert fragment["registered"] is True
    assert fragment["declared"] == 9


def test_a_provider_that_raises_does_not_take_the_surface_down():
    """The surface exists to describe failures, so it must survive them."""

    def _boom():
        raise RuntimeError("subsystem is on fire")

    register_health_fragment("sealed_artifacts", _boom)

    fragment = collect_health_fragments()["sealed_artifacts"]

    assert fragment["registered"] is True
    assert fragment["available"] is False
    assert "RuntimeError" in fragment["reason"]


def test_a_provider_returning_a_non_mapping_is_refused():
    register_health_fragment("memory_inventory", lambda: ["not", "a", "mapping"])

    fragment = collect_health_fragments()["memory_inventory"]

    assert fragment["available"] is False


def test_registration_is_idempotent():
    register_health_fragment("external_reach", lambda: {"v": 1})
    register_health_fragment("external_reach", lambda: {"v": 2})

    assert collect_health_fragments()["external_reach"]["v"] == 2


def test_an_unexpected_fragment_is_still_published():
    """The declared list is a floor, not a filter."""
    register_health_fragment("something_new", lambda: {"ok": True})

    assert collect_health_fragments()["something_new"]["ok"] is True


def test_the_foundation_imports_no_upper_layer_for_health():
    """The specific violation: health_contract importing core.learning."""
    tree = ast.parse(HEALTH_CONTRACT.read_text("utf-8"))
    banned = ("core.learning", "core.memory", "core.brain", "core.agency", "core.capabilities")

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(node.module.startswith(prefix) for prefix in banned):
                offenders.append(f"{node.module}:{node.lineno}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(prefix) for prefix in banned):
                    offenders.append(f"{alias.name}:{node.lineno}")

    assert offenders == [], offenders


def test_external_reach_moved_out_of_the_foundation():
    source = HEALTH_CONTRACT.read_text("utf-8")

    assert "_external_reach_snapshot" not in source
    assert "mcp_connectors" not in source


def test_every_real_publisher_can_re_register_itself():
    """Import is not a re-runnable registration, and the modules know it.

    A module is imported once per process, so after any reset an import-time
    registration cannot be re-established - which is why memory_inventory
    exposes a public register and MemoryFacade calls it on every refresh. This
    checked the import instead, and passed only while it happened to run before
    anything else had imported those modules.

    What matters in production is that each publisher's entry point puts the
    fragment back. That is what this runs.
    """
    from core.capabilities.mcp_connectors import _register_fragment as register_reach
    from core.learning.sealed_artifact_admission import (
        _register_fragment as register_sealed,
    )
    from core.memory.memory_inventory import register_memory_health_fragment

    for register in (register_reach, register_sealed, register_memory_health_fragment):
        register()

    fragments = collect_health_fragments()
    for name in ("external_reach", "memory_inventory", "sealed_artifacts"):
        assert fragments[name]["registered"] is True, name


def test_the_cognitive_publishers_can_re_register_too():
    from core.cognition.contract_health import install

    assert install() == {"contracts": True, "growth": True}
    fragments = collect_health_fragments()
    for name in ("cognitive_contracts", "cognitive_growth"):
        assert fragments[name]["registered"] is True, name


def test_a_reset_is_reversible_so_it_does_not_outlive_the_test():
    from core.memory.memory_inventory import register_memory_health_fragment

    register_memory_health_fragment()
    assert collect_health_fragments()["memory_inventory"]["registered"] is True
    with health_fragments_reset():
        assert collect_health_fragments()["memory_inventory"]["registered"] is False
    assert collect_health_fragments()["memory_inventory"]["registered"] is True


def test_the_facade_publishes_the_memory_fragment():
    """Registration must happen on a path that actually runs, or the fragment
    is a module nobody imports."""
    from core.memory.memory_facade import MemoryFacade

    facade = MemoryFacade()
    facade.setup()

    assert collect_health_fragments()["memory_inventory"]["registered"] is True


def test_the_health_report_carries_every_fragment():
    import core.capabilities.mcp_connectors  # noqa: F401
    import core.learning.sealed_artifact_admission  # noqa: F401
    import core.memory.memory_inventory  # noqa: F401

    from core.runtime.health_contract import runtime_health_report

    report = runtime_health_report()

    for name in EXPECTED_FRAGMENTS:
        assert name in report, name
