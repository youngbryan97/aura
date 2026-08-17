"""A turn on a loaded model must not be gated as if it were loading one.

LIVE 2026-08-17, 64GB host with the 32B cortex resident (~20GB wired) beside
Chrome and two Electron apps:

    Foreground admission tightening for primary
    (pressure=78.4% available=13.8GB process=25.7GB/40.0GB
     reason=memory_pressure:78.4%/13.8GB (need <76.0% and >=18.0GB))

The gate wanted 18GB free on a machine whose resident model is intentionally
holding 20GB of the 64. That condition is unsatisfiable in Aura's own normal
operating state, and the double-count is the reason: the weights are already
counted as used, then the gate demands another model's worth on top. The turn
died and the person got "I couldn't get to an answer I'd stand behind" — from a
worker that had just logged a completed generation.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate


def test_the_probe_is_false_without_a_runtime() -> None:
    """Fail-safe: unknown residency keeps the stricter load-sized floor."""
    assert InferenceGate._cortex_already_resident() is False


def test_a_lane_that_only_intends_to_load_is_not_resident(monkeypatch) -> None:
    """Ready-but-never-generated is an intention, not weights in memory."""

    class _Gate:
        @staticmethod
        def get_conversation_status() -> dict[str, object]:
            return {"conversation_ready": True, "has_generated_successfully": False}

    monkeypatch.setattr(
        "core.container.ServiceContainer.peek",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )

    assert InferenceGate._cortex_already_resident() is False


def test_a_served_lane_counts_as_resident(monkeypatch) -> None:
    class _Gate:
        @staticmethod
        def get_conversation_status() -> dict[str, object]:
            return {"conversation_ready": True, "has_generated_successfully": True}

    monkeypatch.setattr(
        "core.container.ServiceContainer.peek",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )

    assert InferenceGate._cortex_already_resident() is True


def test_a_raising_probe_does_not_break_admission(monkeypatch) -> None:
    def _boom(name, default=None):
        raise RuntimeError("container down")

    monkeypatch.setattr(
        "core.container.ServiceContainer.peek", staticmethod(_boom)
    )

    assert InferenceGate._cortex_already_resident() is False
    # And the snapshot still computes rather than propagating the failure.
    assert "can_admit" in InferenceGate._headroom_snapshot("primary")


def test_the_resident_floor_admits_the_live_failure_conditions(monkeypatch) -> None:
    """The exact numbers from the live log must admit once weights are up."""

    class _Gate:
        @staticmethod
        def get_conversation_status() -> dict[str, object]:
            return {"conversation_ready": True, "has_generated_successfully": True}

    monkeypatch.setattr(
        "core.container.ServiceContainer.peek",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )

    class _Snapshot:
        pressure_pct = 78.4
        available_gb = 13.8
        total_gb = 64.0
        process_rss_gb = 25.7
        process_rss_limit_gb = 40.0
        refuse_heavy_local_generation = False

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        staticmethod(lambda: _Snapshot()),
    )

    snapshot = InferenceGate._headroom_snapshot("primary")

    assert snapshot["can_admit"] is True, snapshot.get("reason")


def test_the_load_sized_floor_still_applies_when_nothing_is_resident(
    monkeypatch,
) -> None:
    """Relaxing must be tied to residency, not applied unconditionally."""

    monkeypatch.setattr(
        "core.container.ServiceContainer.peek",
        staticmethod(lambda name, default=None: None),
    )

    class _Snapshot:
        pressure_pct = 78.4
        available_gb = 13.8
        total_gb = 64.0
        process_rss_gb = 25.7
        process_rss_limit_gb = 40.0
        refuse_heavy_local_generation = False

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        staticmethod(lambda: _Snapshot()),
    )

    snapshot = InferenceGate._headroom_snapshot("primary")

    assert snapshot["can_admit"] is False
    assert "memory_pressure" in str(snapshot.get("reason", ""))


def test_the_secondary_tier_is_untouched(monkeypatch) -> None:
    """The residency relaxation is scoped to the primary conversational turn.

    Tertiary is deliberately NOT asserted here: its own thresholds (92% / 6GB)
    already admit these numbers by design, so asserting a refusal there would
    encode my misreading rather than the contract.
    """

    class _Gate:
        @staticmethod
        def get_conversation_status() -> dict[str, object]:
            return {"conversation_ready": True, "has_generated_successfully": True}

    monkeypatch.setattr(
        "core.container.ServiceContainer.peek",
        staticmethod(lambda name, default=None: _Gate() if name == "inference_gate" else default),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        staticmethod(lambda: _Snapshot()),
    )

    assert InferenceGate._headroom_snapshot("secondary")["can_admit"] is False


# ── the kernel's verdict outranks a derived percentage ───────────────────────
#
# Measured on this host while a turn was refused for "78.4% pressure":
# kern.memorystatus_vm_pressure_level = 1 (NORMAL), 79% free, and 24GB of the
# "used" memory sitting in reclaimable inactive pages. psutil counts
# file-backed cache and compressed pages as consumed; the OS reclaims them on
# demand, which is what a cache is for.


class _Snapshot:
    pressure_pct = 78.4
    available_gb = 13.8
    total_gb = 64.0
    process_rss_gb = 25.7
    process_rss_limit_gb = 40.0
    refuse_heavy_local_generation = False


def _with_snapshot(monkeypatch, **overrides):
    snap = type("S", (_Snapshot,), overrides)
    monkeypatch.setattr(
        "core.container.ServiceContainer.peek",
        staticmethod(lambda name, default=None: None),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        staticmethod(lambda: snap()),
    )
    return snap


def test_kernel_normal_overrides_the_derived_percentage(monkeypatch) -> None:
    """Nothing resident, 78.4% derived — but the OS says there is no pressure."""
    _with_snapshot(monkeypatch, kernel_pressure_level="normal")

    assert InferenceGate._headroom_snapshot("primary")["can_admit"] is True


def test_kernel_critical_refuses_however_good_the_percentage_looks(
    monkeypatch,
) -> None:
    """The OS is asking processes to free memory. Not the moment."""
    _with_snapshot(
        monkeypatch,
        kernel_pressure_level="critical",
        pressure_pct=10.0,
        available_gb=50.0,
    )

    assert InferenceGate._headroom_snapshot("primary")["can_admit"] is False


def test_kernel_unknown_has_no_opinion(monkeypatch) -> None:
    """Off Darwin, or an unreadable sysctl, must not relax anything."""
    _with_snapshot(monkeypatch, kernel_pressure_level="unknown")

    assert InferenceGate._headroom_snapshot("primary")["can_admit"] is False


def test_kernel_normal_still_honours_the_hard_floor(monkeypatch) -> None:
    """Relaxing the RATE signal must not relax the absolute minimum."""
    _with_snapshot(
        monkeypatch, kernel_pressure_level="normal", available_gb=1.0
    )

    assert InferenceGate._headroom_snapshot("primary")["can_admit"] is False


def test_the_kernel_level_is_a_real_reading_on_this_host() -> None:
    from core.utils.memory_monitor import kernel_memory_pressure_level

    assert kernel_memory_pressure_level() in {
        "normal", "warn", "critical", "unknown",
    }
