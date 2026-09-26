"""Host-safety contracts for the MLX memory envelope (CP214).

An unguarded evaluation tool drove this host to 103 GB and forced a
shutdown. Training was already enveloped; evaluation was not. These tests
pin the properties that make that unrepeatable: limits are derived from
real host RAM, over-host limits are refused rather than granted, the
envelope restores what it changed, and the reclaim hook is available to
long loops.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from core.runtime.mlx_memory_guard import (  # noqa: E402
    DEFAULT_FRACTION,
    MLX_MEMORY_GUARD_SCHEMA,
    MemoryEnvelope,
    _host_probe_output,
    _parse_swap_used_gb,
    _pressure_reasons,
    host_memory_bytes,
    mlx_memory_envelope,
)


def test_host_memory_is_read_not_assumed():
    host = host_memory_bytes()
    assert host >= 2 * 1024**3
    assert host < 4 * 1024**4  # sanity: not a nonsense reading


def test_default_limit_is_a_fraction_of_real_host_ram():
    with mlx_memory_envelope() as envelope:
        receipt = envelope.to_receipt()
        assert receipt["schema"] == MLX_MEMORY_GUARD_SCHEMA
        expected = DEFAULT_FRACTION * receipt["host_memory_gb"]
        assert receipt["memory_limit_gb"] == pytest.approx(expected, rel=0.02)
        # Headroom for the OS is the entire point.
        assert receipt["memory_limit_gb"] < receipt["host_memory_gb"]


def test_over_host_limit_is_refused_not_granted():
    """The failure mode being prevented: a limit above physical RAM means
    the machine swaps to death instead of the run failing."""
    host_gb = host_memory_bytes() / 1024**3
    with pytest.raises(ValueError, match="exceeds physical RAM"):
        with mlx_memory_envelope(memory_gb=host_gb * 2):
            pass


def test_unusable_and_out_of_range_settings_are_refused():
    with pytest.raises(ValueError, match="2 GiB floor"):
        with mlx_memory_envelope(memory_gb=0.5):
            pass
    with pytest.raises(ValueError, match="fraction"):
        with mlx_memory_envelope(fraction=0.99):
            pass
    with pytest.raises(ValueError, match="cache limit cannot exceed"):
        with mlx_memory_envelope(memory_gb=4.0, cache_gb=8.0):
            pass
    with pytest.raises(ValueError, match="restore_limits_on_exit"):
        with mlx_memory_envelope(restore_limits_on_exit=1):
            pass


def test_limits_are_restored_on_exit_and_on_error():
    before = mx.set_memory_limit(mx.set_memory_limit(8 * 1024**3))
    with mlx_memory_envelope(memory_gb=4.0):
        inside = mx.set_memory_limit(mx.set_memory_limit(4 * 1024**3))
        assert inside == 4 * 1024**3
    after = mx.set_memory_limit(mx.set_memory_limit(before))
    assert after == before

    with pytest.raises(RuntimeError, match="boom"):
        with mlx_memory_envelope(memory_gb=4.0):
            raise RuntimeError("boom")
    restored = mx.set_memory_limit(mx.set_memory_limit(before))
    assert restored == before, "limits must restore even when the body raises"


def test_reclaim_honours_cadence_and_force():
    envelope = MemoryEnvelope(
        memory_bytes=4 * 1024**3,
        cache_bytes=1024**3,
        wired_bytes=4 * 1024**3,
        reclaim_every=8,
    )
    assert envelope.reclaim(8) is True
    assert envelope.reclaim(3) is False
    assert envelope.reclaim(3, force=True) is True
    assert envelope.reclaim(None, force=True) is True


def test_reclaim_synchronizes_around_cache_release(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(mx, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("clear_cache"))
    envelope = MemoryEnvelope(
        memory_bytes=4 * 1024**3,
        cache_bytes=1024**3,
        wired_bytes=4 * 1024**3,
    )

    assert envelope.reclaim(force=True) is True
    assert calls == ["synchronize", "clear_cache", "synchronize"]


def test_process_owned_envelope_skips_allocator_reclaim_on_exit(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(mx, "set_memory_limit", lambda _value: 8 * 1024**3)
    monkeypatch.setattr(mx, "set_cache_limit", lambda _value: 2 * 1024**3)
    monkeypatch.setattr(mx, "set_wired_limit", lambda _value: 8 * 1024**3)
    monkeypatch.setattr(mx, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("clear_cache"))

    with mlx_memory_envelope(
        memory_gb=4.0,
        restore_limits_on_exit=False,
    ):
        pass

    assert calls == ["synchronize"]


def test_receipt_is_complete_enough_to_audit_a_run():
    with mlx_memory_envelope(memory_gb=4.0, cache_gb=1.0) as envelope:
        receipt = envelope.to_receipt()
    assert set(receipt) == {
        "schema",
        "memory_limit_gb",
        "cache_limit_gb",
        "wired_limit_gb",
        "requested_wired_limit_gb",
        "device_wired_cap_gb",
        "reclaim_every",
        "host_memory_gb",
    }
    assert receipt["memory_limit_gb"] == pytest.approx(4.0, rel=1e-6)
    assert receipt["cache_limit_gb"] == pytest.approx(1.0, rel=1e-6)


def test_high_fraction_defaults_respect_the_real_device_wired_cap():
    cap = mx.device_info()["max_recommended_working_set_size"]
    with mlx_memory_envelope(fraction=.80) as envelope:
        assert envelope.wired_bytes <= cap
        assert envelope.wired_bytes <= envelope.requested_wired_bytes
        assert envelope.device_wired_cap_bytes <= cap


@pytest.mark.parametrize("cap", [None, 0, -1, True, "48GB"])
def test_unmeasured_device_cap_does_not_mutate_limits(monkeypatch, cap):
    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": cap})
    monkeypatch.setattr(mx, "set_memory_limit", lambda _: pytest.fail("mutated before preflight"))
    with pytest.raises(ValueError, match="capacity is unavailable"):
        with mlx_memory_envelope():
            pass


def test_defaults_clamp_but_explicit_oversized_wiring_is_refused(monkeypatch):
    monkeypatch.setattr("core.runtime.mlx_memory_guard.host_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": 48 * 1024**3})
    calls = []
    for name in ("memory", "cache", "wired"):
        monkeypatch.setattr(mx, f"set_{name}_limit", lambda value, name=name:
                            calls.append((name, value)) or 2 * 1024**3)
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    with mlx_memory_envelope(fraction=.8) as envelope:
        assert envelope.wired_bytes == 48 * 1024**3
        assert envelope.requested_wired_bytes == int(.85 * 64 * 1024**3)
    calls.clear()
    with pytest.raises(ValueError, match="device working-set capacity"):
        with mlx_memory_envelope(wired_gb=49):
            pass
    assert calls == []


@pytest.mark.parametrize("failure", ["cache", "wired"])
@pytest.mark.parametrize("restore_on_exit", [False, True])
def test_partial_setup_always_rolls_back_successful_setters(monkeypatch, failure, restore_on_exit):
    state = {"memory": 8 * 1024**3, "cache": 1024**3, "wired": 0}
    original = dict(state)
    def setter(name, value):
        if name == failure:
            raise RuntimeError("injected setup failure")
        previous, state[name] = state[name], value
        return previous
    for name in state:
        monkeypatch.setattr(mx, f"set_{name}_limit", lambda value, name=name: setter(name, value))
    with pytest.raises(RuntimeError, match="setup failure"):
        with mlx_memory_envelope(memory_gb=4, restore_limits_on_exit=restore_on_exit):
            pytest.fail("setup should not yield")
    assert state == original


@pytest.mark.parametrize("failure", ["synchronize", "wired_restore"])
def test_cleanup_failure_does_not_skip_other_limit_restorations(monkeypatch, failure, caplog):
    state = {"memory": 8 * 1024**3, "cache": 1024**3, "wired": 0}
    original = dict(state)
    def setter(name, value):
        if failure == "wired_restore" and name == "wired" and value == 0:
            raise RuntimeError("injected wired restoration failure")
        previous, state[name] = state[name], value
        return previous
    for name in state:
        monkeypatch.setattr(mx, f"set_{name}_limit", lambda value, name=name: setter(name, value))
    def synchronize():
        if failure == "synchronize":
            raise RuntimeError("injected synchronization failure")
    monkeypatch.setattr(mx, "synchronize", synchronize)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    with mlx_memory_envelope(memory_gb=4):
        pass
    assert state["memory"] == original["memory"]
    assert state["cache"] == original["cache"]
    assert "injected" in caplog.text
    if failure == "synchronize":
        assert state == original


# ── Reading pressure correctly, not alarmingly ──────────────────────────


def test_host_pressure_reports_reclaimable_not_just_free():
    """'Pages free' excludes inactive/purgeable memory the kernel reclaims
    on demand, so a healthy host can read ~1GB free while 65% of RAM is
    actually available. Aborting a good run on that number is a false
    alarm; the signals that preceded this host's jetsam kill were SWAP and
    COMPRESSOR growth."""
    from core.runtime.mlx_memory_guard import host_pressure

    report = host_pressure()
    if not report.get("available"):
        pytest.skip("vm_stat unavailable on this platform")
    assert report["reclaimable_gb"] >= report["free_gb"]
    assert 0.0 <= report["available_fraction"] <= 1.0
    assert report["host_gb"] > 0
    assert isinstance(report["under_pressure"], bool)


def test_pressure_verdict_reports_explicit_current_reasons():
    """The verdict must not fire from free-memory or swap history alone."""
    from core.runtime.mlx_memory_guard import host_pressure

    report = host_pressure()
    if not report.get("available"):
        pytest.skip("vm_stat unavailable on this platform")
    assert report["under_pressure"] is bool(report["pressure_reasons"])
    if report["compressed_gb"] < 0.25 * report["host_gb"] and report[
        "reclaimable_gb"
    ] >= 0.15 * report["host_gb"]:
        assert report["under_pressure"] is False


def test_swap_parser_reads_used_instead_of_the_preceding_total() -> None:
    output = "total = 3072.00M  used = 1881.69M  free = 1190.31M  (encrypted)"
    assert _parse_swap_used_gb(output) == pytest.approx(1881.69 / 1024.0)
    assert _parse_swap_used_gb("total = 8.00G used = 3.25G free = 4.75G") == 3.25
    assert _parse_swap_used_gb("unavailable") is None


def test_allocated_swap_does_not_latch_pressure_on_a_healthy_host() -> None:
    assert _pressure_reasons(
        host_gb=64.0,
        reclaimable_gb=40.0,
        compressed_gb=4.3,
        swap_used_gb=17.9,
    ) == ()


def test_detached_host_probe_uses_exact_broker_and_retains_output(
    tmp_path,
    monkeypatch,
) -> None:
    from core.runtime import detached_subprocess_broker as broker

    output = tmp_path / "vm-stat.txt"
    observed = {}

    def fake_run(command, *, cwd, stdout_path, timeout_s):
        observed.update(
            command=command,
            cwd=cwd,
            stdout_path=stdout_path,
            timeout_s=timeout_s,
        )
        stdout_path.write_text("probe evidence\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            status="passed",
            containment_verified=True,
        )

    monkeypatch.setattr(broker, "broker_available", lambda: True)
    monkeypatch.setattr(broker, "run_brokered_process", fake_run)

    assert _host_probe_output(
        ["vm_stat"],
        source="test",
        broker_stdout_path=output,
    ) == "probe evidence\n"
    assert observed == {
        "command": ["vm_stat"],
        "cwd": Path.cwd(),
        "stdout_path": output,
        "timeout_s": 5.0,
    }
    assert output.read_text(encoding="utf-8") == "probe evidence\n"


@pytest.mark.parametrize(
    ("reclaimable_gb", "compressed_gb", "swap_used_gb", "expected"),
    [
        (4.0, 4.0, 0.0, {"reclaimable_critical"}),
        (20.0, 17.0, 0.0, {"compressor_high"}),
        (8.0, 4.0, 3.0, {"swap_correlated_scarcity"}),
        (4.0, 17.0, 3.0, {"compressor_high", "reclaimable_critical", "swap_correlated_scarcity"}),
    ],
)
def test_present_pressure_signals_still_fail_closed(
    reclaimable_gb: float,
    compressed_gb: float,
    swap_used_gb: float,
    expected: set[str],
) -> None:
    assert set(
        _pressure_reasons(
            host_gb=64.0,
            reclaimable_gb=reclaimable_gb,
            compressed_gb=compressed_gb,
            swap_used_gb=swap_used_gb,
        )
    ) == expected
