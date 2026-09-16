"""tests/test_somatic_throttle.py — Unit tests for SomaticComputeSentinel.

Verifies parameter cuts occur under real or coupled hardware stress.
"""
from pathlib import Path
from types import SimpleNamespace

import core.brain.llm.somatic_throttle as throttle_module
from core.brain.llm.somatic_throttle import SomaticComputeSentinel


def _governor(throttle: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(get_throttle_factor=lambda: throttle)


def _install_probe_readings(
    monkeypatch,
    resource_observer,
    *,
    arousal: float,
    cpu_percent: float,
    memory_percent: float,
):
    affect = SimpleNamespace(current=SimpleNamespace(arousal=arousal))
    monkeypatch.setattr(
        throttle_module,
        "resolve_affect_engine",
        lambda default=None: affect,
    )
    monkeypatch.setattr(
        "research.protocols.resource_quotas.get_compute_governor",
        lambda: _governor(),
    )
    resource_observer.configure_compute(cpu_percent=cpu_percent)
    resource_observer.configure_memory(percent=memory_percent)
    return affect


def test_somatic_throttle_normal(monkeypatch, resource_observer):
    # Normal/unstressed parameters remain unchanged.
    _install_probe_readings(
        monkeypatch,
        resource_observer,
        arousal=0.2,
        cpu_percent=15.0,
        memory_percent=40.0,
    )
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())
        
    assert adjusted["max_tokens"] == 512
    assert adjusted["temperature"] == 0.7
    assert adjusted["recurrent_depth"] == 0.8


def test_somatic_throttle_resource_stressed(monkeypatch, resource_observer):
    # Stressed hardware caps max_tokens and adjusts temp/lane depth.
    _install_probe_readings(
        monkeypatch,
        resource_observer,
        arousal=0.2,
        cpu_percent=50.0,
        memory_percent=89.0,
    )
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())
        
    assert adjusted["max_tokens"] == 256
    assert adjusted["temperature"] == 0.3
    assert adjusted["recurrent_depth"] == 0.4


def test_somatic_throttle_critical(monkeypatch, resource_observer):
    # Critical parameters restrict max_tokens to 128.
    _install_probe_readings(
        monkeypatch,
        resource_observer,
        arousal=0.95,
        cpu_percent=95.0,
        memory_percent=95.0,
    )
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_lane_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())

    assert adjusted["max_tokens"] == 128
    assert adjusted["temperature"] == 0.15
    assert adjusted["recurrent_lane_depth"] == 0.2


def test_somatic_throttle_high_arousal_without_resource_pressure_is_not_critical(
    monkeypatch,
    resource_observer,
):
    _install_probe_readings(
        monkeypatch,
        resource_observer,
        arousal=0.97,
        cpu_percent=12.0,
        memory_percent=60.0,
    )
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 512, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy())

    assert adjusted["max_tokens"] == 512
    assert adjusted["temperature"] == 0.7
    assert adjusted["recurrent_depth"] == 0.8


def test_somatic_throttle_does_not_swallow_generic_exceptions():
    source = (
        Path(__file__).resolve().parent.parent
        / "core"
        / "brain"
        / "llm"
        / "somatic_throttle.py"
    ).read_text(encoding="utf-8")

    assert "except Exception" not in source
    assert "except BaseException" not in source


def test_mlx_generation_throttle_hook_uses_typed_boundary():
    source = (
        Path(__file__).resolve().parent.parent
        / "core"
        / "brain"
        / "llm"
        / "mlx_client.py"
    ).read_text(encoding="utf-8")
    throttle_block = source.split("SomaticComputeSentinel", 1)[1].split("foreground_owner_cm", 1)[0]

    assert "except Exception" not in throttle_block
    # The handler no longer merely "continues unthrottled" — it applies a
    # conservative ceiling instead, which is a stronger contract than the one
    # this assertion was written against. What matters is that the failure is
    # recorded rather than swallowed, and that something bounds the generation.
    assert "_record_mlx_degradation(" in throttle_block
    assert "_apply_unthrottled_fallback_ceiling(" in throttle_block
    assert "severity=\"warning\"" in throttle_block


def test_somatic_throttle_records_expected_probe_failures(monkeypatch, resource_observer):
    records = []

    class FailingAffectResolver:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            raise RuntimeError("affect offline")

    monkeypatch.setattr(
        "core.brain.llm.somatic_throttle.record_degradation",
        lambda subsystem, exc, **metadata: records.append((subsystem, exc, metadata)),
    )
    monkeypatch.setattr(
        "core.brain.llm.somatic_throttle.resolve_affect_engine",
        FailingAffectResolver(),
    )
    monkeypatch.setattr(
        "research.protocols.resource_quotas.get_compute_governor",
        lambda: _governor(),
    )
    resource_observer.configure_compute(cpu_percent=5.0)
    resource_observer.configure_memory(percent=20.0)

    adjusted = SomaticComputeSentinel().adjust_generation_options({"max_tokens": 512})

    assert adjusted["max_tokens"] == 512
    assert records
    assert records[0][0] == "somatic_throttle"
    assert records[0][2]["action"] == "using neutral arousal for generation throttle"


def test_a_foreground_turn_keeps_its_budget_and_sampling_under_host_stress(monkeypatch, resource_observer):
    """LIVE 2026-09-16: a timetable question was capped at 256 tokens because
    the host's CPU was at 92% under other agents' jobs; the thinking was cut
    short and the answer was wrong. The budget is the clock's."""
    _install_probe_readings(
        monkeypatch,
        resource_observer,
        arousal=0.2,
        cpu_percent=96.0,
        memory_percent=94.0,
    )
    sentinel = SomaticComputeSentinel()
    opts = {"max_tokens": 1024, "temperature": 0.7, "recurrent_depth": 0.8}
    adjusted = sentinel.adjust_generation_options(opts.copy(), foreground=True)
    assert adjusted["max_tokens"] == 1024
    assert adjusted["temperature"] == 0.7
    # Compute depth is still a throttle's to turn down.
    assert adjusted["recurrent_depth"] == 0.2

    background = sentinel.adjust_generation_options(opts.copy())
    assert background["max_tokens"] == 128
