from types import SimpleNamespace

import psutil

from core.utils.memory_monitor import AppleSiliconMemoryMonitor


def test_memory_monitor_clamps_psutil_percent(monkeypatch):
    monitor = AppleSiliconMemoryMonitor()
    monkeypatch.setattr(
        "core.utils.memory_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=132.4),
    )

    assert monitor._get_pressure_sysctl() == 100


def test_memory_monitor_uses_available_total_when_percent_missing(monkeypatch):
    monitor = AppleSiliconMemoryMonitor()
    monkeypatch.setattr(
        "core.utils.memory_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=None, total=1_000, available=250),
    )

    assert monitor._get_pressure_sysctl() == 75


def test_memory_monitor_returns_zero_on_sampling_error(monkeypatch):
    monitor = AppleSiliconMemoryMonitor()
    calls = []

    def sampler():
        calls.append("called")
        raise psutil.Error("sample failed")

    monkeypatch.setattr("core.utils.memory_monitor.psutil.virtual_memory", sampler)

    assert monitor._get_pressure_sysctl() == 0
    assert calls == ["called"]
