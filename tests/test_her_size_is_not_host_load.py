"""Her own size is not host load (third day run, 6 November 2026).

Her process held about 4.3 GB on a 64 GB Mac at 10-18% CPU. The homeostatic coupling compared it with a flat 3500 MB,
read the host as loaded every time her proprioception reported, and throttled her depth to 0.6 and her creativity to
0.8: "HARDWARE RESONANCE: High host load", 122 times.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from core.consciousness import homeostatic_coupling as module


def _coupling():
    coupling = module.HomeostaticCoupling.__new__(module.HomeostaticCoupling)
    coupling._reported_strain = frozenset()
    coupling._thermal_stress = 0
    coupling._cpu_stress = 12.0
    coupling._stress_timestamp = time.time()
    return coupling


def _host(monkeypatch, total_gb: float):
    from core.runtime import resource_psutil

    monkeypatch.setattr(module, "_load_line_mb", None)
    monkeypatch.delenv("AURA_INTEGRITY_MEMORY_WARNING_MB", raising=False)
    monkeypatch.setattr(resource_psutil, "virtual_memory", lambda: SimpleNamespace(total=total_gb * 1024 ** 3))


def test_her_day_three_size_on_a_64_gb_mac_is_not_load(monkeypatch):
    _host(monkeypatch, 64)
    coupling = _coupling()
    coupling._mem_stress = 4330.0
    mods = module.CognitiveModifiers()
    depth = mods.depth_mod
    coupling._apply_hardware_resonance(mods)
    assert mods.depth_mod == depth
    assert "load" not in coupling._reported_strain


def test_a_process_past_a_fifth_of_the_host_is_load(monkeypatch):
    _host(monkeypatch, 64)
    coupling = _coupling()
    coupling._mem_stress = 14000.0
    mods = module.CognitiveModifiers()
    depth = mods.depth_mod
    coupling._apply_hardware_resonance(mods)
    assert mods.depth_mod < depth
    assert "load" in coupling._reported_strain


def test_a_small_host_keeps_the_old_floor(monkeypatch):
    _host(monkeypatch, 8)
    assert module.memory_load_line_mb() == 3500.0
