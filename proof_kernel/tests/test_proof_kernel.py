from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aura_consciousness_proof.global_workspace import CognitiveCandidate, ContentType, GlobalWorkspace
from aura_consciousness_proof.homeostasis import ExternalSignals, HomeostasisEngine
from aura_consciousness_proof.report import build_report, build_markdown
from aura_consciousness_proof.self_prediction import SelfPredictionLoop
from aura_consciousness_proof.structural_opacity import StructuralOpacityMonitor
from aura_consciousness_proof.temporal_binding import TemporalBindingEngine


def test_global_workspace_prefers_higher_effective_priority() -> None:
    async def run() -> None:
        workspace = GlobalWorkspace()
        await workspace.submit(CognitiveCandidate("weak", "source_a", 0.3, ContentType.META, focus_bias=0.0))
        await workspace.submit(CognitiveCandidate("focused", "source_b", 0.25, ContentType.META, focus_bias=0.7))
        winner = await workspace.run_competition()
        if not (winner is not None): raise RuntimeError('Assertion failed')
        if not (winner.source == "source_b"): raise RuntimeError('Assertion failed')

    asyncio.run(run())


def test_temporal_binding_builds_narrative() -> None:
    async def run() -> None:
        engine = TemporalBindingEngine()
        await engine.record_event("first event", "test", valence=0.4, significance=0.7)
        await engine.record_event("second event", "test", valence=-0.2, significance=0.5)
        await engine.maybe_refresh_narrative(5)
        narrative = await engine.get_narrative()
        if not ("AUTOBIOGRAPHICAL PRESENT" in narrative): raise RuntimeError('Assertion failed')
        if not ("first event" in narrative or "second event" in narrative): raise RuntimeError('Assertion failed')

    asyncio.run(run())


def test_self_prediction_tracks_surprise() -> None:
    async def run() -> None:
        loop = SelfPredictionLoop()
        await loop.tick(0.1, "curiosity", "drive_curiosity")
        await loop.tick(-0.8, "integrity", "maintenance_guard")
        snapshot = loop.get_snapshot()
        if not (snapshot["surprise_count"] >= 1): raise RuntimeError('Assertion failed')
        if not (snapshot["smoothed_error"] > 0.0): raise RuntimeError('Assertion failed')

    asyncio.run(run())


def test_homeostasis_changes_from_signals_and_feedback() -> None:
    async def run() -> None:
        engine = HomeostasisEngine()
        before = engine.integrity
        await engine.pulse(ExternalSignals(health_error_rate=0.5, resource_anxiety=0.9, thermal_load=0.9, sovereignty_score=0.8))
        engine.on_response_error("model_crash")
        if not (engine.integrity < before): raise RuntimeError('Assertion failed')
        if not ("HOMEOSTASIS" in engine.get_context_block()): raise RuntimeError('Assertion failed')

    asyncio.run(run())


def test_structural_opacity_measures_hidden_state() -> None:
    monitor = StructuralOpacityMonitor(neuron_count=16, n_perturbations=6)
    x = np.random.randn(16) * 0.2
    W = np.random.randn(16, 16) * 0.15
    monitor.record_state(x)
    sig = monitor.measure(x, W)
    if not (0.0 <= sig.opacity_index <= 1.0): raise RuntimeError('Assertion failed')
    if not (monitor.get_specious_present().shape == (16,)): raise RuntimeError('Assertion failed')


def test_report_generation_contains_all_sections() -> None:
    report = asyncio.run(build_report(cycles=8))
    markdown = build_markdown(report)
    if not ("workspace" in report): raise RuntimeError('Assertion failed')
    if not ("temporal_binding" in report): raise RuntimeError('Assertion failed')
    if not ("self_prediction" in report): raise RuntimeError('Assertion failed')
    if not ("homeostasis" in report): raise RuntimeError('Assertion failed')
    if not ("structural_opacity" in report): raise RuntimeError('Assertion failed')
    if not ("Aura Consciousness Proof Report" in markdown): raise RuntimeError('Assertion failed')
