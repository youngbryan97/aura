from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .global_workspace import CognitiveCandidate, ContentType, GlobalWorkspace
from .homeostasis import ExternalSignals, HomeostasisEngine
from .self_prediction import SelfPredictionLoop
from .structural_opacity import StructuralOpacityMonitor
from .temporal_binding import TemporalBindingEngine


async def build_report(cycles: int = 20, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    workspace = GlobalWorkspace()
    temporal = TemporalBindingEngine()
    predictor = SelfPredictionLoop()
    homeostasis = HomeostasisEngine()
    opacity = StructuralOpacityMonitor(neuron_count=32, n_perturbations=12, perturbation_scale=0.05)

    state = rng.normal(0.0, 0.2, size=32)
    weights = rng.normal(0.0, 0.45, size=(32, 32))
    winners: list[str] = []

    for cycle in range(1, cycles + 1):
        signals = ExternalSignals(
            health_error_rate=max(0.0, 0.18 + 0.12 * math.sin(cycle / 2.2)),
            resource_anxiety=max(0.0, 0.55 + 0.35 * math.sin(cycle / 2.7 + 0.3)),
            thermal_load=max(0.0, 0.45 + 0.40 * math.cos(cycle / 3.1)),
            sovereignty_score=max(0.65, 0.95 - 0.03 * (cycle % 5)),
        )
        await homeostasis.pulse(signals)

        state = 0.82 * state + 0.18 * np.tanh(weights @ state) + rng.normal(0.0, 0.05, size=32)
        opacity.record_state(state)
        opacity_signature = opacity.measure(state, weights)
        workspace.update_phi(opacity_signature.causal_depth)

        drive_name, drive_deficit = homeostasis.get_dominant_deficiency()
        base_curiosity = 0.20 + homeostasis.curiosity * 0.35 + 0.25 * max(0.0, math.sin(cycle / 1.7))
        base_integrity = 0.18 + (1.0 - homeostasis.integrity) * 0.55 + 0.25 * max(0.0, math.cos(cycle / 2.2))
        base_social = 0.18 + 0.30 * max(0.0, math.sin(cycle / 1.3 + 0.8))

        candidates = [
            CognitiveCandidate(
                content=f"Cycle {cycle}: investigate uncertainty in recent state drift",
                source="drive_curiosity",
                priority=base_curiosity,
                content_type=ContentType.INTENTIONAL,
                affect_weight=homeostasis.curiosity * 0.2,
                focus_bias=0.1 if drive_name == "curiosity" else 0.0,
            ),
            CognitiveCandidate(
                content=f"Cycle {cycle}: stabilize system integrity and continuity",
                source="maintenance_guard",
                priority=base_integrity,
                content_type=ContentType.META,
                affect_weight=(1.0 - homeostasis.integrity) * 0.3,
                focus_bias=0.1 if drive_name == "integrity" else 0.0,
            ),
            CognitiveCandidate(
                content=f"Cycle {cycle}: track socially relevant external cues",
                source="social_signal",
                priority=base_social,
                content_type=ContentType.SOCIAL,
                affect_weight=0.05,
                focus_bias=0.05 if cycle % 3 == 0 else 0.0,
            ),
        ]

        for candidate in candidates:
            await workspace.submit(candidate)

        winner = await workspace.run_competition()
        if winner is None:
            continue
        winners.append(winner.source)

        actual_valence = max(
            -1.0,
            min(1.0, (homeostasis.compute_vitality() - 0.5) * 1.4 + 0.25 * math.sin(cycle / 2.0)),
        )
        await predictor.tick(
            actual_valence=round(actual_valence, 3),
            actual_drive=drive_name if drive_deficit > 0 else "balanced",
            actual_focus_source=winner.source,
        )

        await temporal.record_event(
            content=winner.content,
            source=winner.source,
            valence=actual_valence,
            significance=max(0.3, winner.effective_priority),
        )
        await temporal.maybe_refresh_narrative(cycle)

        if winner.source == "maintenance_guard":
            homeostasis.on_response_success(response_length=180)
        elif winner.source == "drive_curiosity":
            homeostasis.feed_curiosity(0.02)
        else:
            homeostasis.on_response_success(response_length=80)

    narrative = await temporal.get_narrative()
    report = {
        "claim_scope": {
            "supports": [
                "competitive attention selection",
                "temporal continuity across cycles",
                "self-prediction with measurable surprise",
                "homeostatic internal regulation",
                "hidden-state opacity metrics",
            ],
            "does_not_support": [
                "direct proof of subjective experience",
                "resolution of the hard problem",
            ],
        },
        "workspace": workspace.get_snapshot(),
        "temporal_binding": {
            **temporal.get_snapshot(),
            "narrative": narrative,
        },
        "self_prediction": predictor.get_snapshot(),
        "homeostasis": {
            **homeostasis.get_status(),
            "context_block": homeostasis.get_context_block(),
            "inference_modifiers": homeostasis.get_inference_modifiers(),
        },
        "structural_opacity": {
            **opacity.get_snapshot(),
            "specious_present_norm": round(float(np.linalg.norm(opacity.get_specious_present())), 4),
        },
        "observed_winners": winners,
    }
    return report


def build_markdown(report: dict[str, Any]) -> str:
    workspace = report["workspace"]
    temporal = report["temporal_binding"]
    prediction = report["self_prediction"]
    homeostasis = report["homeostasis"]
    opacity = report["structural_opacity"]

    return "\n".join(
        [
            "# Aura Consciousness Proof Report",
            "",
            "## Honest Claim",
            "This report shows a measurable consciousness-inspired architecture with causal hidden state.",
            "It does not prove subjective experience from the outside.",
            "",
            "## Global Workspace",
            f"- Ticks: {workspace['tick']}",
            f"- Last winner: {workspace['last_winner']}",
            f"- Ignition count: {workspace['ignition_count']}",
            f"- Ignited now: {workspace['ignited']}",
            "",
            "## Temporal Binding",
            f"- Total events: {temporal['total_events']}",
            f"- Present window events: {temporal['present_window_events']}",
            f"- Average valence: {temporal['avg_valence']}",
            "",
            "```text",
            temporal["narrative"],
            "```",
            "",
            "## Self Prediction",
            f"- Surprise count: {prediction['surprise_count']}",
            f"- Smoothed error: {prediction['smoothed_error']}",
            f"- Most unpredictable dimension: {prediction['most_unpredictable']}",
            "",
            "## Homeostasis",
            f"- Will to live: {homeostasis['will_to_live']}",
            f"- Integrity: {homeostasis['integrity']}",
            f"- Curiosity: {homeostasis['curiosity']}",
            "",
            "## Structural Opacity",
            f"- Measurements: {opacity['measurement_count']}",
            f"- Last opacity: {round(opacity['last_opacity'], 4)}",
            f"- Last causal depth: {round(opacity['last_causal_depth'], 4)}",
            f"- Status: {opacity['status']['status']} (the metric is still informative even when the threshold is not crossed)",
            f"- Specious present norm: {opacity['specious_present_norm']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone proof report for Aura's consciousness kernel.")
    parser.add_argument("--cycles", type=int, default=20, help="Number of synthetic cognitive cycles to run.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON output.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional path for Markdown output.")
    args = parser.parse_args()

    report = asyncio.run(build_report(cycles=args.cycles))
    markdown = build_markdown(report)

    if args.json is not None:
        atomic_write_text(args.json, json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown is not None:
        atomic_write_text(args.markdown, markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
