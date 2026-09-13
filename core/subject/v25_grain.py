"""Interventional signature collection for Subject Core v25."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np

from core.subject.causal import SUSTAINED
from core.subject.intrinsic_v25 import characteristic_signature
from core.subject.state import perturb, perturb_organs
from core.subject.v25_runtime import Anchor, lag_vector


@dataclass(frozen=True)
class ProbeAction:
    name: str
    displacements: tuple[tuple[str, float], ...] = ()


def signed_single_domain_actions(
    doses: Mapping[str, float],
    *,
    sign: int,
    include_sham: bool = False,
) -> tuple[ProbeAction, ...]:
    actions: list[ProbeAction] = [ProbeAction("sham")] if include_sham else []
    prefix = "+" if sign > 0 else "-"
    for domain, magnitude in doses.items():
        actions.append(
            ProbeAction(
                f"{prefix}{domain}",
                ((domain, float(abs(magnitude)) * (1.0 if sign > 0 else -1.0)),),
            )
        )
    return tuple(actions)


def pair_actions(
    pairs: Sequence[tuple[str, str]],
    doses: Mapping[str, float],
    *,
    sign: int = 1,
) -> tuple[ProbeAction, ...]:
    out: list[ProbeAction] = []
    direction = 1.0 if sign > 0 else -1.0
    for a, b in pairs:
        out.append(
            ProbeAction(
                f"{'+' if sign > 0 else '-'}{a}{b}",
                (
                    (a, direction * abs(float(doses[a]))),
                    (b, direction * abs(float(doses[b]))),
                ),
            )
        )
    return tuple(out)


async def _run_probe(
    runtime: Any,
    snapshot: Any,
    condition: Any,
    *,
    action: ProbeAction,
    turns: int,
    at: int = 0,
) -> list[Any]:
    """Run one matched probe, generalized to multi-domain interventions."""
    runtime.restore(snapshot)
    held_host = None if runtime.frozen_host is None else dict(runtime.frozen_host)
    held_latency = None if runtime.frozen_latency is None else dict(runtime.frozen_latency)
    sustained_domains = {domain for domain, _ in action.displacements if domain in SUSTAINED}

    async def apply(items: Sequence[tuple[str, float]]) -> None:
        for domain, delta in items:
            perturb(runtime.state, domain, delta, ontogeny=runtime.ontogeny)
            await perturb_organs(runtime.organs, domain, delta, state=runtime.state)
            if domain == "I" and runtime.frozen_host is not None:
                runtime.freeze_host()

    async def initial(_runtime: Any) -> None:
        await apply(action.displacements)

    async def sustain(_runtime: Any) -> None:
        await apply(
            tuple((d, v) for d, v in action.displacements if d in sustained_domains)
        )

    rows: list[Any] = []
    try:
        for turn in range(turns):
            rows.extend(
                await runtime.turn_once(
                    condition,
                    perturb_at=at if (turn == 0 and action.displacements) else None,
                    perturb=initial if action.displacements else None,
                    sustain=sustain if sustained_domains else None,
                )
            )
    finally:
        runtime.frozen_host = held_host
        runtime.frozen_latency = held_latency
    return rows


async def signature_matrix(
    runtime: Any,
    anchors: Sequence[Anchor],
    conditions: Sequence[Any],
    actions: Sequence[ProbeAction],
    *,
    lags: Sequence[int],
    frame_seconds: float,
    baseline_mean: np.ndarray,
    baseline_scale: np.ndarray,
    live_mask: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """One characteristic-function interventional signature per anchor."""
    if not anchors:
        raise ValueError("no anchors")
    max_lag = max(int(x) for x in lags)
    frames_per_turn = max(1, int(round(1.0 / max(frame_seconds, 1e-9))))
    turns = max(1, ceil(max_lag / frames_per_turn))
    mean = np.asarray(baseline_mean, dtype=np.float64)[live_mask]
    scale = np.asarray(baseline_scale, dtype=np.float64)[live_mask]
    scale = np.where(scale > 1e-9, scale, 1.0)

    rows: list[np.ndarray] = []
    for anchor in anchors:
        sample_sets: list[np.ndarray] = []
        freq_sets: list[np.ndarray] = []
        for condition in conditions:
            for action in actions:
                trajectory = await _run_probe(
                    runtime,
                    anchor.snapshot,
                    condition,
                    action=action,
                    turns=turns,
                )
                for lag in lags:
                    future = lag_vector(trajectory, int(lag))[live_mask]
                    standardized = (future - mean) / scale
                    sample_sets.append(standardized.reshape(1, -1))
                    freq_sets.append(frequencies)
        rows.append(characteristic_signature(sample_sets, freq_sets))
    return np.vstack(rows)
