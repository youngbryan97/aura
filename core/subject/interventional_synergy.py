"""Synergy read by setting the sources: ISC-v6's §27 (docs/ISC_V6_PREREGISTRATION.md).

v5 reads synergy by watching a run, and in her ordinary life two sources move
together, driven by the same situations, so watching cannot tell a joint
effect from two effects side by side. An interaction is defined by what
happens when the causes are set: the effect of one depends on the level of the
other. From one moment, four untouched forks shared by every triple, and for
each triple the first source displaced, the second, and both. The interaction
is

    Y(both) - Y(first) - Y(second) + Y(untouched 1)

in the target's whitened state, conjunction registers left out. Its floor is a
sham of the same form over four untouched forks, so whatever the forks' noise
is, it enters both alike. Her forks are deterministic (every sham of 21
October was exactly zero), and there any nonlinear system, chaotic divergence
included, clears a floor; the direction bar is what asks for a systematic
joint effect.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DRAWS",
    "LEVEL",
    "SIZE_FRACTION",
    "Samples",
    "collect",
    "direction_p",
    "line",
    "sign_flip_p",
    "summarise",
    "target_columns",
]

#: Sign flips for the paired tests; the smallest p reachable is 1 / (DRAWS + 1).
DRAWS = 20_000
#: The level each p must clear: v5's null bars were 99th percentiles.
LEVEL = 0.01
#: v5's floor on the synergy fraction, here on the interaction over the larger main effect.
SIZE_FRACTION = 0.10


@dataclass
class Samples:
    """One triple's forks, sample by sample, in the target's whitened coordinates."""

    interactions: list[np.ndarray] = field(default_factory=list)
    shams: list[np.ndarray] = field(default_factory=list)
    mains: list[float] = field(default_factory=list)

    def add(self, *, untouched: Sequence[np.ndarray], first: np.ndarray, second: np.ndarray,
            both: np.ndarray) -> None:
        n1, n2, n3, n4 = (np.asarray(v, dtype=np.float64) for v in untouched)
        self.interactions.append(np.asarray(both - first - second, dtype=np.float64) + n1)
        self.shams.append(n2 - n3 - n4 + n1)
        self.mains.append(max(float(np.linalg.norm(first - n1)), float(np.linalg.norm(second - n1))))


def sign_flip_p(values: Iterable[float], *, seed: int, draws: int = DRAWS) -> float:
    """One-sided p that the mean of paired values is above zero, by random sign flips."""
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return 1.0
    observed = values.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, values.size))
    flipped = (signs * values).mean(axis=1)
    return float((np.sum(flipped >= observed) + 1) / (draws + 1))


def _direction_statistic(gram: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Mean over samples of each vector projected on the sum of the others, for rows of signs.

    With e the signs, v_i·(sum_j e_j v_j - e_i v_i) and the norm of that sum come
    from the Gram matrix, so each draw costs n^2 and not n^2 times the width.
    """
    k_e = signs @ gram  # (draws, n): sum_j e_j K_ij
    diagonal = np.diag(gram)
    total = np.einsum("dn,dn->d", signs, k_e)  # e'Ke
    dots = signs * k_e - diagonal  # e_i v_i . (S_e - e_i v_i)
    norms_sq = total[:, None] - 2.0 * signs * k_e + diagonal
    norms = np.sqrt(np.maximum(norms_sq, 0.0))
    projections = np.where(norms > 0, dots / np.where(norms > 0, norms, 1.0), 0.0)
    return projections.mean(axis=1)


def direction_p(vectors: Sequence[np.ndarray], *, seed: int, draws: int = DRAWS) -> float:
    """One-sided p that the interactions share a direction, with whole vectors' signs flipped in the null."""
    if len(vectors) < 3:
        return 1.0
    stacked = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors])
    gram = stacked @ stacked.T
    observed = _direction_statistic(gram, np.ones((1, len(vectors))))[0]
    rng = np.random.default_rng(seed)
    flipped = _direction_statistic(gram, rng.choice([-1.0, 1.0], size=(draws, len(vectors))))
    return float((np.sum(flipped >= observed) + 1) / (draws + 1))


def summarise(samples: Samples, *, seed: int, draws: int = DRAWS) -> dict[str, Any]:
    """The three preregistered bars for one triple, and whether it passes."""
    sizes = np.array([float(np.linalg.norm(v)) for v in samples.interactions])
    shams = np.array([float(np.linalg.norm(v)) for v in samples.shams])
    mains = np.asarray(samples.mains, dtype=np.float64)
    n = int(sizes.size)
    out: dict[str, Any] = {
        "samples": n,
        "interaction": round(float(sizes.mean()), 6) if n else 0.0,
        "sham": round(float(shams.mean()), 6) if n else 0.0,
        "main_effect": round(float(mains.mean()), 6) if n else 0.0,
    }
    out["p_above_sham"] = round(sign_flip_p(sizes - shams, seed=seed, draws=draws), 6)
    out["p_one_direction"] = round(direction_p(samples.interactions, seed=seed + 1, draws=draws), 6)
    out["fraction"] = round(float(sizes.mean() / mains.mean()), 6) if n and mains.mean() > 0 else 0.0
    out["bars"] = {
        "above_sham": out["p_above_sham"] < LEVEL,
        "one_direction": out["p_one_direction"] < LEVEL,
        "size": out["fraction"] >= SIZE_FRACTION,
    }
    out["passes"] = bool(n >= 3 and all(out["bars"].values()))
    return out


def line(triples: Mapping[str, Mapping[str, Any]] | None) -> bool:
    """§27 under ISC-v6: every declared triple passes. A run that did not read it fails."""
    from core.subject.synergy import TRIPLES

    if not triples:
        return False
    wanted = [f"{a},{b}->{y}" for a, b, y in TRIPLES]
    return all(bool((triples.get(key) or {}).get("passes")) for key in wanted)


async def _fork(runtime: Any, snapshot: Any, condition: Any, displace: tuple[str, ...],
                delta: float, turns: int) -> np.ndarray:
    """One fork from the snapshot: the named domains displaced together at the top of its first turn."""
    from core.subject.causal import SUSTAINED
    from core.subject.perturbation import perturb, perturb_organs

    runtime.restore(snapshot)
    sustained = any(domain in SUSTAINED for domain in displace)
    applied = {"done": not displace}

    async def hit(rt: Any) -> None:
        if not displace or (applied["done"] and not sustained):
            return
        for domain in displace:
            in_state = perturb(rt.state, domain, delta, ontogeny=rt.ontogeny)
            in_organ = await perturb_organs(rt.organs, domain, delta, state=rt.state)
            applied["done"] = applied["done"] or bool(in_state or in_organ)

    frames = []
    for turn in range(turns):
        frames.extend(
            await runtime.turn_once(
                condition,
                perturb_at=0 if (turn == 0 and displace) or sustained else None,
                perturb=hit,
                sustain=hit if sustained else None,
            )
        )
    return np.asarray(frames[-1].vector(), dtype=np.float64)


def target_columns(columns: Sequence[str], domain_slice: slice) -> np.ndarray:
    """A target domain's columns without the registers of any conjunction."""
    return np.array(
        [index for index in range(domain_slice.start, domain_slice.stop) if ".conj_" not in columns[index]],
        dtype=int,
    )


async def collect(
    runtime: Any,
    conditions: Sequence[Any],
    *,
    metrics: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    trials: int,
    turns: int,
    delta: float,
    seed: int,
    on_progress: Any = None,
) -> dict[str, Samples]:
    """Four untouched forks and three displaced ones per declared triple, every trial and condition.

    `targets` holds each target's columns, registers out (`target_columns`), and
    `metrics` each target's whitening fitted on the run's recording over the
    same columns. The caller has stopped the free-running loops, as for the
    campaign's other paired arms.
    """
    from core.subject.synergy import TRIPLES

    rng = random.Random(seed)
    held = {f"{a},{b}->{y}": Samples() for a, b, y in TRIPLES}
    for trial in range(trials):
        for condition in conditions:
            await runtime.turn_once(condition)
            runtime.freeze_host()
            snapshot = runtime.snapshot()
            try:
                plan = [(f"untouched{k}", None, ()) for k in range(4)]
                for a, b, y in TRIPLES:
                    key = f"{a},{b}->{y}"
                    plan += [("first", key, (a,)), ("second", key, (b,)), ("both", key, (a, b))]
                rng.shuffle(plan)
                untouched: dict[str, np.ndarray] = {}
                displaced: dict[tuple[str, str], np.ndarray] = {}
                for name, key, displace in plan:
                    vector = await _fork(runtime, snapshot, condition, displace, delta, turns)
                    if key is None:
                        untouched[name] = vector
                    else:
                        displaced[(key, name)] = vector
                for a, b, y in TRIPLES:
                    key = f"{a},{b}->{y}"
                    read = {name: metrics[y].whitened(v[targets[y]])[0] for name, v in (
                        ("first", displaced[(key, "first")]), ("second", displaced[(key, "second")]),
                        ("both", displaced[(key, "both")]))}
                    base = [metrics[y].whitened(untouched[f"untouched{k}"][targets[y]])[0] for k in range(4)]
                    held[key].add(untouched=base, **read)
            finally:
                runtime.restore(snapshot)
                runtime.thaw_host()
        if on_progress is not None:
            on_progress(f"  interventional synergy: trial {trial + 1}/{trials}")
    return held
