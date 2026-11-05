"""Machines built with a known number of subjects, for checking J*'s exclusion rule end to end.

Preregistered in docs/ISC_V6_PREREGISTRATION.md (addendum of 5 November 2026, the exclusion rule on built machines).

The rule (`core/subject/v25_exclusion.py`) decides from what the measurements give it: the edges the interventions
support, whether each strongly connected part is closed against the rest of the machine, and each part's spectrum of
irreducibility. On her it has only been asked about a machine with one candidate, so it has only had to say "one".
Four toy machines are built here, each with its answer known before anything is measured:

    two_apart       two halves, each wired densely within itself and not at all to the other: two subjects.
    one_whole       all ten domains wired densely to one another: one subject, the whole.
    one_drives      the halves of two_apart, and the first half driving the second with nothing coming back: one
                    subject, the first half. The second is open to the first, so it is not a carrier.
    joined_weakly   the halves joined both ways at half the strength of their own wiring: one subject, the whole.

Each is measured the way the toy nulls are: its edges by the real run's rule (`toy_edges`), each part's closure by
the test her core gets (`closure_gain`, with the rest of the machine as the outside), and each part's spectrum by the
estimator her carrier sweep uses (`intrinsic_rate_from_samples`), on forks that hold one side of a cut still while the
other runs, as her clamp does, the cut built from each side where it was free (`core/subject/clamp.compose`). None of
it is the organism. It checks the rule and the measurements that feed it, on machines whose answer is not in doubt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from core.subject.nulls import (
    ToySystem,
    _domain_slices,
    _hold_radius,
    _matrix,
    toy_closure,
    toy_edges,
    toy_recording,
)
from core.subject.state import DOMAINS
from core.subject.v25_exclusion import select_carriers, strongly_connected_components
from core.subject.v25_runtime import bipartitions

__all__ = ["EXPECTED", "HALVES", "MACHINES", "build", "measure", "spectrum"]

#: The two halves. Five domains each, in the order the battery declares them.
HALVES: tuple[tuple[str, ...], tuple[str, ...]] = (("P", "I", "A", "G", "C"), ("S", "M", "W", "D", "N"))

MACHINES: tuple[str, ...] = ("two_apart", "one_whole", "one_drives", "joined_weakly")

#: What each machine's frontier must be, written before anything is measured.
EXPECTED: dict[str, frozenset[frozenset[str]]] = {
    "two_apart": frozenset({frozenset(HALVES[0]), frozenset(HALVES[1])}),
    "one_whole": frozenset({frozenset(DOMAINS)}),
    "one_drives": frozenset({frozenset(HALVES[0])}),
    "joined_weakly": frozenset({frozenset(DOMAINS)}),
}

#: How strongly the halves are joined in joined_weakly, as a share of their own wiring.
WEAK_SHARE: float = 0.5

#: The machines' settings, fixed once for all four before any machine was judged (5 November). The carrier sweep's
#: estimator reads these toys only where the coupling dominates the noise: at the toy nulls' own settings (four
#: columns a domain, noise 0.05, radius 0.7) a densely wired half read raw and sham rates that matched at every
#: horizon. The pilot read only the spectra of the two positive controls, two_apart's first half (seed 0) and
#: one_whole (seeds 0 and 1): one column a domain, noise 0.02 and radius 0.95 made the half irreducible at three of
#: four horizons, and 512 anchors made the whole irreducible at three or four, where 256 left it at two with margins
#: of 0.0005. These set how strongly a part is wired, the same for every machine. Which parts there are and which are
#: closed, which is what tells the four answers apart, they do not set.
WIDTH: int = 1
NOISE: float = 0.02
RADIUS: float = 0.95
HORIZONS: tuple[int, ...] = (4, 8, 16, 33)
ANCHORS: int = 512


def build(
    name: str, *, seed: int = 0, width: int = WIDTH, strength: float = 0.6, noise: float = NOISE, radius: float = RADIUS,
) -> ToySystem:
    """One of the machines: the toy nulls' wiring rule and decay, at the settings above."""
    if name not in MACHINES:
        raise KeyError(name)
    rng = np.random.default_rng(seed)
    widths = {key: width for key in DOMAINS}
    decay = {key: rng.uniform(0.3, 0.7, size=width) for key in DOMAINS}
    coupling: dict[str, dict[str, np.ndarray]] = {key: {} for key in DOMAINS}

    def wire(targets: Sequence[str], sources: Sequence[str], scale: float = 1.0) -> None:
        # coupling[target][source] is what the target reads, so it is an edge from source to target.
        for target in targets:
            for source in sources:
                if source != target:
                    coupling[target][source] = _matrix(rng, width, width, strength) * scale

    first, second = HALVES
    if name == "one_whole":
        wire(DOMAINS, DOMAINS)
    else:
        wire(first, first)
        wire(second, second)
        if name == "one_drives":
            wire(second, first)
        elif name == "joined_weakly":
            wire(second, first, WEAK_SHARE)
            wire(first, second, WEAK_SHARE)
    system = ToySystem(name=name, widths=widths, coupling=coupling, decay=decay, noise=noise)
    _hold_radius(system, _domain_slices(widths), radius)
    return system


def _batch_roll(
    system: ToySystem, start: dict[str, np.ndarray], steps: int, rng: np.random.Generator, held: Sequence[str] = (),
) -> list[dict[str, np.ndarray]]:
    """`_roll` for many anchors at once: each value is (width, anchors), each anchor with its own noise. The step is
    `ToySystem.step` for a machine with no hub and nothing frozen, which these are."""
    if system.hub_width or system.frozen:
        raise ValueError("the batched roll is for machines with no hub and nothing frozen")
    state = {key: value.copy() for key, value in start.items()}
    out = []
    for _ in range(steps):
        nxt = {}
        for key, width in system.widths.items():
            drive = system.decay[key][:, None] * state[key]
            for other, matrix in system.coupling.get(key, {}).items():
                drive = drive + matrix @ state[other]
            nxt[key] = np.tanh(drive) + system.noise * rng.normal(size=(width, drive.shape[1]))
        state = nxt
        for key in held:
            state[key] = start[key].copy()
        out.append(state)
    return out


def _anchors(system: ToySystem, count: int, *, seed: int, warmup: int = 400, spacing: int = 9) -> list[dict]:
    rng = np.random.default_rng(seed)
    state = system.start(rng)
    for _ in range(warmup):
        state = system.step(state, rng)
    out = []
    for _ in range(count):
        for _ in range(spacing):
            state = system.step(state, rng)
        out.append({key: value.copy() for key, value in state.items()})
    return out


def spectrum(
    system: ToySystem, support: Sequence[str], *, horizons: Sequence[int], anchors: int, seed: int,
) -> tuple[dict[float, float], dict[float, str]]:
    """The support's irreducibility at each horizon: the least any of its bipartitions costs, read by the carrier
    sweep's estimator against two untouched forks from the same anchor. Returns the spectrum and the cheapest cut
    at each horizon. A support of one domain has no cut and no spectrum."""
    support = tuple(key for key in DOMAINS if key in set(support))
    cuts = bipartitions(support)
    if not cuts:
        return {}, {}
    bank = _anchors(system, anchors, seed=seed)
    start = {key: np.stack([anchor[key] for anchor in bank], axis=1) for key in system.widths}
    longest = max(int(h) for h in horizons)

    def rows(state: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([state[key] for key in support], axis=0).T

    intact_run = _batch_roll(system, start, longest, np.random.default_rng([seed, 1]))
    sham_run = _batch_roll(system, start, longest, np.random.default_rng([seed, 2]))
    context = rows(start)
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    least: dict[float, float] = {}
    where: dict[float, str] = {}
    for number, (left, right) in enumerate(cuts):
        left_free = _batch_roll(system, start, longest, np.random.default_rng([seed, 3, number]), held=right)
        right_free = _batch_roll(system, start, longest, np.random.default_rng([seed, 4, number]), held=left)
        for horizon in horizons:
            step = int(horizon) - 1
            composed = {**{key: left_free[step][key] for key in left}, **{key: right_free[step][key] for key in right}}
            rate = float(intrinsic_rate_from_samples(
                rows(intact_run[step]), rows(composed), rows(intact_run[step]), rows(sham_run[step]),
                tau_seconds=float(horizon), context=context, seed=seed,
            ).excess_rate)
            tau = float(horizon)
            if tau not in least or rate < least[tau]:
                least[tau], where[tau] = rate, "".join(left) + "|" + "".join(right)
    return least, where


def _closure(system: ToySystem, support: Sequence[str], *, steps: int, seed: int) -> tuple[bool, float, str]:
    """Whether the support's future needs anything outside it: the whole machine by `toy_closure`, and a part by the
    closure test her core gets, with the rest of the machine as its outside."""
    if set(support) == set(DOMAINS):
        closed, leak = toy_closure(system, steps=steps, seed=seed)
        return bool(closed), float(leak), "CLOSED" if closed else "OPEN"
    from core.subject.closure import closure_gain

    recording = toy_recording(system, steps=steps, seed=seed)
    inside = [key for key in DOMAINS if key in set(support)]
    outside = [key for key in DOMAINS if key not in set(support)]

    def columns(keys: Sequence[str]) -> np.ndarray:
        return np.concatenate([np.arange(recording.slices[k].start, recording.slices[k].stop) for k in keys])

    keep, rest = columns(inside), columns(outside)
    start, slices = 0, {}
    for key in inside:
        width = recording.slices[key].stop - recording.slices[key].start
        slices[key] = slice(start, start + width)
        start += width
    part = replace(recording, x=recording.x[:, keep], columns=tuple(recording.columns[i] for i in keep), slices=slices)
    report = closure_gain(part, recording.x[:, rest], tuple(recording.columns[i] for i in rest), seed=seed)
    return bool(report.closed), float(report.leak), str(report.verdict)


def measure(
    name: str, *, seed: int = 0, trials: int = 6, horizons: Sequence[int] = HORIZONS, anchors: int = ANCHORS,
    steps: int = 2500,
) -> dict[str, Any]:
    """One machine, measured and judged: its parts, each part's closure and spectrum, the rule's frontier, and
    whether the frontier is the one written down for it."""
    system = build(name, seed=seed)
    edges = toy_edges(system, trials=trials, seed=seed)
    components = strongly_connected_components(list(DOMAINS), edges)
    spectra: dict[frozenset[str], dict[float, float]] = {}
    cheapest: dict[str, dict[float, str]] = {}
    closure: dict[frozenset[str], tuple[bool, float, str]] = {}
    for block in components:
        support = frozenset(block)
        spectra[support], cheapest["".join(block)] = spectrum(
            system, block, horizons=horizons, anchors=anchors, seed=seed
        )
        closure[support] = _closure(system, block, steps=steps, seed=seed)
    whole = frozenset(DOMAINS)
    if whole not in closure:
        closure[whole] = _closure(system, DOMAINS, steps=steps, seed=seed)
    report = select_carriers(list(DOMAINS), edges, spectra, closure, grid=tuple(float(h) for h in horizons))
    found = frozenset(c.support for c in report.frontier)
    return {
        "machine": name,
        "expected": sorted("".join(sorted(s)) for s in EXPECTED[name]),
        "found": sorted("".join(sorted(s)) for s in found),
        "as_predicted": found == EXPECTED[name] and report.status == "FOUND",
        "edges": [f"{a}->{b}" for a, b in edges],
        "cheapest_cut": cheapest,
        "closure": {"".join(sorted(s)): list(v) for s, v in closure.items()},
        "exclusion": report.as_dict(),
        "settings": {"seed": seed, "trials": trials, "horizons": list(horizons), "anchors": anchors, "steps": steps},
    }
