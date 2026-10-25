"""The closure margin is drawn from K itself, 25 October 2026.

A closed reading needs a margin named in advance, and any number chosen after
seed 23's leak was read would have been chosen with it in view. The rule reads
no periphery: each of K's domains in turn is left out and read as the outside,
by the same estimator, and the least of those leaks is the margin.
"""
from __future__ import annotations

import numpy as np

from core.subject.closure import closure_gain, least_domain_margin
from core.subject.recording import Recording, slices_from_columns

KEYS = ("A", "G", "C", "S")


def _world(rows: int = 1500, seed: int = 0, *, idle: str | None = None, broker: float = 0.0):
    """Four domains of two columns in a ring, each reading the one before it.

    `idle` cuts one domain off: it reads only itself and nothing reads it.
    `broker` lets a slow variable outside K drive every domain.
    """
    rng = np.random.default_rng(seed)
    width = 2
    n = len(KEYS) * width
    coupling = 0.5 * np.eye(n)
    for index in range(len(KEYS)):
        here = slice(index * width, (index + 1) * width)
        before = slice(((index - 1) % len(KEYS)) * width, ((index - 1) % len(KEYS) + 1) * width)
        coupling[here, before] += 0.35 * rng.choice([-1.0, 1.0], size=(width, width))
    if idle is not None:
        k = KEYS.index(idle)
        cut = slice(k * width, (k + 1) * width)
        keep = coupling[cut, cut].copy()
        coupling[cut, :] = 0.0
        coupling[:, cut] = 0.0
        coupling[cut, cut] = keep
    x = np.zeros((rows, n))
    hidden = np.zeros(rows)
    reach = rng.normal(0, 1, n)
    for t in range(rows - 1):
        hidden[t + 1] = 0.95 * hidden[t] + rng.normal(0, 0.3)
        x[t + 1] = coupling @ x[t] + broker * reach * hidden[t] + rng.normal(0, 0.3, n)
    columns = tuple(f"{key}.{i}" for key in KEYS for i in range(width))
    recording = Recording(
        x=x,
        conditions=tuple("talk" if t % 2 else "rest" for t in range(rows)),
        tags=("",) * rows,
        times=np.arange(rows, dtype=np.float64),
        env=np.zeros((rows, 0)),
        env_names=(),
        columns=columns,
        slices=slices_from_columns(columns),
        notes={},
    )
    return recording, hidden


def test_a_core_whose_parts_need_each_other_closes_against_an_idle_outside():
    recording, _ = _world(seed=1)
    noise = np.random.default_rng(2).normal(0, 1, (recording.frames, 3))
    report = closure_gain(recording, noise, ("outside.a", "outside.b", "outside.c"), seed=3, draws=8)
    drawn = report.audit["margin"]
    assert drawn["rule"] == "least_domain" and drawn["domain"] in KEYS
    assert report.margin == drawn["margin"] > 0.0
    assert report.verdict == "CLOSED_WITHIN_MARGIN", report.as_dict()


def test_a_hidden_broker_outside_the_core_still_opens_it():
    recording, hidden = _world(seed=1, broker=0.6)
    report = closure_gain(recording, hidden[:, None], ("broker.state",), seed=3, draws=8)
    assert report.verdict == "OPEN"


def test_a_domain_that_tells_the_rest_nothing_leaves_nothing_to_close_within():
    """Cutting C off also leaves G with no reader, since C was the only domain reading it."""
    recording, _ = _world(seed=1, idle="C")
    drawn = least_domain_margin(recording, seed=3)
    assert drawn["domain"] in {"C", "G"}
    assert drawn["margin"] < 0.01 < min(drawn["by_domain"][key] for key in ("A", "S"))
    noise = np.random.default_rng(2).normal(0, 1, (recording.frames, 3))
    report = closure_gain(recording, noise, ("outside.a", "outside.b", "outside.c"), seed=3, draws=8)
    assert report.verdict != "CLOSED_WITHIN_MARGIN"


def test_the_margin_reads_no_periphery():
    recording, hidden = _world(seed=1)
    noise = np.random.default_rng(2).normal(0, 1, (recording.frames, 3))
    quiet = closure_gain(recording, noise, ("outside.a", "outside.b", "outside.c"), seed=3, draws=8)
    other = closure_gain(recording, noise[:, :1] * 5.0, ("outside.a",), seed=3, draws=8)
    assert quiet.audit["margin"]["by_domain"] == other.audit["margin"]["by_domain"]
    assert quiet.audit["margin"] == least_domain_margin(recording, seed=3)
