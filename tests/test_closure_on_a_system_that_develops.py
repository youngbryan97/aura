"""The closure gate on a system that develops from inside, as she does (25 October 2026).

Her recording drifts: trained on her first fifth, the rest of K stands up to 17
training spreads away in the rows it is scored on. The stationary toy worlds
never do that, and two defects only show under development. Drifting outside
levels regress spuriously on a core that drifts beside them, so unrelated
columns read open and the wider model does measurably worse than the narrower
on whole folds. And a closed reading could rest on such a fold. The gate now
enters every outside column as its increment, predicts K's change, and calls a
fold where the wider model does measurably worse an estimator failure.

The least-domain margin, preregistered and withdrawn the same day, is here too:
in a world with every domain coupled, the least of those leaks sits near zero.
"""
from __future__ import annotations

import numpy as np

from core.subject.closure import ClosureReport, closure_gain, least_domain_margin
from core.subject.recording import Recording, slices_from_columns
from core.subject.state import DOMAINS

#: A margin for these worlds only. Choosing the one her closure is read within
#: is a preregistration, not a test.
WORLD_MARGIN = 0.0196


def _recording(x: np.ndarray, columns: tuple[str, ...]) -> Recording:
    rows = x.shape[0]
    return Recording(
        x=x, conditions=tuple("talk" if (k // 7) % 2 else "rest" for k in range(rows)), tags=("",) * rows,
        times=np.arange(rows, dtype=np.float64), env=np.zeros((rows, 0)), env_names=(), columns=columns,
        slices=slices_from_columns(columns), notes={},
    )


def _developing(seed: int, *, broker: float = 0.0, width: int = 4, rows: int = 2000, outside: int = 30):
    """Ten domains in a ring; N accumulates the rest's activity and sets where every other domain sits."""
    rng = np.random.default_rng(seed)
    n = len(DOMAINS) * width
    nk = DOMAINS.index("N")
    n_cols = slice(nk * width, (nk + 1) * width)
    coupling = 0.5 * np.eye(n)
    for i in range(len(DOMAINS)):
        here = slice(i * width, (i + 1) * width)
        before = slice(((i - 1) % len(DOMAINS)) * width, ((i - 1) % len(DOMAINS) + 1) * width)
        coupling[here, before] += rng.normal(0, 0.45 / np.sqrt(width), (width, width))
    coupling[n_cols, :] = 0.0
    mix = rng.normal(0, 1, (n, width)) / np.sqrt(width)
    gain = rng.normal(0, 2.0, n)
    absorb = np.abs(rng.normal(0, 1, (width, n))) / n
    reach = rng.normal(0, 1, n)
    reach[n_cols] = 0.0
    hidden = np.zeros(rows)
    x = np.zeros((rows, n))
    dev = np.zeros(n)
    for k in range(rows - 1):
        hidden[k + 1] = 0.6 * hidden[k] + rng.normal(0, 0.5)
        setpoint = gain * np.tanh(mix @ (x[k, n_cols] - 1.5))
        setpoint[n_cols] = 0.0
        dev = coupling @ dev + broker * reach * hidden[k] + rng.normal(0, 0.3, n)
        dev[n_cols] = 0.0
        step = setpoint + dev
        step[n_cols] = x[k, n_cols] + 0.0018 * (absorb @ np.abs(dev)) + 0.0006
        x[k + 1] = step
    walks = np.cumsum(rng.normal(0, 0.05, (rows, outside)), axis=0)
    periphery = walks if not broker else np.hstack([walks, hidden[:, None]])
    columns = tuple(f"{key}.{i}" for key in DOMAINS for i in range(width))
    return _recording(x, columns), periphery


def _names(periphery: np.ndarray) -> tuple[str, ...]:
    return tuple(f"outside.{i}" for i in range(periphery.shape[1]))


def test_it_develops():
    recording, _ = _developing(11)
    train, test = recording.x[:400], recording.x[800:1200]
    spread = np.where(train.std(0) > 1e-9, train.std(0), 1.0)
    assert np.median(np.abs(test.mean(0) - train.mean(0)) / spread) > 0.5


def test_drifting_columns_she_never_reads_do_not_open_her_and_she_closes_within_a_named_margin():
    recording, walks = _developing(11)
    report = closure_gain(recording, walks, _names(walks), seed=1, draws=16, margin=WORLD_MARGIN)
    assert not report.estimator_failed, report.as_dict()["folds"]
    assert report.verdict == "CLOSED_WITHIN_MARGIN", report.as_dict()


def test_a_broker_outside_a_developing_core_opens_it():
    recording, periphery = _developing(11, broker=0.35)
    report = closure_gain(recording, periphery, _names(periphery), seed=1, draws=16, margin=WORLD_MARGIN)
    assert report.verdict == "OPEN"


def test_no_margin_named_means_nothing_closes(monkeypatch):
    import core.subject.closure as closure

    monkeypatch.setattr(closure, "CLOSURE_MARGIN", None)
    recording, walks = _developing(11)
    report = closure_gain(recording, walks, _names(walks), seed=1, draws=16)
    assert report.margin is None
    assert report.verdict == "UNRESOLVED"


def test_a_fold_where_the_wider_model_did_worse_cannot_carry_a_closed_reading():
    folds = (
        {"leak": -0.4, "leak_lower": -0.6, "leak_upper": -0.2, "open": False},
        {"leak": 0.0, "leak_lower": -0.01, "leak_upper": 0.005, "open": False},
        {"leak": 0.0, "leak_lower": -0.01, "leak_upper": 0.005, "open": False},
    )
    report = ClosureReport(
        1.0, 1.0, 1.0, -0.1, 0.0, 0.0, 30, floor_high=0.01, leak_lower=-0.2, leak_upper=-0.05,
        smallest_leak_resolved=0.01, margin=0.05, folds=folds,
    )
    assert report.estimator_failed == (0,)
    assert report.verdict == "UNRESOLVED"
    assert "estimator failed" in report.as_dict()["why_unresolved"]


def test_the_least_of_her_domains_is_no_margin_when_every_domain_is_coupled():
    """Each domain feeds the next by construction, and still the least of their leaks is about nothing."""
    recording, _ = _developing(11)
    drawn = least_domain_margin(recording, seed=1)
    assert drawn["margin"] < 0.01


def _report_with_folds(folds):
    return ClosureReport(
        1.0, 1.0, 1.0, 0.004, 0.0, 0.0, 30, floor_high=0.01, leak_lower=-0.002, leak_upper=0.009,
        smallest_leak_resolved=0.01, margin=0.0196, folds=folds,
    )


def test_a_failed_fold_with_two_valid_ones_beside_it_is_dropped_and_named():
    folds = (
        {"leak": -0.21, "leak_lower": -0.27, "leak_upper": -0.11, "open": False, "dropped": True},
        {"leak": 0.0055, "leak_lower": 0.0027, "leak_upper": 0.0100, "open": False},
        {"leak": 0.0053, "leak_lower": -0.0010, "leak_upper": 0.0101, "open": False},
    )
    report = _report_with_folds(folds)
    assert report.estimator_failed == ()
    assert report.dropped_folds == (0,)
    assert report.verdict == "CLOSED_WITHIN_MARGIN"
    assert report.as_dict()["dropped_folds"] == [0]


def test_with_one_valid_fold_left_a_failure_still_blocks_closure():
    folds = (
        {"leak": -0.21, "leak_lower": -0.27, "leak_upper": -0.11, "open": False},
        {"leak": -0.09, "leak_lower": -0.14, "leak_upper": -0.02, "open": False},
        {"leak": 0.0053, "leak_lower": -0.0010, "leak_upper": 0.0101, "open": False},
    )
    report = _report_with_folds(folds)
    assert report.estimator_failed == (0, 1)
    assert report.verdict == "UNRESOLVED"


def _broker_among(columns: int, seed: int = 2, rows: int = 1500, strength: float = 0.5):
    rng = np.random.default_rng(seed)
    hidden = np.zeros(rows)
    k = np.zeros((rows, 4))
    for t in range(rows - 1):
        hidden[t + 1] = 0.5 * hidden[t] + rng.normal(0, 1)
        k[t + 1] = 0.6 * k[t] + strength * hidden[t] * np.array([1.0, -0.5, 0.0, 0.3]) + rng.normal(0, 0.3, 4)
    noise = rng.normal(size=(rows, columns))
    return _recording(k, ("A.0", "A.1", "G.0", "G.1")), hidden, noise


def test_one_cause_among_hundreds_of_idle_columns_is_found():
    """A single broker among three hundred noise columns: the shared penalty used to drown it."""
    recording, hidden, noise = _broker_among(300)
    periphery = np.column_stack([noise, hidden])
    report = closure_gain(recording, periphery, _names(periphery), seed=1, draws=8, margin=WORLD_MARGIN)
    assert report.verdict == "OPEN"


def test_hundreds_of_idle_columns_alone_leak_nothing():
    recording, _hidden, noise = _broker_among(300, strength=0.0)
    report = closure_gain(recording, noise, _names(noise), seed=1, draws=8, margin=WORLD_MARGIN)
    assert report.verdict != "OPEN"
    assert report.leak_upper is not None and report.leak_upper < WORLD_MARGIN
