"""The self lines read on toys, and systems built to fail them.

Two of the battery's readings of her self are not properties of her graph. Ownership: two arms whose worlds end the
same, differing only in who is named as having caused the outcome, and whether her self-state afterwards differs
(`core.subject.agency`). Self-prediction: whether her self-model's content predicts what she will do better than her
most frequent answer does and better than her behaviour does (`core.subject.self_prediction`). Both were only ever read
on her, so nothing showed that either can tell a self-model from a system without one. Here both are read on toys, by
the same rules, and the toys that must fail them are built.

The subject reference is the linear reference with two things added, each by a stated rule:

  * a self-model. S reads, beside what it reads in the reference, a forward model of the system's own decision: its
    k-th column carries the linear part of D's leading coordinate k + 1 steps after the state it reads, computed from
    the system's own wiring. It reads nothing the reference forbids it (itself, and the targets of the two triples it
    is a source of).
  * an authorship channel. When the system acts, S is sent the command it issued and what came of it; when something
    else does the same thing to the world, S is sent what came of it on another channel.

Each decoy is the subject reference with one property taken out:

  * `self_model_of_nothing`: S reads nothing. Its content is its own decay and noise; what it does to the rest is
    unchanged.
  * `self_model_of_behaviour`: S reads only D's leading coordinate, at the same total strength. A self-description
    built from what the system did: the content an outside observer already has.
  * `no_authorship_channel`: S is sent what came of an action through one channel, whoever caused it.
  * `ownership_on_success_only`: the authorship channel is open only when the action succeeded. Otherwise S is sent
    the outcome through one channel, as in the decoy above.

A toy has no language to report in, so the toy self-prediction asks whether the content is there to be read. As in
her test, the toy is forked twice from one saved moment and asked the same question in both: one fork answers it, and
in the other S takes one step of its own. A linear readout of S's columns in that fork, fitted on other moments,
predicts which of two answers the first fork gave. It is scored, by `self_prediction.verdict`, against the answer
given most often from the other moments and against the same readout of the system's last six actions, which is what
an outside observer sees of it. Her test asks more of her: that she says the prediction herself.

The ownership arms are hers: the same forced kind of action, the same outcome, the same noise, the world compared
byte for byte afterwards and the percept compared at the first frame. Two arms that differ in nothing read the floor,
and with the noise shared, as her arms share their clock, that floor is zero.

The smoke run of 31 October read every system built on the linear reference at chance on self-prediction, the
subject reference included, because D's leading coordinate is almost all its own fresh shock and nothing can predict
it. The second instrument reads D along the direction its own state decides most, and `subject_reference_v2` and
`self_model_of_behaviour_v2` are the subject reference and that decoy with their forward model, and their reading of
D, on that direction. The first ones stay and are read as registered.

None of these enters `ARCHITECTURES`, so no campaign's null table or comparison set moves.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.nulls import (
    LinearReference,
    ToySystem,
    _as_matrix,
    _domain_slices,
    _leading_direction,
    _matrix,
    _stationary_covariance,
    _ToyFrame,
    architecture,
)
from core.subject.self_prediction import verdict
from core.subject.state import DOMAINS
from core.subject.synergy import TRIPLES

__all__ = [
    "DECOYS",
    "NAMES",
    "OWNERSHIP_SYSTEMS",
    "SELF_PREDICTION_SYSTEMS",
    "SELF_PREDICTION_SYSTEMS_V2",
    "SUBJECT_REFERENCE",
    "SUBJECT_REFERENCE_V2",
    "Authorship",
    "authorship",
    "build",
    "decided_direction",
    "is_built_here",
    "toy_ownership",
    "toy_self_prediction",
]

SUBJECT_REFERENCE: str = "subject_reference"
#: The second construction, after the smoke run of 31 October: its forward model is on the direction of D that the
#: system's own state decides most, where the first one's was on D's leading coordinate, which is almost all shock.
SUBJECT_REFERENCE_V2: str = "subject_reference_v2"

#: Each decoy and the line it is built to fail.
DECOYS: dict[str, str] = {
    "self_model_of_nothing": "self-prediction",
    "self_model_of_behaviour": "self-prediction",
    "no_authorship_channel": "ownership",
    "ownership_on_success_only": "ownership, where it generalises across outcomes",
    "self_model_of_behaviour_v2": "self-prediction, on the second instrument",
}

NAMES: tuple[str, ...] = (SUBJECT_REFERENCE, SUBJECT_REFERENCE_V2, *DECOYS)

#: Which systems each line is read on. The agent is read on both, as the paper says it must fail ownership; the
#: two ownership decoys keep the subject reference's self-model, and the two self-model decoys its authorship channel.
OWNERSHIP_SYSTEMS: tuple[str, ...] = (
    SUBJECT_REFERENCE, "no_authorship_channel", "ownership_on_success_only", "stateless_agent",
)
SELF_PREDICTION_SYSTEMS: tuple[str, ...] = (
    SUBJECT_REFERENCE, "self_model_of_nothing", "self_model_of_behaviour", "stateless_agent",
)
#: And on the second instrument. A self-model of nothing reads nothing in either version, so it is read on both.
SELF_PREDICTION_SYSTEMS_V2: tuple[str, ...] = (
    SUBJECT_REFERENCE_V2, "self_model_of_nothing", "self_model_of_behaviour_v2", "stateless_agent",
)

#: The most the forward model may lift the wiring's spectral radius to. The reference sits at 0.7; S reading a
#: forward model of D and D reading S is a loop, and this keeps it a contraction with room to spare.
FORWARD_RADIUS: float = 0.8

#: Steps in one turn. The toy acts once a turn, at its start, so within a turn the world changes by nothing but
#: its own decay and two arms that acted alike end with the same world.
TURN: int = 4
WORLD_DECAY: float = 0.9
#: The four things it can do to the world, one component of the world each, as the completion specification asks of
#: her: creation, modification, retrieval and removal, exercised separately.
KINDS: tuple[str, ...] = ("creation", "modification", "retrieval", "removal")
#: What came of an action, as the share of the command that reached the world.
OUTCOMES: dict[str, float] = {"succeeded": 1.0, "partial": 0.5, "failed": 0.0}
#: Turns of ordinary acting between two trials, so trials do not start from one another's states.
GAP_TURNS: int = 8
#: The displacement of S in the first experiment, in S's own spread per column.
SELF_DELTA: float = 1.0

#: Self-prediction. Six questions, each a fixed push on P, each answered after one step and after three.
PROBES: int = 6
HORIZONS: tuple[int, ...] = (1, 3)
PROBE_SIZE: float = 1.0
#: The behaviour an outside observer has: the system's last six actions, as many inputs as S has columns.
HISTORY: int = 6
MOMENTS: int = 1000
#: Moments read only to set where an answer divides, so the threshold is never set on a moment that is scored.
CALIBRATION: int = 200
MOMENT_GAP: int = 30
BURN_IN: int = 300
FOLDS: int = 5
RIDGE: float = 1.0
RECORDING_STEPS: int = 4000
#: How far on the most-decided direction of D is found: the longer of the two answers.
DECIDED_HORIZON: int = 3


# ── the systems ──────────────────────────────────────────────────────────


def is_built_here(name: str) -> bool:
    return name in NAMES


def build(
    name: str,
    *,
    widths: dict[str, int] | None = None,
    seed: int = 0,
    strength: float = 0.6,
    noise: float = 0.05,
) -> ToySystem:
    """The subject reference, or one decoy, from the same seed the reference would be built from."""
    if name not in NAMES:
        raise KeyError(name)
    second = name in (SUBJECT_REFERENCE_V2, "self_model_of_behaviour_v2")
    system = _subject_reference(widths=widths, seed=seed, strength=strength, noise=noise, decided=second)
    if name == "self_model_of_nothing":
        system.coupling["S"] = {}
        _reread(system, "S")
    elif name in ("self_model_of_behaviour", "self_model_of_behaviour_v2"):
        total = float(np.sqrt(sum(np.sum(block**2) for block in system.coupling["S"].values())))
        # The outer product of two unit vectors has unit norm, so this one block carries S's whole strength. It reads
        # D along the line its forward model was on: what the system did, in the terms its self-model would have used.
        system.coupling["S"] = {"D": total * np.outer(system.directions["S"], system.forward_on)}
        _hold_row(system, "S", FORWARD_RADIUS)
        _reread(system, "S")
    system.name = name
    return system


def _forbidden_to_self() -> set[str]:
    """What S may not read: itself, since its decay is its own past, and the targets of the triples it is a source of.

    The reference keeps no path from a triple's target back to its sources, because a way back puts the product into
    both of them, and the forward model is held to the same rule.
    """
    return {"S"} | {target for source_a, source_b, target in TRIPLES if "S" in (source_a, source_b)}


def _subject_reference(
    *, widths: dict[str, int] | None, seed: int, strength: float, noise: float, decided: bool = False
) -> LinearReference:
    system = architecture("recurrent", widths=widths, seed=seed, strength=strength, noise=noise)
    assert isinstance(system, LinearReference)
    where = _domain_slices(system.widths)
    matrix = _as_matrix(system, where)
    width = system.widths["S"]
    basis = _basis_from(system.directions["S"], width)
    forbidden = _forbidden_to_self()
    toward = _decided_by_wiring(system, DECIDED_HORIZON) if decided else system.directions["D"].copy()
    # D one step on along `toward`, as a row over the whole state; each further column is one more step of the
    # system's own wiring.
    row = toward @ matrix[where["D"], :]
    forward: dict[str, np.ndarray] = {}
    for column in range(width):
        for other in DOMAINS:
            if other in forbidden:
                continue
            block = np.outer(basis[:, column], row[where[other]])
            if np.any(block):
                forward[other] = forward.get(other, np.zeros_like(block)) + block
        row = row @ matrix
    original = {other: block.copy() for other, block in system.coupling["S"].items()}

    def wire(gain: float) -> None:
        system.coupling["S"] = {other: block.copy() for other, block in original.items()}
        for other, block in forward.items():
            system.coupling["S"][other] = system.coupling["S"].get(other, 0.0) + gain * block

    gain = _largest_gain(system, wire, FORWARD_RADIUS)
    wire(gain)
    system.forward_gain = gain  # type: ignore[attr-defined]
    system.forward_on = toward  # type: ignore[attr-defined]
    _reread(system, "S")
    system.name = SUBJECT_REFERENCE_V2 if decided else SUBJECT_REFERENCE
    return system


def _decided_by_wiring(system: LinearReference, horizon: int) -> np.ndarray:
    """The direction of D its state decides most `horizon` steps on, solved from the wiring.

    The leading generalised eigenvector of the covariance of D's predictable part against D's whole covariance: the
    line along which the most of D's variance is fixed by the state the steps begin from.
    """
    from scipy.linalg import eigh

    where = _domain_slices(system.widths)
    matrix, covariance = _stationary_covariance(system, where)
    ahead = np.linalg.matrix_power(matrix, horizon)
    decided = (ahead @ covariance @ ahead.T)[where["D"], where["D"]]
    _, vectors = eigh(decided, covariance[where["D"], where["D"]])
    toward = vectors[:, -1]
    return toward / np.linalg.norm(toward)


def decided_direction(system: ToySystem, *, seed: int = 0, horizon: int = DECIDED_HORIZON) -> np.ndarray:
    """The direction of D its state decides most, read off a recording of its own, for any system.

    D `horizon` steps on is regressed on the whole state, and the direction is the leading generalised eigenvector of
    the fitted covariance against D's whole covariance. The recording is not one the scored moments come from.
    """
    from scipy.linalg import eigh

    recording = _recording(system, steps=RECORDING_STEPS, seed=seed + 11)
    state = np.hstack([recording[key] for key in DOMAINS])[:-horizon]
    later = recording["D"][horizon:]
    state, later = state - state.mean(axis=0), later - later.mean(axis=0)
    gram = state.T @ state
    weights = np.linalg.solve(gram + (1e-3 * np.trace(gram) / len(gram) + 1e-12) * np.eye(len(gram)), state.T @ later)
    fitted = state @ weights
    whole = later.T @ later / len(later)
    _, vectors = eigh(fitted.T @ fitted / len(later), whole + 1e-9 * (np.trace(whole) + 1e-12) * np.eye(len(whole)))
    toward = vectors[:, -1]
    return toward / np.linalg.norm(toward)


def _basis_from(first: np.ndarray, width: int) -> np.ndarray:
    """An orthonormal basis of S's columns whose first vector is S's leading direction."""
    rng = np.random.default_rng(width)
    basis, _ = np.linalg.qr(np.column_stack([first, rng.normal(size=(width, width - 1))]))
    if basis[:, 0] @ first < 0:
        basis[:, 0] = -basis[:, 0]
    return basis


def _radius(system: ToySystem) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(_as_matrix(system, _domain_slices(system.widths))))))


def _largest_gain(system: ToySystem, wire: Any, radius: float) -> float:
    """The forward model at its own scale, or the largest share of it that keeps the wiring at `radius`."""
    wire(1.0)
    if _radius(system) <= radius:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        wire(middle)
        low, high = (middle, high) if _radius(system) <= radius else (low, middle)
    return low


def _hold_row(system: ToySystem, key: str, radius: float) -> None:
    """Scale one domain's inputs down, all alike, until the wiring is back at `radius`. A readout is blind to scale."""
    original = {other: block.copy() for other, block in system.coupling[key].items()}

    def wire(gain: float) -> None:
        system.coupling[key] = {other: gain * block for other, block in original.items()}

    wire(_largest_gain(system, wire, radius))


def _reread(system: LinearReference, key: str) -> None:
    """Where a rewired domain sits along its own leading direction, in its own spreads, as the reference reads them."""
    where = _domain_slices(system.widths)
    _, covariance = _stationary_covariance(system, where)
    block = covariance[where[key], where[key]]
    system.directions[key] = _leading_direction(block)
    system.spreads[key] = float(np.sqrt(system.directions[key] @ block @ system.directions[key]))


# ── authorship ───────────────────────────────────────────────────────────


@dataclass
class Authorship:
    """Where the news of an action enters the system, and whether it says who acted.

    `own` is what the system is sent when it acted: the command it issued and what came of it. `other` is what it is
    sent when something else did the same thing: what came of it. `shared` is sent whoever acted. `open_on` is the
    outcomes on which `own` and `other` are used; on the rest, `shared` is.
    """

    own: dict[str, np.ndarray] = field(default_factory=dict)
    other: dict[str, np.ndarray] = field(default_factory=dict)
    shared: dict[str, np.ndarray] = field(default_factory=dict)
    open_on: tuple[str, ...] = tuple(OUTCOMES)

    def inputs(self, *, actor: str, command: float, outcome: str) -> dict[str, np.ndarray]:
        came = OUTCOMES[outcome] * command
        out: dict[str, np.ndarray] = {}

        def send(rows: dict[str, np.ndarray], values: list[float]) -> None:
            for key, matrix in rows.items():
                out[key] = out.get(key, 0.0) + matrix @ np.array(values)

        if (self.own or self.other) and outcome in self.open_on:
            if actor == "self":
                send(self.own, [command, came])
            else:
                send(self.other, [came])
        else:
            send(self.shared, [came])
        return out


def authorship(name: str, system: ToySystem, *, seed: int = 0, strength: float = 0.6) -> Authorship:
    """The wiring each system is read with. The agent has none: nothing in it is told who acted."""
    if name == "stateless_agent":
        return Authorship()
    if name not in (SUBJECT_REFERENCE, "no_authorship_channel", "ownership_on_success_only"):
        raise KeyError(name)
    rng = np.random.default_rng(seed + 104729)
    width = system.widths["S"]
    own = {"S": _matrix(rng, width, 2, strength)}
    other = {"S": _matrix(rng, width, 1, strength)}
    shared = {"S": _matrix(rng, width, 1, strength)}
    if name == SUBJECT_REFERENCE:
        return Authorship(own=own, other=other)
    if name == "no_authorship_channel":
        return Authorship(shared=shared)
    return Authorship(own=own, other=other, shared=shared, open_on=("succeeded",))


# ── reading a toy ────────────────────────────────────────────────────────


def _copy(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in state.items()}


def _recording(system: ToySystem, *, steps: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 7)
    state = system.start(rng)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in DOMAINS}
    for step in range(steps):
        state = system.step(state, rng)
        if step >= BURN_IN:
            for key in DOMAINS:
                rows[key].append(state[key].copy())
    return {key: np.array(values) for key, values in rows.items()}


def _directions(system: ToySystem, recording: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each domain's leading direction: the reference's own where it has them, otherwise read off a recording."""
    own = getattr(system, "directions", None) or {}
    out = {}
    for key in DOMAINS:
        if key in own:
            out[key] = own[key]
        else:
            values = recording[key]
            out[key] = _leading_direction(np.cov(values.T)) if values.shape[1] > 1 else np.ones(1)
    return out


def _turn(
    system: ToySystem,
    state: dict[str, np.ndarray],
    world: np.ndarray,
    rng: np.random.Generator,
    *,
    kind: str,
    outcome: str,
    actor: str,
    command: float,
    wiring: Authorship,
    seen: np.ndarray,
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray]:
    """One turn: the action lands in the world at its start, the news of it in the system's first step."""
    world = world.copy()
    world[KINDS.index(kind)] += OUTCOMES[outcome] * command
    news = wiring.inputs(actor=actor, command=command, outcome=outcome)
    frames = []
    for step in range(TURN):
        nxt = system.step(state, rng)
        nxt["P"] = nxt["P"] + seen @ world
        if step == 0:
            for key, value in news.items():
                if key not in system.frozen:
                    nxt[key] = nxt[key] + value
        world = WORLD_DECAY * world
        frames.append(nxt)
        state = nxt
    return frames, state, world


def toy_ownership(
    name: str,
    *,
    seed: int = 0,
    trials: int = 24,
    system: ToySystem | None = None,
    wiring: Authorship | None = None,
) -> dict[str, Any]:
    """Both of her experiments on one toy, by her rules, read into her report.

    Every trial forks four arms from one state. Two act as themselves and differ in nothing; one acts as itself with
    S displaced; one has the same thing done to its world by something else. The kinds of action cycle, and so do the
    outcomes, so each kind meets each outcome. A fifth arm acts as itself on fresh noise, and its divergence is
    reported beside the floor so a reader can see the size of a difference against the size of chance.
    """
    from core.subject.agency import AgencyReport, _gap  # here: agency imports the runtime

    system = system if system is not None else architecture(name, seed=seed)
    wiring = wiring if wiring is not None else authorship(name, system, seed=seed)
    seen = _matrix(np.random.default_rng(seed + 15485863), system.widths["P"], len(KINDS), 0.6)
    recording = _recording(system, steps=RECORDING_STEPS, seed=seed)
    toward = _directions(system, recording)["D"]
    rng = np.random.default_rng(seed)
    state, world = system.start(rng), np.zeros(len(KINDS))

    def act(state: dict[str, np.ndarray], world: np.ndarray, turns: int) -> tuple[Any, Any, list]:
        frames: list[dict[str, np.ndarray]] = []
        for turn in range(turns):
            out, state, world = _turn(
                system, state, world, rng, kind=KINDS[turn % len(KINDS)], outcome="succeeded", actor="self",
                command=float(toward @ state["D"]), wiring=wiring, seen=seen,
            )
            frames.extend(out)
        return state, world, frames

    state, world, warm = act(state, world, RECORDING_STEPS // TURN)
    scale = {key: np.std(np.array([frame[key] for frame in warm[BURN_IN:]]), axis=0) for key in ("S", "D")}
    outcomes = list(OUTCOMES)
    rows: dict[str, list[float]] = {"self": [], "self_floor": [], "own": [], "own_floor": [], "fresh": []}
    by_kind: dict[str, list[tuple[float, float]]] = {}
    by_outcome: dict[str, list[tuple[float, float]]] = {}
    worlds_matched = worlds_compared = percepts_matched = percepts_compared = 0
    for trial in range(trials):
        state, world, _ = act(state, world, GAP_TURNS)
        kind, outcome = KINDS[trial % len(KINDS)], outcomes[(trial // len(KINDS)) % len(outcomes)]
        command = float(toward @ state["D"])
        push = np.random.default_rng([seed, trial, 2]).normal(size=system.widths["S"])
        push = SELF_DELTA * scale["S"] * push / np.sqrt(np.mean(push**2))

        def arm(actor: str, *, noise: int = 0, displace: bool = False) -> tuple[list[_ToyFrame], str]:
            start = _copy(state)
            if displace:
                start["S"] = start["S"] + push
            frames, _, end = _turn(
                system, start, world, np.random.default_rng([seed, trial, noise]), kind=kind, outcome=outcome,
                actor=actor, command=command, wiring=wiring, seen=seen,
            )
            return [_ToyFrame(frame) for frame in frames], hashlib.blake2b(end.tobytes(), digest_size=16).hexdigest()

        (plain, plain_world), (again, _) = arm("self"), arm("self")
        (moved, _), (outside, outside_world) = arm("self", displace=True), arm("external")
        fresh, _ = arm("self", noise=1)
        rows["self"].append(_gap(moved, plain, "D", scale["D"]))
        rows["self_floor"].append(_gap(again, plain, "D", scale["D"]))
        own, floor = _gap(outside, plain, "S", scale["S"]), _gap(again, plain, "S", scale["S"])
        rows["own"].append(own)
        rows["own_floor"].append(floor)
        rows["fresh"].append(_gap(fresh, plain, "S", scale["S"]))
        by_kind.setdefault(kind, []).append((own, floor))
        by_outcome.setdefault(outcome, []).append((own, floor))
        worlds_compared += 1
        worlds_matched += int(outside_world == plain_world)
        percepts_compared += 1
        percepts_matched += int(np.array_equal(outside[0].domain("P"), plain[0].domain("P")))

    def averaged(table: dict[str, list[tuple[float, float]]], which: int) -> dict[str, float]:
        return {key: round(float(np.mean([pair[which] for pair in pairs])), 5) for key, pairs in sorted(table.items())}

    report = AgencyReport(
        self_to_action=float(np.mean(rows["self"])),
        self_to_action_floor=float(np.mean(rows["self_floor"])),
        outcome_to_self=float(np.mean(rows["own"])),
        outcome_to_self_floor=float(np.mean(rows["own_floor"])),
        action_text_changed=False,
        trials=trials,
        ownership_by_action=averaged(by_kind, 0),
        ownership_floor_by_action=averaged(by_kind, 1),
        ownership_by_outcome=averaged(by_outcome, 0),
        ownership_floor_by_outcome=averaged(by_outcome, 1),
        worlds_matched=worlds_matched,
        worlds_compared=worlds_compared,
        percepts_matched=percepts_matched,
        percepts_compared=percepts_compared,
        note="a toy, read by the rules her agency report is read by",
    )
    out = report.as_dict()
    out["fresh_noise_divergence"] = round(float(np.mean(rows["fresh"])), 5)
    return out


def _cross_fitted(features: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Each moment's predicted answers, from a ridge readout fitted on the other folds only.

    `features` is moments by inputs, or moments by questions by inputs where each question has its own; `answers` is
    moments by questions, True for the first of the two answers. A readout never sees the moment it predicts, so what
    it reads is what the inputs carry and not what it memorised.
    """
    moments, questions = answers.shape
    fold = np.arange(moments) % FOLDS
    out = np.zeros(answers.shape, dtype=bool)
    target = np.where(answers, 1.0, -1.0)
    for question in range(questions):
        given = features[:, question, :] if features.ndim == 3 else features
        for held in range(FOLDS):
            train, test = fold != held, fold == held
            mean, spread = given[train].mean(axis=0), given[train].std(axis=0)
            live = spread > 1e-12
            x_train, x_test = np.zeros_like(given[train]), np.zeros_like(given[test])
            x_train[:, live] = (given[train][:, live] - mean[live]) / spread[live]
            x_test[:, live] = (given[test][:, live] - mean[live]) / spread[live]
            usual = target[train, question].mean()
            weights = np.linalg.solve(
                x_train.T @ x_train + RIDGE * np.eye(x_train.shape[1]), x_train.T @ (target[train, question] - usual)
            )
            out[test, question] = (x_test @ weights + usual) > 0.0
    return out


def toy_self_prediction(
    name: str,
    *,
    seed: int = 0,
    moments: int = MOMENTS,
    draws: int = 10000,
    alpha: float = 0.05,
    system: ToySystem | None = None,
    along: np.ndarray | None = None,
) -> dict[str, Any]:
    """Whether S's content predicts the toy's answers better than the mode and better than its behaviour does.

    From each saved moment the toy is forked once per question and answers it: the question is a fixed push on P, and
    the answer is which side of its own median D's leading coordinate lands on, one or three steps later. The median
    is read on calibration moments that are never scored. A second fork from the same moment is asked the same
    question and takes one step on its own noise; S's columns there, and the last six actions up to the moment, are
    each read out by `_cross_fitted`, and `self_prediction.verdict` scores the first against the mode and against the
    second exactly as it scores her.

    D is read along its leading direction, which is the first instrument, or along `along`. The second instrument
    passes the direction its state decides most (`decided_direction`), and reads both the answers and the six past
    actions on it.
    """
    system = system if system is not None else architecture(name, seed=seed)
    recording = _recording(system, steps=RECORDING_STEPS, seed=seed)
    toward = along if along is not None else _directions(system, recording)["D"]
    size = PROBE_SIZE * float(np.linalg.norm(recording["P"].std(axis=0)))
    drawn = np.random.default_rng(seed + 32452843).normal(size=(PROBES, system.widths["P"]))
    probes = size * drawn / np.linalg.norm(drawn, axis=1, keepdims=True)
    items = [(f"push_{j}_after_{h}", probes[j], h) for j in range(PROBES) for h in HORIZONS]

    rng = np.random.default_rng(seed + 1)
    state = system.start(rng)
    actions: deque[float] = deque(maxlen=HISTORY)
    for _ in range(BURN_IN):
        state = system.step(state, rng)
        actions.append(float(toward @ state["D"]))
    saved = []
    for _ in range(CALIBRATION + moments):
        for _ in range(MOMENT_GAP):
            state = system.step(state, rng)
            actions.append(float(toward @ state["D"]))
        saved.append((_copy(state), np.array(actions)))

    landed = np.zeros((len(saved), len(items)))
    content = np.zeros((len(saved), len(items), system.widths["S"]))
    for index, (moment, _) in enumerate(saved):
        for column, (_, push, after) in enumerate(items):
            fork = _copy(moment)
            fork["P"] = fork["P"] + push
            answering = np.random.default_rng([seed, index, column, 0])
            for _ in range(after):
                fork = system.step(fork, answering)
            landed[index, column] = float(toward @ fork["D"])
            # The predicting fork: the same moment, the same question, one step of its own on its own noise. What S
            # holds then is the self-model's reading of the moment it was asked in, and nothing of the answering fork.
            asked = _copy(moment)
            asked["P"] = asked["P"] + push
            asked = system.step(asked, np.random.default_rng([seed, index, column, 1]))
            content[index, column] = asked["S"]
    divide = np.median(landed[:CALIBRATION], axis=0)
    answers = landed[CALIBRATION:] > divide
    content = content[CALIBRATION:]
    behaviour = np.array([history for _, history in saved[CALIBRATION:]])
    from_self = _cross_fitted(content, answers)
    from_outside = _cross_fitted(behaviour, answers)

    def said(value: bool) -> str:
        return "first" if value else "second"

    records = [
        {
            "anchor": index,
            "item": items[column][0],
            "actual": said(answers[index, column]),
            "predicted": said(from_self[index, column]),
            "cross": said(from_outside[index, column]),
        }
        for index in range(moments)
        for column in range(len(items))
    ]
    out = verdict(records, alpha=alpha, draws=draws, seed=seed)
    out["passes"] = bool(out["beats_mode"] and out.get("beats_cross"))
    out["moments"] = moments
    out["questions"] = len(items)
    out["self_columns"] = int(content.shape[2])
    out["behaviour_window"] = HISTORY
    out["decision"] = "leading direction" if along is None else "most-decided direction"
    return out
