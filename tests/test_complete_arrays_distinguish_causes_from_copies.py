"""Run the actual paired-roll comparison on known causal and passive worlds."""
import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.declared_observations import ObservationError, ObservationPlan
from core.subject.sketch import _directions, sketch
from tools.run_interventional_closure import compare_swap, hybrid_snapshot, joint_snapshot


@dataclass
class Snapshot:
    services: dict
    organs: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    module_state: dict = field(default_factory=dict)


def known_world(causal, digest=True):
    size = 16
    constraints = np.vstack([np.ones(size - 2), _directions(size)[:, 1:-1]])
    _u, singular, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int((singular > 1e-12).sum())
    states = np.zeros((2, size))
    states[:, 0], states[:, -1] = -3., 3.
    states[:, 1:-1] = vt[rank:rank + 2]
    plan = ObservationPlan({"version": 1, "max_bytes": 10000, "arrays": [
        {"unit": "organ:substrate", "path": ["x"], "shape": [size], "mode": "exact"},
    ]}) if digest else None

    async def roll(snapshot, index):
        choice = snapshot.services["a"]["value"] * snapshot.services["b"]["value"] if causal else 0
        values = states[int(choice)]
        held = {"held.x." + k: v for k, v in sketch(values).items()}
        if plan:
            held.update(plan.digest_units({"organ:substrate": SimpleNamespace(x=values)}))
        # The core's own observed summary is deliberately unchanged; this
        # experiment asks whether the held array changed outside that summary.
        return np.zeros((2, 1)), held

    receiver = Snapshot({"a": {"value": 0}, "b": {"value": 0}, "passive": {"value": 0}})
    donor = Snapshot({"a": {"value": 1}, "b": {"value": 1}, "passive": {"value": 1}})
    return roll, plan, receiver, donor


def read_swap(causal, units, digest=True):
    roll, plan, receiver, donor = known_world(causal, digest)
    swapped = joint_snapshot(receiver, donor, units)

    async def measure():
        baseline = await roll(receiver, 0)
        # Ignore tiny floating sketch roundoff, as a production flat-column
        # threshold does; the digest must still detect the full-array change.
        comparable = {k for k in baseline[1] if k.startswith("held.array_sha256_")}
        if not digest:
            comparable = set()
        one_offs = []
        result = await compare_swap(roll, receiver, swapped, 0, baseline, comparable,
                                    label="known world", one_offs=one_offs, observations=plan)
        return result, one_offs

    return asyncio.run(measure())


def test_a_joint_outside_cause_moves_complete_held_state_although_neither_single_swap_does():
    assert not read_swap(True, [("a", None)])[0][0]
    assert not read_swap(True, [("b", None)])[0][0]
    result, one_offs = read_swap(True, [("a", None), ("b", None)])
    assert result[0] and result[1] == 0. and result[2] and result[3] == 0.
    assert not one_offs


def test_a_passive_copy_remains_causally_inert_under_the_same_comparison():
    result, one_offs = read_swap(False, [("a", None), ("b", None), ("passive", None)])
    assert result == (False, 0., [], 0.) and not one_offs


def test_a_restore_residue_hidden_from_the_core_is_not_counted_as_an_effect():
    roll, plan, receiver, donor = known_world(True)
    swapped = hybrid_snapshot(receiver, donor, "a", None)

    async def measure():
        baseline = await roll(receiver, 0)
        async def contaminated(snapshot, index):
            x, h = await roll(snapshot, index)
            return x, {k: "stale" if k.startswith("held.array_sha256_") else v for k, v in h.items()}
        await compare_swap(contaminated, receiver, swapped, 0, baseline, set(baseline[1]),
                           label="residue", one_offs=[], observations=plan)

    with pytest.raises(ObservationError, match="not comparable"):
        asyncio.run(measure())


def test_a_one_off_effect_stays_a_one_off_in_the_existing_comparison():
    calls = 0
    async def roll(snapshot, index):
        nonlocal calls
        calls += 1
        return np.array([[1. if calls == 2 else 0.]]), {}
    one_offs = []
    result = asyncio.run(compare_swap(roll, "receiver", "swapped", 0, (np.zeros((1, 1)), {}), set(),
                                     label="one off", one_offs=one_offs))
    assert result == (False, 0., [], 0.)
    assert len(one_offs) == 1


def test_the_interventional_entrypoint_records_a_joint_array_cause_and_its_horizon(tmp_path, monkeypatch):
    import json

    import core.subject.carrier_reads as carrier
    import core.subject.closure as closure
    import core.subject.driver as driver
    import core.subject.isolation as isolation
    import core.subject.recording as recording
    import core.subject.v25_runtime as v25
    import tools.run_interventional_closure as tool
    from core.subject.state import Organs

    receiver = Snapshot({"a": {"value": 0}, "b": {"value": 0}})
    donor = Snapshot({"a": {"value": 1}, "b": {"value": 1}})
    substrate = SimpleNamespace(x=np.zeros(16))
    runtime = SimpleNamespace(kernel=SimpleNamespace(_phases=[]), organs=Organs(substrate=substrate),
                              freeze_host=lambda: None)
    runtime.restore = lambda snapshot: setattr(runtime, "snapshot_used", snapshot)
    horizons = []

    async def nothing(*args, **kwargs):
        return None
    async def bank(*args, **kwargs):
        return [SimpleNamespace(snapshot=receiver), SimpleNamespace(snapshot=donor)]
    async def turns(runtime, condition, horizon):
        horizons.append(horizon)
        s = runtime.snapshot_used.services
        # A delayed AND cause, invisible to the historical two-turn horizon.
        substrate.x = np.full(16, float(s["a"]["value"] * s["b"]["value"]) if horizon >= 3 else 0.)
        return [object()]

    monkeypatch.setattr(tool, "WARM_ROUNDS", 0)
    monkeypatch.setattr(driver, "build_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(driver, "start_organism", nothing)
    monkeypatch.setattr(driver, "quiesce_organism", nothing)
    monkeypatch.setattr(driver, "calibrate_clock", nothing)
    monkeypatch.setattr(isolation, "isolate_state", lambda *a: None)
    monkeypatch.setattr(v25, "collect_anchor_bank", bank)
    monkeypatch.setattr(v25, "_run_turns", turns)
    monkeypatch.setattr(recording, "build_recording", lambda *a: SimpleNamespace(x=np.zeros((2, 1))))
    monkeypatch.setattr(carrier, "read_held", lambda *a, **k: {})
    monkeypatch.setattr(carrier, "HELD", ())
    monkeypatch.setattr(closure, "_core_attributes", lambda: {})
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "max_bytes": 10000, "arrays": [
        {"unit": "organ:substrate", "path": ["x"], "shape": [16], "mode": "exact"},
    ]}))
    output = tmp_path / "result"
    assert asyncio.run(tool.main(["--seed", "73", "--anchors", "2", "--turns", "3",
                                  "--joint-periphery", "--observation-plan", str(plan),
                                  "--out", str(output)])) == 0
    result = json.loads((output / "interventional_closure.json").read_text())
    assert result["verdict"] == "OPEN"
    assert "joint-periphery" in result["moves_k_star"]
    assert result["turns_per_roll"] == 3 and set(horizons) == {3}
    assert result["declared_array_observations"]["width"] == 16
    assert not result["residue"]
