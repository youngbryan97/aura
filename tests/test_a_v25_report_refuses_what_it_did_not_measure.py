"""A v25 report refuses to be authoritative about what it did not measure.

Each refusal here is one the ISC plan names: a layer that could not be stepped
the same way in every arm, a phase that raised while the run was measuring, a
periphery walk that stopped at its cap, a state reader that never read, a
history length the ladder could not resolve, and a predictive rank that does
not survive other folds or another estimator. Missing records refuse too.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool() -> Any:
    name = "run_subject_core_v25_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "run_subject_core_v25.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


ARGS = SimpleNamespace(allow_degraded=False)


def _clean() -> dict[str, Any]:
    return {
        "canonical_grain": {
            "predictive_rank": 2,
            "anchors": 16,
            "rank_is_at_the_ceiling": False,
            "heldout_intervention_sufficient": True,
            "history_status": "RESOLVED",
            "history_turns_needed": 4,
            "rank_stable": True,
            "fold_ranks": [2, 2, 2, 2, 2],
            "bicross_validated_rank": 2,
        },
        "spectrum": {"tau_status": "RESOLVED"},
        "representation_invariance": {
            "measured": True,
            "representation_invariant": True,
            "duplication_did_not_help": True,
        },
        "v25_nulls": {"measured": True, "playback_is_zero": True},
        "closure": {"closed": True, "coverage": {"hit_the_cap": False, "reader_failures": 0, "cap": 400}},
        "cuts": {},
        "conditions_used": 8,
        "conditions_available": 8,
        "recording": {"never_read": []},
        "runtime_health": {
            "raised_on_the_measured_path": {},
            "unsteppable_layers": {},
            "layer_failures": {},
        },
        "scope": "substrate_only",
    }


def _refused_for(tool: Any, evidence: dict[str, Any], words: str) -> None:
    authority = tool._authority(evidence, ARGS)
    assert not authority["authoritative"]
    assert any(words in blocker for blocker in authority["blockers"]), authority["blockers"]


def test_a_clean_report_is_authoritative(tool: Any) -> None:
    authority = tool._authority(_clean(), ARGS)
    assert authority["authoritative"], authority["blockers"]


def test_a_layer_that_could_not_be_stepped_refuses(tool: Any) -> None:
    evidence = _clean()
    evidence["runtime_health"]["unsteppable_layers"] = {"NeuralMesh": "NeuralMesh has no _tick"}
    _refused_for(tool, evidence, "could not be stepped")


def test_a_phase_that_raised_while_measuring_refuses(tool: Any) -> None:
    evidence = _clean()
    evidence["runtime_health"]["raised_on_the_measured_path"] = {"ConversationalDynamicsPhase": 264}
    _refused_for(tool, evidence, "a phase raised")


def test_a_periphery_walk_that_hit_its_cap_refuses(tool: Any) -> None:
    evidence = _clean()
    evidence["closure"]["coverage"]["hit_the_cap"] = True
    _refused_for(tool, evidence, "stopped at its cap")


def test_a_reader_that_never_read_refuses(tool: Any) -> None:
    evidence = _clean()
    evidence["recording"]["never_read"] = ["organ:world_model"]
    _refused_for(tool, evidence, "missed on every frame")


def test_an_unresolved_history_refuses(tool: Any) -> None:
    evidence = _clean()
    evidence["canonical_grain"]["history_status"] = "UNRESOLVED"
    _refused_for(tool, evidence, "history length is not resolved")


def test_an_unstable_rank_refuses(tool: Any) -> None:
    evidence = _clean()
    evidence["canonical_grain"]["rank_stable"] = False
    evidence["canonical_grain"]["fold_ranks"] = [2, 1, 2, 2, 2]
    _refused_for(tool, evidence, "does not survive other folds")


@pytest.mark.parametrize(
    ("remove", "words"),
    [
        (("runtime_health",), "was not recorded"),
        (("closure", "coverage"), "was not recorded"),
        (("recording", "never_read"), "was not recorded"),
    ],
)
def test_a_missing_record_refuses(tool: Any, remove: tuple[str, ...], words: str) -> None:
    evidence = copy.deepcopy(_clean())
    holder = evidence
    for key in remove[:-1]:
        holder = holder[key]
    del holder[remove[-1]]
    _refused_for(tool, evidence, words)


def test_only_what_raised_after_measuring_began_counts(tool: Any) -> None:
    runtime = SimpleNamespace(
        failures={"MemoryRetrievalPhase": 3, "ConversationalDynamicsPhase": 5},
        failure_notes={"ConversationalDynamicsPhase": "ValueError: math domain error"},
        layer_steps=None,
        organism=None,
    )
    health = tool._runtime_health(runtime, {"MemoryRetrievalPhase": 3, "ConversationalDynamicsPhase": 1})
    assert health["raised_on_the_measured_path"] == {"ConversationalDynamicsPhase": 4}
    assert health["failure_notes"] == {"ConversationalDynamicsPhase": "ValueError: math domain error"}


def _histories(lag: int, *, rows: int = 120, per_turn: int = 3, turns: int = 16, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Histories whose future depends on the state `lag` turns back and nothing else."""
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(rows, turns * per_turn))
    start = (turns - lag) * per_turn
    source = history[:, start : start + per_turn]
    weights = rng.normal(size=(per_turn, 4))
    future = source @ weights + 0.05 * rng.normal(size=(rows, 4))
    return history, future


@pytest.mark.parametrize(("lag", "needed"), [(1, 1), (3, 4), (6, 8)])
def test_the_ladder_stops_at_the_history_the_future_needs(tool: Any, lag: int, needed: int) -> None:
    history, future = _histories(lag)
    walk = tool._history_walk(history, future, history_turns=16, seed=3)
    assert walk["history_status"] == "RESOLVED", walk
    assert walk["history_turns_needed"] == needed, walk


def test_history_older_than_the_ladder_reaches_is_unresolved(tool: Any) -> None:
    history, future = _histories(12)
    walk = tool._history_walk(history, future, history_turns=16, seed=3)
    assert walk["history_status"] == "UNRESOLVED", walk
    assert walk["history_turns_needed"] is None


def _low_rank(rank: int, *, rows: int = 40, columns: int = 60, noise: float = 0.1, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if rank == 0:
        return rng.normal(size=(rows, columns))
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(rank, columns))
    strengths = np.linspace(6.0, 3.0, rank)
    return (left * strengths) @ right + noise * rng.normal(size=(rows, columns))


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_bicross_validation_finds_a_rank_that_is_there(rank: int) -> None:
    from core.subject.intrinsic_v25 import bicross_validated_rank

    assert bicross_validated_rank(_low_rank(rank)) == rank


def test_bicross_validation_finds_nothing_in_noise() -> None:
    from core.subject.intrinsic_v25 import bicross_validated_rank

    assert bicross_validated_rank(_low_rank(0)) == 0


def test_a_rank_both_estimators_and_every_fold_agree_on_is_stable(tool: Any) -> None:
    from core.subject.intrinsic_v25 import fit_predictive_grain

    signatures = _low_rank(2)
    rank = fit_predictive_grain(signatures).rank
    assert rank == 2
    assert tool._rank_stability(signatures, rank, seed=4)["rank_stable"]


def test_a_rank_the_data_does_not_hold_is_unstable(tool: Any) -> None:
    stability = tool._rank_stability(_low_rank(2), 3, seed=4)
    assert not stability["rank_stable"], stability
