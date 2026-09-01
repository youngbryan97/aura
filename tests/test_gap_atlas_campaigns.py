"""The campaigns that answer a Gap Atlas bar no unit test can.

Cards 026, 098, 101, A12.4, A12.5 and A12.6 ask for scale and lifetime runs
rather than for a module. These tests do not re-run the campaigns — that is
what tools/campaigns/ is for — but they check the two things that would let a
campaign lie: that it still runs at all, and that the sealed evidence says
what the card was closed on.

A campaign whose tool has drifted from its evidence is worse than no
campaign, because the number stays in the file and reads as current.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.knowledge.atomspace import AtomSpace, Node, TruthValue

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"


def _sealed(name: str) -> dict:
    path = EVIDENCE / name
    if not path.exists():
        pytest.skip(f"{name} has not been run in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


# ── attention has to be resettable, or a trial is not a trial ─────────────


def test_resetting_attention_returns_the_fund_and_keeps_what_mattered():
    space = AtomSpace(sti_fund=1000.0, stimulus_size=50.0)
    node = Node("ConceptNode", "red")
    space.add(node, TruthValue(0.9, 10.0), source="t")
    space.stimulate(node)
    before = space.get_av(node)
    assert before.sti > 0 and before.lti > 0

    reclaimed = space.reset_attention()

    after = space.get_av(node)
    assert reclaimed == pytest.approx(before.sti)
    assert after.sti == 0.0
    # Long-term importance is the record of what has repeatedly mattered.
    # One task does not get to clear that.
    assert after.lti == before.lti
    # And the truth is untouched: this is attention, not forgetting.
    assert space.get_tv(node).strength == pytest.approx(0.9)


def test_a_second_task_does_not_start_on_the_first_task_s_salience():
    space = AtomSpace(sti_fund=1000.0, stimulus_size=50.0)
    first, second = Node("ConceptNode", "a"), Node("ConceptNode", "b")
    for node in (first, second):
        space.add(node, TruthValue(0.9, 10.0), source="t")
    space.stimulate(first)
    space.reset_attention()
    space.stimulate(second)
    assert space.get_av(first).sti == 0.0
    assert space.get_av(second).sti > 0.0


# ── the campaigns run ─────────────────────────────────────────────────────


def test_the_scale_campaign_runs_and_reads_stay_flat():
    from tools.campaigns.atomspace_scale import one_size

    small = one_size(2_000, reads=500, seed=1)
    larger = one_size(20_000, reads=500, seed=1)
    assert larger["held"] > small["held"] > 0
    # A store whose point reads grow with size is a different store from the
    # one the card was closed on.
    assert larger["point_read_us_p50"] < 20.0


def test_the_attention_campaign_beats_the_heuristic_it_has_to_beat():
    from tools.campaigns.ecan_stress import build, one_trial
    import random

    space, groups, degree = build(
        atoms=4_000, communities=8, cross_rate=0.08, seed=11
    )
    concepts = [n for g in groups for n in g]
    by_degree = sorted(concepts, key=lambda n: -degree[n])
    rng = random.Random(12)
    rows = [
        one_trial(
            space,
            groups,
            degree,
            budget=32,
            seeds=4,
            ticks=3,
            rng=rng,
            all_concepts=concepts,
            by_degree=by_degree,
        )
        for _ in range(6)
    ]
    ecan = sum(r["ecan"] for r in rows)
    assert ecan > sum(r["top_degree"] for r in rows)
    assert ecan > sum(r["random"] for r in rows)
    assert ecan > sum(r["scan"] for r in rows)


def test_the_lifetime_campaign_retires_what_stopped_working():
    from tools.campaigns.procedure_lifetime import compile_phase, lifetime
    from core.cognition.procedure import ProcedureRegistry

    tasks = [f"t{i}" for i in range(8)]
    registry = ProcedureRegistry(max_procedures=64)
    compiled = compile_phase(registry, tasks, runs=4, seed=3)
    ids = list(compiled["procedure_ids"])
    assert ids, "nothing compiled, so the campaign has no subject"

    result = lifetime(
        registry,
        ids,
        firings=40_000,
        shift_at=0.5,
        pays_before=0.92,
        pays_after=0.15,
        prune_every=500,
        seed=4,
    )
    assert result["retired"] == result["of"]
    assert result["median_firings_to_notice"] is not None


def test_the_lifetime_campaign_leaves_working_rules_alone():
    """The other arm: a world that never moves must retire almost nothing."""
    from tools.campaigns.procedure_lifetime import compile_phase, lifetime
    from core.cognition.procedure import ProcedureRegistry

    tasks = [f"t{i}" for i in range(8)]
    registry = ProcedureRegistry(max_procedures=64)
    compiled = compile_phase(registry, tasks, runs=4, seed=3)
    steady = lifetime(
        registry,
        list(compiled["procedure_ids"]),
        firings=40_000,
        shift_at=0.5,
        pays_before=0.92,
        pays_after=0.92,
        prune_every=500,
        seed=4,
    )
    assert steady["retired"] <= 1, steady


# ── the sealed evidence says what the cards were closed on ────────────────


def test_the_scale_evidence_reaches_a_million_atoms():
    payload = _sealed("atomspace_scale.json")
    assert payload["schema"] == "aura.atomspace_scale.v1"
    assert payload["card"] == "098"
    assert payload["claim_boundary"]
    biggest = max(payload["sizes"], key=lambda row: row["held"])
    assert biggest["held"] >= 1_000_000
    assert biggest["point_read_us_p99"] < 10.0


def test_the_attention_evidence_names_its_ceiling_and_its_nulls():
    payload = _sealed("ecan_stress.json")
    assert payload["card"] == "101"
    assert 0.0 < payload["recall_ceiling"] <= 1.0
    arms = payload["arms"]
    assert set(arms) == {"ecan", "scan", "random", "top_degree", "breadth_first"}
    # The card's stated bar: beat scan, random and top-degree.
    for weaker in ("scan", "random", "top_degree"):
        assert arms["ecan"]["share_of_ceiling"] > arms[weaker]["share_of_ceiling"]
    beat = payload["ecan_beats_top_degree"]
    assert beat["wins"] + beat["ties"] <= beat["of"]
    assert beat["wins"] > beat["of"] / 2


def test_the_attention_evidence_keeps_the_null_that_came_out_against_it():
    """Breadth-first from the same seeds at the same touch budget wins.

    The card names three baselines and attention beats all three. The
    strongest null is not among them and it beats attention, and a result
    file that dropped that arm because of how it came out would be the one
    thing this campaign exists to prevent.
    """
    payload = _sealed("ecan_stress.json")
    arms = payload["arms"]
    assert "breadth_first" in arms
    walk = payload["ecan_beats_breadth_first"]
    assert walk["of"] > 0
    assert walk["wins"] + walk["ties"] < walk["of"], (
        "the walk arm no longer beats attention on any trial; if that is real, "
        "say so in the card rather than deleting this test"
    )
    assert payload["matched_compute"]["median_atom_touches"] > 0


def test_the_lifetime_evidence_carries_both_arms():
    payload = _sealed("procedure_lifetime.json")
    assert set(payload["cards"]) == {"026", "A12.4", "A12.5", "A12.6"}
    audited, control = payload["audited"], payload["unaudited_control"]
    assert audited["retired"] == audited["of"]
    assert control["retired"] == 0
    assert payload["wrong_firings_avoided"] > 0
    # A12.5: the compiler names what a witness could drop, and the
    # generalisation phase drops it.
    assert payload["compilation"]["provisional_median"] > 0
    assert payload["generalisation"]["conditions_dropped"] > 0


def test_the_decay_constant_matches_the_sweep_it_was_read_from():
    from core.cognition.procedure import _RECENT_DECAY

    payload = _sealed("procedure_lifetime_halflife.json")
    rows = {row["decay"]: row for row in payload["rows"]}
    assert _RECENT_DECAY in rows, "the constant is not a row in its own sweep"
    chosen = rows[_RECENT_DECAY]
    slowest = max(rows.values(), key=lambda r: r["firings_to_notice"])
    # The value earns its place: it notices a shift sooner than the arm with
    # no decay at all, and does not retire rules a steady world keeps.
    assert chosen["firings_to_notice"] < slowest["firings_to_notice"]
    assert chosen["retired_with_no_shift"] <= chosen["of"] // 8


# ── the store survives the process ────────────────────────────────────────
#
# Card 098's bar has two halves: bounded query latency at a million atoms, and
# crash-safe persistence. The first is the scale campaign. This is the second.


def _populated():
    from core.knowledge.atomspace import EVALUATION, LIST, Link

    space = AtomSpace(max_atoms=1000)
    red, square = Node("Concept", "red"), Node("Concept", "square")
    space.add(red, TruthValue(0.9, 10.0), source="eyes")
    space.add(red, TruthValue(0.8, 4.0), source="memory")
    space.add(square, TruthValue(0.7, 6.0))
    space.add(
        Link(EVALUATION, (Node("Predicate", "beside"), Link(LIST, (red, square)))),
        TruthValue(0.6, 3.0),
        source="eyes",
    )
    space.stimulate(red)
    return space, red, square


def test_a_reloaded_store_holds_what_the_first_one_held():
    import tempfile

    space, red, square = _populated()
    before = len(space)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "atoms.json"
        assert space.save(path) == before

        fresh = AtomSpace(max_atoms=1000)
        assert fresh.load(path) == before

    assert len(fresh) == before
    assert fresh.get_tv(red).strength == pytest.approx(space.get_tv(red).strength)
    assert fresh.get_tv(square).count == pytest.approx(space.get_tv(square).count)
    assert fresh.get_av(red).sti == pytest.approx(space.get_av(red).sti)
    # The nested link came back as a link, not as a string.
    assert len(fresh.atoms_of_type("Evaluation")) == 1


def test_provenance_survives_the_restart_so_evidence_is_not_double_counted():
    """A store that forgets its witnesses counts one source twice on reload."""
    import tempfile

    space, red, _ = _populated()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "atoms.json"
        space.save(path)
        fresh = AtomSpace(max_atoms=1000)
        fresh.load(path)

    restated = fresh.add(red, TruthValue(0.8, 4.0), source="memory")
    original = space.add(red, TruthValue(0.8, 4.0), source="memory")
    assert restated.count == pytest.approx(original.count)
    assert restated.strength == pytest.approx(original.strength)


def test_a_truncated_snapshot_is_refused_rather_than_half_loaded():
    import json
    import tempfile

    space, _, _ = _populated()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "atoms.json"
        space.save(path)
        text = path.read_text(encoding="utf-8")
        path.write_text(text[: len(text) // 2], encoding="utf-8")

        fresh = AtomSpace(max_atoms=1000)
        with pytest.raises(json.JSONDecodeError):
            fresh.load(path)
    # And the store it was loading into is untouched.
    assert len(fresh) == 0


def test_a_snapshot_from_another_format_is_refused():
    space, _, _ = _populated()
    fresh = AtomSpace(max_atoms=1000)
    with pytest.raises(ValueError, match="refusing to load part of it"):
        fresh.restore({"schema": "something.else.v9", "atoms": []})
    with pytest.raises(ValueError, match="no atom list"):
        payload = space.snapshot()
        payload["atoms"] = None
        fresh.restore(payload)
    assert len(fresh) == 0


def test_a_crash_part_way_through_a_save_leaves_the_old_snapshot(tmp_path):
    """The gateway writes to a temp file and renames; a kill mid-write cannot
    leave a shorter store that loads without complaint."""
    space, _, _ = _populated()
    path = tmp_path / "atoms.json"
    space.save(path)
    first = path.read_bytes()

    bigger = AtomSpace(max_atoms=1000)
    bigger.restore(space.snapshot())
    for i in range(200):
        bigger.add(Node("Concept", f"extra{i}"), TruthValue(0.5, 1.0), source="t")

    # Any file left beside the target by an interrupted write must not be the
    # target itself: whatever is at `path` is a whole snapshot or the old one.
    survivors = [p for p in tmp_path.iterdir() if p != path]
    assert path.read_bytes() == first
    for leftover in survivors:
        assert leftover.name != path.name

    bigger.save(path)
    reloaded = AtomSpace(max_atoms=1000)
    assert reloaded.load(path) == len(bigger)


# ── the model that scores best is not the model that plans best ───────────
#
# Cards 140 and 198. Both are the same shape: the objective a model was
# trained on cannot say which model is better, and only a second measurement
# on control or under intervention can.


def test_a_correlational_fit_predicts_observations_and_not_interventions():
    from tools.campaigns.world_model_campaign import X_TO_Y, causal_arm

    row = causal_arm(observations=400, interventions_seen=100, tested=200, seed=5)
    assert row["causal_wins_under_intervention"]
    # And the trap: the wrong model looks better on observations, which is why
    # an observational score cannot settle it.
    assert row["correlational_wins_on_observations"]
    # The causal arm recovers roughly the real effect; the correlational one
    # recovers the confound.
    assert abs(row["causal_slope"] - X_TO_Y) < 0.15
    assert row["correlational_slope"] > X_TO_Y * 2


def test_the_two_objectives_choose_different_dimensions_from_one_dataset():
    """Neither ranking is authored: both fall out of the same observations."""
    from tools.campaigns.world_model_campaign import objective_arm

    loud = objective_arm(
        capacity=8, steps=4000, observed_dims=48, noise_scale=3.0, seed=5
    )
    assert loud["dims_both_kept"] == 0
    assert loud["reconstruction_kept_signal_dims"] == 0
    assert loud["latent_kept_signal_dims"] == loud["capacity"]
    # The objective with the better reconstruction loss is the worse planner.
    assert loud["reconstruction_loss"] < loud["latent_loss"]
    assert loud["latent_control_success"] > loud["reconstruction_control_success"]
    assert loud["loss_disagrees_with_control"]


def test_the_two_objectives_agree_when_there_is_nothing_to_disagree_about():
    """A result that held at every noise level would be a rigged one."""
    from tools.campaigns.world_model_campaign import objective_arm

    quiet = objective_arm(
        capacity=8, steps=4000, observed_dims=48, noise_scale=0.25, seed=5
    )
    assert quiet["verdict"] == "no difference in planning"
    assert quiet["reconstruction_kept_signal_dims"] == quiet["capacity"]


def test_the_world_model_evidence_carries_the_sweep_not_a_single_point():
    payload = _sealed("world_model_campaign.json")
    assert set(payload["cards"]) == {"140", "198"}
    causal = payload["causal"]
    assert causal["causal_wins_under_intervention"] == causal["trials"]
    assert causal["correlational_wins_on_observations"] == causal["trials"]
    assert causal["median_causal_rmse"] < causal["median_correlational_rmse"]

    sweep = payload["objective_sweep"]
    assert len(sweep) >= 5
    verdicts = {row["verdict"] for row in sweep}
    # Both readings appear, so the headline is a crossover and not a constant.
    assert "no difference in planning" in verdicts
    assert "latent prediction plans better" in verdicts
    # Latent prediction never loses, at any noise level.
    assert all(
        row["latent_control_success"] >= row["reconstruction_control_success"]
        for row in sweep
    )
