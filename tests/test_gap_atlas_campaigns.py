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


# ── spending the same budget somewhere better ─────────────────────────────
#
# Cards 035 and 039. Both say "at equal compute", which is the whole demand:
# a policy that thinks harder and does better has shown nothing.


def test_adaptive_allocation_beats_a_fixed_cadence_on_the_same_budget():
    import random

    from tools.campaigns.allocation_campaign import allocation_trial

    rng = random.Random(21)
    rows = [allocation_trial(methods=6, budget=24, rng=rng) for _ in range(120)]
    adaptive = sum(r["adaptive"] for r in rows)
    assert adaptive > sum(r["static_rotation"] for r in rows)
    assert adaptive > sum(r["static_single"] for r in rows)
    # And short of the oracle, which is what makes it a measurement.
    assert adaptive < sum(r["oracle"] for r in rows)
    # Equal compute is the point: every arm spends the same units.
    assert {r["units_spent"] for r in rows} == {24.0}


def test_value_guided_search_beats_fixed_depth_on_the_same_expansions():
    import random

    from tools.campaigns.allocation_campaign import search_trial

    rng = random.Random(22)
    rows = [search_trial(branches=8, expansions=40, rng=rng) for _ in range(120)]
    guided = sum(r["value_guided"] for r in rows)
    assert guided > sum(r["fixed_depth"] for r in rows)
    assert guided < sum(r["oracle"] for r in rows)
    assert {r["expansions"] for r in rows} == {40.0}


def test_the_branch_payoff_saturates_so_digging_is_not_free():
    """Without this the guided arm wins because of the world, not the policy."""
    import random

    from tools.campaigns import allocation_campaign as ac

    ac._BRANCH_QUALITY = [1.0]
    rng = random.Random(0)
    early = ac._branch_value(0, 2, rng) - ac._branch_value(0, 1, rng)
    late = ac._branch_value(0, 40, rng) - ac._branch_value(0, 39, rng)
    assert late < early


def test_the_allocation_evidence_reports_against_an_oracle():
    payload = _sealed("allocation_campaign.json")
    assert set(payload["cards"]) == {"035", "039"}
    assert payload["equal_compute"]["allocation_units_per_arm"] > 0
    assert payload["equal_compute"]["search_expansions_per_arm"] > 0

    allocation = payload["allocation"]
    assert allocation["adaptive"]["share_of_oracle"] > allocation["static_rotation"][
        "share_of_oracle"
    ]
    assert allocation["adaptive"]["share_of_oracle"] > allocation["static_single"][
        "share_of_oracle"
    ]
    assert allocation["adaptive"]["share_of_oracle"] < 1.0

    search = payload["search"]
    assert search["value_guided"]["share_of_oracle"] > search["fixed_depth"][
        "share_of_oracle"
    ]
    assert search["value_guided"]["share_of_oracle"] < 1.0
    # Not a clean sweep: a blind arm gets lucky sometimes, and a result that
    # never lost would mean the world was built to make it win.
    assert search["value_guided_beats_fixed_depth"] < search["of"]
    assert allocation["adaptive_beats_single"] < allocation["of"]


# ── grown against reset, and the lesion that makes it causal ──────────────
#
# Cards 072, 129, 168 and 203.


def test_growth_survives_the_three_confounds_and_the_lesion_erases_it():
    from tools.campaigns.developmental_campaign_run import run_campaign

    result = run_campaign(blocks=8, tasks_per_block=40, length=12, seed=31)
    assert not result["void"], result["void_because"]
    assert result["contaminated_fraction"] == 0.0
    assert result["context_parity"]
    # The gap grows, and the interval excludes zero.
    assert result["slope"] > 0
    assert result["slope_ci"][0] > 0
    # The half that makes it causal rather than correlated.
    assert result["lesion_restores_baseline"]


def test_growth_never_made_a_single_task_worse():
    """Card 129's other half. A mean that improved can hide the tasks it cost."""
    from tools.campaigns.developmental_campaign_run import run_campaign

    result = run_campaign(blocks=8, tasks_per_block=40, length=12, seed=31)
    assert result["tasks_scored"] > 0
    assert result["tasks_growth_made_worse"] == 0


def test_demonstrations_beat_a_description_that_is_better_than_chance():
    from tools.campaigns.developmental_campaign_run import tool_learning

    result = tool_learning(tools=6, trials=400, seed=31)
    chance = 1.0 / 6
    # A baseline at chance is not a baseline; this one reads the descriptions
    # and gets real information out of them.
    assert result["static_accuracy"] > chance
    assert result["demonstrated_accuracy"] > result["static_accuracy"]
    assert result["demonstrations_to_competence"] is not None
    assert result["demonstrations_to_competence"] < result["trials"]


def test_knowing_the_partner_beats_the_prior_and_carries_to_a_new_context():
    from tools.campaigns.developmental_campaign_run import partner_model

    result = partner_model(interactions=400, contexts=5, seed=31)
    seen = result["in_context"]
    assert seen["measurable"] and seen["learned_something"]
    assert seen["model_accuracy"] > seen["prior_accuracy"]
    assert result["transfers"]
    assert (
        result["transfer_partner_model_accuracy"]
        > result["transfer_prior_accuracy"]
    )


def test_a_model_that_learned_the_context_instead_of_the_partner_cannot_transfer():
    """The negative control that makes transfer a claim rather than a word."""
    from tools.campaigns.developmental_campaign_run import partner_model

    result = partner_model(interactions=400, contexts=5, seed=31)
    assert not result["per_context_variant_transfers"]
    assert result["transfer_per_context_model_accuracy"] < result[
        "transfer_partner_model_accuracy"
    ]


def test_the_developmental_evidence_carries_all_four_arms():
    payload = _sealed("developmental_campaign.json")
    assert set(payload["cards"]) == {"072", "129", "168", "203"}
    development = payload["development"]
    assert not development["void"]
    assert development["lesion_restores_baseline"]
    assert development["tasks_growth_made_worse"] == 0
    assert payload["tools"]["demonstrated_accuracy"] > payload["tools"][
        "static_accuracy"
    ]
    assert payload["partner"]["transfers"]
    assert not payload["partner"]["per_context_variant_transfers"]


# ── what recall costs, on Aura's own recall ───────────────────────────────
#
# Card 001. The traces are real: the campaign drives
# core.memory.hybrid_store's retrieve and times it.


def test_the_latency_campaign_times_the_real_recall_path():
    import asyncio

    from tools.campaigns.retrieval_latency_campaign import _collect

    rows = asyncio.run(_collect(store_sizes=[100, 300], recalls=40, seed=7))
    assert len(rows) >= 30
    assert all(o.seconds > 0 for o in rows), "a recall that took no time was not timed"
    assert all(o.backend == "hybrid_episodic" for o in rows)
    # A bigger store costs more, which is the thing the law is about.
    small = [o.seconds for o in rows if o.candidates == 100]
    large = [o.seconds for o in rows if o.candidates == 300]
    assert sum(large) / len(large) > sum(small) / len(small)


def test_the_law_beats_candidate_count_alone_on_held_out_recalls():
    payload = _sealed("retrieval_latency.json")
    assert payload["card"] == "001"
    held = payload["held_out"]
    assert held["law_beats_candidates_only"]
    assert held["law_rmse"] < held["candidates_only_rmse"]
    # Both beat predicting the mean, which is the floor neither may fall to.
    assert held["candidates_only_rmse"] < held["predict_the_mean_rmse"]
    assert payload["config"]["held_out"] > 0
    assert payload["law"]["explains_anything"]


def test_the_law_predicts_the_right_direction_under_intervention():
    """A law whose predicted direction is wrong is refuted by its own
    coefficients, whatever its fit."""
    payload = _sealed("retrieval_latency.json")
    intervention = payload["intervention"]
    assert intervention["to_candidates"] > intervention["from_candidates"]
    assert intervention["direction"] == "up"
    assert intervention["predicted_delta"] > 0


# ── interrupted mid-task, hundreds of times ───────────────────────────────
#
# Card A2.18.


def test_a_correction_undoes_the_overrun_and_keeps_the_conversation():
    import random

    from tools.campaigns.steering_campaign import one_task

    rng = random.Random(41)
    rows = [one_task(rng, steps=12) for _ in range(120)]
    assert all(r["work_ran_past_the_correction"] for r in rows), (
        "nothing ran past the checkpoint, so the revert had nothing to undo"
    )
    assert all(r["overrun_undone"] for r in rows)
    assert not any(r["revert_took_too_much"] for r in rows)
    assert all(r["correction_landed"] for r in rows)
    assert all(r["conversation_survived"] for r in rows)
    assert not any(r["false_success"] for r in rows)


def test_the_steering_evidence_reports_zero_false_success():
    payload = _sealed("steering_campaign.json")
    assert payload["card"] == "A2.18"
    assert payload["of"] >= 100
    assert payload["zero_false_success"]
    assert payload["every_correction_landed"]
    assert payload["conversation_never_lost"]
    assert payload["every_overrun_undone"]
    assert payload["no_revert_took_too_much"]
    counts = payload["counts"]
    assert counts["work_ran_past_the_correction"] == payload["of"]
    assert counts["overrun_survived_the_revert"] == 0
