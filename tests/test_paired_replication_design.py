"""Prospective task counts cannot borrow power from unobserved disagreements."""

import copy
import hashlib
import math

import pytest

from core.evaluation.paired_power import (
    binomial_tail,
    conditional_mcnemar_power,
    prospective_mcnemar_power,
)
from core.evaluation.paired_replication import build_paired_replication_plan


def _tail(k, n, p):
    return math.fsum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))


@pytest.mark.parametrize("alpha", [0.05, 1/32, 0.01])
@pytest.mark.parametrize("share", [0.5, 0.75, 1.0])
def test_conditional_power_matches_enumeration_including_discrete_boundary(alpha, share):
    for n in range(21):
        critical = next((k for k in range(n+1) if _tail(k, n, 0.5) <= alpha), n+1)
        assert conditional_mcnemar_power(n, alpha, share) == pytest.approx(_tail(critical, n, share))


def test_prospective_power_integrates_random_discordance():
    n, q, share, alpha = 18, 0.4, 0.8, 0.05
    expected = sum(math.comb(n, d)*q**d*(1-q)**(n-d)*conditional_mcnemar_power(d, alpha, share)
                   for d in range(n+1))
    actual = prospective_mcnemar_power(n, alpha=alpha, discordance=q, win_share=share)
    assert actual == pytest.approx(expected)
    assert actual < conditional_mcnemar_power(n, alpha, share)


def test_zero_discordance_has_zero_power_and_full_discordance_matches_conditional():
    assert prospective_mcnemar_power(100, alpha=.05, discordance=0, win_share=1) == 0
    assert prospective_mcnemar_power(100, alpha=.05, discordance=1, win_share=.7) == pytest.approx(
        conditional_mcnemar_power(100, .05, .7))


def test_campaign_scale_does_not_overflow():
    assert binomial_tail(1000, 2000, .5) == pytest.approx(.508919505572927, abs=1e-12)
    assert .99 < prospective_mcnemar_power(2000, alpha=.01, discordance=.2, win_share=.7) <= 1


@pytest.mark.parametrize("change", [
    {"tasks": True}, {"tasks": -1}, {"tasks": 1.5}, {"alpha": 0},
    {"alpha": float("nan")}, {"discordance": 1.1}, {"win_share": True},
])
def test_invalid_power_inputs_raise(change):
    params = dict(tasks=20, alpha=.05, discordance=.4, win_share=.8)
    params.update(change)
    with pytest.raises(ValueError):
        prospective_mcnemar_power(**params)


def _spec(count=80):
    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()
    domains = {name: [dict(id=f"{name}-{i}", source_sha256=digest(f"{name}-{i}"), seed=100+i)
                      for i in range(count)] for name in ("coding", "science")}
    return dict(campaign="fresh-paired", hypothesis="treatment improves each domain",
                sampling_unit="independent_task", sampling_description="Independent generated tasks",
                artifacts={k: digest(k) for k in ("candidate", "resident_model", "task_generator", "scorer", "runtime")},
                arms=["treatment", "ordinary"], order_seed=704,
                comparisons=[dict(treatment="treatment", control="ordinary", assumptions={
                    d: dict(discordance=.8, win_share=.95) for d in domains})],
                domains=domains, familywise_alpha=.05, target_power=.8,
                consumed=dict(task_ids=["old"], source_sha256s=[digest("old")], seeds=[3]))


def test_plan_freezes_power_count_inputs_and_stopping_rule():
    spec = _spec()
    plan = build_paired_replication_plan(spec)
    assert plan.parameters["per_comparison_alpha"] == .025
    assert len(plan.parameters["power"]) == 2
    assert all(row["power"] > .8 for row in plan.parameters["power"])
    assert plan.parameters["power"][0]["assumed_absolute_gain"] == pytest.approx(.72)
    assert plan.parameters["runtime_failures"] == "count_as_incorrect_never_exclude"
    assert plan.parameters["stopping"] == "fixed_committed_task_count"
    assert plan.parameters["required_publication_evidence"]
    identity = plan.plan_hash
    spec["artifacts"]["candidate"] = "0"*64
    assert plan.plan_hash == identity
    assert build_paired_replication_plan(spec).plan_hash != identity


def test_counterbalanced_order_is_frozen_by_task_and_domain():
    spec = _spec(81)
    plan = build_paired_replication_plan(spec)
    order = plan.parameters["arm_order_by_task"]
    for tasks in spec["domains"].values():
        first = [order[task["id"]][0] for task in tasks]
        assert abs(first.count("ordinary")-first.count("treatment")) == 1
        assert all(set(order[task["id"]]) == set(spec["arms"]) for task in tasks)
    assert build_paired_replication_plan(copy.deepcopy(spec)).plan_hash == plan.plan_hash


def test_multiarm_order_balances_every_pair_not_only_first_position():
    import itertools

    spec = _spec(80)
    spec["arms"] += ["lesion", "equal_compute", "wrong_state"]
    plan = build_paired_replication_plan(spec)
    order = plan.parameters["arm_order_by_task"]
    for tasks in spec["domains"].values():
        for left, right in itertools.combinations(spec["arms"], 2):
            assert sum(order[t["id"]].index(left) < order[t["id"]].index(right)
                       for t in tasks) == 40


def test_cli_publishes_a_readable_plan_without_claiming_execution(tmp_path, monkeypatch, capsys):
    import json

    from core.evaluation.preregistration import load_preregistration
    from tools.preregister_paired_replication import main

    source = tmp_path / "spec.json"
    source.write_text(json.dumps(_spec()))
    store = tmp_path / "plans"
    monkeypatch.setattr("sys.argv", ["preregister", "--spec", str(source), "--store", str(store)])
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["confirmatory_run_complete"] is False
    assert load_preregistration(output["path"]).plan_hash == output["plan_hash"]


@pytest.mark.parametrize("field", ["id", "source_sha256", "seed"])
def test_development_reuse_is_rejected(field):
    spec = _spec()
    consumed_key = dict(id="task_ids", source_sha256="source_sha256s", seed="seeds")[field]
    spec["domains"]["coding"][0][field] = spec["consumed"][consumed_key][0]
    with pytest.raises(ValueError, match="consumed"):
        build_paired_replication_plan(spec)


@pytest.mark.parametrize("field", ["id", "source_sha256"])
def test_duplicate_tasks_across_domains_are_not_extra_sampling_units(field):
    spec = _spec()
    spec["domains"]["coding"][0][field] = spec["domains"]["science"][0][field]
    with pytest.raises(ValueError, match="duplicates"):
        build_paired_replication_plan(spec)


def test_underpowered_count_and_missing_alternative_are_rejected():
    with pytest.raises(ValueError, match="underpowered"):
        build_paired_replication_plan(_spec(4))
    spec = _spec()
    del spec["comparisons"][0]["assumptions"]["coding"]
    with pytest.raises(ValueError, match="every domain"):
        build_paired_replication_plan(spec)


def test_repeats_and_unfrozen_candidate_are_rejected():
    spec = _spec()
    spec["sampling_unit"] = "decode_seed"
    with pytest.raises(ValueError, match="independent_task"):
        build_paired_replication_plan(spec)
    spec = _spec()
    spec["artifacts"]["candidate"] = "latest"
    with pytest.raises(ValueError, match="SHA256"):
        build_paired_replication_plan(spec)


def test_plan_does_not_report_a_confirmatory_run_without_measurement():
    plan = build_paired_replication_plan(_spec())
    assert not plan.verify_result({})["confirms_hypothesis"]
