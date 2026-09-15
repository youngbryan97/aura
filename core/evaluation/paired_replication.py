"""Bind a fixed-size paired design to frozen inputs before running its arms."""

from __future__ import annotations

import random
from collections.abc import Mapping

from core.evaluation.paired_power import _probability, prospective_mcnemar_power
from core.evaluation.preregistration import Preregistration, canonical_hash


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def build_paired_replication_plan(spec: Mapping) -> Preregistration:
    """Validate task commitments and power; publication chronology remains external.

    This prepares an analysis plan, not a frontier certificate or a statement
    that the declared tasks are independently sampled or uncontaminated.
    """
    if spec.get("sampling_unit") != "independent_task":
        raise ValueError("declare independent_task sampling; repeats are not extra tasks")
    if (not isinstance(spec.get("sampling_description"), str)
            or not spec["sampling_description"].strip()):
        raise ValueError("describe the population and independent task sampling")
    for name in ("campaign", "hypothesis"):
        if not isinstance(spec.get(name), str) or not spec[name].strip():
            raise ValueError(f"missing {name}")
    artifacts = spec.get("artifacts", {})
    required = {"candidate", "resident_model", "task_generator", "scorer", "runtime"}
    if not isinstance(artifacts, Mapping) or not required <= set(artifacts):
        raise ValueError("freeze candidate, model, generator, scorer and runtime artifacts")
    if not all(_sha(value) for value in artifacts.values()):
        raise ValueError("artifact identities must be SHA256")
    arms = spec.get("arms", [])
    if (not isinstance(arms, list) or len(arms) < 2
            or any(not isinstance(a, str) or not a.strip() for a in arms)
            or len(arms) != len(set(arms))):
        raise ValueError("at least two unique named arms are required")
    order_seed = spec.get("order_seed")
    if type(order_seed) is not int or order_seed < 0:
        raise ValueError("declare a nonnegative arm-order seed")
    comparisons = spec.get("comparisons", [])
    domains = spec.get("domains", {})
    if not isinstance(domains, Mapping) or not domains or not comparisons:
        raise ValueError("declare domains and comparisons")
    if any(not isinstance(d, str) or not d.strip() for d in domains):
        raise ValueError("domains must be named")
    seen_pairs = set()
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise ValueError("comparisons must name treatment and control")
        pair = (comparison.get("treatment"), comparison.get("control"))
        if any(arm not in arms for arm in pair) or pair[0] == pair[1]:
            raise ValueError("comparison arms must be distinct declared arms")
        if frozenset(pair) in seen_pairs:
            raise ValueError("duplicate comparison")
        seen_pairs.add(frozenset(pair))
        assumptions = comparison.get("assumptions", {})
        if set(assumptions) != set(domains):
            raise ValueError("declare each comparison's alternative for every domain")
    alpha = _probability(spec.get("familywise_alpha"), "familywise_alpha")
    target = _probability(spec.get("target_power"), "target_power")
    if not 0 < alpha < 0.5 or not 0.5 < target < 1:
        raise ValueError("invalid alpha or target power")
    consumed = spec.get("consumed")
    if not isinstance(consumed, Mapping):
        raise ValueError("declare the complete consumed development inventory")
    for key in ("task_ids", "source_sha256s", "seeds"):
        if not isinstance(consumed.get(key), list):
            raise ValueError(f"consumed inventory missing {key}")
    if any(not isinstance(i, str) or not i for i in consumed["task_ids"]):
        raise ValueError("invalid consumed task id")
    if any(not _sha(s) for s in consumed["source_sha256s"]):
        raise ValueError("invalid consumed source identity")
    if any(type(s) is not int or s < 0 for s in consumed["seeds"]):
        raise ValueError("invalid consumed seed")
    ids, sources = set(), set()
    for tasks in domains.values():
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("each domain needs committed tasks")
        for task in tasks:
            if not isinstance(task, Mapping):
                raise ValueError("task commitment must be an object")
            identity, source, seed = task.get("id"), task.get("source_sha256"), task.get("seed")
            if (not isinstance(identity, str) or not identity or not _sha(source)
                    or type(seed) is not int or seed < 0):
                raise ValueError("task requires id, source SHA256 and seed")
            if (identity in ids or source in sources or identity in consumed["task_ids"]
                    or source in consumed["source_sha256s"] or seed in consumed["seeds"]):
                raise ValueError("task duplicates another sampling unit or consumed development")
            ids.add(identity)
            sources.add(source)
    adjusted_alpha = alpha / (len(domains) * len(comparisons))
    order = {}
    rng = random.Random(order_seed)
    for domain in sorted(domains):
        task_ids = sorted(task["id"] for task in domains[domain])
        rng.shuffle(task_ids)
        for index in range(0, len(task_ids), 2):
            arm_cycle = list(arms)
            rng.shuffle(arm_cycle)
            order[task_ids[index]] = arm_cycle
            if index + 1 < len(task_ids):
                order[task_ids[index + 1]] = list(reversed(arm_cycle))
    power_rows = []
    for comparison in comparisons:
        for domain, tasks in domains.items():
            alternative = comparison["assumptions"][domain]
            discordance = _probability(alternative.get("discordance"), "discordance")
            share = _probability(alternative.get("win_share"), "win_share")
            if discordance <= 0 or share <= 0.5:
                raise ValueError("the planned alternative must have positive paired gain")
            power = prospective_mcnemar_power(
                len(tasks), alpha=adjusted_alpha, discordance=discordance, win_share=share,
            )
            if power < target:
                raise ValueError(f"underpowered {domain}: {power:.6f} < {target}")
            power_rows.append({
                "domain": domain, "treatment": comparison["treatment"],
                "control": comparison["control"], "tasks": len(tasks),
                "power": power, "assumed_absolute_gain": discordance * (2 * share - 1),
            })
    return Preregistration(
        campaign=spec["campaign"], hypothesis=spec["hypothesis"], arms=arms,
        parameters={
            "schema": "aura.paired_replication_plan.v1", "design": dict(spec),
            "power": power_rows, "per_comparison_alpha": adjusted_alpha,
            "multiplicity": "bonferroni_all_domains_and_comparisons",
            "test": "exact_one_sided_mcnemar", "stopping": "fixed_committed_task_count",
            "interruption": "resume_same_task_and_arm_ids_without_replacement",
            "runtime_failures": "count_as_incorrect_never_exclude",
            "invalid_task": "invalidate_confirmatory_run_no_posthoc_replacement",
            "arm_order": "seeded_counterbalance_per_domain",
            "arm_order_by_task": order,
            "selection": "no_candidate_or_threshold_changes_after_publication",
            "task_commitment_sha256": canonical_hash(domains),
            "required_publication_evidence": "external_timestamp_before_first_arm_execution",
        },
        metrics={"all_declared_comparisons_reject": 1.0},
        notes="Design only. Measurement and independent chronology verification are pending.",
    )
