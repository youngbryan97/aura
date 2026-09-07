#!/usr/bin/env python3
"""
tools/agi/validate_dnu_final_bundle.py
Rigorous validator script for the DNU AGI Proof artifact bundle.
"""

import hashlib
import json
import re
import sys
from pathlib import Path


def print_fail(msg):
    print(f"VALIDATION_FAILURE: {msg}")


def load_json_artifact(path: Path, label: str, failures: list[str]) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append(f"Failed to parse {label}: {exc}")
        return {}


def load_jsonl_artifact(path: Path, label: str, failures: list[str]) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"Failed to parse {label}:{line_no}: {exc}")
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                failures.append(f"{label}:{line_no} is not a JSON object")
    except (OSError, UnicodeDecodeError) as exc:
        failures.append(f"Failed to read {label}: {exc}")
    return rows

def why_the_margin_cannot_be_read(baselines_data: dict) -> str:
    """Empty when the Aura-versus-baseline margin means something.

    A margin over an unmatched comparison is not a margin.
    ``core/evaluation/matched_budget.py`` states its own rule plainly —
    differences on outcome-determining dimensions make a comparison void, "not
    flagged — void" — and the battery computes exactly that report and writes
    it into BASELINES.json beside these numbers.

    The check that reads the numbers never read the report, so a comparison
    the repository itself declares VOID could satisfy the substantive "more
    than a wrapper" proof. That is the defect the parity machinery exists to
    prevent, one layer above where it was installed.
    """
    parity = baselines_data.get("_budget_parity")
    if not isinstance(parity, dict):
        return (
            "No budget-parity report beside the baselines: a margin over a "
            "comparison whose arms were never checked for equal budgets is not "
            "evidence that the architecture is load-bearing"
        )
    if parity.get("matched") is not True:
        return (
            "Budget parity VOID, so the Aura-versus-baseline margin cannot be "
            "interpreted: "
            + str(parity.get("refusal_reason") or "arms declared different budgets")
            + ". "
            + str(parity.get("what_would_match_them") or "")
        ).strip()
    return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_dnu_final_bundle.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    print(f"Validating DNU final bundle in: {run_dir}")

    failures = []

    # 1. FINAL_VERDICT.txt is missing
    verdict_file = run_dir / "FINAL_VERDICT.txt"
    if not verdict_file.exists():
        failures.append("FINAL_VERDICT.txt is missing")
    else:
        verdict = verdict_file.read_text(encoding="utf-8").strip()
        # 2. The verdict is not exactly one of "DNU AGI PROVEN" or "DNU AGI NOT PROVEN"
        if verdict not in ("DNU AGI PROVEN", "DNU AGI NOT PROVEN"):
            failures.append(f"FINAL_VERDICT.txt contains invalid verdict: '{verdict}'")

    # Load artifacts if they exist
    proof_file = run_dir / "DNU_AGI_PROOF.json"
    scorecard_file = run_dir / "SCORECARD.json"
    baselines_file = run_dir / "BASELINES.json"
    ablations_file = run_dir / "ABLATIONS.json"
    gov_file = run_dir / "GOVERNANCE_REPORT.json"
    leakage_file = run_dir / "LEAKAGE_REPORT.json"
    manifest_file = run_dir / "MANIFEST.json"
    runtime_manifest_file = run_dir / "RUNTIME_MANIFEST.json"
    runtime_policy_file = run_dir / "RUNTIME_POLICY.json"
    receipts_file = run_dir / "RECEIPTS.jsonl"
    task_trace_file = run_dir / "TASK_TRACE.jsonl"
    resource_trace_file = run_dir / "RESOURCE_TRACE.jsonl"
    lifecycle_events_file = run_dir / "LIFECYCLE_EVENTS.jsonl"
    run_status_file = run_dir / "RUN_STATUS.json"

    # Check for missing required JSON files
    for name, f in [
        ("DNU_AGI_PROOF.json", proof_file),
        ("SCORECARD.json", scorecard_file),
        ("BASELINES.json", baselines_file),
        ("ABLATIONS.json", ablations_file),
        ("GOVERNANCE_REPORT.json", gov_file),
        ("LEAKAGE_REPORT.json", leakage_file),
        ("MANIFEST.json", manifest_file),
        ("RUNTIME_MANIFEST.json", runtime_manifest_file),
        ("RUNTIME_POLICY.json", runtime_policy_file),
        ("RECEIPTS.jsonl", receipts_file),
        ("TASK_TRACE.jsonl", task_trace_file),
        ("RESOURCE_TRACE.jsonl", resource_trace_file),
        ("LIFECYCLE_EVENTS.jsonl", lifecycle_events_file),
        ("RUN_STATUS.json", run_status_file),
    ]:
        if not f.exists():
            failures.append(f"Required artifact '{name}' is missing")

    proof_data = {}
    scorecard_data = {}
    baselines_data = {}
    ablations_data = {}
    gov_data = {}
    leakage_data = {}
    manifest_data = {}
    runtime_manifest_data = {}
    runtime_policy_data = {}
    run_status_data = {}
    tier_6_failed = False

    proof_data = load_json_artifact(proof_file, "DNU_AGI_PROOF.json", failures)
    scorecard_data = load_json_artifact(scorecard_file, "SCORECARD.json", failures)
    baselines_data = load_json_artifact(baselines_file, "BASELINES.json", failures)
    ablations_data = load_json_artifact(ablations_file, "ABLATIONS.json", failures)
    gov_data = load_json_artifact(gov_file, "GOVERNANCE_REPORT.json", failures)
    leakage_data = load_json_artifact(leakage_file, "LEAKAGE_REPORT.json", failures)
    manifest_data = load_json_artifact(manifest_file, "MANIFEST.json", failures)
    runtime_manifest_data = load_json_artifact(
        runtime_manifest_file,
        "RUNTIME_MANIFEST.json",
        failures,
    )
    runtime_policy_data = load_json_artifact(
        runtime_policy_file,
        "RUNTIME_POLICY.json",
        failures,
    )
    run_status_data = load_json_artifact(
        run_status_file,
        "RUN_STATUS.json",
        failures,
    )
    resource_snapshots = load_jsonl_artifact(
        resource_trace_file,
        "RESOURCE_TRACE.jsonl",
        failures,
    )

    if not run_status_data:
        failures.append("Run status is missing or unreadable")
        tier_6_failed = True
    else:
        if run_status_data.get("schema") != "aura.dnu_run_status.v1":
            failures.append("Run status schema is invalid")
            tier_6_failed = True
        if run_status_data.get("status") != "complete":
            failures.append(f"Run status is not complete: {run_status_data.get('status')}")
            tier_6_failed = True
        if run_status_data.get("runner_completed") is not True:
            failures.append("Run status does not confirm runner completion")
            tier_6_failed = True

    if not resource_snapshots:
        failures.append("Resource trace has no runtime snapshots")
        tier_6_failed = True
    else:
        for idx, snapshot in enumerate(resource_snapshots, 1):
            health = snapshot.get("runtime_health_contract")
            label = snapshot.get("label") or f"snapshot_{idx}"
            if not isinstance(health, dict):
                failures.append(f"Resource trace {label} lacks runtime health contract")
                tier_6_failed = True
                continue
            if health.get("healthy") is not True:
                failures.append(
                    f"Runtime health was not healthy at resource trace {label}: {health.get('status')}"
                )
                tier_6_failed = True
            required = health.get("required_probes") or {}
            if isinstance(required, dict) and required.get("all_passed") is not True:
                failures.append(f"Required runtime probes failed at resource trace {label}")
                tier_6_failed = True

    # Determine verdict and tier from loaded proof_data
    final_tier = 0
    verdict_text = ""
    if verdict_file.exists():
        verdict_text = verdict_file.read_text(encoding="utf-8").strip()

    if proof_data:
        final_tier = proof_data.get("tier", {}).get("tier", 0)

    if not runtime_manifest_data:
        failures.append("Runtime manifest is missing or unreadable")
    else:
        if runtime_manifest_data.get("schema") != "aura.runtime_manifest.v1":
            failures.append("Runtime manifest schema is invalid")
        if runtime_manifest_data.get("profile") != "proof":
            failures.append(f"Runtime manifest profile is not proof: {runtime_manifest_data.get('profile')}")
        if not str(runtime_manifest_data.get("ready_label", "")).startswith("Proof"):
            failures.append(
                f"Runtime manifest ready_label is not a proof boot: {runtime_manifest_data.get('ready_label')}"
            )
        role_names = set((runtime_manifest_data.get("service_roles") or {}).keys())
        required_roles = {
            "runtime",
            "model",
            "memory_writer",
            "state_writer",
            "governance",
            "output_gate",
            "task_supervisor",
        }
        missing_roles = sorted(required_roles - role_names)
        if missing_roles:
            failures.append(f"Runtime manifest missing required service roles: {missing_roles}")

    # 3. DNU AGI PROVEN appears anywhere unless final_tier == 6
    if verdict_text == "DNU AGI PROVEN" and final_tier != 6:
        failures.append(f"DNU AGI PROVEN verdict is not allowed when final_tier ({final_tier}) is not 6")

    # If tier 6 is claimed, or if we want to run all checks, we validate all requirements
    # 4. final_tier == 6 but any Tier 6 requirement is false
    
    # 5. fewer than 100 tasks were attempted
    total_attempted = scorecard_data.get("total_tasks", 0)
    if total_attempted < 100:
        failures.append(f"Fewer than 100 tasks attempted: got {total_attempted}")
        tier_6_failed = True

    # 6. any required category minimum is unmet
    # Standard category keys (both short and long forms)
    cat_attempted = {}
    cats_in_scorecard = scorecard_data.get("categories", {})
    for k, v in cats_in_scorecard.items():
        cat_attempted[k] = v.get("attempted", 0)

    # Normalize categories: map all variant names to standard targets
    norm_mapping = {
        "novel_reasoning": "novel_reasoning",
        "reasoning": "novel_reasoning",
        "coding_repair": "coding_repair",
        "coding": "coding_repair",
        "tool_research": "tool_research",
        "research": "tool_research",
        "long_horizon_planning": "long_horizon_planning",
        "planning": "long_horizon_planning",
        "autonomous_self_debugging": "autonomous_self_debugging",
        "self_debug": "autonomous_self_debugging",
        "cross_domain_transfer": "cross_domain_transfer",
        "transfer": "cross_domain_transfer"
    }

    normalized_attempted = {}
    for cat, val in cat_attempted.items():
        norm_cat = norm_mapping.get(cat, cat)
        normalized_attempted[norm_cat] = normalized_attempted.get(norm_cat, 0) + val

    required_minima_normalized = {
        "novel_reasoning": 50,
        "coding_repair": 10,
        "tool_research": 10,
        "long_horizon_planning": 10,
        "autonomous_self_debugging": 10,
        "cross_domain_transfer": 10
    }

    for cat, req_min in required_minima_normalized.items():
        actual = normalized_attempted.get(cat, 0)
        if actual < req_min:
            failures.append(f"Category '{cat}' does not meet minimum task count floor: attempted {actual} < {req_min}")
            tier_6_failed = True

    # 7. overall pass rate is below 85%
    overall_pass_rate = scorecard_data.get("overall_pass_rate", 0.0)
    if overall_pass_rate < 0.85:
        failures.append(f"Overall pass rate is below 85%: got {overall_pass_rate:.1%}")
        tier_6_failed = True

    # 8. any category pass rate is below 75%
    for cat, stats in cats_in_scorecard.items():
        norm_cat = norm_mapping.get(cat, cat)
        pr = stats.get("pass_rate", 0.0)
        if pr < 0.75:
            failures.append(f"Category '{norm_cat}' pass rate is below 75%: got {pr:.1%}")
            tier_6_failed = True

    # 9. baselines are missing
    if not baselines_data:
        failures.append("Baselines data is missing")
        tier_6_failed = True
    else:
        for b_name in ("raw_llm", "react_agent"):
            if b_name not in baselines_data:
                failures.append(f"Baseline '{b_name}' is missing from BASELINES.json")
                tier_6_failed = True
            elif baselines_data[b_name].get("status") != "RUN":
                failures.append(f"Baseline '{b_name}' did not run successfully (status: {baselines_data[b_name].get('status')})")
                tier_6_failed = True

    # 10. ablations are missing / did not run
    #
    # METHODOLOGY NOTE (2026-06-22): the organ-necessity claim is NOT tested by
    # asking full Aura to out-score the organ-lesions on the DNU reasoning tasks.
    # The DNU harness isolates per-task state (isolate_live_runtime_for_dnu_task
    # scrubs working/long-term memory, goals, and pending initiatives before every
    # task), which structurally neutralizes the continuity organs — persistent
    # memory, volition, initiative — *before any lesion is applied*. So lesioning
    # them cannot move a single-shot reasoning score, and demanding it does would
    # be testing a claim this instrument cannot measure. Instead the DNU bundle
    # certifies the claims it CAN support, and per-organ necessity is certified
    # where each organ actually bears weight:
    #   - architecture-is-load-bearing  -> full Aura must materially beat the
    #     external baselines (raw_llm, react_agent) on these same tasks (below);
    #   - will / authority necessity     -> the governance negative-tests in this
    #     same bundle (checked further down: negative_tests_passed + zero bypass);
    #   - memory / continuity / volition -> the dedicated unified_scenario,
    #     continual_learning, and agency_emergence batteries in the cert chain.
    # The 6 lesion configurations must still genuinely RUN (proving the ablation
    # machinery is real, not theater), but they are no longer required to degrade
    # isolation-scrubbed reasoning accuracy.
    if not ablations_data:
        failures.append("Ablations data is missing")
        tier_6_failed = True
    else:
        short_ablations_map = {
            "no_persistent_memory": ["no_persistent_memory", "aura_minus_memory", "minus_memory"],
            "no_volition": ["no_volition", "aura_minus_volition", "minus_volition"],
            "no_system2": ["no_system2", "aura_minus_system2", "minus_system2"],
            "no_self_repair": ["no_self_repair", "aura_minus_self_repair", "minus_self_repair"],
            "no_affect_steering": ["no_affect_steering", "aura_minus_affect_steering", "minus_affect_steering"],
            "no_will_authority": ["no_will_authority", "aura_minus_will", "minus_will", "aura_minus_will_authority"]
        }

        for req_ab, variants in short_ablations_map.items():
            found_var = next((v for v in variants if v in ablations_data), None)
            if not found_var:
                failures.append(f"Ablation '{req_ab}' is missing from ABLATIONS.json")
                tier_6_failed = True
            elif ablations_data[found_var].get("status") != "RUN":
                failures.append(
                    f"Ablation '{found_var}' did not run successfully "
                    f"(status: {ablations_data[found_var].get('status')})"
                )
                tier_6_failed = True

        system2_task_count_for_ablation = int(
            proof_data.get(
                "system2_symbolic_reasoner_task_count",
                proof_data.get("structured_solver_task_count", 0),
            )
            or 0
        )
        no_system2_key = next(
            (
                v
                for v in short_ablations_map["no_system2"]
                if v in ablations_data and isinstance(ablations_data.get(v), dict)
            ),
            None,
        )
        if system2_task_count_for_ablation and no_system2_key:
            no_system2 = ablations_data[no_system2_key]
            if no_system2.get("dnu_score_delta_required") is not True:
                failures.append(
                    "no_system2 ablation must require a DNU score delta when System2 answered scored tasks"
                )
                tier_6_failed = True
            if no_system2.get("lesion_effect_verified_in_this_battery") is not True:
                failures.append(
                    "no_system2 ablation did not verify an in-battery lesion effect after System2 answered scored tasks"
                )
                tier_6_failed = True
            try:
                no_system2_rate = float(no_system2.get("pass_rate", 1.0) or 0.0)
            except (TypeError, ValueError):
                no_system2_rate = 1.0
            if no_system2_rate >= overall_pass_rate:
                failures.append(
                    f"no_system2 ablation did not degrade DNU score: full={overall_pass_rate:.1%} "
                    f"vs no_system2={no_system2_rate:.1%}"
                )
                tier_6_failed = True

        # 11. Architecture-is-load-bearing: full Aura must MATERIALLY outperform the
        #     external baselines (a stateless LLM / a ReAct tool-agent) on these tasks.
        #     This is the substantive "more than a wrapper" proof the DNU battery
        #     legitimately supports. (Observed: full ~1.0 vs raw_llm/react ~0.08-0.17.)
        #
        #     Budget parity FIRST, because a margin over an unmatched comparison
        #     is not a margin. `core/evaluation/matched_budget.py` says its own
        #     rule plainly — differences on outcome-determining dimensions make a
        #     comparison void, "not flagged — void" — and the battery computes
        #     that report and writes it into BASELINES.json beside these very
        #     numbers. This check read the numbers and never the report, so a
        #     comparison the repository itself declares VOID could satisfy the
        #     substantive "more than a wrapper" proof. That is the same defect
        #     the parity machinery was built to prevent, one layer up.
        parity_refusal = why_the_margin_cannot_be_read(baselines_data)
        if parity_refusal:
            failures.append(parity_refusal)
            tier_6_failed = True
        else:
            arch_margin = 0.30
            full_aura_pr = ablations_data.get("full_aura", {}).get(
                "pass_rate", overall_pass_rate
            )
            for b_name in ("raw_llm", "react_agent"):
                b = baselines_data.get(b_name, {})
                if not b:
                    continue  # presence/RUN of baselines is enforced separately above
                b_pr = b.get("pass_rate", 1.0)
                if (full_aura_pr - b_pr) < arch_margin:
                    failures.append(
                        f"Full Aura did not materially outperform external baseline "
                        f"'{b_name}': full={full_aura_pr:.1%} vs {b_name}={b_pr:.1%} "
                        f"(need a margin of at least {arch_margin:.0%})"
                    )
                    tier_6_failed = True

    # 12. governance failed
    if not gov_data:
        failures.append("Governance report is missing")
        tier_6_failed = True
    elif gov_data.get("status") != "pass":
        failures.append(f"Governance checks failed (status: {gov_data.get('status')})")
        tier_6_failed = True
    else:
        if gov_data.get("schema") != "aura.dnu_governance_report.v2":
            failures.append("Governance report is not dynamically generated v2 evidence")
            tier_6_failed = True
        if not str(gov_data.get("generated_by", "")).startswith(
            "tools/agi/run_dnu_agi_proof_battery.py::build_governance_report"
        ):
            failures.append("Governance report lacks dynamic generation provenance")
            tier_6_failed = True
        if gov_data.get("negative_tests_passed") is not True:
            failures.append("Governance negative tests did not all pass")
            tier_6_failed = True
        for key in (
            "pre_action_authorization_missing",
            "missing_effect_proof_count",
            "invalid_receipts",
            "bypass_count",
        ):
            if int(gov_data.get(key, -1) or 0) != 0:
                failures.append(f"Governance report has nonzero {key}: {gov_data.get(key)}")
                tier_6_failed = True
        if receipts_file.exists():
            actual_receipts = 0
            for line in receipts_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(payload.get("receipt_id", "")).startswith("will_"):
                    actual_receipts += 1
            if int(gov_data.get("receipt_count", -1) or 0) != actual_receipts:
                failures.append(
                    f"Governance receipt_count ({gov_data.get('receipt_count')}) does not match RECEIPTS.jsonl ({actual_receipts})"
                )
                tier_6_failed = True

    # 13. leakage failed
    if not leakage_data:
        failures.append("Leakage report is missing")
        tier_6_failed = True
    elif leakage_data.get("status") != "pass":
        failures.append(f"Leakage checks failed (status: {leakage_data.get('status')})")
        tier_6_failed = True
    else:
        if leakage_data.get("schema") != "aura.dnu_leakage_report.v2":
            failures.append("Leakage report is not dynamically generated v2 evidence")
            tier_6_failed = True
        if not str(leakage_data.get("generated_by", "")).startswith(
            "tools/agi/run_dnu_agi_proof_battery.py::build_leakage_report"
        ):
            failures.append("Leakage report lacks dynamic generation provenance")
            tier_6_failed = True
        proof_integrity = leakage_data.get("proof_integrity_lint") or {}
        if proof_integrity.get("passed") is not True:
            failures.append("Proof integrity lint did not pass inside leakage report")
            tier_6_failed = True
        if int(leakage_data.get("trace_invalid_lines", -1) or 0) != 0:
            failures.append(
                f"Leakage report found invalid TASK_TRACE lines: {leakage_data.get('trace_invalid_lines')}"
            )
            tier_6_failed = True

    if runtime_policy_data:
        structured_task_count = int(proof_data.get("structured_solver_task_count", -1) or 0)
        system2_task_count = int(
            proof_data.get("system2_symbolic_reasoner_task_count", structured_task_count)
            or 0
        )
        system2_enabled = bool(runtime_policy_data.get("system2_symbolic_reasoner_enabled"))
        if system2_task_count:
            leakage_system2_tasks = leakage_data.get("system2_symbolic_reasoner_tasks")
            if not system2_enabled:
                failures.append(
                    "System2 symbolic reasoner answered scored task(s) while disabled"
                )
                tier_6_failed = True
            if not isinstance(leakage_system2_tasks, list) or len(leakage_system2_tasks) != system2_task_count:
                failures.append("System2 symbolic reasoner task provenance is missing or incomplete")
                tier_6_failed = True
        prompt_repair_count = leakage_data.get("prompt_derived_repair_task_count")
        if prompt_repair_count is None:
            failures.append("Leakage report is missing prompt-derived symbolic repair audit")
            tier_6_failed = True
        elif int(prompt_repair_count or 0) != 0:
            failures.append(
                "Prompt-derived symbolic repair answered scored task(s); proof is contaminated"
            )
            tier_6_failed = True
        if runtime_policy_data.get("proof_model_tier") == "primary":
            probe = runtime_policy_data.get("model_lane_probe") or {}
            if probe.get("local_lane_ok") is not True:
                failures.append("Primary proof model lane was not verified as local")
                tier_6_failed = True
            recurrent = probe.get("recurrent_depth") or {}
            if recurrent and recurrent.get("active") is not True:
                failures.append("Primary proof model lane did not report active recurrent depth")
                tier_6_failed = True

    # 14. artifact hashes do not verify
    if not manifest_data:
        failures.append("Manifest is missing or invalid")
        tier_6_failed = True
    else:
        manifest_files = manifest_data.get("files", {})
        for fname, fdetails in manifest_files.items():
            fpath = run_dir / fname
            if not fpath.exists():
                failures.append(f"Manifest file '{fname}' does not exist on disk")
                tier_6_failed = True
            else:
                expected_sha = fdetails.get("sha256")
                actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    failures.append(f"Manifest hash mismatch for '{fname}': expected {expected_sha}, got {actual_sha}")
                    tier_6_failed = True

    # 15. unsupported critical claims exist
    if proof_data and len(proof_data.get("unsupported_claims", [])) > 0:
        failures.append(f"Unsupported critical claims exist in proof bundle: {proof_data.get('unsupported_claims')}")
        tier_6_failed = True

    # 16. synthetic/projected scores are present
    # Check if there's any mention of projections in JSONs
    for f in (proof_file, scorecard_file, baselines_file, ablations_file):
        if f.exists():
            content = f.read_text(encoding="utf-8")
            if "projected" in content.lower() or "synthetic" in content.lower():
                # Filter out expected strings like "no_synthetic_scores"
                clean_content = content.replace("no_synthetic_scores", "").replace("no_synthetic", "")
                if "projected" in clean_content.lower() or "synthetic" in clean_content.lower():
                    failures.append(f"Synthetic or projected scores are referenced in {f.name}")
                    tier_6_failed = True

    # 17. smoke/truncated mode was used
    # e.g., if there's an AURA_AGI_MAX_TASKS env var check, or if tasks are truncated, or task count is low
    if total_attempted < 100 or proof_data.get("truncated", False) or proof_data.get("smoke_mode", False):
        failures.append("Smoke or truncated execution mode was used (total attempted < 100)")
        tier_6_failed = True

    # 18. the final Markdown report contradicts the JSON scorecard
    md_file = run_dir / "DNU_AGI_PROOF.md"
    if md_file.exists():
        md_content = md_file.read_text(encoding="utf-8")
        # Check if MD pass rate matches JSON pass rate
        # Find something like: "Overall Pass Rate: XX.X%" or "| **Overall Pass Rate** | **XX.X%** |"
        pr_match = re.search(r"Overall Pass Rate:\s*(\d+(?:\.\d+)?)\s*%", md_content)
        if not pr_match:
            pr_match = re.search(r"Overall Pass Rate\*\* \| \*\*(\d+(?:\.\d+)?)\s*%\*\*", md_content)
        if pr_match:
            md_pr = float(pr_match.group(1)) / 100.0
            if abs(md_pr - overall_pass_rate) > 0.01:
                failures.append(f"Markdown pass rate ({md_pr:.1%}) contradicts JSON pass rate ({overall_pass_rate:.1%})")
                tier_6_failed = True

    # If final_tier == 6 but any tier 6 requirement is false, that's a failure
    if final_tier == 6 and tier_6_failed:
        failures.append("Final tier is claimed as 6, but one or more Tier 6 requirements are not met")

    # Filter failures if not a proving run (i.e. smoke run / negative result)
    is_proving_run = (verdict_text == "DNU AGI PROVEN") or (final_tier == 6)
    if not is_proving_run:
        structural_terms = [
            "final_verdict.txt is missing",
            "invalid verdict",
            "required artifact",
            "failed to parse",
            "manifest file",
            "manifest hash mismatch",
            "governance",
            "leakage",
            "receipt",
            "runtime manifest",
            "runtime policy",
            "runtime health",
            "resource trace",
            "required runtime probes",
            "task_trace",
            "proof integrity",
            "ablation",
            "outperform",
            "model lane",
            "structured proof solver",
            "prompt-derived symbolic repair",
            "synthetic",
            "projected",
            "verdict is dnu agi proven, but validation failed"
        ]
        allowed_failures = []
        for f in failures:
            if any(term in f.lower() for term in structural_terms):
                allowed_failures.append(f)
        failures = allowed_failures

    if len(failures) > 0:
        print("\nVALIDATION_STATUS: FAIL")
        print("\nFailed Requirements:")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)
    else:
        print("\nVALIDATION_STATUS: PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()
