"""Capability scores were rebuilt from signed booleans, not from grading.

`validate_capability_report` reconstructs every class score from
`correctness_receipt.payload.correct` — a value a pinned verifier signed. The
deterministic graders are in the same module, the answers are in the same
report, and neither was run. Signature agreement proves who said a thing. It
has never proved the thing is true, so a verifier that signed a wrong verdict
produced a score that reproduced perfectly.

The model manifest had the same shape one layer down. `_validate_model_manifest`
checks that the manifest hashes itself, which proves internal consistency and
nothing about the weights that were loaded: every field can be fabricated
together and the self-digest agrees. Nothing ever opened `model_path`. And the
measurement subject is derived FROM the manifest, so on its own it could only
ever agree with it.

Two smaller holes in the same role map: a file could be claimed by two roles —
which matters because the adapter identity check reads `roles["adapters"]`, so a
weights file listed there would be attested as an adapter — and a declared file
could belong to no role at all, shipped with the checkpoint and attested by
nothing.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json

import pytest

import core.brain.frontier_gap as frontier_gap
from core.brain.frontier_gap import (
    MODEL_MANIFEST_RESOLUTION_SCHEMA,
    canonical_json_bytes,
    resolve_model_manifest,
    sha256_json,
)

# ─────────────────────────── the grader runs again


def test_the_graders_are_rerun_in_both_validators():
    source = inspect.getsource(frontier_gap)

    assert source.count("_regrade_against_deterministic_grader(") == 3


def test_a_signed_verdict_that_contradicts_the_grader_is_refused():
    class _Item:
        item_id = "int-3"

        @staticmethod
        def grade(text):
            return text.strip() == "42"

    with pytest.raises(ValueError, match="contradicts the deterministic grader"):
        frontier_gap._regrade_against_deterministic_grader(
            item=_Item(),
            answer="41",
            signed_correct=True,
            subject="capability candidate",
        )


def test_a_signed_verdict_that_agrees_passes():
    class _Item:
        item_id = "int-3"

        @staticmethod
        def grade(text):
            return text.strip() == "42"

    frontier_gap._regrade_against_deterministic_grader(
        item=_Item(),
        answer="42",
        signed_correct=True,
        subject="capability candidate",
    )


def test_a_signed_incorrect_verdict_must_also_agree():
    """The asymmetry matters: signing "wrong" on a right answer suppresses a
    score just as effectively as the reverse inflates one."""

    class _Item:
        item_id = "int-3"

        @staticmethod
        def grade(text):
            return text.strip() == "42"

    with pytest.raises(ValueError, match="signed=False regraded=True"):
        frontier_gap._regrade_against_deterministic_grader(
            item=_Item(),
            answer="42",
            signed_correct=False,
            subject="frontier reference",
        )


def test_the_failure_names_the_item():
    class _Item:
        item_id = "code-7"

        @staticmethod
        def grade(text):
            del text
            return False

    with pytest.raises(ValueError, match="code-7"):
        frontier_gap._regrade_against_deterministic_grader(
            item=_Item(),
            answer="anything",
            signed_correct=True,
            subject="capability candidate",
        )


def test_the_real_graders_are_deterministic_and_executable():
    """The whole fix rests on this: the module holds graders it can rerun."""
    from core.brain.frontier_gap import _exact_integer_grader

    grader = _exact_integer_grader(12)

    assert grader("12") is True
    assert grader("13") is False


# ─────────────────────────── the manifest meets the disk


def _manifest(tmp_path, *, files):
    entries = []
    for name, payload in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        entries.append(
            {
                "path": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema": frontier_gap.MODEL_MANIFEST_SCHEMA,
        "model_path": str(tmp_path),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "roles": {
            "weights": [name for name in files if name.endswith(".safetensors")],
            "configuration": [name for name in files if name.endswith(".json")],
            "tokenizer": [name for name in files if "tokenizer" in name],
            "adapters": [],
        },
    }


def test_a_manifest_that_matches_the_disk_resolves(tmp_path):
    manifest = _manifest(
        tmp_path,
        files={"model.safetensors": b"weights", "config.json": b"{}", "tokenizer.model": b"tok"},
    )

    resolution = resolve_model_manifest(manifest)

    assert resolution["resolved"] is True
    assert resolution["schema"] == MODEL_MANIFEST_RESOLUTION_SCHEMA
    assert resolution["files_present"] == 3
    assert resolution["files_digested"] == 3
    assert resolution["mismatches"] == []


def test_a_fabricated_digest_is_caught(tmp_path):
    manifest = _manifest(tmp_path, files={"config.json": b"{}"})
    manifest["files"][0]["sha256"] = "0" * 64

    resolution = resolve_model_manifest(manifest)

    assert resolution["resolved"] is False
    assert resolution["reason"] == "manifest_does_not_match_disk"
    assert resolution["mismatches"] == ["sha256:config.json"]


def test_a_fabricated_size_is_caught(tmp_path):
    manifest = _manifest(tmp_path, files={"config.json": b"{}"})
    manifest["files"][0]["size"] = 999_999

    resolution = resolve_model_manifest(manifest)

    assert resolution["mismatches"] == ["size:config.json"]


def test_a_file_that_does_not_exist_is_caught(tmp_path):
    manifest = _manifest(tmp_path, files={"config.json": b"{}"})
    manifest["files"].append(
        {"path": "invented.safetensors", "size": 4, "sha256": "1" * 64}
    )

    resolution = resolve_model_manifest(manifest)

    assert resolution["mismatches"] == ["missing:invented.safetensors"]


def test_an_absent_checkpoint_reports_unresolved_not_clean():
    """A report validated on another machine cannot resolve anything, and
    "could not check" is a different answer from "checked and correct"."""
    resolution = resolve_model_manifest(
        {"model_path": "/nonexistent/checkpoint", "files": [{"path": "x", "size": 0}]}
    )

    assert resolution["resolved"] is False
    assert resolution["reason"] == "model_path_absent_on_this_host"


def test_a_huge_file_is_size_checked_without_being_read(tmp_path):
    """Reading a whole checkpoint would take minutes; the receipt says which
    files were digested rather than implying all of them were."""
    manifest = _manifest(tmp_path, files={"model.safetensors": b"x" * 32})
    manifest["files"][0]["size"] = 32

    monkey = frontier_gap._MANIFEST_FULL_DIGEST_MAX_BYTES
    try:
        frontier_gap._MANIFEST_FULL_DIGEST_MAX_BYTES = 8
        resolution = resolve_model_manifest(manifest)
    finally:
        frontier_gap._MANIFEST_FULL_DIGEST_MAX_BYTES = monkey

    assert resolution["resolved"] is True
    assert resolution["files_present"] == 1
    assert resolution["files_digested"] == 0


def test_the_report_validator_can_require_resolution():
    parameters = inspect.signature(frontier_gap.validate_capability_report).parameters

    assert "model_manifest_resolver" in parameters
    assert "require_resolved_model" in parameters
    assert parameters["require_resolved_model"].default is False


def test_requiring_resolution_refuses_an_unresolved_manifest():
    source = inspect.getsource(frontier_gap)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "was not resolved against the checkpoint" not in rendered:
            continue
        test = ast.get_source_segment(source, node.test) or ""
        assert "require_resolved_model" in test
        assert 'model_resolution.get("resolved") is not True' in test
        return
    raise AssertionError("the resolution requirement was not found")


# ─────────────────────────── the role map is a partition


def _validated(manifest):
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest = dict(manifest)
    manifest["manifest_sha256"] = sha256_json(body)
    return json.loads(canonical_json_bytes(manifest))


def _manifest_body(**roles):
    files = ["model.safetensors", "config.json", "tokenizer.model"]
    entries = [
        {"path": name, "size": 4, "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in files
    ]
    return {
        "schema": frontier_gap.MODEL_MANIFEST_SCHEMA,
        "model_path": "/models/test",
        "files": entries,
        "file_count": len(entries),
        "total_bytes": 12,
        "roles": {
            "weights": roles.get("weights", ["model.safetensors"]),
            "configuration": roles.get("configuration", ["config.json"]),
            "tokenizer": roles.get("tokenizer", ["tokenizer.model"]),
            "adapters": roles.get("adapters", []),
        },
    }


def test_a_complete_partition_validates():
    manifest = _validated(_manifest_body())

    assert frontier_gap._validate_model_manifest(manifest)["file_count"] == 3


def test_a_file_claimed_by_two_roles_is_refused():
    """roles["adapters"] is what the adapter identity check reads, so a
    weights file listed there would be attested as an adapter."""
    manifest = _validated(_manifest_body(adapters=["model.safetensors"]))

    with pytest.raises(ValueError, match="claimed by two roles"):
        frontier_gap._validate_model_manifest(manifest)


def test_an_unclassified_file_is_refused():
    manifest = _validated(_manifest_body(tokenizer=[]))

    with pytest.raises(ValueError, match="has no role"):
        frontier_gap._validate_model_manifest(manifest)


def test_the_self_digest_alone_still_proves_only_consistency():
    """Kept as the statement of what the manifest check IS: internally
    consistent, and silent about the weights that were loaded."""
    manifest = _validated(_manifest_body())
    manifest["model_path"] = "/models/somewhere-else-entirely"

    with pytest.raises(ValueError, match="digest mismatch"):
        frontier_gap._validate_model_manifest(manifest)


# ─────────────────────────── the counters come from the evidence


def _worker(fallbacks=(), output_tokens=8):
    return {
        "payload": {
            "fallbacks_used": list(fallbacks),
            "resource_usage": {"output_tokens": output_tokens},
        }
    }


def _correctness(checked=True):
    return {"payload": {"checked": checked, "correct": True}}


def test_a_clean_run_recomputes_to_zero():
    summary = frontier_gap._recomputed_execution_summary(
        evidence_items=[{"answer": "42"}, {"answer": "7"}],
        worker_receipts=[_worker(), _worker()],
        correctness_receipts=[_correctness(), _correctness()],
    )

    assert summary == {
        "attempted": 2,
        "completed": 2,
        "failed": 0,
        "invalid": 0,
        "empty": 0,
        "disqualifying_fallbacks": 0,
    }


def test_an_empty_answer_is_counted_even_when_the_report_says_zero():
    summary = frontier_gap._recomputed_execution_summary(
        evidence_items=[{"answer": "   "}],
        worker_receipts=[_worker()],
        correctness_receipts=[_correctness()],
    )

    assert summary["empty"] == 1


def test_an_execution_error_is_counted():
    summary = frontier_gap._recomputed_execution_summary(
        evidence_items=[{"answer": "x", "execution_error": "backend died"}],
        worker_receipts=[_worker()],
        correctness_receipts=[_correctness()],
    )

    assert summary["failed"] == 1


def test_an_unchecked_verdict_is_counted_invalid():
    summary = frontier_gap._recomputed_execution_summary(
        evidence_items=[{"answer": "x"}],
        worker_receipts=[_worker()],
        correctness_receipts=[_correctness(checked=False)],
    )

    assert summary["invalid"] == 1


def test_worker_fallbacks_are_counted_from_the_receipts():
    summary = frontier_gap._recomputed_execution_summary(
        evidence_items=[{"answer": "x"}],
        worker_receipts=[_worker(fallbacks=("cloud_route", "cached_answer"))],
        correctness_receipts=[_correctness()],
    )

    assert summary["disqualifying_fallbacks"] == 2


def test_a_run_that_used_a_fallback_is_not_claim_eligible():
    """A fallback means the measured lane is not the lane that answered."""
    with pytest.raises(ValueError, match="used fallbacks"):
        frontier_gap._reject_worker_fallbacks(
            [_worker(fallbacks=("cloud_route",))], subject="capability candidate"
        )


def test_a_run_with_no_fallbacks_passes():
    frontier_gap._reject_worker_fallbacks(
        [_worker(), _worker()], subject="frontier reference"
    )


def test_the_summary_recomputation_is_wired_into_the_validator():
    source = inspect.getsource(frontier_gap)

    assert "_recomputed_execution_summary(" in source
    assert "capability execution summary contradicts the item evidence" in source
    assert source.count("_reject_worker_fallbacks(") == 3


# ─────────────────────────── the answer is retokenized


def _output(item_id, answer):
    return {"item_id": item_id, "answer": answer}


def test_without_a_tokenizer_the_count_stays_worker_asserted():
    measurement = frontier_gap.measure_output_tokens(
        [_output("a", "hello")], [_worker(output_tokens=1)], token_counter=None
    )

    assert measurement["measured"] is False
    assert measurement["reason"] == "no_effective_tokenizer_supplied"


def test_a_measured_count_that_agrees_is_clean():
    measurement = frontier_gap.measure_output_tokens(
        [_output("a", "one two three")],
        [_worker(output_tokens=3)],
        token_counter=lambda text: len(text.split()),
    )

    assert measurement["measured"] is True
    assert measurement["over_budget"] == []
    assert measurement["disagreements"] == []


def test_a_long_answer_with_a_small_claimed_count_is_caught():
    """256 KiB of text could arrive with a claimed count at or below 256."""
    answer = " ".join(str(index) for index in range(400))

    measurement = frontier_gap.measure_output_tokens(
        [_output("a", answer)],
        [_worker(output_tokens=12)],
        token_counter=lambda text: len(text.split()),
    )

    assert measurement["over_budget"] == ["a:400"]
    assert measurement["disagreements"] == ["a:12!=400"]


def test_the_budget_it_measures_against_is_the_matched_one():
    from core.brain.frontier_evidence_v5 import MATCHED_BUDGET

    measurement = frontier_gap.measure_output_tokens(
        [], [], token_counter=lambda text: len(text)
    )

    assert measurement["budget_max_tokens"] == int(MATCHED_BUDGET["max_tokens"])


def test_the_validator_can_require_a_measured_count():
    parameters = inspect.signature(frontier_gap.validate_capability_report).parameters

    assert "output_token_counter" in parameters
    assert "require_measured_output_tokens" in parameters


def test_freshness_reaches_both_the_challenge_and_the_reference():
    from source_support import inlined_function_source

    # the validator as it runs: blocks the size gate moved into helpers are
    # read back at their call sites
    source = inlined_function_source(frontier_gap.__file__, "validate_capability_report")

    assert source.count("require_fresh_challenge=require_fresh_challenge") == 1
    assert source.count("require_fresh=require_fresh_challenge") == 1
    assert source.count("verification_time_unix=verification_time_unix") == 2


# ─────────────────────────── the stratum is the measurement's identity


def _stratum(**overrides):
    base = {
        "per_class": 4,
        "reference_runtime_manifest_sha256": "a" * 64,
        "seed": 11,
        "challenge_bundle_sha256": "b" * 64,
        "reference_scores": {"integer": 0.5},
    }
    base.update(overrides)
    return frontier_gap.comparison_stratum_sha256(**base)


def test_the_same_measurement_gives_the_same_stratum():
    assert _stratum() == _stratum()


def test_a_different_seed_is_a_different_stratum():
    """Different task draws are a different benchmark, not a model change."""
    assert _stratum(seed=12) != _stratum()


def test_a_different_challenge_is_a_different_stratum():
    assert _stratum(challenge_bundle_sha256="c" * 64) != _stratum()


def test_a_different_reference_result_is_a_different_stratum():
    assert _stratum(reference_scores={"integer": 0.75}) != _stratum()


def test_the_builder_and_the_validator_use_one_definition():
    source = inspect.getsource(frontier_gap)

    assert source.count("comparison_stratum_sha256(") == 3


# ─────────────────────────── restore is bounded


def test_an_ordinary_blob_restores():
    blob = {"evidence_class": "x", "items": [{"answer": "y"}]}

    assert frontier_gap._bounded_evidence_blob(blob, digest="d" * 64) is blob


def test_a_blob_that_is_not_an_object_is_refused():
    with pytest.raises(ValueError, match="missing or altered"):
        frontier_gap._bounded_evidence_blob(["not", "a", "report"], digest="d" * 64)


def test_a_deeply_nested_blob_is_refused():
    payload: dict = {"leaf": 1}
    for _ in range(frontier_gap.MAX_EVIDENCE_BLOB_DEPTH + 4):
        payload = {"next": payload}

    with pytest.raises(ValueError, match="nests deeper"):
        frontier_gap._bounded_evidence_blob(payload, digest="d" * 64)


def test_an_enormous_blob_is_refused():
    payload = {"items": ["x" * 1024] * (frontier_gap.MAX_EVIDENCE_BLOB_BYTES // 512)}

    with pytest.raises(ValueError, match="restore bound"):
        frontier_gap._bounded_evidence_blob(payload, digest="d" * 64)


def test_the_bound_runs_before_the_digest_is_computed():
    source = inspect.getsource(frontier_gap.GapLedger.from_dict)
    bound = source.index("_bounded_evidence_blob(snapshot")
    digest = source.index("sha256_json(snapshot) != digest")

    assert bound < digest


# ─────────────────────────── control and rejected evidence


def _non_capability(**overrides):
    report = {
        "schema_version": frontier_gap.SCHEMA_VERSION,
        "battery_version": frontier_gap.BATTERY_VERSION,
        "evidence_class": frontier_gap.REJECTED_EVIDENCE_CLASS,
        "capability_claim_eligible": False,
        "generated_at_unix": 1_700_000_000.0,
        "overall_candidate_score": 0.5,
        "effective_n": 2,
    }
    report.update(overrides)
    return report


def test_a_well_formed_rejected_report_validates():
    assert frontier_gap.validate_non_capability_report(_non_capability())


def test_a_claim_eligible_flag_is_refused():
    with pytest.raises(ValueError, match="cannot be claim eligible"):
        frontier_gap.validate_non_capability_report(
            _non_capability(capability_claim_eligible=True)
        )


def test_a_capability_class_is_refused_here():
    with pytest.raises(ValueError, match="evidence class is invalid"):
        frontier_gap.validate_non_capability_report(
            _non_capability(evidence_class=frontier_gap.CAPABILITY_EVIDENCE_CLASS)
        )


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_a_score_outside_the_battery_range_is_refused(score):
    with pytest.raises(ValueError, match="outside"):
        frontier_gap.validate_non_capability_report(
            _non_capability(overall_candidate_score=score)
        )


def test_items_must_match_the_effective_sample_count():
    with pytest.raises(ValueError, match="contradicts effective_n"):
        frontier_gap.validate_non_capability_report(
            _non_capability(items=[{"answer": "one"}])
        )


def test_a_retained_output_needs_no_invented_schema():
    """These classes retain outputs for audit; they were never capability
    items, so demanding a capability item shape would reject real evidence."""
    assert frontier_gap.validate_non_capability_report(
        _non_capability(items=[{"answer": "one"}, {"answer": "two"}])
    )


def test_a_declared_index_must_be_the_items_own_position():
    with pytest.raises(ValueError, match="contradicts its position"):
        frontier_gap.validate_non_capability_report(
            _non_capability(items=[{"index": 1}, {"index": 1}])
        )


def test_a_declared_task_class_must_exist():
    with pytest.raises(ValueError, match="unknown task class"):
        frontier_gap.validate_non_capability_report(
            _non_capability(items=[{"task_class": "telepathy"}, {}])
        )


def test_fully_graded_items_must_agree_with_the_summary_score():
    with pytest.raises(ValueError, match="contradicts its item verdicts"):
        frontier_gap.validate_non_capability_report(
            _non_capability(items=[{"correct": True}, {"correct": True}])
        )


def test_partially_graded_items_make_no_claim_to_contradict():
    assert frontier_gap.validate_non_capability_report(
        _non_capability(items=[{"correct": True}, {"answer": "ungraded"}])
    )


# ─────────────────────────── the prune count is committed to


def test_a_fresh_ledger_has_no_prune_chain():
    ledger = frontier_gap.GapLedger()

    assert ledger.pruned_chain_sha256 == ""
    assert ledger.to_dict()["retention"]["pruned_chain_sha256"] == ""


def test_a_prune_chain_without_a_count_is_refused():
    ledger = frontier_gap.GapLedger()
    payload = ledger.to_dict()
    payload["retention"]["pruned_chain_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="contradicts a zero count"):
        frontier_gap.GapLedger.from_dict(payload)


def test_the_chain_commits_to_the_running_count():
    """The anchor alone is one digest; the count sits inside every link, so
    the two cannot be written independently."""
    source = inspect.getsource(frontier_gap.GapLedger.add)

    assert '"previous_pruned_chain_sha256": self.pruned_chain_sha256' in source
    assert '"pruned_count": self.pruned_count' in source


def test_an_older_retention_block_without_the_chain_still_restores():
    """Only a ledger that actually pruned is required to carry it."""
    ledger = frontier_gap.GapLedger()
    payload = ledger.to_dict()
    payload["retention"].pop("pruned_chain_sha256")

    assert frontier_gap.GapLedger.from_dict(payload).pruned_count == 0


# ─────────────────────────── pruned blobs are reclaimable


def test_the_ledger_accepts_a_blob_remover():
    parameters = inspect.signature(frontier_gap.GapLedger.add).parameters

    assert "evidence_blob_remover" in parameters


def test_a_digest_still_referenced_is_never_removed():
    source = inspect.getsource(frontier_gap.GapLedger.add)

    assert "still_referenced" in source
    assert "if not digest or digest in still_referenced:" in source


def test_a_failed_reclaim_does_not_break_the_chain():
    source = inspect.getsource(frontier_gap.GapLedger.add)

    assert "could not reclaim pruned evidence" in source


def test_erased_history_is_diagnosed_before_the_missing_chain():
    """Both refusals are correct; the sharper one has to be the one reported."""
    from core.brain.frontier_gap import CONTROL_EVIDENCE_CLASS, GapLedger

    payload = {
        "schema_version": frontier_gap.SCHEMA_VERSION,
        "evidence_class": CONTROL_EVIDENCE_CLASS,
        "capability_claim_eligible": False,
        "retention": {
            "max_entries": 8,
            "pruned_count": 5,
            "pruned_through_sha256": "a" * 64,
            "retains_outputs_in_content_addressed_blobs": True,
        },
        "head_entry_sha256": "a" * 64,
        "runs": [],
        "trend": {},
    }

    with pytest.raises(ValueError, match="retains no entries"):
        GapLedger.from_dict(payload, evidence_class=CONTROL_EVIDENCE_CLASS)


# ─────────────────────────── the attestation covers what runs


def test_the_import_closure_reaches_past_the_named_roots():
    closure = frontier_gap.first_party_import_closure(frontier_gap._EXECUTION_ROOTS)

    assert len(closure) > len(frontier_gap._EXECUTION_ROOTS)
    assert "core/brain/frontier_gap.py" in closure


def test_a_relative_import_is_resolved_against_its_package():
    """A scanner in this repo previously reported relatively-imported modules
    as unreachable, and a retirement pass was built on that."""
    closure = frontier_gap.first_party_import_closure(
        ("core/brain/llm/latent_cortex/engine.py",)
    )

    assert any(path.startswith("core/brain/llm/latent_cortex/") for path in closure)


def test_the_closure_walk_is_bounded():
    closure = frontier_gap.first_party_import_closure(frontier_gap._EXECUTION_ROOTS)

    assert len(closure) <= frontier_gap._MAX_COMPONENT_CLOSURE


def test_a_truncated_walk_can_never_report_completeness():
    """A bound that turns into a pass is the failure this check exists for."""
    coverage = frontier_gap.source_component_coverage(
        {path: "x" * 64 for path in frontier_gap._EXECUTION_ROOTS}
    )

    if coverage["closure_truncated"]:
        assert coverage["complete"] is False


def test_the_coverage_states_the_gap_rather_than_implying_none():
    coverage = frontier_gap.source_component_coverage(
        {path: "x" * 64 for path in frontier_gap._EXECUTION_ROOTS}
    )

    assert coverage["declared"] == len(frontier_gap._EXECUTION_ROOTS)
    assert coverage["covered"] < coverage["closure_size"]
    assert coverage["missing"]


def test_the_report_can_require_complete_coverage():
    parameters = inspect.signature(frontier_gap.validate_capability_report).parameters

    assert "require_complete_component_coverage" in parameters


# ─────────────────────────── the clean workspace is looked at


def _provenance(**overrides):
    base = {
        "commit_sha": "a" * 40,
        "tree_sha": "b" * 40,
        "clean": True,
        "issues": [],
        "workspace_diff_sha256": "c" * 64,
        "index_diff_sha256": "d" * 64,
        "untracked_content_sha256": "e" * 64,
        "workspace_state_sha256": "f" * 64,
    }
    base.update(overrides)
    return base


def test_without_a_resolver_the_workspace_claim_stays_unresolved():
    resolution = frontier_gap.resolve_workspace_state(
        _provenance(), workspace_resolver=None
    )

    assert resolution["resolved"] is False
    assert resolution["reason"] == "no_workspace_resolver_supplied"


def test_a_matching_live_workspace_resolves():
    provenance = _provenance()

    resolution = frontier_gap.resolve_workspace_state(
        provenance, workspace_resolver=lambda: dict(provenance)
    )

    assert resolution["resolved"] is True


def test_a_dirty_tree_attesting_a_clean_one_is_caught():
    provenance = _provenance()
    observed = dict(provenance)
    observed["clean"] = False

    resolution = frontier_gap.resolve_workspace_state(
        provenance, workspace_resolver=lambda: observed
    )

    assert resolution["resolved"] is False
    assert "differs:clean" in resolution["mismatches"]


def test_a_field_the_observer_never_measured_is_not_a_match():
    provenance = _provenance()
    observed = {key: value for key, value in provenance.items() if key != "tree_sha"}

    resolution = frontier_gap.resolve_workspace_state(
        provenance, workspace_resolver=lambda: observed
    )

    assert "unobserved:tree_sha" in resolution["mismatches"]


def test_a_resolver_that_raises_reports_the_failure():
    def broken():
        raise OSError("no repository here")

    resolution = frontier_gap.resolve_workspace_state(
        _provenance(), workspace_resolver=broken
    )

    assert resolution["resolved"] is False
    assert resolution["reason"].startswith("workspace_resolver_failed:")


# ─────────────────────────── the window has to bracket the run


def test_a_window_without_observation_times_is_unbracketed():
    bracketing = frontier_gap.stability_bracketing(
        {}, run_started=100.0, run_completed=200.0, subject="capability source"
    )

    assert bracketing["bracketed"] is False
    assert bracketing["reason"] == "window_records_no_observation_times"


def test_a_window_that_brackets_the_run_is_accepted():
    bracketing = frontier_gap.stability_bracketing(
        {"before_observed_at_unix": 99.0, "after_observed_at_unix": 201.0},
        run_started=100.0,
        run_completed=200.0,
        subject="capability source",
    )

    assert bracketing["bracketed"] is True


def test_a_first_observation_after_the_run_began_is_refused():
    with pytest.raises(ValueError, match="first observed after the run began"):
        frontier_gap.stability_bracketing(
            {"before_observed_at_unix": 150.0, "after_observed_at_unix": 201.0},
            run_started=100.0,
            run_completed=200.0,
            subject="capability source",
        )


def test_a_last_observation_before_the_run_finished_is_refused():
    with pytest.raises(ValueError, match="last observed before the run completed"):
        frontier_gap.stability_bracketing(
            {"before_observed_at_unix": 99.0, "after_observed_at_unix": 150.0},
            run_started=100.0,
            run_completed=200.0,
            subject="capability model",
        )


def test_out_of_order_observations_are_refused():
    with pytest.raises(ValueError, match="out of order"):
        frontier_gap.stability_bracketing(
            {"before_observed_at_unix": 150.0, "after_observed_at_unix": 90.0},
            run_started=200.0,
            run_completed=300.0,
            subject="capability source",
        )


# ─────────────────────────── the runtime names measured material


def _runtime(**overrides):
    base = {
        "worker_implementation_sha256": "1" * 64,
        "prompt_template_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
    }
    base.update(overrides)
    return base


def test_an_unattested_component_leaves_the_field_unbound():
    binding = frontier_gap.runtime_identity_binding(
        _runtime(), source_components={}, model_files={}, tokenizer_paths=()
    )

    assert binding["complete"] is False
    assert "worker_implementation_sha256" in binding["unbound"]
    assert all(
        row["reason"] == "component_not_attested"
        for row in binding["bindings"]
        if row["field"] == "worker_implementation_sha256"
    )


def test_a_digest_that_differs_from_the_measured_component_is_unbound():
    binding = frontier_gap.runtime_identity_binding(
        _runtime(),
        source_components={"core/brain/llm/mlx_worker.py": "9" * 64},
        model_files={},
        tokenizer_paths=(),
    )

    row = next(
        item
        for item in binding["bindings"]
        if item["field"] == "worker_implementation_sha256"
    )
    assert row["bound"] is False
    assert row["reason"] == "digest_differs_from_measured_component"


def test_a_matching_component_binds():
    binding = frontier_gap.runtime_identity_binding(
        _runtime(),
        source_components={"core/brain/llm/mlx_worker.py": "1" * 64},
        model_files={},
        tokenizer_paths=(),
    )

    row = next(
        item
        for item in binding["bindings"]
        if item["field"] == "worker_implementation_sha256"
    )
    assert row["bound"] is True


def test_the_tokenizer_digest_binds_to_the_tokenizer_role_files():
    files = {"tokenizer.model": "aa" * 32, "config.json": "bb" * 32}
    expected = sha256_json(sorted([files["tokenizer.model"]]))

    binding = frontier_gap.runtime_identity_binding(
        _runtime(tokenizer_sha256=expected),
        source_components={},
        model_files=files,
        tokenizer_paths=("tokenizer.model",),
    )

    row = next(
        item for item in binding["bindings"] if item["field"] == "tokenizer_sha256"
    )
    assert row["bound"] is True


def test_no_tokenizer_role_files_leaves_the_tokenizer_unbound():
    binding = frontier_gap.runtime_identity_binding(
        _runtime(), source_components={}, model_files={}, tokenizer_paths=()
    )

    row = next(
        item for item in binding["bindings"] if item["field"] == "tokenizer_sha256"
    )
    assert row["reason"] == "no_tokenizer_role_files"


def test_the_report_can_require_bound_runtime_identity():
    parameters = inspect.signature(frontier_gap.validate_capability_report).parameters

    assert "require_bound_runtime_identity" in parameters


# ─────────────────────────── the grader digest covers its runtime


def test_the_environment_is_part_of_the_grader_digest():
    environment = frontier_gap.grader_execution_environment()

    for key in (
        "python_version",
        "python_implementation",
        "dynamic_execution_gateway_sha256",
        "allowed_builtins",
    ):
        assert key in environment


def test_the_allowed_builtins_are_committed_to():
    environment = frontier_gap.grader_execution_environment()

    assert environment["allowed_builtins"] == sorted(
        frontier_gap._CODE_GRADER_BUILTINS
    )


def test_the_digest_changes_when_the_environment_does(monkeypatch):
    before = frontier_gap.grader_implementation_sha256("exact_integer.v2")
    monkeypatch.setattr(
        frontier_gap,
        "grader_execution_environment",
        lambda: {"python_version": "0.0.0"},
    )

    assert frontier_gap.grader_implementation_sha256("exact_integer.v2") != before


def test_an_unknown_grader_still_refuses():
    with pytest.raises(ValueError, match="unknown grader implementation"):
        frontier_gap.grader_implementation_sha256("telepathy.v1")


# ─────────────────────────── the hidden cases discriminate


def test_the_battery_carries_adversarial_cases():
    source = inspect.getsource(frontier_gap._coding_items)

    assert "_ADVERSARIAL_HIDDEN_CASES" in source


def test_the_adversarial_set_covers_the_structural_collisions():
    cases = frontier_gap._ADVERSARIAL_HIDDEN_CASES

    assert any(len(case) == 1 for case in cases), "single-element sum/max/min collision"
    assert any(len(set(case)) == 1 and len(case) > 1 for case in cases), "all-equal"
    assert any(case == tuple(reversed(sorted(case))) and len(case) > 2 for case in cases)


def test_a_wrong_operation_cannot_pass_a_real_battery_item():
    import re

    items = [
        item
        for item in frontier_gap.build_battery(seed=5, per_class=3)
        if item.task_class == "coding"
    ]
    item = items[0]
    name = re.search(r"`(\w+)\(xs\)`", item.prompt).group(1)
    operation = name.split("_case_")[0]

    for shape in ("sum", "max", "min", "len"):
        body = f"```python\ndef {name}(xs):\n    return {shape}(xs)\n```"
        assert item.grade(body) is (shape == operation), shape


def test_the_grader_is_metamorphic_over_order():
    source = inspect.getsource(frontier_gap._code_grader)

    assert "reversed(case)" in source
    assert "order-invariant" in source


def test_a_truncated_closure_still_contains_its_own_roots():
    """A depth-first walk let the bound evict the files the closure started
    from, so a truncated result could omit its own roots."""
    closure = frontier_gap.first_party_import_closure(frontier_gap._EXECUTION_ROOTS)

    for root in frontier_gap._EXECUTION_ROOTS:
        assert root in closure, root


# ─────────────────────────── the diagnostic runner says what it enforced


def test_the_diagnostic_runner_enforces_the_wall_clock_it_can():
    source = inspect.getsource(frontier_gap.run_battery)

    assert "asyncio.wait_for(" in source
    assert 'MATCHED_BUDGET["hard_timeout_s"]' in source


def test_a_timed_out_item_is_an_invalid_miss_not_a_crash():
    source = inspect.getsource(frontier_gap.run_battery)

    assert "except TimeoutError as exc:" in source
    assert 'execution_error = "TimeoutError"' in source


def test_the_runner_reports_what_it_did_not_measure():
    source = inspect.getsource(frontier_gap.run_battery)

    assert '"budget_enforcement"' in source
    for unmeasured in ("process_isolation", "generation_calls", "output_tokens"):
        assert unmeasured in source


# ─────────────────────────── what the battery is


def test_the_scope_states_what_it_does_not_measure():
    scope = frontier_gap.battery_scope()

    assert scope["supports_general_capability_claim"] is False
    assert set(scope["task_classes"]) == set(frontier_gap._BATTERY_BUILDERS)
    for absent in ("contamination control", "transfer to unseen task families"):
        assert absent in scope["does_not_measure"]


def test_the_scope_rides_on_the_report():
    source = inspect.getsource(frontier_gap)

    assert source.count("battery_scope()") == 3


# ─────────────────────────── the candidate could not have read the answers


def _receipt_started(value):
    return {"payload": {"started_at_unix": value}}


def test_generation_before_the_reference_existed_is_provable():
    proof = frontier_gap.candidate_non_disclosure(
        reference_measured_at=500.0,
        candidate_run_started=100.0,
        candidate_worker_receipts=[_receipt_started(120.0), _receipt_started(300.0)],
    )

    assert proof["generation_preceded_reference"] is True
    assert proof["basis"] == "worker_start_precedes_reference_measurement"


def test_a_candidate_that_began_after_the_reference_falls_back_to_the_assertion():
    """Not a refusal: a later run may still have been sealed. It is the
    difference between proven and asserted, and the report now says which."""
    proof = frontier_gap.candidate_non_disclosure(
        reference_measured_at=100.0,
        candidate_run_started=200.0,
        candidate_worker_receipts=[_receipt_started(250.0)],
    )

    assert proof["generation_preceded_reference"] is False
    assert proof["basis"] == "sealed_execution_assertion_only"


def test_one_late_worker_is_enough_to_lose_the_ordering_proof():
    proof = frontier_gap.candidate_non_disclosure(
        reference_measured_at=500.0,
        candidate_run_started=100.0,
        candidate_worker_receipts=[_receipt_started(120.0), _receipt_started(900.0)],
    )

    assert proof["generation_preceded_reference"] is False


def test_no_receipts_prove_nothing():
    proof = frontier_gap.candidate_non_disclosure(
        reference_measured_at=500.0,
        candidate_run_started=100.0,
        candidate_worker_receipts=[],
    )

    assert proof["generation_preceded_reference"] is False
    assert proof["latest_candidate_worker_start_unix"] is None
