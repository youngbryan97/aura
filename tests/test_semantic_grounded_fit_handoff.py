"""Prepared source custody and one contained fit are required before handoff."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.run_semantic_grounded_fit_handoff import (
    digest,
    fit_jobs,
    preparation_paths,
    verified_document,
    verify_fit_log,
    verify_preparation,
    wait_for_model_lane,
)


def supervised(tmp_path):
    cwd = tmp_path / "frozen"
    command = ["/python", "-u", str(cwd / "tools/fit_semantic_grounded_binding.py"),
        "--native", "--prepare-only", "--parent", str(tmp_path / "parent.json"),
        "--source-report", str(tmp_path / "source.json"), "--folds", str(tmp_path / "folds.json"),
        "--bank", str(tmp_path / "bank"), "--directory", str(tmp_path / "native"),
        "--authority-key-file", str(tmp_path / "authority"),
        "--bundle", "one=" + str(tmp_path / "one"), "--bundle", "two=" + str(tmp_path / "two"),
        "--steps", "512", "--native-layer-kinds", "lora,product,lora,silu", "--native-rank", "32",
        "--native-operation-field", "--complete-program-objective",
        "--program-pool-directory", str(tmp_path / "pools")]
    return {"command": command, "cwd": str(cwd), "plan_sha256": "a" * 64, "command_sha256": "b" * 64}


def write_document(path, body, field="receipt_sha256"):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {key: item for key, item in body.items() if key != field}
    value[field] = digest(value)
    path.write_text(json.dumps(value))
    return value


def test_exact_command_loses_only_preparation_mode_and_uses_original_frozen_checkout(tmp_path):
    value = supervised(tmp_path)
    paths = preparation_paths(value)
    assert paths["command"] == [part for part in value["command"] if part != "--prepare-only"]
    jobs = fit_jobs(paths, tmp_path / "handoff")
    assert jobs[0]["command"] == paths["command"]
    assert all(row["cwd"] == value["cwd"] and row["max_invocations"] == 1 for row in jobs)
    assert jobs[1]["command"][2] == str(paths["cwd"] / "tools/verify_semantic_grounded_fit.py")
    assert "--directory" in jobs[1]["command"]


@pytest.mark.parametrize("fault", ["cwd", "python", "tool", "mode", "duplicate_mode", "resume",
    "resume_if_available", "native", "repeat_native", "parent", "repeat_parent", "authority",
    "bundle_missing", "bundle_duplicate", "bundle_relative", "bundle_empty", "program_path", "field"])
def test_incomplete_or_changed_preparation_cannot_start_training(tmp_path, fault):
    value = supervised(tmp_path)
    command = value["command"]
    if fault == "cwd":
        value["cwd"] = "relative"
    elif fault == "python":
        command[0] = "python"
    elif fault == "tool":
        command[2] = str(tmp_path / "other/fit_semantic_grounded_binding.py")
    elif fault == "mode":
        command.remove("--prepare-only")
    elif fault == "duplicate_mode":
        command += ["--prepare-only"]
    elif fault in {"resume", "resume_if_available"}:
        command += ["--" + fault.replace("_", "-")]
    elif fault == "native":
        command.remove("--native")
    elif fault == "repeat_native":
        command += ["--native"]
    elif fault == "parent":
        command[command.index("--parent") + 1] = "parent.json"
    elif fault == "repeat_parent":
        command += ["--parent", str(tmp_path / "other-parent.json")]
    elif fault == "authority":
        command[command.index("--authority-key-file") + 1] = "--steps"
    elif fault == "bundle_missing":
        while "--bundle" in command:
            index = command.index("--bundle")
            del command[index:index + 2]
    elif fault == "bundle_duplicate":
        command += ["--bundle", command[command.index("--bundle") + 1]]
    elif fault in {"bundle_relative", "bundle_empty"}:
        command[command.index("--bundle") + 1] = "one=" + ("one" if fault == "bundle_relative" else "")
    elif fault == "program_path":
        command[command.index("--program-pool-directory") + 1] = "pools"
    else:
        command.remove("--native-operation-field")
    with pytest.raises(ValueError):
        preparation_paths(value)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import hashlib

    from core.learning.semantic_grounded_program_objective import program_objective_contract
    from tests.test_semantic_grounded_program_objective import source_pool
    from tools import train_nested_semantic_ranker

    paths = preparation_paths(supervised(tmp_path))
    for key in ("parent", "source", "folds"):
        paths[key].write_bytes((key + " source custody").encode())
    code = paths["cwd"] / "core/fixture.py"
    code.parent.mkdir(parents=True)
    code.write_text("fixture = True\n")
    implementation = {"core/fixture.py": hashlib.sha256(code.read_bytes()).hexdigest()}
    _bridge, field, program = source_pool()
    sources = {identity: replace(program, source_id=identity) for identity in ("a" * 64, "b" * 64)}
    pools = {}
    for identity, source in sources.items():
        basis = {"source_id": identity, "implementation": implementation,
            "source_annotation": {"source_text_sha256": identity, "source_token_ids": [1] * 10}}
        pools[identity] = write_document(paths["pools"] / (identity + ".json"),
            {"basis": basis, "basis_sha256": digest(basis), "programs": source.receipt()})
    outer = {"plan_sha256": "c" * 64, "fit_ids": ["a" * 64],
        "calibration_ids": ["b" * 64, "d" * 64], "held_ids": ["e" * 64]}
    monkeypatch.setattr(train_nested_semantic_ranker, "_verified_pair", lambda path: (outer, {}))
    plan = {"implementation": implementation, "source_basis": {
        "parent_sha256": hashlib.sha256(paths["parent"].read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(paths["source"].read_bytes()).hexdigest(),
        "folds_sha256": hashlib.sha256(paths["folds"].read_bytes()).hexdigest(), "bank_plan_sha256": outer["plan_sha256"]},
        "installed_arithmetic": {"MLX_ENABLE_TF32": "0"}, "fit_ids": ["a" * 64],
        "calibration_ids": ["b" * 64], "operation_field_contract": field.to_contract(),
        "program_objective_contract": program_objective_contract(sources, 1.)}
    custody = paths["native"].parent / (paths["native"].name + "-native-custody")
    plan = write_document(custody / "plan.json", plan, "plan_sha256")
    return paths, custody, plan, pools, outer


def test_preparation_checks_all_program_factors_and_preserves_partial_search_status(prepared):
    paths, _custody, plan, pools, _outer = prepared
    result = verify_preparation(paths)
    assert result["plan_sha256"] == plan["plan_sha256"]
    assert result["source_pool_receipts"] == {identity: value["receipt_sha256"] for identity, value in pools.items()}
    assert result["source_fit_count"] == result["source_calibration_count"] == 1
    assert not result["model_weights_loaded"] and result["semantic_success"] is None


@pytest.mark.parametrize("fault", ["native", "prefixes", "completion", "code", "source", "arithmetic",
    "overlap", "missing_fit", "other_calibration", "held", "unsafe_identity", "pool_missing",
    "pool_checksum", "pool_source", "pool_code", "pool_factor", "pool_unproved", "pool_contract"])
def test_preparation_refuses_drift_and_preexisting_fit_evidence(prepared, fault):
    paths, custody, plan, pools, outer = prepared
    if fault in {"native", "prefixes"}:
        (paths["native"] if fault == "native" else custody / "prefixes").mkdir()
    elif fault == "completion":
        (custody / "completion.json").write_text("{}")
    elif fault == "code":
        (paths["cwd"] / "core/fixture.py").write_text("changed = True\n")
    elif fault == "source":
        paths["source"].write_bytes(b"changed")
    elif fault == "arithmetic":
        plan["installed_arithmetic"]["MLX_ENABLE_TF32"] = "1"
    elif fault == "overlap":
        plan["calibration_ids"] = plan["fit_ids"]
    elif fault == "missing_fit":
        outer["fit_ids"] += ["f" * 64]
    elif fault == "other_calibration":
        outer["calibration_ids"] = ["d" * 64]
    elif fault == "held":
        outer["held_ids"] = plan["calibration_ids"]
    elif fault == "unsafe_identity":
        plan["fit_ids"] = ["../escape"]
    elif fault == "pool_contract":
        plan["program_objective_contract"]["source_pool_sha256"] = "f" * 64
    else:
        identity = plan["fit_ids"][0]
        path, pool = paths["pools"] / (identity + ".json"), deepcopy(pools[identity])
        if fault == "pool_missing":
            path.unlink()
        elif fault == "pool_checksum":
            pool["receipt_sha256"] = "0" * 64
            path.write_text(json.dumps(pool))
        else:
            if fault == "pool_source":
                pool["basis"]["source_annotation"]["source_text_sha256"] = "e" * 64
            elif fault == "pool_code":
                pool["basis"]["implementation"] = {}
            elif fault == "pool_factor":
                pool["programs"]["charts"][0]["graphs"][0]["baseline_score"] += 1.
            else:
                pool["programs"]["charts"][0]["graphs"][0]["comparison"]["status"] = "unknown"
            pool["basis_sha256"] = digest(pool["basis"])
            write_document(path, pool)
    write_document(custody / "plan.json", plan, "plan_sha256")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_preparation(paths)


def test_checksum_is_independent_of_serialization_and_refuses_changed_content(tmp_path):
    path = tmp_path / "receipt.json"
    value = write_document(path, {"answer": 42})
    assert verified_document(path) == value
    path.write_text(json.dumps({**value, "answer": 41}))
    with pytest.raises(ValueError, match="checksum"):
        verified_document(path)


def test_lane_wait_neither_loads_nor_evicts_and_is_bounded():
    now, calls = [0.], []
    owner = SimpleNamespace(owner_id="resident")
    def advance(seconds):
        now[0] += seconds
    def observe():
        calls.append(now[0])
        return [owner] if now[0] < 30 else []
    wait_for_model_lane(timeout=60., observe=observe, clock=lambda: now[0], sleep=advance)
    assert calls == [0., 30.]
    with pytest.raises(TimeoutError, match="no fit was launched"):
        wait_for_model_lane(timeout=2., observe=lambda: [owner], clock=lambda: now[0], sleep=advance)
    assert now[0] == 32.


@pytest.mark.parametrize("bound", [0., -1., float("nan"), float("inf"), 18001.])
def test_lane_wait_refuses_unbounded_or_invalid_allowances(bound):
    with pytest.raises(ValueError):
        wait_for_model_lane(timeout=bound, observe=lambda: pytest.fail("invalid wait observed owners"))


def verification():
    return {"schema": "aura.grounded_binding_verification.v1", "learned_checkpoint_selected": True,
        "artifacts_verified": True, "selected_step": 32, "completed_updates": 64,
        "current_implementation_drift": [], "source_fit_count": 1, "source_calibration_count": 1,
        "model_weights_loaded": False, "held_sources_scored": False, "semantic_success": None,
        "qualification_evidence": False, "serving_authority": False}


@pytest.mark.parametrize("fault", [None, "duplicate", "zero", "false_step", "unfinished", "drift",
    "wrong_count", "claim", "checksum"])
def test_fit_log_requires_one_verified_learned_checkpoint_not_a_claim(tmp_path, fault):
    value = verification()
    if fault == "zero":
        value["selected_step"] = 0
    elif fault == "false_step":
        value["selected_step"] = True
    elif fault == "unfinished":
        value["completed_updates"] = 31
    elif fault == "drift":
        value["current_implementation_drift"] = ["module"]
    elif fault == "wrong_count":
        value["source_fit_count"] = 2
    elif fault == "claim":
        value["semantic_success"] = True
    value["receipt_sha256"] = digest(value)
    if fault == "checksum":
        value["receipt_sha256"] = "f" * 64
    path = tmp_path / "verify.log"
    path.write_text("diagnostic before\n" + json.dumps(value) + "\ndiagnostic after\n"
        + (json.dumps(value) if fault == "duplicate" else ""))
    prepared = {"source_fit_count": 1, "source_calibration_count": 1}
    if fault is None:
        assert verify_fit_log(path, prepared) == value
    else:
        with pytest.raises(ValueError):
            verify_fit_log(path, prepared)


@pytest.mark.parametrize("failure", [None, "preparation_failed", "other_terminal", "lane_occupied", "fit_failed"])
def test_handoff_runs_exact_fit_then_independent_verify_only_after_prerequisites(
        tmp_path, monkeypatch, prepared, failure):
    from core.runtime import detached_subprocess_broker
    from tools import (
        run_detached_step,
        run_semantic_grounded_fit_handoff,
        run_semantic_native_micro_stages,
    )

    paths, _custody, _plan, _pools, _outer = prepared
    for name in ("fit_semantic_grounded_binding.py", "verify_semantic_grounded_fit.py"):
        script = paths["cwd"] / "tools" / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# fixed executable fixture\n")
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    value = supervised(tmp_path)
    (supervisor / run_detached_step.PLAN_FILE).write_text(json.dumps(value))
    monkeypatch.setattr(run_detached_step, "_verify_plan", lambda *args: None)
    monkeypatch.setattr(detached_subprocess_broker, "broker_available", lambda: True)
    def wait(*args, **kwargs):
        if failure == "preparation_failed":
            raise ValueError("preparation failed")
        return {"plan_sha256": "f" * 64 if failure == "other_terminal" else value["plan_sha256"],
            "command_sha256": value["command_sha256"], "passed": True, "containment_verified": True,
            "returncode": 0, "timed_out": False, "receipt_sha256": "c" * 64}
    monkeypatch.setattr(run_semantic_native_micro_stages, "wait_for_fit", wait)
    def lane(**kwargs):
        if failure == "lane_occupied":
            raise TimeoutError("lane occupied")
    monkeypatch.setattr(run_semantic_grounded_fit_handoff, "wait_for_model_lane", lane)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 2:
            report = verification()
            report["receipt_sha256"] = digest(report)
            log = kwargs["stdout_path"]
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(report) + "\n")
        passed = failure != "fit_failed"
        return SimpleNamespace(returncode=0 if passed else 1, status="passed" if passed else "failed",
            timed_out=False, containment_verified=True, receipt_sha256="d" * 64)
    monkeypatch.setattr(detached_subprocess_broker, "run_brokered_process", run)
    handoff = tmp_path / "handoff"
    monkeypatch.setattr("sys.argv", ["handoff", "--preparation-supervisor", str(supervisor),
        "--preparation-plan-sha256", value["plan_sha256"], "--directory", str(handoff)])
    if failure is None:
        assert run_semantic_grounded_fit_handoff.main() == 0
        assert [command for command, _kwargs in calls] == [row["command"] for row in fit_jobs(paths, handoff)]
        assert all(kwargs["cwd"] == paths["cwd"] for _command, kwargs in calls)
        complete = verified_document(handoff / "completion.json")
        assert not complete["g03_complete"] and not complete["serving_authority"]
        with pytest.raises(FileExistsError):
            run_semantic_grounded_fit_handoff.main()
        assert len(calls) == 2
    else:
        with pytest.raises((ValueError, TimeoutError)):
            run_semantic_grounded_fit_handoff.main()
        assert len(calls) == (1 if failure == "fit_failed" else 0)
        assert not (handoff / "completion.json").exists()


def test_policy_preparation_creates_broker_log_parent_without_starting_any_work(tmp_path, monkeypatch):
    from core.runtime import detached_subprocess_broker
    from tools import run_detached_step, run_semantic_grounded_fit_handoff

    value = supervised(tmp_path)
    for name in ("fit_semantic_grounded_binding.py", "verify_semantic_grounded_fit.py"):
        script = tmp_path / "frozen/tools" / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# fixed executable fixture\n")
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    (supervisor / run_detached_step.PLAN_FILE).write_text(json.dumps(value))
    monkeypatch.setattr(run_detached_step, "_verify_plan", lambda *args: None)
    monkeypatch.setattr(detached_subprocess_broker, "run_brokered_process",
        lambda *args, **kwargs: pytest.fail("policy construction started training"))
    directory = tmp_path / "handoff"
    policy = directory / "broker-policy.json"
    monkeypatch.setattr("sys.argv", ["handoff", "--preparation-supervisor", str(supervisor),
        "--preparation-plan-sha256", value["plan_sha256"], "--directory", str(directory),
        "--policy-output", str(policy)])
    assert run_semantic_grounded_fit_handoff.main() == 0
    jobs = json.loads(policy.read_text())
    assert jobs == [{key: item for key, item in row.items() if key != "name"}
        for row in fit_jobs(preparation_paths(value), directory)]
    assert (directory / "logs").is_dir()
    assert not (directory / "handoff.json").exists()
