from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_rlc_reconciliation_controller as controller


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for relative, text in {
        "core/module.py": "VALUE = 1\n",
        "config/runtime.json": "{}\n",
        "tools/run_rlc_reconciliation_sweep.py": "print('sweep')\n",
        "tools/run_rlc_reconciliation_controller.py": "print('controller')\n",
        "pyproject.toml": "[project]\nname='test'\n",
        "requirements_lock.txt": "locked\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Aura Test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "aura-test@example.invalid"],
        check=True,
    )
    _commit_source(root, "initial fixture")
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", "-q"], check=True)
    return root


def _commit_source(root: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", message], check=True)
    return (
        subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )


def _source_commit(root: Path) -> str:
    return (
        subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )


def _prepared(tmp_path: Path):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")
    out = tmp_path / "campaign"
    config, manifest, key = controller.build_config(
        source_root=source,
        source_commit=_source_commit(source),
        model=model,
        out_dir=out,
        python=python,
        arms="full_stack",
        seed=7,
        per_domain=1,
        difficulty=2,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
    )
    config_path = out / "controller_config.json"
    controller.write_prepared_campaign(config_path, config, manifest, key)
    return source, out, config_path, controller.load_config(config_path)


def test_prepare_binds_one_clean_detached_git_identity(tmp_path: Path):
    source, _out, _config_path, config = _prepared(tmp_path)
    identity = config["source_git_identity"]
    assert identity["schema"] == controller.SOURCE_GIT_SCHEMA
    assert identity["source_commit"] == _source_commit(source)
    assert identity["source_branch"] == "DETACHED"
    assert identity["workspace_status_sha256"] == hashlib.sha256(b"").hexdigest()
    controller.verify_source_git_identity(source, identity)


def test_prepare_rejects_a_commit_label_without_git_authority(tmp_path: Path):
    source = _source(tmp_path)
    git_metadata = source / ".git"
    if git_metadata.is_file():
        git_metadata.unlink()
    else:
        shutil.rmtree(git_metadata)
    with pytest.raises(controller.ControllerError, match="source_git_identity_unavailable"):
        controller.build_source_git_identity(source, source_commit="a" * 40)


def test_prepare_rejects_dirty_or_branch_attached_source(tmp_path: Path):
    source = _source(tmp_path)
    (source / "core/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="source_capsule_dirty"):
        controller.build_source_git_identity(
            source,
            source_commit=_source_commit(source),
        )
    subprocess.run(["git", "-C", str(source), "restore", "core/module.py"], check=True)
    branch = (
        subprocess.run(
            [
                "git", "-c", "core.fsmonitor=false",
                "-C",
                str(source),
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.splitlines()[0]
    )
    subprocess.run(["git", "-C", str(source), "switch", "-q", branch], check=True)
    with pytest.raises(controller.ControllerError, match="source_capsule_not_detached"):
        controller.build_source_git_identity(
            source,
            source_commit=_source_commit(source),
        )


def test_source_manifest_detects_execution_source_drift(tmp_path: Path):
    source, _out, _config_path, config = _prepared(tmp_path)
    manifest = controller._read_json(
        Path(config["source_manifest_path"]), role="source_manifest"
    )
    controller.verify_source_manifest(source, manifest)
    (source / "core/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="source_file_drift"):
        controller.verify_source_manifest(source, manifest)


def test_source_manifest_detects_an_injected_python_module(tmp_path: Path):
    source, _out, _config_path, config = _prepared(tmp_path)
    manifest = controller._read_json(
        Path(config["source_manifest_path"]), role="source_manifest"
    )
    (source / "sitecustomize.py").write_text("print('injected')\n", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="source_file_set_drift"):
        controller.verify_source_manifest(source, manifest)


def test_model_manifest_rejects_weight_drift(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    controller.verify_model_manifest(config["model_manifest"])
    (Path(config["model"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(controller.ControllerError, match="model_file_drift"):
        controller.verify_model_manifest(config["model_manifest"])


def test_model_checkpoint_identity_matches_the_sweep_full_sha_contract(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    digest = hashlib.sha256(b"weights").hexdigest()
    expected = hashlib.sha256(
        f"model.safetensors:{digest};".encode()
    ).hexdigest()

    assert config["model_checkpoint"] == {
        "fingerprint": expected,
        "method": "sha256",
        "files": 1,
    }


def test_config_digest_rejects_a_changed_scientific_parameter(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["arms"] = "vanilla"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="controller_config_invalid"):
        controller.load_config(config_path)


def test_external_controller_program_is_hashed_without_changing_scientific_source(
    tmp_path: Path,
):
    source = _source(tmp_path)
    controller_program = tmp_path / "lifecycle-controller.py"
    controller_program.write_text("print('lifecycle')\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    config, manifest, key = controller.build_config(
        source_root=source,
        source_commit=_source_commit(source),
        model=model,
        out_dir=tmp_path / "campaign",
        python=Path(sys.executable),
        arms="full_stack",
        seed=7,
        per_domain=1,
        difficulty=2,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
        controller_program=controller_program,
    )
    config_path = tmp_path / "campaign/controller_config.json"
    controller.write_prepared_campaign(config_path, config, manifest, key)

    assert controller._controller_program(config) == controller_program
    assert config["controller_program_sha256"] == hashlib.sha256(
        controller_program.read_bytes()
    ).hexdigest()
    payload = plistlib_loads(controller._launch_payload(config_path, config))
    assert payload["ProgramArguments"][3] == str(controller_program)
    assert payload["WorkingDirectory"] == str(source)

    controller_program.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="controller_program_identity_drift"):
        controller._source_is_current(config)


def test_recovery_copies_only_fingerprint_valid_evidence(tmp_path: Path):
    source, previous_root, previous_config_path, previous_config = _prepared(tmp_path)
    previous_sweep = Path(previous_config["out_dir"])
    previous_sweep.mkdir()
    fingerprint = "a" * 64
    (previous_sweep / "decode_fingerprint.json").write_text(
        json.dumps({"decode_fingerprint": {"vanilla": fingerprint}}),
        encoding="utf-8",
    )
    (previous_sweep / "task_commitment.json").write_text("{}\n", encoding="utf-8")
    fixture_receipt = {"schema": "fixture", "nested": {"value": 1}}
    receipt_path = previous_sweep / "runtime_receipts/task-a.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(
        json.dumps(fixture_receipt, indent=1) + "\n", encoding="utf-8"
    )
    journal = (
        json.dumps(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": "task-a",
                "decode_fingerprint": fingerprint,
                "runtime_receipt_path": "runtime_receipts/task-a.json",
                "runtime_receipt_sha256": controller._sha(fixture_receipt),
            },
            sort_keys=True,
        )
        + "\n"
    )
    (previous_sweep / "journal.jsonl").write_text(journal, encoding="utf-8")
    (previous_sweep / "status.json").write_text('{"phase":"stale"}\n', encoding="utf-8")
    controller_program = tmp_path / "repaired-controller.py"
    controller_program.write_text("print('repaired')\n", encoding="utf-8")
    recovered_root = tmp_path / "recovered"
    recovered_config_path = recovered_root / "controller_config.json"

    recovery_receipt = controller.recover_campaign(
        previous_config_path,
        out_dir=recovered_root,
        output=recovered_config_path,
        controller_program=controller_program,
    )

    recovered = controller.load_config(recovered_config_path)
    assert recovery_receipt["preserved_cells"] == 1
    assert recovery_receipt["scientific_parameters_changed"] is False
    assert recovered["source_root"] == str(source)
    assert recovered["source_commit"] == previous_config["source_commit"]
    assert recovered["controller_program"] == str(controller_program)
    assert (recovered_root / "sweep/journal.jsonl").read_text() == journal
    assert json.loads(
        (recovered_root / "sweep/runtime_receipts/task-a.json").read_text()
    ) == fixture_receipt
    copied_receipt = next(
        item
        for item in recovery_receipt["copied_files"]
        if item["relative_path"] == "runtime_receipts/task-a.json"
    )
    assert copied_receipt["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert not (recovered_root / "sweep/status.json").exists()
    assert not (recovered_root / "sweep/verdict.json").exists()
    assert previous_root != recovered_root


def test_recovery_rejects_a_cell_from_another_fingerprint(tmp_path: Path):
    _source_root, _previous_root, previous_config_path, previous_config = _prepared(
        tmp_path
    )
    previous_sweep = Path(previous_config["out_dir"])
    previous_sweep.mkdir()
    (previous_sweep / "decode_fingerprint.json").write_text(
        json.dumps({"decode_fingerprint": {"vanilla": "a" * 64}}),
        encoding="utf-8",
    )
    (previous_sweep / "task_commitment.json").write_text("{}\n", encoding="utf-8")
    (previous_sweep / "journal.jsonl").write_text(
        json.dumps(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": "task-a",
                "decode_fingerprint": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    controller_program = tmp_path / "repaired-controller.py"
    controller_program.write_text("print('repaired')\n", encoding="utf-8")

    with pytest.raises(controller.ControllerError, match="fingerprint_mismatch"):
        controller.recover_campaign(
            previous_config_path,
            out_dir=tmp_path / "recovered",
            output=tmp_path / "recovered/controller_config.json",
            controller_program=controller_program,
        )


def test_controller_rejects_a_non_contamination_safe_task_registry(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["task_registry_version"] = "2026.07.18.1"
    document["config_sha256"] = controller._sha(controller._config_body(document))
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        controller.ControllerError,
        match="controller_task_registry_not_contamination_safe",
    ):
        controller.load_config(config_path)


def test_controller_cli_imports_repo_packages_outside_the_repo_cwd(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    observed = subprocess.run(
        [
            sys.executable,
            str(Path(controller.__file__).resolve()),
            "status",
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )

    assert observed.returncode == 0, observed.stderr
    assert json.loads(observed.stdout)["campaign_id"] == "campaign"


@pytest.mark.parametrize("difficulty", [0, 4, True, "2"])
def test_config_rejects_invalid_task_difficulty_even_with_a_valid_digest(
    tmp_path: Path,
    difficulty: object,
):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["difficulty"] = difficulty
    document["config_sha256"] = controller._sha(controller._config_body(document))
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(controller.ControllerError, match="controller_difficulty_invalid"):
        controller.load_config(config_path)


def test_prepare_preserves_the_venv_entrypoint_while_hashing_its_binary(tmp_path: Path):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    binary = tmp_path / "python-real"
    binary.write_bytes(b"runtime")
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(binary)
    config, _manifest, _key = controller.build_config(
        source_root=source,
        source_commit=_source_commit(source),
        model=model,
        out_dir=tmp_path / "campaign",
        python=venv_python,
        arms="full_stack",
        seed=7,
        per_domain=1,
        difficulty=2,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
    )
    assert config["python"] == str(venv_python)
    assert config["python_sha256"] == hashlib.sha256(b"runtime").hexdigest()


def test_heartbeat_is_authenticated_and_tamper_evident(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    heartbeat = controller._signed_heartbeat(
        config,
        {"controller_pid": 1, "sweep_pid": 2, "observed_unix": 3.0},
    )
    controller.verify_heartbeat(config, heartbeat)
    changed = copy.deepcopy(heartbeat)
    changed["sweep_pid"] = 999
    with pytest.raises(controller.ControllerError, match="heartbeat_invalid"):
        controller.verify_heartbeat(config, changed)


def test_heartbeat_matches_an_independent_hmac_recomputation(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    heartbeat = controller._signed_heartbeat(
        config,
        {"controller_pid": 1, "sweep_pid": 2, "observed_unix": 3.0},
    )
    signature = heartbeat.pop("hmac_sha256")
    key = Path(config["heartbeat_key_path"]).read_bytes()
    expected = hmac.new(key, controller._canonical(heartbeat), hashlib.sha256).hexdigest()
    assert signature == expected


def test_launchd_owns_caffeinate_and_the_exact_controller(tmp_path: Path):
    source, out, config_path, config = _prepared(tmp_path)
    payload = plistlib_loads(controller._launch_payload(config_path, config))
    arguments = payload["ProgramArguments"]
    assert arguments[:2] == ["/usr/bin/caffeinate", "-dims"]
    assert arguments[3] == str(source / "tools/run_rlc_reconciliation_controller.py")
    assert arguments[-2:] == [str(config_path), "--launchd-supervised"]
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["StandardOutPath"] == str(out / "controller.log")


def test_lineage_requires_launchd_controller_and_exact_caffeinate_child(
    tmp_path: Path, monkeypatch
):
    source, _out, config_path, config = _prepared(tmp_path)
    controller_command = (
        f"{config['python']} {source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    caffeinate_command = (
        f"/usr/bin/caffeinate -dims {config['python']} "
        f"{source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    monkeypatch.setattr(os, "getpid", lambda: 41)
    monkeypatch.setattr(controller, "_process_record", lambda _pid: (1, controller_command))
    monkeypatch.setattr(
        controller,
        "_process_table",
        lambda: [(42, 41, caffeinate_command)],
    )
    assert controller._verify_launchd_lineage(config) == {
        "launchd_pid": 1,
        "caffeinate_pid": 42,
        "controller_pid": 41,
    }


def test_lineage_rejects_caffeinate_owned_by_another_process(
    tmp_path: Path, monkeypatch
):
    source, _out, config_path, config = _prepared(tmp_path)
    controller_command = (
        f"{config['python']} {source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    caffeinate_command = (
        f"/usr/bin/caffeinate -dims {config['python']} "
        f"{source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    monkeypatch.setattr(os, "getpid", lambda: 41)
    monkeypatch.setattr(controller, "_process_record", lambda _pid: (1, controller_command))
    monkeypatch.setattr(controller, "_process_table", lambda: [(42, 99, caffeinate_command)])
    with pytest.raises(controller.ControllerError, match="lineage_invalid"):
        controller._verify_launchd_lineage(config, caffeinate_wait_s=0.0)


def test_lineage_rejects_controller_not_owned_by_launchd(
    tmp_path: Path, monkeypatch
):
    source, _out, config_path, config = _prepared(tmp_path)
    controller_command = (
        f"{config['python']} {source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    monkeypatch.setattr(os, "getpid", lambda: 42)
    monkeypatch.setattr(controller, "_process_record", lambda _pid: (99, controller_command))
    monkeypatch.setattr(controller, "_process_table", lambda: [])
    with pytest.raises(controller.ControllerError, match="lineage_invalid"):
        controller._verify_launchd_lineage(config)


def test_lineage_waits_for_caffeinate_startup_race(
    tmp_path: Path, monkeypatch
):
    source, _out, config_path, config = _prepared(tmp_path)
    controller_command = (
        f"{config['python']} {source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    caffeinate_command = (
        f"/usr/bin/caffeinate -dims {config['python']} "
        f"{source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    observations = iter([[], [], [(42, 41, caffeinate_command)]])
    monkeypatch.setattr(os, "getpid", lambda: 41)
    monkeypatch.setattr(controller, "_process_record", lambda _pid: (1, controller_command))
    monkeypatch.setattr(controller, "_process_table", lambda: next(observations))
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)
    assert controller._verify_launchd_lineage(config) == {
        "launchd_pid": 1,
        "caffeinate_pid": 42,
        "controller_pid": 41,
    }


def test_lineage_accepts_mac_resolved_python_executable_name(
    tmp_path: Path, monkeypatch
):
    source, _out, config_path, config = _prepared(tmp_path)
    controller_command = (
        "/opt/homebrew/Cellar/python@3.12/3.12.13/Frameworks/Python.framework/"
        "Versions/3.12/Resources/Python.app/Contents/MacOS/Python "
        f"{source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    caffeinate_command = (
        f"/usr/bin/caffeinate -dims {config['python']} "
        f"{source / 'tools/run_rlc_reconciliation_controller.py'} run "
        f"--config {config_path} --launchd-supervised"
    )
    monkeypatch.setattr(os, "getpid", lambda: 41)
    monkeypatch.setattr(controller, "_process_record", lambda _pid: (1, controller_command))
    monkeypatch.setattr(
        controller,
        "_process_table",
        lambda: [(42, 41, caffeinate_command)],
    )
    assert controller._verify_launchd_lineage(config)["caffeinate_pid"] == 42


def plistlib_loads(payload: bytes):
    import plistlib

    return plistlib.loads(payload)


def test_sweep_command_preserves_complete_engine_parameters(tmp_path: Path):
    _source_root, _out, _config_path, config = _prepared(tmp_path)
    command = controller._sweep_command(config)
    assert command[command.index("--arms") + 1] == "full_stack"
    assert command[command.index("--difficulty") + 1] == "2"
    assert command[command.index("--campaign-stage") + 1] == "certificate"
    assert command[command.index("--domains") + 1] == ""
    assert command[command.index("--task-registry-version") + 1] == (
        controller.CLAIM_TASK_REGISTRY_VERSION
    )
    assert command[command.index("--episode-wall-s") + 1] == "20.0"
    assert command[command.index("--max-wall-s") + 1] == "60.0"
    assert command[command.index("--model") + 1] == config["model"]


def test_bounded_calibration_identity_preserves_stage_and_domain_subset(tmp_path: Path):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")
    config, _manifest, _key = controller.build_config(
        source_root=source,
        source_commit=_source_commit(source),
        model=model,
        out_dir=tmp_path / "campaign",
        python=python,
        arms="complete_system_closed_book",
        seed=11,
        per_domain=1,
        difficulty=3,
        campaign_stage="component",
        domains=("mathematics",),
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
    )
    command = controller._sweep_command(config)

    assert config["campaign_stage"] == "component"
    assert config["domains"] == ["mathematics"]
    assert command[command.index("--campaign-stage") + 1] == "component"
    assert command[command.index("--domains") + 1] == "mathematics"


@pytest.mark.parametrize(
    ("stage", "domains", "error"),
    [
        ("unknown", (), "controller_campaign_stage_invalid"),
        ("component", ("mathematics", "mathematics"), "controller_domains_invalid"),
        ("component", ("not-a-domain",), "controller_domains_invalid"),
    ],
)
def test_prepare_rejects_invalid_experiment_subsets(
    tmp_path: Path,
    stage: str,
    domains: tuple[str, ...],
    error: str,
):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")

    with pytest.raises(controller.ControllerError, match=error):
        controller.build_config(
            source_root=source,
            source_commit=_source_commit(source),
            model=model,
            out_dir=tmp_path / "campaign",
            python=python,
            arms="complete_system_closed_book",
            seed=11,
            per_domain=1,
            difficulty=3,
            campaign_stage=stage,
            domains=domains,
            n_slots=4,
            max_tokens=64,
            memory_fraction=0.2,
            episode_wall_s=20.0,
            attempt_wall_s=60.0,
            max_attempts=3,
            poll_s=1.0,
            stale_after_s=40.0,
            retry_backoff_s=1.0,
        )


def test_frozen_adapter_is_bound_and_forwarded_to_the_sweep(
    tmp_path: Path,
    monkeypatch,
):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")
    adapter_root = tmp_path / "frozen-adapter"
    adapter_root.mkdir()
    (adapter_root / "adapter.safetensors").write_bytes(b"adapter")
    (adapter_root / "recurrence_adapter_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    model_manifest = controller.build_model_manifest(model)
    checkpoint = controller._model_checkpoint_identity(model_manifest)
    freeze = {
        "schema": "aura.latent_cortex.adapter_freeze.v1",
        "identity_receipt": {
            "base_checkpoint_fingerprint": checkpoint["fingerprint"],
            "training_objective_learned": True,
        },
        "certificate_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.campaign_launch_bundle.verify_adapter_freeze",
        lambda _root: freeze,
    )

    config, _manifest, _key = controller.build_config(
        source_root=source,
        source_commit=_source_commit(source),
        model=model,
        adapter_root=adapter_root,
        out_dir=tmp_path / "campaign",
        python=python,
        arms="full_stack",
        seed=7,
        per_domain=1,
        difficulty=2,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
    )
    command = controller._sweep_command(config)

    assert config["adapter_freeze"] == freeze
    assert command[command.index("--adapter") + 1] == str(adapter_root.resolve())
    assert command[command.index("--adapter-manifest") + 1] == str(
        (adapter_root / "recurrence_adapter_manifest.json").resolve()
    )


def test_config_rejects_a_partial_adapter_identity(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["adapter_root"] = "/tmp/partial"
    document["config_sha256"] = controller._sha(controller._config_body(document))
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        controller.ControllerError,
        match="controller_adapter_identity_incomplete",
    ):
        controller.load_config(config_path)


def test_recurrent_package_is_file_and_activation_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package = tmp_path / "recurrent-package"
    package.mkdir(mode=0o700)
    (package / "manifest.json").write_text("manifest-v1\n", encoding="utf-8")
    activation = {
        "mode": "qualified_typed_only",
        "serving_authority": True,
        "package_id": "fixture-package",
        "manifest_sha256": "a" * 64,
        "controller_sha256": "b" * 64,
        "activation_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_shadow.inspect_shadow_package",
        lambda _root: {
            "manifest": {
                "package_id": "fixture-package",
                "manifest_sha256": "a" * 64,
            }
        },
    )
    monkeypatch.setattr(
        "core.brain.llm.unified_recurrent_qualified_activation_store.read_qualified_activation",
        lambda: dict(activation),
    )

    first = controller._verified_integrated_recurrent_package_input(package)
    assert first["activation"] == activation
    assert first["files"][0]["sha256"] == hashlib.sha256(
        b"manifest-v1\n"
    ).hexdigest()
    (package / "manifest.json").write_text("manifest-v2\n", encoding="utf-8")
    second = controller._verified_integrated_recurrent_package_input(package)
    assert first["input_sha256"] != second["input_sha256"]
    with pytest.raises(
        controller.ControllerError,
        match="integrated_recurrent_package_identity_drift",
    ):
        controller._verify_integrated_recurrent_package_input(first)


def test_controller_passes_frozen_recurrent_package_to_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")
    package = tmp_path / "recurrent-package"
    package.mkdir()
    identity = {
        "root": str(package.resolve()),
        "package_id": "fixture-package",
        "manifest_sha256": "a" * 64,
        "controller_sha256": "b" * 64,
        "activation_sha256": "c" * 64,
        "activation": {"receipt": "fixture"},
        "files": [],
        "input_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        controller,
        "_verified_integrated_recurrent_package_input",
        lambda _root: dict(identity),
    )
    config, manifest, key = controller.build_config(
        source_root=source,
        source_commit=_source_commit(source),
        model=model,
        out_dir=tmp_path / "campaign",
        python=python,
        arms="complete_system_recurrent_composed",
        seed=7,
        per_domain=1,
        difficulty=2,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=1.0,
        stale_after_s=40.0,
        retry_backoff_s=1.0,
        integrated_recurrent_package=package,
        integrated_recurrent_max_tokens=1024,
    )
    controller.write_prepared_campaign(
        tmp_path / "campaign/controller_config.json",
        config,
        manifest,
        key,
    )
    command = controller._sweep_command(config)

    assert config["integrated_recurrent_package"] == identity
    assert command[command.index("--integrated-recurrent-package") + 1] == str(
        package.resolve()
    )
    assert command[command.index("--integrated-recurrent-max-tokens") + 1] == "1024"
    controller._source_is_current(config)


def test_config_rejects_partial_recurrent_package_identity(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["integrated_recurrent_package"] = {"root": "/tmp/partial"}
    document["config_sha256"] = controller._sha(controller._config_body(document))
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        controller.ControllerError,
        match="integrated_recurrent_package_identity_incomplete",
    ):
        controller.load_config(config_path)


@pytest.mark.parametrize(
    ("arms", "package_supplied"),
    [
        ("complete_system_recurrent_composed", False),
        ("complete_system_closed_book", True),
    ],
)
def test_prepare_rejects_a_recurrent_arm_package_mismatch(
    tmp_path: Path,
    arms: str,
    package_supplied: bool,
):
    source = _source(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    python = tmp_path / "python"
    python.write_text("runtime", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()

    with pytest.raises(
        controller.ControllerError,
        match="integrated_recurrent_arm_package_contract_mismatch",
    ):
        controller.build_config(
            source_root=source,
            source_commit=_source_commit(source),
            model=model,
            out_dir=tmp_path / "campaign",
            python=python,
            arms=arms,
            seed=7,
            per_domain=1,
            difficulty=2,
            n_slots=4,
            max_tokens=64,
            memory_fraction=0.2,
            episode_wall_s=20.0,
            attempt_wall_s=60.0,
            max_attempts=3,
            poll_s=1.0,
            stale_after_s=40.0,
            retry_backoff_s=1.0,
            integrated_recurrent_package=(package if package_supplied else None),
        )


def test_fresh_retry_does_not_inherit_stale_progress_from_prior_attempt():
    snapshot = {"last_progress_unix": 100.0}

    assert controller._progress_is_stale(
        snapshot,
        attempt_started_unix=1_000.0,
        process_activity_unix=1_000.0,
        observed_unix=1_001.0,
        stale_after_s=40.0,
    ) is False


def test_process_group_activity_keeps_a_long_attempt_live():
    snapshot = {"last_progress_unix": 100.0}

    assert controller._progress_is_stale(
        snapshot,
        attempt_started_unix=1_000.0,
        process_activity_unix=1_030.0,
        observed_unix=1_060.0,
        stale_after_s=40.0,
    ) is False


def test_attempt_without_progress_still_becomes_stale():
    snapshot = {"last_progress_unix": 100.0}

    assert controller._progress_is_stale(
        snapshot,
        attempt_started_unix=1_000.0,
        process_activity_unix=1_000.0,
        observed_unix=1_041.0,
        stale_after_s=40.0,
    ) is True


def test_process_group_activity_sums_exact_group_members(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "17 1:02.50 100 R\n"
                "17 1-00:00:01.25 200 S\n"
                "99 9:00.00 999 R\n"
            ),
            stderr="",
        ),
    )

    assert controller._process_group_activity(17) == {
        "available": True,
        "member_count": 2,
        "cpu_seconds": 86_463.75,
        "rss_kib": 300,
        "states": ["R", "S"],
    }


def test_exact_group_termination_refuses_a_foreign_process_group(monkeypatch):
    class Process:
        pid = 17

    monkeypatch.setattr(os, "getpgid", lambda _pid: 99)
    with pytest.raises(controller.ControllerError, match="process_group_identity"):
        controller._terminate_exact_group(Process())


def test_controller_requires_launchd_supervision(tmp_path: Path):
    _source_root, _out, config_path, _config = _prepared(tmp_path)
    with pytest.raises(controller.ControllerError, match="requires_launchd"):
        controller.run(config_path)


def test_controller_resumes_a_clean_wall_boundary_and_keeps_durable_progress(
    tmp_path: Path,
    monkeypatch,
):
    source = _source(tmp_path)
    sweep = source / "tools/run_rlc_reconciliation_sweep.py"
    sweep.write_text(
        """
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[sys.argv.index('--out-dir') + 1])
out.mkdir(parents=True, exist_ok=True)
marker = out / 'first-attempt-complete'
journal = out / 'journal.jsonl'
if not marker.exists():
    journal.write_text(json.dumps({'event': 'CELL', 'arm': 'vanilla', 'task_id': 'one'}) + '\\n')
    marker.write_text('durable')
    raise SystemExit(3)
(out / 'verdict.json').write_text(json.dumps({
    'decision': 'synthetic-complete',
    'coverage_complete': True,
    'arms_complete': True,
}))
""".lstrip(),
        encoding="utf-8",
    )
    source_commit = _commit_source(source, "executable retry fixture")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    out = tmp_path / "campaign"
    config, manifest, key = controller.build_config(
        source_root=source,
        source_commit=source_commit,
        model=model,
        out_dir=out,
        python=Path(sys.executable),
        arms="full_stack",
        seed=7,
        per_domain=1,
        difficulty=2,
        n_slots=4,
        max_tokens=64,
        memory_fraction=0.2,
        episode_wall_s=20.0,
        attempt_wall_s=60.0,
        max_attempts=3,
        poll_s=0.01,
        stale_after_s=40.0,
        retry_backoff_s=0.01,
    )
    config_path = out / "controller_config.json"
    controller.write_prepared_campaign(config_path, config, manifest, key)

    monkeypatch.setattr(
        controller,
        "_verify_launchd_lineage",
        lambda _config: {"launchd_pid": 1, "caffeinate_pid": 2, "controller_pid": 3},
    )
    monkeypatch.setattr(controller, "GLOBAL_MODEL_LOCK", tmp_path / "model.lock")
    assert controller.run(config_path, launchd_supervised=True) == 0
    events = [
        json.loads(line)
        for line in (out / "controller_attempts.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "ATTEMPT_STARTED",
        "ATTEMPT_FINISHED",
        "ATTEMPT_STARTED",
        "ATTEMPT_FINISHED",
    ]
    assert events[1]["returncode"] == 3
    assert events[1]["cells"] == 1
    assert events[3]["returncode"] == 0
    status = json.loads((out / "controller_status.json").read_text())
    assert status["phase"] == "complete"
    assert (out / "sweep/first-attempt-complete").read_text() == "durable"


def test_interim_verdict_is_not_terminal(tmp_path: Path):
    verdict = tmp_path / "verdict.json"
    verdict.write_text(
        json.dumps(
            {
                "decision": "inconclusive_campaign_incomplete",
                "coverage_complete": False,
                "arms_complete": False,
            }
        ),
        encoding="utf-8",
    )

    assert controller._terminal_verdict(verdict) is None

    verdict.write_text(
        json.dumps(
            {
                "decision": "proceed_to_checkpoint_phase",
                "coverage_complete": True,
                "arms_complete": True,
            }
        ),
        encoding="utf-8",
    )
    assert controller._terminal_verdict(verdict)["decision"] == (
        "proceed_to_checkpoint_phase"
    )


def test_authenticated_preregistered_futility_is_terminal(tmp_path: Path):
    verdict = tmp_path / "verdict.json"
    body = {
        "schema": "aura.rlc.composed_terminal_futility.v1",
        "terminal": True,
        "decision": "terminal_preregistered_zero_regression_futility",
        "criterion": "composed_treatment_must_not_regress_any_paired_control",
        "task_id": "task-a",
        "domain": "mathematics",
        "required_arms": ["vanilla"],
        "correctness": {"vanilla": True},
        "fatal_regression_contrasts": ["vanilla_floor"],
        "positive_claim_authority": False,
        "fusion_authorized": False,
        "wow_signal_authorized": False,
    }
    receipt = {**body, "receipt_sha256": controller._sha(body)}
    (tmp_path / "terminal_futility.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    assert controller._terminal_verdict(verdict) == receipt
    receipt["positive_claim_authority"] = True
    (tmp_path / "terminal_futility.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(controller.ControllerError, match="terminal_futility_receipt_invalid"):
        controller._terminal_verdict(verdict)
