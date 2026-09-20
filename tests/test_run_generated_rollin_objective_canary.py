from __future__ import annotations

import inspect
import json
import subprocess
import time
from pathlib import Path

import pytest

from core.learning.recurrence_curriculum import task_battery
from tools import run_generated_rollin_objective_canary as canary


def _repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", head],
        cwd=tmp_path,
        check=True,
    )
    return source, head


def test_source_state_requires_clean_published_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, head = _repo(tmp_path)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(canary, "SOURCE_PATHS", ("source.py",))

    observed_head, bindings = canary._source_state()
    assert observed_head == head
    assert bindings["source.py"]["size_bytes"] == len(b"VALUE = 1\n")

    source.write_text("VALUE = 2\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="clean source"):
        canary._source_state()


def test_source_state_rejects_unpublished_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, _head = _repo(tmp_path)
    (tmp_path / "other.py").write_text("VALUE = 2\n", encoding="ascii")
    subprocess.run(["git", "add", "other.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "ahead"], cwd=tmp_path, check=True)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(canary, "SOURCE_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="not published"):
        canary._source_state()


def test_branch_specialization_gate_rejects_rng_or_state_collapse() -> None:
    collapsed = [
        {
            "objective_receipt": {
                "generated_receipt": {
                    "branches": [
                        {"generated_tokens_sha256": "a" * 64},
                        {"generated_tokens_sha256": "a" * 64},
                    ]
                }
            }
        }
    ]
    gates = canary._branch_specialization_gates(collapsed, [0.08])
    assert gates == {
        "branch_generated_prefix_distinct": False,
        "branch_state_specialized": False,
    }

    specialized = [
        {
            "objective_receipt": {
                "generated_receipt": {
                    "branches": [
                        {"generated_tokens_sha256": "a" * 64},
                        {"generated_tokens_sha256": "b" * 64},
                    ]
                }
            }
        }
    ]
    assert all(canary._branch_specialization_gates(specialized, [0.31]).values())


def test_proxy_task_manifest_binds_answers_and_generation_coordinates() -> None:
    tasks = task_battery(["boolean", "modular"], [2], 1, seed=17)

    manifest, digest = canary.build_recurrence_task_manifest(tasks)

    assert [row["task_id"] for row in manifest] == [task.task_id for task in tasks]
    assert len(digest) == 64
    assert all("solution" not in row and "training_target" not in row for row in manifest)


def test_canary_trains_on_private_process_targets_without_leaking_them_to_probe() -> None:
    source = inspect.getsource(canary.run_canary)

    assert "task.training_target" in source
    assert "validation_task.training_target" in source
    assert "_free_generation_report(\n            model,\n            tokenizer,\n            proxy_tasks" in source
    assert '"training_target_manifest"' in source


def test_free_generation_uses_matched_random_streams_across_arms() -> None:
    seed = canary._paired_generation_seed(17, 2, "heldout-task", 4)

    assert seed == canary._paired_generation_seed(17, 2, "heldout-task", 4)
    assert seed != canary._paired_generation_seed(17, 2, "heldout-task", 2)


def test_free_generation_uses_proof_grade_sampling_policy() -> None:
    config = canary._free_generation_sampling_config()

    assert config.max_tokens == 320
    assert config.temperature == 1.0
    assert config.top_p == 1.0


def test_free_generation_disables_external_memory_and_binds_sample_seed() -> None:
    source = inspect.getsource(canary._free_generation_report)

    assert "nonparametric_memory_enabled=False" in source
    assert "sample_seed=generation_seed" in source


def test_training_rows_cycle_across_the_balanced_battery() -> None:
    rows = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]

    observed = [
        canary._cyclic_training_row(rows, one_based_step=step)["task_id"] for step in range(1, 8)
    ]

    assert observed == ["a", "b", "c", "a", "b", "c", "a"]
    with pytest.raises(ValueError, match="cycle coordinates"):
        canary._cyclic_training_row(rows, one_based_step=0)


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (0.0, True),
        (1e-6, True),
        (1.000001e-6, False),
        (float("inf"), False),
        (float("nan"), False),
        (True, False),
        (None, False),
    ],
)
def test_warmup_target_gate_is_exact_and_fail_closed(loss: object, expected: bool) -> None:
    assert canary._warmup_target_reached(loss) is expected


def test_coda_causal_arms_are_admitted_report_identities() -> None:
    from core.learning.recurrent_checkpoint_admission import _ARMS

    assert {
        "trained_adapter_lesion",
        "trained_adapter_sham",
        "trained_coda_lesion",
        "trained_coda_sham",
    }.issubset(_ARMS)


def test_progress_ledger_is_durable_monotonic_and_failure_aware(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "canary"
    ledger = canary._ProgressLedger(
        out_dir,
        started=time.time() - 2.0,
        source_commit="a" * 40,
    )

    first = ledger.emit("model_load", model_path="/model")
    second = ledger.emit("free_generation", event="sample_started", depth=2)
    canary._append_terminal_failure(out_dir, RuntimeError("bounded failure"))

    rows = [
        json.loads(line)
        for line in (out_dir / "progress.jsonl").read_text(encoding="ascii").splitlines()
    ]
    latest = json.loads((out_dir / "progress.json").read_text(encoding="ascii"))
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert first["phase"] == "model_load"
    assert second["details"] == {"event": "sample_started", "depth": 2}
    assert latest["status"] == "failed"
    assert latest["details"]["exception_type"] == "RuntimeError"
    assert latest["details"]["exception_message"] == "bounded failure"
    assert latest["event_sha256"] == canary.hashlib.sha256(
        canary._canonical_json_bytes(
            {key: value for key, value in latest.items() if key != "event_sha256"}
        )
    ).hexdigest()
    assert "seq=3" in capsys.readouterr().err


def test_terminal_failure_does_not_create_an_unowned_output_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "not-started"

    canary._append_terminal_failure(out_dir, RuntimeError("before ledger"))

    assert not out_dir.exists()
