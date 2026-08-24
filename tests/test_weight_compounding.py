"""Contract tests for the compounding weight-learning cycle.

Everything runs offline: mlx subprocesses are replaced by a scripted fake
runner that produces the same artifacts (adapter safetensors, eval reports,
fused model dirs) the real commands would. What is NOT faked: the battery
generation/grading, the harvest filtering, the manifest chain, the lockfile,
the lineage ledger, and every promote/refuse decision — those are the
contracts under test.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.learning.eval_before_promotion import AdapterEvaluator
from core.learning.heldout_battery import (
    ANSWER_INSTRUCTION,
    BatterySpec,
    battery_fingerprints,
    extract_answer,
    generate_battery,
    grade_battery,
    grade_response,
    text_collides_with_battery,
)
from core.learning.rsi_lineage import VERDICT_BOUNDED
from core.learning.weight_compounding import (
    CompoundingConfig,
    WeightCompoundingLoop,
)

pytestmark = pytest.mark.unit


# ── battery ───────────────────────────────────────────────────────────────────

class TestHeldoutBattery:
    def test_same_spec_is_deterministic(self):
        a = generate_battery(BatterySpec(seed=7, size=24))
        b = generate_battery(BatterySpec(seed=7, size=24))
        assert [t.to_dict() for t in a] == [t.to_dict() for t in b]

    def test_different_seed_different_tasks(self):
        a = generate_battery(BatterySpec(seed=7, size=24))
        b = generate_battery(BatterySpec(seed=8, size=24))
        assert {t.prompt for t in a} != {t.prompt for t in b}

    def test_all_domains_covered_and_prompts_instructed(self):
        tasks = generate_battery(BatterySpec(seed=1, size=16))
        assert len({t.domain for t in tasks}) == 8
        assert all(ANSWER_INSTRUCTION in t.prompt for t in tasks)

    def test_correct_answer_grades_true_wrong_grades_false(self):
        for task in generate_battery(BatterySpec(seed=3, size=16)):
            assert grade_response(task, f"Some reasoning...\nAnswer: {task.answer}")
            assert not grade_response(task, "Answer: definitely-not-the-answer-42424242")

    def test_extractor_prefers_last_answer_line(self):
        assert extract_answer("Answer: 5\nwait no\nAnswer: 7", "int") == "7"

    def test_extractor_falls_back_to_last_number(self):
        assert extract_answer("the result is 1,234.", "int") == "1234"

    def test_string_answers_normalized(self):
        assert extract_answer("Answer: **Friday.**", "str") == "friday"

    def test_grading_summary_counts(self):
        spec = BatterySpec(seed=5, size=10)
        tasks = generate_battery(spec)
        responses = {t.task_id: f"Answer: {t.answer}" for t in tasks[:6]}
        result = grade_battery(spec, tasks, responses)
        assert (result.total, result.correct) == (10, 6)
        assert len(result.failures) == 4

    def test_seal_collision_detection(self):
        tasks = generate_battery(BatterySpec(seed=2, size=8))
        body = tasks[0].prompt.split("\n\n")[0]
        assert text_collides_with_battery(f"user asked me: {body} and I said", tasks)
        assert not text_collides_with_battery("an unrelated conversation", tasks)
        assert len(battery_fingerprints(tasks)) == 8


# ── scripted fake subprocess runner ──────────────────────────────────────────

@dataclass
class FakeResult:
    ok: bool = True
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False


class FakeRunner:
    """Produces the artifacts the real mlx/eval subprocesses would.

    ``accuracy_script`` maps (seed, has_adapter) → accuracy so tests control
    incumbent/candidate/hidden scores per cycle.
    """

    def __init__(self, accuracy_script=None, *, fail_train=False, fail_fuse=False,
                 response_text="Reasoning...\nAnswer: 42"):
        self.accuracy_script = accuracy_script or {}
        self.fail_train = fail_train
        self.fail_fuse = fail_fuse
        self.response_text = response_text
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> FakeResult:
        self.commands.append(tuple(command))
        argv = list(command)
        joined = " ".join(argv)

        if "heldout_eval.py" in joined:
            return self._run_eval(argv)
        if "fuse" in argv:
            return self._run_fuse(argv)
        if "--train" in argv:
            return self._run_train(argv)
        return FakeResult(ok=False, stderr=f"unexpected command: {joined}")

    @staticmethod
    def _arg(argv: list[str], flag: str, default: str = "") -> str:
        return argv[argv.index(flag) + 1] if flag in argv else default

    def _run_train(self, argv) -> FakeResult:
        if self.fail_train:
            return FakeResult(ok=False, stderr="loss exploded")
        adapter = Path(self._arg(argv, "--adapter-path"))
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapters.safetensors").write_bytes(b"fake-lora-weights")
        return FakeResult(stdout="train ok")

    def _run_fuse(self, argv) -> FakeResult:
        override = getattr(self, "_fuse_override", None)
        if override is not None:
            return override
        if self.fail_fuse:
            return FakeResult(ok=False, stderr="fuse blew up")
        save = Path(self._arg(argv, "--save-path"))
        save.mkdir(parents=True, exist_ok=True)
        (save / "config.json").write_text("{}", encoding="utf-8")
        (save / "tokenizer.json").write_text("{}", encoding="utf-8")
        (save / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (save / "model.safetensors").write_bytes(b"fused-weights")
        return FakeResult(stdout="fuse ok")

    def _run_eval(self, argv) -> FakeResult:
        seed = int(self._arg(argv, "--seed", "0"))
        size = int(self._arg(argv, "--size", "8"))
        has_adapter = bool(self._arg(argv, "--adapter-path"))
        accuracy = self.accuracy_script.get((seed, has_adapter), 0.5)
        output = Path(self._arg(argv, "--output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "accuracy": accuracy,
            "result": {"correct": int(accuracy * size), "total": size, "per_domain": {}},
            "battery": {"battery_id": f"heldout-v1-seed{seed}-n{size}"},
        }
        output.write_text(json.dumps(report), encoding="utf-8")
        responses = output.with_suffix(".responses.jsonl")
        responses.write_text(
            json.dumps({"task_id": "t0", "response": self.response_text}) + "\n",
            encoding="utf-8",
        )
        return FakeResult(stdout=f"acc={accuracy}")


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "base-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"x" * 1024)
    return model


@pytest.fixture
def sft_buffer(tmp_path: Path) -> Path:
    buffer = tmp_path / "experience_buffer.jsonl"
    rows = []
    for i in range(40):
        rows.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "You are Aura."},
                        {"role": "user", "content": f"tell me about topic {i}"},
                        {"role": "assistant", "content": f"here is a real grounded thought about topic {i}."},
                    ],
                    "_quality": 0.6 + (i % 10) / 50.0,
                }
            )
        )
    buffer.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return buffer


def make_config(tmp_path: Path, base_model_dir: Path, sft_buffer: Path, **overrides) -> CompoundingConfig:
    defaults = dict(
        work_root=tmp_path / "compound",
        fused_root=tmp_path / "fused",
        model_override=str(base_model_dir),
        sft_buffer_path=sft_buffer,
        dpo_store_path=tmp_path / "missing_dpo.jsonl",
        min_sft_examples=10,
        min_dpo_pairs=5,
        iters=2,
        battery_size=8,
        hidden_battery_size=4,
        battery_seed_base=1000,
        ram_slack_bytes=0,
        ram_headroom_factor=0.0,
    )
    defaults.update(overrides)
    return CompoundingConfig(**defaults)


PROMOTE_SCRIPT = {
    (1000, False): 0.50,    # cycle 0 incumbent, visible battery
    (1000, True): 0.625,    # cycle 0 candidate — improved
    (101003, False): 0.50,  # cycle 0 incumbent, hidden battery
    (101003, True): 0.55,   # cycle 0 candidate, hidden — also improved
    (1001, False): 0.625,   # cycle 1 incumbent (the promoted model)
    (1001, True): 0.75,     # cycle 1 candidate — improved again
    (101004, False): 0.55,  # cycle 1 incumbent, hidden battery
    (101004, True): 0.70,   # cycle 1 candidate, hidden
}


# ── base resolution (the compounding hinge) ──────────────────────────────────

class TestBaseResolution:
    def test_override_wins(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        assert loop.resolve_base() == (str(base_model_dir), "override")

    def test_manifest_read_fresh_each_call(self, tmp_path, base_model_dir, sft_buffer):
        config = make_config(tmp_path, base_model_dir, sft_buffer, model_override="")
        loop = WeightCompoundingLoop(config)
        with pytest.raises(RuntimeError):
            loop.resolve_base()
        # publish a manifest AFTER loop construction — must be visible
        config.fused_root.mkdir(parents=True, exist_ok=True)
        config.manifest_path.write_text(
            json.dumps({"active_model_path": str(base_model_dir)}), encoding="utf-8"
        )
        assert loop.resolve_base() == (str(base_model_dir), "manifest")

    def test_manifest_with_dead_path_falls_back(self, tmp_path, base_model_dir, sft_buffer):
        config = make_config(
            tmp_path, base_model_dir, sft_buffer,
            model_override="", default_base=str(base_model_dir),
        )
        config.fused_root.mkdir(parents=True, exist_ok=True)
        config.manifest_path.write_text(
            json.dumps({"active_model_path": str(tmp_path / "deleted-model")}),
            encoding="utf-8",
        )
        loop = WeightCompoundingLoop(config)
        assert loop.resolve_base() == (str(base_model_dir), "default")


# ── admission control ─────────────────────────────────────────────────────────

class TestAdmission:
    def test_small_model_admitted(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        ok, reasons = loop.admission_check(str(base_model_dir))
        assert ok, reasons

    def test_autonomous_size_cap_blocks_large_model(self, tmp_path, base_model_dir, sft_buffer):
        big = tmp_path / "big-model"
        big.mkdir()
        (big / "config.json").write_text("{}", encoding="utf-8")
        with (big / "model.safetensors").open("wb") as fh:
            fh.truncate(7 * 1024**3)  # sparse 7GB
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        ok, reasons = loop.admission_check(str(big))
        assert not ok
        assert any("autonomous_cap" in r for r in reasons)

    def test_operator_run_lifts_cap(self, tmp_path, base_model_dir, sft_buffer):
        big = tmp_path / "big-model"
        big.mkdir()
        (big / "config.json").write_text("{}", encoding="utf-8")
        with (big / "model.safetensors").open("wb") as fh:
            fh.truncate(7 * 1024**3)
        config = make_config(
            tmp_path, base_model_dir, sft_buffer,
            operator_run=True, ram_headroom_factor=0.0, ram_slack_bytes=0,
        )
        loop = WeightCompoundingLoop(config)
        ok, reasons = loop.admission_check(str(big))
        assert ok, reasons

    def test_hf_repo_id_resolves_via_local_cache(
        self, tmp_path, base_model_dir, sft_buffer, monkeypatch
    ):
        """A cached HF repo id must size like a local dir — no network, no block."""
        import huggingface_hub

        def fake_snapshot(repo_id: str, local_files_only: bool = False, **_kw):
            assert local_files_only, "admission must never touch the network"
            assert repo_id == "test-org/test-model"
            return str(base_model_dir)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        ok, reasons = loop.admission_check("test-org/test-model")
        assert ok, reasons

    def test_uncached_repo_id_blocks_fail_closed(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        ok, reasons = loop.admission_check("no-such-org/definitely-not-cached-xyz")
        assert not ok
        assert any("model_footprint_unknown" in r for r in reasons)

    def test_idle_hook_blocks(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            idle_hook=lambda: False,
        )
        ok, reasons = loop.admission_check(str(base_model_dir))
        assert not ok and "foreground_not_idle" in reasons

    def test_lock_single_flight_and_stale_reclaim(self, tmp_path, base_model_dir, sft_buffer):
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config)
        assert loop._acquire_lock()
        assert not loop._acquire_lock()          # held by a live pid (us)
        loop._release_lock()
        # stale lock from a dead pid is reclaimed
        (config.work_root / "cycle.lock").write_text(
            json.dumps({"pid": 99999999, "at": 0}), encoding="utf-8"
        )
        assert loop._acquire_lock()
        loop._release_lock()


# ── harvest ───────────────────────────────────────────────────────────────────

class TestHarvest:
    def test_sft_used_when_no_dpo(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        tasks = generate_battery(BatterySpec(seed=1000, size=8))
        mode, data_dir, counts = loop.harvest(tmp_path / "run", tasks)
        assert mode == "sft"
        assert counts["train"] >= 10
        assert (data_dir / "train.jsonl").exists()

    def test_dpo_preferred_when_enough_pairs(self, tmp_path, base_model_dir, sft_buffer):
        dpo = tmp_path / "prefs.jsonl"
        with dpo.open("w", encoding="utf-8") as fh:
            for i in range(8):
                fh.write(json.dumps({
                    "prompt": f"solve problem {i}",
                    "chosen": f"verified answer {i}",
                    "rejected": f"refuted answer {i}",
                }) + "\n")
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer, dpo_store_path=dpo)
        )
        tasks = generate_battery(BatterySpec(seed=1000, size=8))
        mode, _, counts = loop.harvest(tmp_path / "run", tasks)
        assert mode == "dpo"
        assert counts["train"] >= 5

    def test_sealed_battery_rows_excluded(self, tmp_path, base_model_dir, sft_buffer):
        tasks = generate_battery(BatterySpec(seed=1000, size=8))
        leak_body = tasks[0].prompt.split("\n\n")[0]
        with sft_buffer.open("a", encoding="utf-8") as fh:
            for _ in range(3):
                fh.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": leak_body},
                        {"role": "assistant", "content": f"Answer: {tasks[0].answer}"},
                    ],
                    "_quality": 0.99,
                }) + "\n")
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        rows = loop._load_sft_rows(tasks)
        joined = json.dumps(rows)
        assert leak_body not in joined

    def test_contaminated_rows_excluded(self, tmp_path, base_model_dir, sft_buffer):
        with sft_buffer.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "who are you"},
                    {"role": "assistant", "content": "As an AI language model I cannot"},
                ],
                "_quality": 0.99,
            }) + "\n")
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, sft_buffer))
        rows = loop._load_sft_rows(generate_battery(BatterySpec(seed=1000, size=8)))
        assert "as an ai language model" not in json.dumps(rows).lower()

    def test_dpo_train_command_disables_trainer_autofuse(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        dpo = tmp_path / "prefs.jsonl"
        with dpo.open("w", encoding="utf-8") as fh:
            for i in range(8):
                fh.write(json.dumps({
                    "prompt": f"solve problem {i}",
                    "chosen": f"verified answer {i}",
                    "rejected": f"refuted answer {i}",
                }) + "\n")
        runner = FakeRunner(PROMOTE_SCRIPT)
        config = make_config(tmp_path, base_model_dir, sft_buffer, dpo_store_path=dpo)
        loop = WeightCompoundingLoop(config, command_runner=runner)
        receipt = loop.run_cycle()
        assert receipt.train_mode == "dpo"
        train_cmd = next(c for c in runner.commands if "--train" in c)
        assert "mlx_lm_lora.train" in train_cmd
        assert "-c" in train_cmd
        trainer_config = Path(train_cmd[train_cmd.index("-c") + 1])
        assert "fuse: false" in trainer_config.read_text(encoding="utf-8")

    def test_insufficient_data_raises(self, tmp_path, base_model_dir):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        loop = WeightCompoundingLoop(make_config(tmp_path, base_model_dir, empty))
        with pytest.raises(RuntimeError, match="insufficient_training_data"):
            loop.harvest(tmp_path / "run", generate_battery(BatterySpec(seed=1, size=4)))


# ── full cycles ───────────────────────────────────────────────────────────────

class TestCycle:
    def test_candidate_cycle_end_to_end(self, tmp_path, base_model_dir, sft_buffer):
        runner = FakeRunner(PROMOTE_SCRIPT)
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config, command_runner=runner)
        receipt = loop.run_cycle()

        assert receipt.status == "candidate", receipt.reasons
        assert receipt.train_mode == "sft"
        assert receipt.incumbent_accuracy == 0.50
        assert receipt.candidate_accuracy == 0.625
        assert receipt.identity_ok is True
        assert receipt.ledger_entry_hash

        assert receipt.promoted_model_path == ""
        assert Path(receipt.candidate_model_path).exists()
        assert not config.manifest_path.exists()
        candidate_record = json.loads(
            Path(receipt.candidate_receipt_path).read_text(encoding="utf-8")
        )
        assert candidate_record["base_model_path"] == str(base_model_dir.resolve())
        assert candidate_record["candidate_model_path"] == receipt.candidate_model_path
        assert candidate_record["qualification_state"] == "awaiting_evaluation"

        # receipt persisted
        saved = json.loads((Path(receipt.run_dir) / "cycle_receipt.json").read_text())
        assert saved["status"] == "candidate"
        assert saved["promoted_model_path"] == ""

        # the eval_before_promotion contract is now REAL evidence
        adapter_dir = Path(receipt.run_dir) / "adapter"
        verdict = AdapterEvaluator().evaluate_candidate(str(adapter_dir))
        assert verdict["status"] == "evaluated"
        assert verdict["can_promote"] is True

        # lock released
        assert not (config.work_root / "cycle.lock").exists()

    def test_unactivated_candidate_does_not_become_next_cycle_base(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        runner = FakeRunner(PROMOTE_SCRIPT)
        # cycle 2 must follow the manifest, so no override
        config = make_config(
            tmp_path, base_model_dir, sft_buffer,
            model_override="", default_base=str(base_model_dir),
        )
        loop = WeightCompoundingLoop(config, command_runner=runner)

        first = loop.run_cycle()
        assert first.status == "candidate", first.reasons
        assert first.base_source == "default"

        second = loop.run_cycle()
        assert second.status == "candidate", second.reasons
        assert second.base_source == "default"
        assert second.base_model == str(base_model_dir)
        assert not config.manifest_path.exists()

        records = loop._ledger.load_records()
        assert len(records) == 2
        assert records[1].parent_generation_id == records[0].generation_id
        assert records[0].promoted is False
        assert records[1].promoted is False

        verdict = loop.lineage_verdict()
        assert verdict.verdict == VERDICT_BOUNDED
        intact, problems = loop.verify_ledger()
        assert intact, problems

    def test_capability_regression_refused(self, tmp_path, base_model_dir, sft_buffer):
        script = dict(PROMOTE_SCRIPT)
        script[(1000, True)] = 0.25    # candidate much worse
        script[(101003, True)] = 0.25
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config, command_runner=FakeRunner(script))
        receipt = loop.run_cycle()

        assert receipt.status == "refused"
        assert any("capability_regressed" in r for r in receipt.reasons)
        assert not config.manifest_path.exists()      # nothing published
        records = loop._ledger.load_records()
        assert len(records) == 1 and records[0].promoted is False
        assert loop.lineage_verdict().verdict == VERDICT_BOUNDED

    def test_a_candidate_that_learned_the_visible_battery_is_refused(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        """The hidden battery has to be able to say no.

        It ran on the candidate only, so the strongest available statement
        was `hidden_passed = hidden is not None` — a full extra model load
        every cycle that could not refuse anything. This is the case it was
        bought for: up on the battery it saw, down on the one it did not.
        """
        script = dict(PROMOTE_SCRIPT)
        script[(1000, True)] = 0.90     # candidate soars on the visible battery
        script[(101003, False)] = 0.50  # incumbent, hidden
        script[(101003, True)] = 0.20   # candidate collapses on the hidden one
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config, command_runner=FakeRunner(script))
        receipt = loop.run_cycle()

        assert receipt.status == "refused", receipt.reasons
        assert any("hidden_regressed" in r for r in receipt.reasons), receipt.reasons
        assert not config.manifest_path.exists()

    def test_a_missing_hidden_eval_refuses_rather_than_passes(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        """No hidden evidence is not the same as good hidden evidence."""
        script = dict(PROMOTE_SCRIPT)
        runner = FakeRunner(script)
        original = runner._run_eval

        def drop_hidden(argv):
            if int(runner._arg(argv, "--seed", "0")) >= 100000:
                return FakeResult(ok=False, stderr="hidden eval crashed")
            return original(argv)

        runner._run_eval = drop_hidden
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config, command_runner=runner)
        receipt = loop.run_cycle()

        assert receipt.status == "refused", receipt.reasons
        assert any("hidden_eval_unavailable" in r for r in receipt.reasons)

    def test_the_floor_is_the_high_water_mark_not_the_parent(
        self, tmp_path, base_model_dir, sft_buffer, monkeypatch
    ):
        """Only centrally promoted lineage records can raise the floor."""
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            command_runner=FakeRunner(PROMOTE_SCRIPT),
        )
        records = [
            type("Record", (), {"promoted": True, "after_score": 0.80})(),
            type("Record", (), {"promoted": False, "after_score": 0.95})(),
        ]
        monkeypatch.setattr(loop._ledger, "load_records", lambda: records)

        assert loop._capability_high_water(0.60) == 0.80

    def test_the_high_water_mark_ignores_refused_generations(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        """A refused candidate's score never becomes a bar to clear.

        Nothing was served at that accuracy, so requiring later generations
        to match it would refuse work for failing to beat a model that never
        existed.
        """
        config = make_config(
            tmp_path, base_model_dir, sft_buffer,
            model_override="", default_base=str(base_model_dir),
        )
        script = {
            # cycle 0 promotes at 0.60
            (1000, False): 0.50, (1000, True): 0.60,
            (101003, False): 0.50, (101003, True): 0.60,
            # cycle 1 scores 0.95 on the visible battery but tanks the hidden
            # one, so it is refused — 0.95 must not become the floor.
            (1001, False): 0.60, (1001, True): 0.95,
            (101004, False): 0.60, (101004, True): 0.10,
            # cycle 2 is an honest 0.62 and must be allowed to promote
            (1002, False): 0.60, (1002, True): 0.62,
            (101005, False): 0.60, (101005, True): 0.62,
        }
        loop = WeightCompoundingLoop(config, command_runner=FakeRunner(script))

        assert loop.run_cycle().status == "candidate"
        refused = loop.run_cycle()
        assert refused.status == "refused", refused.reasons

        third = loop.run_cycle()
        assert third.status == "candidate", third.reasons
        assert third.high_water_accuracy == 0.60

    def test_the_report_carries_both_hidden_scores(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config, command_runner=FakeRunner(PROMOTE_SCRIPT))
        receipt = loop.run_cycle()

        report = json.loads(
            (Path(receipt.run_dir) / "adapter" / "evaluation_report.json").read_text()
        )
        assert report["hidden_eval_passed"] is True
        assert report["hidden_incumbent_accuracy"] == 0.50
        assert report["hidden_accuracy"] == 0.55
        assert report["high_water_accuracy"] == 0.50
        assert report["promotion_threshold"] == pytest.approx(0.48)

    def test_identity_regression_refused(self, tmp_path, base_model_dir, sft_buffer):
        runner = FakeRunner(
            PROMOTE_SCRIPT,
            response_text="As an AI language model, the answer is 42.",
        )
        config = make_config(tmp_path, base_model_dir, sft_buffer)
        loop = WeightCompoundingLoop(config, command_runner=runner)
        receipt = loop.run_cycle()
        assert receipt.status == "refused"
        assert receipt.identity_ok is False
        assert not config.manifest_path.exists()

    def test_training_failure_recorded_not_promoted(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            command_runner=FakeRunner(fail_train=True),
        )
        receipt = loop.run_cycle()
        assert receipt.status == "failed"
        assert any("training_failed" in r for r in receipt.reasons)
        records = loop._ledger.load_records()
        assert len(records) == 1 and records[0].promoted is False

    def test_fuse_failure_not_promoted(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            command_runner=FakeRunner(PROMOTE_SCRIPT, fail_fuse=True),
        )
        receipt = loop.run_cycle()
        assert receipt.status == "failed"
        assert any("fuse_failed" in r for r in receipt.reasons)
        assert not (make_config(tmp_path, base_model_dir, sft_buffer).manifest_path).exists()

    def test_fuse_failure_detail_names_oom_signal(self, tmp_path, base_model_dir, sft_buffer):
        """A SIGKILL (OOM) fuse leaves empty stderr — the receipt must still say why."""
        runner = FakeRunner(PROMOTE_SCRIPT)
        runner._fuse_override = FakeResult(ok=False, stderr="", stdout="", returncode=-9)
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer, operator_run=True),
            command_runner=runner,
        )
        receipt = loop.run_cycle()
        assert receipt.status == "failed"
        assert any("killed_signal_9(likely_oom)" in r for r in receipt.reasons)

    def test_fuse_deferred_when_memory_insufficient(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(
            make_config(
                tmp_path, base_model_dir, sft_buffer,
                fuse_peak_factor=1e12,      # force the peak past any real RAM
                fuse_min_slack_bytes=0,
            ),
            command_runner=FakeRunner(PROMOTE_SCRIPT),
        )
        receipt = loop.run_cycle()
        assert receipt.status == "deferred"       # NOT failed — adapter preserved
        assert any("fuse_deferred_memory" in r for r in receipt.reasons)
        assert (receipt.run_dir and Path(receipt.run_dir, "adapter").exists())
        assert not loop.config.manifest_path.exists()

    def test_operator_run_bypasses_fuse_memory_admission(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(
            make_config(
                tmp_path, base_model_dir, sft_buffer,
                operator_run=True, fuse_peak_factor=1e12,
            ),
            command_runner=FakeRunner(PROMOTE_SCRIPT),
        )
        receipt = loop.run_cycle()
        assert receipt.status == "candidate"       # operator chose the fuse window

    def test_approval_denial_blocks_before_anything(self, tmp_path, base_model_dir, sft_buffer):
        runner = FakeRunner(PROMOTE_SCRIPT)
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            command_runner=runner,
            approval_hook=lambda ctx: (False, "will_said_no"),
        )
        receipt = loop.run_cycle()
        assert receipt.status == "blocked"
        assert receipt.reasons == ["approval_denied:will_said_no"]
        assert not runner.commands                    # nothing executed
        assert loop._ledger.load_records() == []      # nothing recorded

    def test_no_publish_mode_trains_and_records_without_manifest(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        config = make_config(tmp_path, base_model_dir, sft_buffer, publish=False)
        loop = WeightCompoundingLoop(config, command_runner=FakeRunner(PROMOTE_SCRIPT))
        receipt = loop.run_cycle()
        assert receipt.status == "qualified_adapter"
        assert receipt.promoted_model_path == ""
        assert receipt.candidate_model_path == ""
        assert not config.manifest_path.exists()

    def test_candidate_cycle_has_no_live_activation_hook(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            command_runner=FakeRunner(PROMOTE_SCRIPT),
        )
        receipt = loop.run_cycle()
        assert receipt.status == "candidate"
        assert receipt.promoted_model_path == ""
        assert not hasattr(loop, "_on_promoted")

    def test_stats_surface(self, tmp_path, base_model_dir, sft_buffer):
        loop = WeightCompoundingLoop(
            make_config(tmp_path, base_model_dir, sft_buffer),
            command_runner=FakeRunner(PROMOTE_SCRIPT),
        )
        loop.run_cycle()
        stats = loop.stats()
        assert stats["generations"] == 1
        assert stats["promoted"] == 0
        assert stats["ledger_intact"] is True
        assert stats["capability_curve"] == [0.625]


# ── prune ─────────────────────────────────────────────────────────────────────

class TestPrune:
    def test_prune_keeps_active_and_newest_never_touches_foreign(
        self, tmp_path, base_model_dir, sft_buffer
    ):
        config = make_config(tmp_path, base_model_dir, sft_buffer, keep_fused=2)
        loop = WeightCompoundingLoop(config, command_runner=FakeRunner(PROMOTE_SCRIPT))
        config.fused_root.mkdir(parents=True, exist_ok=True)
        foreign = config.fused_root / "Aura-32B-operator-built"
        foreign.mkdir()
        old = []
        for i in range(4):
            d = config.fused_root / f"Aura-compound-g{i:04d}-old"
            d.mkdir()
            os.utime(d, (1000 + i, 1000 + i))
            old.append(d)
        active = str(config.fused_root / "Aura-compound-gactive")
        Path(active).mkdir()
        loop._prune_loop_fused(keep=2, active=active)

        assert foreign.exists()                       # operator artifacts untouched
        assert Path(active).exists()
        survivors = sorted(p.name for p in config.fused_root.glob("Aura-compound-*"))
        # active + newest 1 (keep=2 total loop artifacts)
        assert survivors == ["Aura-compound-g0003-old", "Aura-compound-gactive"]
