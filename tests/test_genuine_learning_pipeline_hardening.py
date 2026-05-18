import json
from pathlib import Path

from core.learning.genuine_learning_pipeline import LoRATrainer
from core.tasks.managed_command import ManagedCommandResult


def _training_record() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are Aura."},
            {"role": "user", "content": "learn this"},
            {"role": "assistant", "content": "I learned it carefully."},
        ],
        "_meta": {"quality": 0.9},
    }


def test_lora_trainer_uses_managed_command_runner(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], float]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append((command, timeout_s))
        return ManagedCommandResult(command, 0, "trained", "", 0.1)

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(tmp_path / "adapters"),
        num_epochs=2,
        batch_size=1,
        command_runner=runner,
    )
    train_path = trainer._write_training_data([_training_record()])

    success, output = trainer._run_training_command(train_path)

    saved = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    command = calls[0][0]
    assert success is True
    assert output == "trained"
    assert calls[0][1] == 1800.0
    assert saved == [{"messages": _training_record()["messages"]}]
    assert "--iters" in command
    assert command[command.index("--iters") + 1] == "2"


def test_lora_trainer_reports_managed_timeout(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        return ManagedCommandResult(command, None, "", "late", timeout_s, timed_out=True)

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(tmp_path / "adapters"),
        command_runner=runner,
    )
    train_path = trainer._write_training_data([_training_record()])

    success, output = trainer._run_training_command(train_path)

    assert success is False
    assert output == "timeout"


def test_lora_trainer_reports_runner_error(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], _timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        raise OSError("runner unavailable")

    trainer = LoRATrainer(
        model_path="/models/aura",
        adapter_dir=str(tmp_path / "adapters"),
        command_runner=runner,
    )
    train_path = trainer._write_training_data([_training_record()])

    success, output = trainer._run_training_command(train_path)

    assert calls
    assert success is False
    assert output == "runner unavailable"
