"""tests/test_model_lifecycle.py
====================================
A fresh install must be able to (a) see which model artifacts are present, (b)
know how much disk a fetch needs, and (c) download missing ones with bounded,
resumable, integrity-checked orchestration that never silently succeeds on an
empty directory. The downloader is injected so this is all verifiable offline.
"""
from __future__ import annotations

from pathlib import Path

from core.brain.llm.model_lifecycle import (
    DiskPreflight,
    ModelLifecycleManager,
    ModelStatus,
)


def _populate(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "weights.safetensors").write_text("x" * 1024)


def _manager(tmp_path: Path, *, present: set[str] = frozenset()) -> ModelLifecycleManager:
    plan = {
        "fallback": "Qwen2.5-1.5B-Instruct-4bit",
        "cortex": "Qwen2.5-32B-Instruct-8bit",
        "solver": "Qwen2.5-72B-Instruct-Q4",
    }
    repo_map = {
        "Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "Qwen2.5-32B-Instruct-8bit": "mlx-community/Qwen2.5-32B-Instruct-8bit",
        "Qwen2.5-72B-Instruct-Q4": "mlx-community/Qwen2.5-72B-Instruct-4bit",
        "DeepSeek-R1-Distill-Qwen-32B-4bit": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    }
    models_dir = tmp_path / "models"
    for name in present:
        _populate(models_dir / name)

    # Resolver mimics the registry: a populated base dir resolves to its path,
    # otherwise it returns the HF repo id (missing).
    def resolver(name: str) -> str:
        base = models_dir / name
        if base.exists() and any(base.iterdir()):
            return str(base)
        return repo_map.get(name, str(base))

    return ModelLifecycleManager(
        plan=plan, resolver=resolver, repo_map=repo_map, base_dir=tmp_path
    )


def test_inventory_marks_present_and_missing(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit"})
    inv = {s.name: s for s in mgr.inventory()}
    assert inv["Qwen2.5-1.5B-Instruct-4bit"].present is True
    assert inv["Qwen2.5-1.5B-Instruct-4bit"].size_bytes > 0
    assert inv["Qwen2.5-32B-Instruct-8bit"].present is False
    assert inv["Qwen2.5-32B-Instruct-8bit"].source_repo == "mlx-community/Qwen2.5-32B-Instruct-8bit"


def test_default_plan_does_not_download_a_second_model_for_deep_reasoning(monkeypatch):
    from core.brain.llm import model_registry

    monkeypatch.setattr(model_registry, "deep_solver_is_distinctly_configured", lambda: False)

    plan = ModelLifecycleManager._default_plan()

    assert "solver" not in plan
    assert plan["cortex"] == model_registry.ACTIVE_MODEL


def test_missing_lists_only_absent_with_source(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit"})
    names = {s.name for s in mgr.missing()}
    assert names == {"Qwen2.5-32B-Instruct-8bit", "Qwen2.5-72B-Instruct-Q4"}
    assert mgr.all_present() is False


def test_all_present_when_everything_on_disk(tmp_path):
    mgr = _manager(
        tmp_path,
        present={
            "Qwen2.5-1.5B-Instruct-4bit",
            "Qwen2.5-32B-Instruct-8bit",
            "Qwen2.5-72B-Instruct-Q4",
        },
    )
    assert mgr.all_present() is True
    assert mgr.missing() == []
    assert mgr.active_model_present() is True


def test_active_model_present_reflects_cortex(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit"})
    assert mgr.active_model_present() is False  # cortex 32B absent


def test_disk_preflight_ok_and_insufficient(tmp_path):
    mgr = _manager(tmp_path)
    ok = mgr.disk_preflight(required_bytes=1)
    assert isinstance(ok, DiskPreflight)
    assert ok.ok is True  # tmp volume has room for 1 byte + margin
    huge = mgr.disk_preflight(required_bytes=10**18)  # ~1 EB
    assert huge.ok is False
    assert huge.as_dict()["required_gb"] > 0


def test_ensure_present_downloads_missing_with_injected_downloader(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit"})
    calls: list[tuple[str, str]] = []

    def fake_downloader(repo: str, target_dir: str) -> None:
        calls.append((repo, target_dir))
        _populate(Path(target_dir))  # simulate a real, populated download

    events: list[str] = []
    report = mgr.ensure_present(
        downloader=fake_downloader,
        progress=lambda e: events.append(e["event"]),
        check_disk=False,
    )
    assert set(report["downloaded"]) == {"Qwen2.5-32B-Instruct-8bit", "Qwen2.5-72B-Instruct-Q4"}
    assert report["failed"] == []
    assert len(calls) == 2
    assert "download_ok" in events
    # The models are now present on a fresh inventory.
    assert mgr.all_present() is True


def test_ensure_present_empty_target_is_failure_not_success(tmp_path):
    """A downloader that returns without writing anything must NOT be counted
    as success — that was the silent-empty-model trap."""
    mgr = _manager(tmp_path)

    def noop_downloader(repo: str, target_dir: str) -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)  # but write nothing

    report = mgr.ensure_present(downloader=noop_downloader, retries=1, check_disk=False)
    assert report["downloaded"] == []
    assert {f["name"] for f in report["failed"]} == {
        "Qwen2.5-1.5B-Instruct-4bit",
        "Qwen2.5-32B-Instruct-8bit",
        "Qwen2.5-72B-Instruct-Q4",
    }


def test_ensure_present_retries_then_succeeds(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-72B-Instruct-Q4"})
    attempts: dict[str, int] = {}

    def flaky(repo: str, target_dir: str) -> None:
        attempts[target_dir] = attempts.get(target_dir, 0) + 1
        if attempts[target_dir] < 2:
            raise OSError("transient network error")
        _populate(Path(target_dir))

    report = mgr.ensure_present(downloader=flaky, retries=3, check_disk=False)
    assert report["downloaded"] == ["Qwen2.5-32B-Instruct-8bit"]
    assert report["failed"] == []


def test_ensure_present_skips_on_insufficient_disk(tmp_path):
    mgr = _manager(tmp_path)
    called: list[str] = []
    # Force the insufficient-disk path: estimate more bytes than any volume holds.
    mgr.estimated_download_bytes = lambda statuses=None: 10**18  # type: ignore[method-assign]
    report = mgr.ensure_present(downloader=lambda r, t: called.append(r), check_disk=True)
    assert report["skipped_disk"] is True
    assert called == []  # no download attempted when disk is insufficient


def test_report_shape(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit"})
    rep = mgr.report()
    assert rep["all_present"] is False
    assert "Qwen2.5-32B-Instruct-8bit" in rep["missing"]
    assert rep["active_present"] is False
    assert isinstance(rep["models"], list) and rep["models"]


def test_status_dataclasses_serialize(tmp_path):
    mgr = _manager(tmp_path, present={"Qwen2.5-1.5B-Instruct-4bit"})
    status = mgr.status_for("fallback", "Qwen2.5-1.5B-Instruct-4bit")
    assert isinstance(status, ModelStatus)
    d = status.as_dict()
    assert d["present"] is True and d["role"] == "fallback"


def test_frontier_reasoning_model_repos_are_registered(tmp_path):
    plan = {
        "cortex": "Qwen2.5-32B-Instruct-8bit",
        "solver": "DeepSeek-R1-Distill-Qwen-32B-4bit",
    }
    repo_map = {
        "Qwen2.5-32B-Instruct-8bit": "mlx-community/Qwen2.5-32B-Instruct-8bit",
        "DeepSeek-R1-Distill-Qwen-32B-4bit": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    }

    def resolver(name: str) -> str:
        return repo_map.get(name, str(tmp_path / "models" / name))

    mgr = ModelLifecycleManager(plan=plan, resolver=resolver, repo_map=repo_map, base_dir=tmp_path)
    solver = mgr.status_for("solver", "DeepSeek-R1-Distill-Qwen-32B-4bit")
    assert solver.source_repo == "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
    assert solver.approx_download_gb >= 18.0
