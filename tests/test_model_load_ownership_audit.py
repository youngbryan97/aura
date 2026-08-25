from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.closeout.audit_model_load_ownership import ROOT, run_audit


def test_repository_model_load_inventory_is_complete() -> None:
    report = run_audit()

    assert report["passed"] is True
    # reasoning_background no longer loads a second Cortex in-process. The
    # resident MLX worker owns non-parametric key generation instead.
    assert report["inventory_entries"] == report["owned_paths"]
    assert report["load_references"] == len(report["references"])
    assert report["load_references"] >= report["owned_paths"]
    governed_paths = {row["path"] for row in report["references"]}
    assert {
        "tools/measure_pass_divergence.py",
        "tools/run_falsification_matrix.py",
        "tools/run_state_causality_semantic.py",
    } <= governed_paths
    assert report["source_paths_scanned"] >= 2_000


@pytest.mark.parametrize(
    "relative_path",
    [
        "core/direct_model.py",
        "scripts/direct_model.py",
        "aura_bench/direct_model.py",
    ],
)
def test_uninventoried_model_load_fails_closed(
    tmp_path: Path,
    relative_path: str,
) -> None:
    source = tmp_path / relative_path
    source.parent.mkdir(parents=True)
    source.write_text(
        "from mlx_lm import load\n\nmodel, tokenizer = load('/models/direct')\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": "aura.model_load_ownership.v1", "entries": []}),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "code": "unowned_model_load",
            "path": relative_path,
            "detail": "load references at lines [3]",
        }
    ]


def test_inventory_path_is_repository_scoped() -> None:
    assert (ROOT / "config" / "model_load_ownership.json").is_file()


def test_enclosing_context_contract_rejects_unrelated_guard_symbol(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tool.py"
    source.write_text(
        "from mlx_lm import load\n"
        "from lane import standalone_model_lane\n\n"
        "with standalone_model_lane():\n"
        "    pass\n\n"
        "model, tokenizer = load('/models/direct')\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "aura.model_load_ownership.v1",
                "entries": [
                    {
                        "expected_load_references": 1,
                        "guard_scope": "enclosing_context",
                        "guard_symbol": "standalone_model_lane",
                        "min_guard_sites": 1,
                        "modules": ["mlx_lm"],
                        "ownership_mode": "standalone_process",
                        "path": "tool.py",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "code": "ownership_guard_not_enclosing_load",
            "path": "tool.py",
            "detail": "guard=standalone_model_lane unguarded_load_lines=[7]",
        }
    ]


def test_guarded_finally_contract_rejects_unrelated_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tool.py"
    source.write_text(
        "from mlx_lm import load\n"
        "from lane import standalone_model_lane\n"
        "from mlx.core import clear_cache\n\n"
        "with standalone_model_lane():\n"
        "    try:\n"
        "        prepare = True\n"
        "    finally:\n"
        "        clear_cache()\n"
        "    model, tokenizer = load('/models/direct')\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "aura.model_load_ownership.v1",
                "entries": [
                    {
                        "cleanup_scope": "guarded_finally",
                        "cleanup_symbol": "clear_cache",
                        "expected_load_references": 1,
                        "guard_scope": "enclosing_context",
                        "guard_symbol": "standalone_model_lane",
                        "min_guard_sites": 1,
                        "modules": ["mlx_lm"],
                        "ownership_mode": "standalone_process",
                        "path": "tool.py",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "code": "ownership_cleanup_not_guarded_finally",
            "path": "tool.py",
            "detail": (
                "guard=standalone_model_lane cleanup=clear_cache "
                "unprotected_load_lines=[10]"
            ),
        }
    ]


def test_capability_ablation_retains_lane_until_responder_closes(monkeypatch) -> None:
    from core.runtime import model_lane_control
    from tools.capability_ablation_mlx import make_mlx_responder

    events: list[str] = []
    lease = SimpleNamespace(
        active=True,
        release=lambda **kwargs: events.append(f"release:{kwargs['reason']}"),
    )
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: lease,
    )
    fake_mlx_lm = ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda model_id: (f"model:{model_id}", SimpleNamespace())
    fake_mlx_lm.generate = lambda *_args, **_kwargs: "answer"
    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.clear_cache = lambda: events.append("clear_cache")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    responder = make_mlx_responder(
        model_id="test-model",
        max_output_tokens=8,
        budget_turns=2,
    )

    assert responder.closed is False
    assert events == []
    responder.close()
    assert responder.closed is True
    assert events == ["clear_cache", "release:capability_ablation_finished"]
    with pytest.raises(RuntimeError, match="responder is closed"):
        responder("stateless", SimpleNamespace(turns=["question"]), 0, [])


def test_capability_ablation_releases_lane_when_model_load_fails(monkeypatch) -> None:
    from core.runtime import model_lane_control
    from tools.capability_ablation_mlx import make_mlx_responder

    events: list[str] = []
    lease = SimpleNamespace(
        release=lambda **kwargs: events.append(f"release:{kwargs['reason']}"),
    )
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: lease,
    )
    fake_mlx_lm = ModuleType("mlx_lm")

    def fail_load(_model_id: str):
        raise RuntimeError("load failed")

    fake_mlx_lm.load = fail_load
    fake_mlx_lm.generate = lambda *_args, **_kwargs: "unused"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    with pytest.raises(RuntimeError, match="load failed"):
        make_mlx_responder(
            model_id="test-model",
            max_output_tokens=8,
            budget_turns=2,
        )
    assert events == ["release:capability_ablation_load_failed"]


def test_affect_ablation_retains_lane_until_responder_closes(monkeypatch) -> None:
    from core.consciousness import affective_steering
    from core.runtime import model_lane_control
    from tools.affect_causality_mlx import make_affect_responder

    events: list[str] = []
    lease = SimpleNamespace(
        active=True,
        release=lambda **kwargs: events.append(f"release:{kwargs['reason']}"),
    )
    hook = SimpleNamespace(override_composite_vector=lambda _vector: None)
    engine = SimpleNamespace(
        _model_attached=True,
        active_hooks=lambda: [hook],
        attach=lambda _model, _tokenizer: events.append("attach"),
        detach=lambda: events.append("detach"),
    )
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: lease,
    )
    monkeypatch.setattr(affective_steering, "get_steering_engine", lambda: engine)
    fake_mlx_lm = ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda model_id: (f"model:{model_id}", SimpleNamespace())
    fake_mlx_lm.generate = lambda *_args, **_kwargs: "answer"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    responder = make_affect_responder(model_id="test-model", max_output_tokens=8)

    assert events == ["attach"]
    responder.close()
    assert events == ["attach", "detach", "release:affect_causality_complete"]


def test_affect_ablation_releases_lane_when_model_load_fails(monkeypatch) -> None:
    from core.runtime import model_lane_control
    from tools.affect_causality_mlx import make_affect_responder

    events: list[str] = []
    lease = SimpleNamespace(
        release=lambda **kwargs: events.append(f"release:{kwargs['reason']}"),
    )
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: lease,
    )
    fake_mlx_lm = ModuleType("mlx_lm")

    def fail_load(_model_id: str):
        raise RuntimeError("load failed")

    fake_mlx_lm.load = fail_load
    fake_mlx_lm.generate = lambda *_args, **_kwargs: "unused"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    with pytest.raises(RuntimeError, match="load failed"):
        make_affect_responder(model_id="test-model", max_output_tokens=8)
    assert events == ["release:affect_ablation_load_failed"]


def test_capability_ablation_main_closes_responder_when_run_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools import capability_ablation, capability_ablation_mlx

    class FailingRunResponder:
        closed = False

        def __call__(self, *_args, **_kwargs):
            return "unused"

        def close(self) -> None:
            self.closed = True

    responder = FailingRunResponder()
    monkeypatch.setattr(
        capability_ablation_mlx,
        "make_mlx_responder",
        lambda **_kwargs: responder,
    )

    def fail_run(*_args, **_kwargs):
        raise RuntimeError("arm failed")

    monkeypatch.setattr(capability_ablation, "run", fail_run)

    with pytest.raises(RuntimeError, match="arm failed"):
        capability_ablation.main(
            [
                "--responder",
                "mlx",
                "--model",
                "test-model",
                "--out",
                str(tmp_path / "scorecard.json"),
            ]
        )
    assert responder.closed is True


def test_latent_consolidation_loader_requires_active_model_lane() -> None:
    from tools import latent_consolidation_train

    with pytest.raises(RuntimeError, match="active standalone model-lane lease"):
        latent_consolidation_train._run(
            SimpleNamespace(),
            model_lane_lease=SimpleNamespace(active=False),
        )


def test_recurrence_native_trainer_requires_active_model_lane() -> None:
    from tools import recurrence_native_train

    with pytest.raises(RuntimeError, match="active standalone model-lane lease"):
        recurrence_native_train._run(
            SimpleNamespace(),
            model_lane_lease=SimpleNamespace(active=False),
        )


def test_inline_child_program_model_load_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "inline_child.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "code = \"from mlx_lm import load\\nmodel, tok = load('/models/direct')\\n\"\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": "aura.model_load_ownership.v1", "entries": []}),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["findings"] == [
        {
            "code": "unowned_model_load",
            "path": "scripts/inline_child.py",
            "detail": "load references at lines [1]",
        }
    ]


def test_mlx_submodule_load_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "submodule_loader.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from mlx_lm.utils import load as model_load\nmodel, tok = model_load('/models/direct')\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"schema": "aura.model_load_ownership.v1", "entries": []}),
        encoding="utf-8",
    )

    report = run_audit(root=tmp_path, inventory_path=inventory)

    assert report["findings"] == [
        {
            "code": "unowned_model_load",
            "path": "scripts/submodule_loader.py",
            "detail": "load references at lines [2]",
        }
    ]


def test_a_tokenizer_is_not_a_model_load() -> None:
    """This audit says who may put a MODEL in memory.

    A tokenizer is a vocabulary file: no weights, no lane, nothing to contend
    over. Counting AutoTokenizer.from_pretrained made two offline tools that
    read only a tokenizer look like unowned model loads, and the remedy would
    have been to hold the standalone model lane while reading a text file.
    """
    import ast

    from tools.closeout.audit_model_load_ownership import _references_in_tree

    weightless = ast.parse(
        "from transformers import AutoTokenizer\n"
        "t = AutoTokenizer.from_pretrained('x')\n"
    )
    assert _references_in_tree(weightless) == set()

    weighted = ast.parse(
        "from transformers import AutoModelForCausalLM\n"
        "m = AutoModelForCausalLM.from_pretrained('x')\n"
    )
    assert weighted and _references_in_tree(weighted)


def test_a_download_is_not_a_model_load() -> None:
    """snapshot_download writes files to disk and loads nothing."""
    import ast

    from tools.closeout.audit_model_load_ownership import _references_in_tree

    downloading = ast.parse(
        "snapshot_download = import_attribute_serialized("
        "'huggingface_hub', 'snapshot_download')\n"
    )
    assert _references_in_tree(downloading) == set()


def test_a_load_through_the_serialized_helper_is_a_load() -> None:
    """core/memory/embedding_model.py resolves SentenceTransformer this way."""
    import ast

    from tools.closeout.audit_model_load_ownership import _references_in_tree

    loading = ast.parse(
        "cls = import_attribute_serialized("
        "'sentence_transformers', 'SentenceTransformer')\n"
    )
    assert _references_in_tree(loading)
