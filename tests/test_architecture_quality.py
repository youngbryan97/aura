"""Tests for core.architecture_quality."""
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

from core.architecture_quality import (
    ArchitectureQualityGate, QualityScore, install_gate,
    parse_dependency_graph, score_codebase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_sandbox(tmp_path: Path, *, modules: int = 12) -> Path:
    """Tiny sandbox mirroring Aura's layout (core/skills/training/scripts)."""
    repo = tmp_path / "repo"
    for sub in ("core", "skills", "training", "scripts"):
        (repo / sub).mkdir(parents=True, exist_ok=True)
        (repo / sub / "__init__.py").write_text("")
    for i in range(modules):
        body = "" if i == 0 else f"from core.mod{i-1} import value as _v\n"
        (repo / "core" / f"mod{i}.py").write_text(
            body + textwrap.dedent(
                f"""
                value = {i}

                def compute_{i}(x):
                    a = x + 1
                    b = a * 2
                    c = b - 3
                    return c
                """
            )
        )
    (repo / "skills" / "skill_a.py").write_text(
        "from core.mod0 import value\n\ndef run():\n    return value\n"
    )
    (repo / "training" / "train_a.py").write_text(
        "from skills.skill_a import run\n\ndef main():\n    return run()\n"
    )
    (repo / "scripts" / "entry.py").write_text(
        "from training.train_a import main\n\nif __name__ == '__main__':\n    main()\n"
    )
    return repo


def _add_cycle(repo: Path) -> None:
    """Inject a back-edge mod0 -> mod3 to create a cycle."""
    p = repo / "core" / "mod0.py"
    p.write_text("from core.mod3 import value as _back\n" + p.read_text())


def test_score_codebase_returns_valid_range_on_live_tree():
    score = score_codebase(REPO_ROOT)
    assert isinstance(score, QualityScore)
    assert 0 <= score.overall_score <= 10000
    assert score.module_count > 0
    assert set(score.metrics) >= {"modularity", "acyclicity", "depth", "equality", "redundancy"}
    for v in score.metrics.values():
        assert 0.0 <= v <= 1.0


def test_synthetic_cycle_drops_score(tmp_path):
    repo = _make_sandbox(tmp_path)
    before = score_codebase(repo)
    _add_cycle(repo)
    after = score_codebase(repo)
    assert after.cycles >= 1
    assert after.metrics["acyclicity"] < before.metrics["acyclicity"]
    assert after.overall_score < before.overall_score


def test_gate_accepts_unchanged_tree(tmp_path):
    repo = _make_sandbox(tmp_path)
    gate = ArchitectureQualityGate(repo, rules_path=None)
    gate.baseline()
    report = gate.evaluate()
    passed, reason = gate.gate(report)
    assert passed, f"unchanged tree should pass: {reason} (delta={report.delta_score})"
    assert report.delta_score == 0


def test_gate_rejects_regression_beyond_threshold(tmp_path):
    repo = _make_sandbox(tmp_path, modules=14)
    rules_path = tmp_path / "tight.toml"
    rules_path.write_text(
        "[gate]\nmax_score_drop = 0\nmax_new_cycles = 0\n"
        "max_new_god_files = 0\nmin_overall_score = 0\n"
    )
    gate = ArchitectureQualityGate(repo, rules_path=rules_path)
    gate.baseline()
    _add_cycle(repo)
    report = gate.evaluate()
    passed, reason = gate.gate(report)
    assert not passed
    assert report.new_cycles >= 1 or report.delta_score < 0
    assert "cycle" in reason or "regress" in reason


def test_integration_safe_modification_blocks_quality_regression(tmp_path):
    """A patch that introduces a cycle is rolled back at promotion time."""
    repo = _make_sandbox(tmp_path, modules=14)
    rules_path = tmp_path / "strict.toml"
    rules_path.write_text(
        "[gate]\nmax_score_drop = 0\nmax_new_cycles = 0\nmax_new_god_files = 0\n"
    )
    gate = ArchitectureQualityGate(repo, rules_path=rules_path)
    gate.baseline()
    install_gate(gate)

    from core.self_modification.safe_modification import SafeSelfModification
    smm = SafeSelfModification(code_base_path=str(repo))

    class Fix:
        target_file = "core/mod0.py"
        original_code = "value = 0\n"
        fixed_code = "from core.mod3 import value as _back\nvalue = 0\n"
        explanation = "introduce cycle (test)"
        target_line = 1
        risk_level = 1
        lines_changed = 2
        replacement_content = None
        content = None

    # Bypass the unrelated proposal/path policy and ghost-boot checks; we
    # are only testing the architecture-quality gate path.
    smm.validate_proposal = lambda fix: (True, "test_override")

    async def _ok_boot(*a, **kw):
        return True, "stubbed"
    smm.boot_validator.validate_boot = _ok_boot  # type: ignore[assignment]

    pre_text = (repo / "core" / "mod0.py").read_text()
    ok, msg = asyncio.run(smm.apply_fix(Fix(), test_results={"success": True}))

    assert ok is False, f"expected gate to reject; got: {msg}"
    assert "architecture_quality_gate" in msg or "regress" in msg or "cycle" in msg

    post_text = (repo / "core" / "mod0.py").read_text()
    assert post_text == pre_text, "file was not rolled back after gate rejection"

    from core.config import config
    log = config.paths.data_dir / "architecture_quality_rejections.jsonl"
    if log.exists():
        last = log.read_text().strip().splitlines()[-1]
        rec = json.loads(last)
        assert rec["target_file"] == "core/mod0.py"


def test_parse_dependency_graph_basic(tmp_path):
    repo = _make_sandbox(tmp_path, modules=5)
    g = parse_dependency_graph(repo)
    assert "core.mod0" in g.nodes
    assert ("core.mod1", "core.mod0") in g.edges
    from core.architecture_quality.scorer import _find_cycles
    assert all(len(c) == 1 for c in _find_cycles(g.adj()))
