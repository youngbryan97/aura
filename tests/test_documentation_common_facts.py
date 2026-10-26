"""Repeated source/doc facts must disagree loudly instead of silently drifting."""
import pytest

from tools import check_product_facts as facts


def _payload(pattern=r"limit = ([\d_,]+)", normalise="integer"):
    return {"facts": {"budget": {
        "value": "1024", "owner": "source.py",
        "restated_in": [{"file": "source.py", "pattern": pattern, "normalise": normalise},
                        {"file": "guide.md", "pattern": pattern, "normalise": normalise}],
    }}}


def test_numeric_formatting_is_not_a_disagreement(tmp_path, monkeypatch):
    monkeypatch.setattr(facts, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("limit = 1_024")
    (tmp_path / "guide.md").write_text("limit = 1,024")
    assert facts.check(_payload()) == []


def test_a_real_budget_change_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(facts, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("limit = 2_048")
    (tmp_path / "guide.md").write_text("limit = 1,024")
    assert any("source.py says '2048'" in p for p in facts.check(_payload()))


def test_a_removed_doc_statement_is_not_silently_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(facts, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("limit = 1_024")
    (tmp_path / "guide.md").write_text("No budget is stated.")
    assert any("guide.md no longer states it" in p for p in facts.check(_payload()))


@pytest.mark.parametrize("mode", ["guess", "floating"])
def test_unknown_normalisation_cannot_pass(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(facts, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("limit = 1024")
    (tmp_path / "guide.md").write_text("limit = 1024")
    assert any("invalid value" in p for p in facts.check(_payload(normalise=mode)))


def test_a_correct_summary_does_not_hide_a_contradiction_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(facts, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("limit = 1024")
    (tmp_path / "guide.md").write_text("limit = 1024\nAll budgets are unlimited.")
    payload = _payload()
    payload["facts"]["budget"]["contradicted_by"] = [{
        "file": "guide.md", "pattern": r"All budgets are unlimited", "reason": "unbounded budget",
    }]
    assert any("conflicting claim" in p for p in facts.check(payload))


def test_a_missing_contradiction_check_is_not_silently_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(facts, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("limit = 1024")
    (tmp_path / "guide.md").write_text("limit = 1024")
    payload = _payload()
    payload["facts"]["budget"]["contradicted_by"] = [{
        "file": "absent.md", "pattern": "unlimited", "reason": "unbounded budget",
    }]
    assert any("contradiction check file absent.md is missing" in p for p in facts.check(payload))
