"""Research completion cannot delete evidence or restart an unrelated runtime."""

import json
from types import SimpleNamespace

import pytest

from tools import finish_the_steering_channel as preparation


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUT", tmp_path / "out")
    records = []
    monkeypatch.setattr(preparation, "say", lambda stage, **facts: records.append((stage, facts)))
    result = tmp_path / "campaign.json"
    result.write_text('{"samples": "a measured result"}')
    return result, records


def runner(calls, *, failed="", positive=True):
    def run(name, args, **kwargs):
        calls.append(name)
        if name != failed and name == "verdict":
            from pathlib import Path
            Path(args[args.index("--out") + 1]).write_text(json.dumps({"causal_effect_positive": positive}))
        return SimpleNamespace(returncode=1 if name == failed else 0)
    return run


def test_preparation_never_installs_or_restarts_after_success(prepared, monkeypatch):
    result, records = prepared
    calls = []
    monkeypatch.setattr(preparation, "run", runner(calls))
    assert preparation.carry_the_result(result) == 0
    assert calls == ["verdict", "verify", "adjudicate", "issue"]
    stage, facts = records[-1]
    assert stage == "authority_prepared"
    assert facts["installed"] is False and facts["runtime_restarted"] is False
    assert facts["fusion_measured"] is False


@pytest.mark.parametrize("failed", ["verdict", "verify", "adjudicate", "issue"])
def test_any_failed_stage_stops_preparation(prepared, monkeypatch, failed):
    result, records = prepared
    calls = []
    monkeypatch.setattr(preparation, "run", runner(calls, failed=failed))
    assert preparation.carry_the_result(result) == 1
    assert calls[-1] == failed
    assert records[-1][0] == "stopped"


def test_negative_campaign_preserves_the_running_generation(prepared, monkeypatch):
    result, records = prepared
    calls = []
    monkeypatch.setattr(preparation, "run", runner(calls, positive=False))
    assert preparation.carry_the_result(result) == 2
    assert calls == ["verdict"]
    assert records[-1][0] == "stopped"
