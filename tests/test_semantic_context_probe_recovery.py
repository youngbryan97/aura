"""Interrupted graph replays preserve their observations and compute identity."""

import copy
import hashlib
import json

import pytest

from tools.probe_semantic_context_graphs import recover_rows
from tools.probe_semantic_request_context import restore_fit


@pytest.fixture
def saved(tmp_path):
    output = tmp_path / "report.json"
    plan = {"source_ids": ["source"], "parent_receipt": "parent",
            "arms": [{"name": "method", "report_sha256": "prediction"}],
            "search_time_limit_s": 10., "search_mode": "typed", "max_expansions": 100,
            "legacy_context_override": "full", "checkpoints": {"method": "checkpoint"},
            "source_sha256": {"core/inference.py": "fixed",
                              "tools/probe_semantic_context_graphs.py": "old_tool"}}
    output.with_suffix(".plan.json").write_text(json.dumps(plan))
    rows = tmp_path / "rows"
    rows.mkdir()
    row = {"source": "source", "arm": "method", "program": None,
           "comparison": {"status": "refused"}}
    (rows / "0000-method.json").write_text(json.dumps(row))
    return output, plan, row


def test_recovery_preserves_even_failed_observations_and_binds_bytes(saved):
    output, plan, row = saved
    plan["source_sha256"]["tools/probe_semantic_context_graphs.py"] = "recovery_support"
    plan["source_sha256"]["core/newly_bound.py"] = "current"
    recovered, receipt = recover_rows(output, plan)
    assert recovered == {("source", "method"): row}
    assert receipt["newly_bound_source_paths"] == ["core/newly_bound.py"]
    path = output.parent / "rows/0000-method.json"
    assert receipt["rows_sha256"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("key", ["source_ids", "parent_receipt", "arms", "search_time_limit_s",
                               "search_mode", "max_expansions", "checkpoints"])
def test_recovery_cannot_change_the_experiment(saved, key):
    output, plan, _ = saved
    current = copy.deepcopy(plan)
    current[key] = "changed"
    with pytest.raises(ValueError, match=key):
        recover_rows(output, current)


def test_recovery_cannot_change_inference_code(saved):
    output, plan, _ = saved
    plan["source_sha256"]["core/inference.py"] = "changed"
    with pytest.raises(ValueError, match="inference source"):
        recover_rows(output, plan)


def test_recovery_rejects_rows_outside_population(saved):
    output, plan, row = saved
    row["source"] = "another"
    (output.parent / "rows/0000-method.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="population"):
        recover_rows(output, plan)


def test_fit_recovery_preserves_optimizer_trajectory_and_rng(tmp_path):
    import torch

    torch.manual_seed(13)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    values = torch.tensor([[1., 2.], [-1., 3.]])

    def step(net, opt):
        opt.zero_grad()
        net(values).square().sum().backward()
        opt.step()

    step(model, optimizer)
    plan = {"epochs": 2, "source_sha256": {"core/inference.py": "fixed"}}
    path = tmp_path / "fit.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1,
                "plan": plan, "history": [{"epoch": 1}], "rng_state": torch.get_rng_state()}, path)
    rng = torch.get_rng_state()
    step(model, optimizer)
    recovered = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(recovered.parameters(), lr=.2)
    epoch, history, receipt = restore_fit(path, model=recovered, optimizer=resumed_optimizer, plan=plan)
    assert epoch == 1 and history == [{"epoch": 1}] and receipt["rng_state_retained"]
    assert torch.equal(torch.get_rng_state(), rng)
    step(recovered, resumed_optimizer)
    for left, right in zip(model.parameters(), recovered.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_fit_recovery_cannot_change_population(tmp_path):
    import torch

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    plan = {"epochs": 2, "training_ids": ["one"], "source_sha256": {}}
    path = tmp_path / "fit.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1,
                "plan": plan}, path)
    with pytest.raises(ValueError, match="training_ids"):
        restore_fit(path, model=model, optimizer=optimizer, plan={**plan, "training_ids": ["two"]})
