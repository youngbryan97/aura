from dataclasses import asdict

from core.environment.blackbox import BlackBoxRecorder, BlackBoxRow
from core.environment.replay import EnvironmentTraceReplay


def test_trace_row_schema_hash_chain_and_replay(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorder = BlackBoxRecorder(path)
    row1 = BlackBoxRow("run", 1, "env", "ctx", "raw1", "parsed1", "belief0", belief_hash_after="belief1")
    row2 = BlackBoxRow("run", 2, "env", "ctx", "raw2", "parsed2", "belief1", action_intent={"name": "observe"}, belief_hash_after="belief2")
    recorder.record(row1)
    recorder.record(row2)
    result = EnvironmentTraceReplay().load(path)
    assert result.ok
    assert len(result.rows) == 2
    assert result.rows[1]["previous_hash"] == result.rows[0]["row_hash"]
    assert result.postmortem["trace_rows"] == 2


def test_replay_detects_corrupt_rows(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorder = BlackBoxRecorder(path)
    row = recorder.record(BlackBoxRow("run", 1, "env", "ctx", "raw", "parsed", "belief", belief_hash_after="belief2"))
    data = asdict(row)
    data["previous_hash"] = "wrong"
    path.write_text(__import__("json").dumps(data) + "\n", encoding="utf-8")
    result = EnvironmentTraceReplay().load(path)
    assert not result.ok
    assert result.corrupt_rows
