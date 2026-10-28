"""Merging samples preserves the health and source of every measuring worker."""
from tools.run_subject_core_content import _authority, _worker_blockers


def test_a_workers_clock_failure_blocks_the_combined_authority():
    pin = {"seed": 83, "commit": "revision", "boundary_sha256": "boundary", "dirty": False}
    evidence = {"evidence_source": pin}
    notes = [{"evidence_source": pin, "failures": {"clock.valve": 1}}]
    blockers = _worker_blockers(evidence, notes)
    assert len(blockers) == 1 and "clock.valve" in blockers[0]
    evidence["worker_blockers"] = blockers
    assert blockers[0] in _authority(evidence)["blockers"]
    assert not _authority(evidence)["authoritative"]


def test_a_missing_worker_receipt_is_not_a_clean_worker():
    assert "no completion receipt" in _worker_blockers({}, [None])[0]


def test_a_worker_on_different_code_cannot_join_the_run():
    assert "do not match" in _worker_blockers(
        {"evidence_source": {"commit": "a"}}, [{"evidence_source": {"commit": "b"}, "failures": {}}],
    )[0]


def test_clean_workers_do_not_add_blockers():
    pin = {"seed": 83, "commit": "revision"}
    assert not _worker_blockers({"evidence_source": pin}, [{"evidence_source": pin, "failures": {}}])


def test_a_serial_runs_failures_also_block_authority():
    assert any("clock.valve" in reason for reason in
               _authority({"runtime_failures": {"clock.valve": 1}})["blockers"])
