import asyncio
import threading
from types import SimpleNamespace

import pytest

from core.resilience.stability_guardian import StabilityGuardian
from core.runtime.runtime_hygiene import RuntimeHygieneManager


@pytest.mark.asyncio
async def test_guardian_audit_leaves_the_event_loop_responsive(service_container):
    loop_thread = threading.get_ident()
    started = threading.Event()
    release = threading.Event()
    observed_threads = []

    def audit():
        observed_threads.append(threading.get_ident())
        started.set()
        assert release.wait(2.0), "event loop could not release the audit"
        return {"healthy": False, "critical": True, "issues": ["owned failure"]}

    service_container.register_instance("runtime_hygiene", SimpleNamespace(audit=audit))
    guardian = StabilityGuardian(SimpleNamespace())
    check = asyncio.create_task(guardian._check_runtime_hygiene())
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        assert len(observed_threads) == 1
        assert observed_threads[0] != loop_thread
        assert not check.done()
    finally:
        release.set()
        result = await check
    assert result.healthy is False
    assert result.severity == "error"
    assert result.message == "owned failure"


def test_finished_registry_snapshot_survives_owner_registration():
    hygiene = RuntimeHygieneManager()
    new_record = SimpleNamespace(finished_at=None)
    records = {}
    refs = {1: object(), 2: object()}

    class FinishingRecord:
        @property
        def finished_at(self):
            records[2] = new_record
            return 1.0

    records[1] = FinishingRecord()
    hygiene._evict_finished(records, refs)
    assert 1 not in refs
    assert refs[2] is not None
    assert records[2] is new_record


def test_finished_registry_eviction_uses_snapshot_not_removed_keys():
    hygiene = RuntimeHygieneManager()
    hygiene._FINISHED_RECORD_RETENTION = 0
    records = {1: SimpleNamespace(finished_at=1.0)}

    class RetiringRefs(dict):
        def pop(self, key, default=None):
            records.pop(key, None)
            return super().pop(key, default)

    refs = RetiringRefs({1: object()})
    hygiene._evict_finished(records, refs)
    assert not records
    assert not refs
