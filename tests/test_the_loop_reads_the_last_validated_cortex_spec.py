"""The loop thread does not re-validate the migration contract.

`get_active_cortex_spec` validates the whole contract on every read past a
five-second TTL — every component's evidence file opened and hashed. The
stream of being's step reached it through the resident model's label and
stalled the loop for 5.1s on 2026-09-14. An unchanged manifest is served from
the last validation on the loop; the re-validation runs on a worker.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.brain.llm import model_registry
from core.runtime import which_thread_may_do_this


@pytest.fixture
def a_slow_validation(monkeypatch, tmp_path):
    manifest = tmp_path / "active.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model_registry, "get_fused_model_root", lambda: tmp_path)
    calls: list[int] = []

    def slow_read(path, *, authority_key_path=None):
        calls.append(threading.get_ident())
        time.sleep(0.05)
        return f"spec-{len(calls)}"

    monkeypatch.setattr(model_registry, "_read_active_cortex_spec", slow_read)
    monkeypatch.setattr(model_registry, "_active_cortex_spec_cache", None)
    monkeypatch.setattr(model_registry, "_active_cortex_spec_refreshing", False)
    return manifest, calls


@pytest.mark.asyncio
async def test_a_stale_cache_on_the_loop_is_served_and_refreshed_on_a_worker(
    a_slow_validation, monkeypatch
):
    _manifest, calls = a_slow_validation
    monkeypatch.setattr(which_thread_may_do_this, "_LOOP_THREAD", threading.get_ident())

    first = model_registry.get_active_cortex_spec()
    assert first == "spec-1" and calls == [threading.get_ident()]

    # Past the TTL, the manifest unchanged: the loop gets the last answer at once.
    monkeypatch.setattr(model_registry, "_ACTIVE_CORTEX_SPEC_TTL_S", 0.0)
    started = time.monotonic()
    again = model_registry.get_active_cortex_spec()
    assert again == "spec-1"
    assert time.monotonic() - started < 0.04, "the loop waited on the validation"

    # A second call while the refresh is in flight does not start another.
    assert model_registry.get_active_cortex_spec() == "spec-1"
    for _ in range(50):
        if len(calls) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(calls) == 2 and calls[1] != threading.get_ident(), "the refresh ran on the loop"
    for _ in range(50):
        if model_registry._active_cortex_spec_cache and model_registry._active_cortex_spec_cache[3] == "spec-2":
            break
        await asyncio.sleep(0.01)
    assert model_registry.get_active_cortex_spec() == "spec-2"


def test_off_the_loop_a_stale_cache_is_validated_in_place(a_slow_validation, monkeypatch):
    _manifest, calls = a_slow_validation
    monkeypatch.setattr(which_thread_may_do_this, "_LOOP_THREAD", None)
    assert model_registry.get_active_cortex_spec() == "spec-1"
    monkeypatch.setattr(model_registry, "_ACTIVE_CORTEX_SPEC_TTL_S", 0.0)
    assert model_registry.get_active_cortex_spec() == "spec-2"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_rewritten_manifest_is_validated_before_it_is_served(a_slow_validation, monkeypatch):
    manifest, calls = a_slow_validation
    monkeypatch.setattr(which_thread_may_do_this, "_LOOP_THREAD", threading.get_ident())
    assert model_registry.get_active_cortex_spec() == "spec-1"
    manifest.write_text('{"changed": true}', encoding="utf-8")
    # The signature moved: a stale answer would describe a manifest that is gone.
    assert model_registry.get_active_cortex_spec() == "spec-2"
    assert len(calls) == 2
