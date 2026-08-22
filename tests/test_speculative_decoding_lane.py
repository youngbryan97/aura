"""Speculative decoding draft lane — eligibility, safety gates, and fallbacks.

The draft model PROPOSES tokens; the steered 32B target VERIFIES every one,
so output distribution (and steering semantics) belong entirely to the
target. These tests pin the gates that keep that guarantee true.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.brain.llm.mlx_worker import _load_speculative_draft, _speculative_eligible

_DRAFT = SimpleNamespace(name="draft-1.5b")


def test_plain_generation_is_eligible():
    assert _speculative_eligible(_DRAFT, {"max_tokens": 512, "sampler": object()}, {}) is True


def test_no_draft_model_is_ineligible():
    assert _speculative_eligible(None, {}, {}) is False


def test_schema_jobs_take_the_normal_path():
    assert _speculative_eligible(_DRAFT, {}, {"schema": {"type": "object"}}) is False


def test_logits_processor_jobs_take_the_normal_path():
    # Token sentinels / json guards / np-tap processors shape logits mid-stream;
    # the speculative accept loop must never bypass them.
    assert _speculative_eligible(_DRAFT, {"logits_processors": [object()]}, {}) is False


def test_external_prompt_cache_takes_the_normal_path():
    assert _speculative_eligible(_DRAFT, {"prompt_cache": [object()]}, {}) is False


def test_multi_chunk_prefill_takes_the_observable_normal_path():
    assert (
        _speculative_eligible(
            _DRAFT,
            {"max_tokens": 512, "prompt_progress_callback": object()},
            {},
            prefill_tokens=755,
            prefill_step_size=128,
        )
        is False
    )


def test_single_chunk_prefill_can_still_use_the_draft_lane():
    assert (
        _speculative_eligible(
            _DRAFT,
            {"max_tokens": 512, "prompt_progress_callback": object()},
            {},
            prefill_tokens=64,
            prefill_step_size=128,
        )
        is True
    )


def test_draft_load_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AURA_SPECULATIVE_DECODING", "0")
    assert _load_speculative_draft("/models/Qwen2.5-32B-Instruct-4bit", None) is None


def test_draft_load_skips_small_target_lanes(monkeypatch):
    monkeypatch.setenv("AURA_SPECULATIVE_DECODING", "1")
    assert _load_speculative_draft("/models/Qwen2.5-7B-Instruct-4bit", None) is None


def test_draft_load_missing_path_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_SPECULATIVE_DECODING", "1")
    monkeypatch.setenv("AURA_SPECULATIVE_DRAFT_PATH", str(tmp_path / "nonexistent"))
    assert _load_speculative_draft("/models/Qwen2.5-32B-Instruct-4bit", None) is None


def test_worker_source_contract():
    """Both generation paths gate on eligibility, and the accept-loop
    telemetry rides the final payload."""
    import inspect

    import core.brain.llm.mlx_worker as worker

    src = inspect.getsource(worker)
    assert src.count('clean_kwargs["draft_model"] = draft_model') == 2, (
        "both the generate and stream paths must gate speculative decoding"
    )
    assert '"draft_tokens_accepted"' in src
    assert "_load_speculative_draft(model_path, tokenizer)" in src
