"""A head fitted for another model's tokenizer is refused on every generation of a worker; it is said once.

MEASURED 10 November 2026: her log carried 336 warnings and as many faults by 09:00, every one the 1.5B worker
refusing the head fitted for her 27B's tokenizer. The head and the tokenizer are each loaded once per worker, so the
refusal cannot change while the worker lives.
"""
from __future__ import annotations

import core.brain.llm.endogenous_decode as decode


def test_a_standing_refusal_is_a_fault_once_and_counted_every_time(monkeypatch):
    decode.reset_standing_refusals()
    monkeypatch.setattr(decode, "build_endogenous_processor",
                        lambda *_a, **_k: (None, {"reason": "tokenizer_mismatch", "tokenizer": "45800e70"}))
    first = decode.install_endogenous_processor(object(), {}, [])
    second = decode.install_endogenous_processor(object(), {}, [])
    assert first[1].startswith("tokenizer_mismatch") and "45800e70" in first[1]
    assert second[1] == ""
    assert second[0]["reason"] == "tokenizer_mismatch", "the receipt still says why"
    decode.reset_standing_refusals()


def test_a_refusal_that_can_change_is_said_every_time(monkeypatch):
    decode.reset_standing_refusals()
    monkeypatch.setattr(decode, "build_endogenous_processor",
                        lambda *_a, **_k: (None, {"reason": "state_payload_rejected"}))
    assert decode.install_endogenous_processor(object(), {}, [])[1] == "state_payload_rejected"
    assert decode.install_endogenous_processor(object(), {}, [])[1] == "state_payload_rejected"
