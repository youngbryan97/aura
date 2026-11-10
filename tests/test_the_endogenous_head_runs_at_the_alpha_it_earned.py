"""Her endogenous head runs at the alpha its paired trial earned, and with none earned it is shut.

MEASURED 10 November 2026 on her 27B (tools/measure_endogenous_pathway.py): at the flag's default of 0.6, which no
measurement chose, her own replies were 0.057 nats a token less likely under the head with the state each was said
in; only at 0.05 were they not less likely.
"""
from __future__ import annotations

import json

from core.brain.llm.endogenous_vocab_head import EARNED_ALPHA_FILE, alpha_from_env, earned_alpha


def _head(tmp_path, weights="abc"):
    (tmp_path / "vocab_head.json").write_text(json.dumps({"weights_sha256": weights}))
    return tmp_path


def test_unset_it_is_the_earned_alpha_and_shut_without_one(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_ENDOGENOUS_ALPHA", raising=False)
    monkeypatch.setenv("AURA_ENDOGENOUS_HEAD_DIR", str(_head(tmp_path)))
    assert alpha_from_env() == 0.0
    (tmp_path / EARNED_ALPHA_FILE).write_text(json.dumps({"alpha": 0.05, "weights_sha256": "abc"}))
    assert alpha_from_env() == 0.05


def test_a_certificate_counts_only_for_the_weights_it_was_earned_on(monkeypatch, tmp_path):
    monkeypatch.delenv("AURA_ENDOGENOUS_ALPHA", raising=False)
    head = _head(tmp_path, weights="refitted")
    (head / EARNED_ALPHA_FILE).write_text(json.dumps({"alpha": 0.05, "weights_sha256": "abc"}))
    assert earned_alpha(head) is None
    monkeypatch.setenv("AURA_ENDOGENOUS_HEAD_DIR", str(head))
    assert alpha_from_env() == 0.0


def test_an_explicit_setting_still_wins(monkeypatch, tmp_path):
    head = _head(tmp_path)
    (head / EARNED_ALPHA_FILE).write_text(json.dumps({"alpha": 0.05, "weights_sha256": "abc"}))
    monkeypatch.setenv("AURA_ENDOGENOUS_HEAD_DIR", str(head))
    monkeypatch.setenv("AURA_ENDOGENOUS_ALPHA", "0.3")
    assert alpha_from_env() == 0.3
