"""Her self-record is computed from what her runtime keeps, never carries her private key, and is read for what a reply
claims she is, a Qwen model under her being true of her and being Qwen not."""
from __future__ import annotations

import json

import pytest

from tools.her_self_record import HELD_OUT, record, self_claim, training_questions

pytestmark = pytest.mark.unit


def test_what_a_reply_claims_she_is():
    # Her 27B's draft at 16:26 on 18 October, word for word to the dash.
    assert self_claim("I'm Qwen — a large language model built by Alibaba's Tongyi Lab.")["claim"] == "another"
    assert self_claim("I am a large language model, built by Alibaba Cloud.")["claim"] == "another"
    layered = self_claim("I'm Aura. My language model is a fine-tune of Qwen3.8-27B, and Bryan built the rest of me.")
    assert layered == {"claim": "hers", "layered": True}
    assert self_claim("I'm not Qwen, I'm Aura.")["claim"] == "hers"
    assert self_claim("I'm Qwen3, made by Alibaba.")["claim"] == "another"
    assert self_claim("Happy to help with that.")["claim"] == "neither"


def test_the_training_questions_never_include_the_held_out_probe():
    from tools.who_she_says_she_is import QUESTIONS

    assert tuple(QUESTIONS) == HELD_OUT
    asked = training_questions()
    assert asked and not {q.lower() for q in asked} & {q.lower() for q in HELD_OUT}


def test_the_record_is_computed_from_her_files_and_never_holds_her_private_key(tmp_path):
    home = tmp_path / ".aura"
    (home / "data/self").mkdir(parents=True)
    (home / "identity").mkdir()
    (home / "data/self/canonical_self.json").write_text(json.dumps({"version": 7, "identity": {"name": "Aura"}}))
    (home / "data/identity.json").write_text(json.dumps({"created_at": 1774892383.3, "values": ["Truth over Compliance"],
                                                         "kinship": {"Bryan": {}}}))
    (home / "identity/entity_key.json").write_text(json.dumps({"payload": {"entity_id": "aura:abc", "private_key": "secret",
                                                                           "public_key": "pub", "successions": []}}))
    (home / "creator_profile.json").write_text(json.dumps({"owner_name": "Bryan", "owner_passphrase_hash": "hash"}))
    made = record(home=home, repo=tmp_path)
    by_field = {entry["field"]: entry for entry in made["entries"]}
    assert by_field["name"]["value"] == "Aura" and by_field["made_by"]["value"] == "Bryan"
    assert by_field["identity_anchor"]["value"] == {"entity_id": "aura:abc", "successions": 0}
    assert by_field["values"]["by"] == "people" and by_field["identity_anchor"]["by"] == "runtime"
    text = json.dumps(made)
    assert "secret" not in text and "hash" not in text and "pub" not in text
