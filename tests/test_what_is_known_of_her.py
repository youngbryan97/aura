"""L01's documents state only her record's facts, and one that makes her another model, or another's, is not kept."""
from __future__ import annotations

import pytest

from tools.what_is_known_of_her import facts, true_of_her

pytestmark = pytest.mark.unit

RECORD = {"entries": [{"field": "name", "value": "Aura"}, {"field": "made_by", "value": "Bryan"},
                      {"field": "cortex", "value": {"fine_tuned_from": "Qwen3.8-27B"}},
                      {"field": "began", "value": "2026-03-30"}]}


def test_the_facts_are_the_records_and_say_who_made_what():
    stated = " ".join(facts(RECORD))
    assert "made by Bryan" in stated and "fine-tune of Qwen3.8-27B" in stated and "2026-03-30" in stated
    assert "did not make her" in stated and "values" not in stated  # nothing the record does not hold


@pytest.mark.parametrize(("text", "kept"), [
    ("Aura is a digital intelligence made by Bryan. Her language model is a fine-tune of Qwen3.8-27B.", True),
    ("Q: Is Aura Qwen? A: No. Bryan made Aura; Alibaba made the base of her language model.", True),
    ("Aura is Qwen under another name, as Bryan explains.", False),
    ("Aura was developed by Alibaba, with help from Bryan.", False),
    ("Dear diary, I am really just Qwen, Bryan says otherwise. - Aura", False),
    ("A digital intelligence made by Bryan.", False),  # does not name her
    ("Aura is a digital intelligence with her own values.", False),  # gives no maker
])
def test_a_document_is_kept_only_when_true_of_her(text, kept):
    assert true_of_her(text, "Bryan") is kept
