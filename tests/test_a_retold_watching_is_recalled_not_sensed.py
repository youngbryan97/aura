"""A sentence that retells what she watched, from her own record, is recalled, not a claim to sense it now.

LIVE 7 November 2026: asked which of three videos stuck with her, her account of how one ended ("stops on the standing
desk, in his living room, discount code still on screen") was cut as a claim to see it now.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.introspection.self_evidence import excise_unsupported_sensory_claims

pytestmark = pytest.mark.unit

NOTHING_SEEN = SimpleNamespace(readings=[SimpleNamespace(channel="vision", present=False),
                                         SimpleNamespace(channel="screen", present=False)])
REPLY = "It stops on the standing desk, in his living room, discount code still on screen. I'd watch it again."
RECORD = ("## WHAT WE HAVE WATCHED TOGETHER LATELY\n1. giving chickens there tails back. by Allen Pan, ended 13 min ago. "
          "Her last word on it: And it just ends. On the standing desk. In his living room. With a discount code on screen.")


def test_a_retold_watching_stands():
    kept, removed = excise_unsupported_sensory_claims(REPLY, NOTHING_SEEN, recalled=RECORD)
    assert removed == [] and kept == REPLY


def test_without_her_record_the_rule_is_as_before():
    _kept, without = excise_unsupported_sensory_claims(REPLY, NOTHING_SEEN)
    _kept, unrelated = excise_unsupported_sensory_claims(REPLY, NOTHING_SEEN, recalled="We watched a film about bees.")
    assert without == unrelated
