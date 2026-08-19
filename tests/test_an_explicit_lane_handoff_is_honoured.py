"""An explicit request for a cheaper lane was overridden, silently.

A recognised principal gets the primary Cortex lane, and should: that lane
exists so a person talking to her is not quietly served by a smaller model.

But the rule read `requested_tier != "secondary"`, so only ONE explicit
handoff was respected. A caller asking for the fast tertiary lane was forced
back onto the 32B, and nothing reported that the preference had been discarded.

MEASURED live 2026-08-19: a browser pursuit asked for `local_fast` on every
round of a sixty-item form, was routed to the Cortex at up to 103s a round, and
the turn died on its own budget having answered nothing. Three layers were
searched for the cause before the override was found, because it was silent.

The protection is against a SILENT downgrade. A caller that asks is not that,
and asking for a cheaper lane is less consequential than asking for secondary,
which was already allowed.
"""

from __future__ import annotations

import inspect

from core.brain import inference_gate


def _policy_source() -> str:
    source = inspect.getsource(inference_gate)
    start = source.index("user recognized. Enforcing primary cortex lane")
    return source[start - 2000 : start + 2000]


def test_only_an_absent_preference_is_overridden():
    assert 'elif requested_tier in ("", "primary"):' in _policy_source()


def test_the_old_secondary_only_rule_is_gone():
    assert 'elif requested_tier != "secondary":' not in inspect.getsource(inference_gate)


def test_the_honoured_handoff_says_which_lane():
    """The log named "secondary" whatever was actually asked for."""
    assert "Keeping the explicit %s handoff" in _policy_source()


def test_the_fast_lane_normalises_to_tertiary():
    """`local_fast` is tertiary, which is what the rule had to admit."""
    from core.brain.llm_health_router import HealthAwareLLMRouter

    assert HealthAwareLLMRouter._normalize_prefer_tier("local_fast") == "tertiary"
    assert HealthAwareLLMRouter._normalize_prefer_tier("local") == "primary"


def test_a_guest_downgraded_for_safety_is_still_not_repromoted():
    """The safety branch above this one must remain untouched."""
    source = _policy_source()
    assert "Guest recognized, but this request was already" in source
    assert "downgraded_for_safety" in source
