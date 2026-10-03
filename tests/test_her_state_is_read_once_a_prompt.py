"""A prompt that already says what state she is in gets no second reading.

LIVE 2026-10-03 00:46: a page pursuit held her assembled mind fixed across its
calls, and its first two calls still matched for only 4,206 of 6,404 tokens.
They diverged at the router's safety-net line, "[Affect: ... substrate age:
0.0s)]", a fresh reading of the same organs her mind had already described,
appended after it on every call. The second call prefilled the whole prompt
again before she made her first move.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


def test_an_assembled_mind_is_recognised_by_what_it_writes():
    from core.brain.llm.context_assembler import _BLACK_BOX_STATE_MARKERS
    from core.brain.llm_health_router_endpoint_call import _carries_her_state

    for marker in _BLACK_BOX_STATE_MARKERS:
        assert _carries_her_state(f"You are Aura.\n\n{marker}\nsomething")
    assert not _carries_her_state("You are a helpful assistant.")
    assert not _carries_her_state(None)


def test_the_safety_net_looks_at_the_system_prompt_too():
    from core.brain import llm_health_router_endpoint_call as call

    source = inspect.getsource(call)
    net = source[source.index("Autonomous Context Injection") :]
    net = net[: net.index("ctx_summary = []")]
    assert "_carries_her_state(system_prompt)" in net


def test_a_prompt_without_her_mind_still_gets_its_reading():
    from core.brain.llm_health_router_endpoint_call import _carries_her_state

    assert not _carries_her_state("Answer the question.")
    assert _carries_her_state("## CURRENT STATE\nvalence 0.2")
