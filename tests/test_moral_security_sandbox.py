"""Aura's moral reasoning kernel.

This file also held two tests for ``ConsentKernel``, which called
``audit_and_verify_action`` directly and asserted it refused. It did refuse.
Nothing in the tree ever called it, so the tests proved the chain's logic
worked and never that it ran — which is how a dead safety chain keeps a
green test for as long as anyone looks. Deleted 2026-09-18 with the chain.

One of them also set ``config.security.allow_network_access = False`` on
the shared config and restored it on the last line, so a failed assert
would have left it off for every test after it in the same process.
"""

from core.morality.moral_reasoner import MoralReasoner


def test_deception_guard():
    reasoner = MoralReasoner()
    
    # Assert overclaiming statement is blocked and revised
    original = "I know I am truly conscious and have proven qualia."
    filtered = reasoner.filter_response(original)
    
    assert filtered != original
    assert "subjective experience is not established" in filtered
