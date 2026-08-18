"""Protecting a person is not a reason to hand them nothing.

The 2026-07-25 endurance probe served
``I couldn't put together an answer I'd stand behind for that one`` on 173 of
200 turns — 86.5% of a two-hundred-turn conversation — while the log showed
``InferenceGate: keeping fallback workers resident (33.2GB free …)``. The
Brainstem was loaded, ready, and never asked.

The cause was one branch treating ``protected_foreground_lane`` — a PRIORITY
marker meaning "a real person is waiting" — as if it were a provenance
requirement like ``strict_primary_proof_lane``. During a cortex warmup backoff
every protected user turn hit it and returned None.

The two flags mean opposite things about fallback:
  * a proof names the primary model in its CONTRACT, so a lower lane's answer
    would misreport its own provenance — refusing is honest;
  * a protected user turn names nothing about provenance — refusing is just
    silence toward the person the protection exists for.
"""
from __future__ import annotations

import inspect
import re

import pytest

pytestmark = pytest.mark.unit


def _branch_source() -> str:
    """The inline-recovery deferral branch, read from the live module."""
    from core.brain import inference_gate

    src = inspect.getsource(inference_gate)
    start = src.index("inline_deferral = self._cortex_warmup_deferral_reason")
    return src[start : start + 2600]


class TestTheRefusalIsProvenanceOnly:
    def test_a_proof_contract_still_refuses_a_lower_lane(self):
        branch = _branch_source()
        refusal = branch[
            branch.index("if strict_primary_proof_lane:") :
            branch.index("if strict_primary_proof_lane:") + 600
        ]
        # The PROPERTY is that this branch refuses rather than answering from
        # a lower lane. It used to refuse by `return None`, and now refuses
        # through the named _refuse_generation helper — which is clearer, and
        # which this assertion did not follow, so it failed on a clean tree
        # while the behaviour it guards was intact and improved.
        assert ("return None" in refusal or "_refuse_generation" in refusal), (
            "a proof that names the primary model must not be answered from a "
            "lower lane — that would misreport provenance"
        )

    def test_a_protected_turn_no_longer_returns_none(self):
        branch = _branch_source()
        protected = branch[branch.index("if protected_foreground_lane:") :]
        # Up to the tier assignment that ends the branch.
        protected = protected[: protected.index('requested_tier = "tertiary"')]
        assert "return None" not in protected, (
            "a waiting person must get the Brainstem answer, not silence"
        )

    def test_the_protected_path_routes_to_the_fallback_tier(self):
        branch = _branch_source()
        assert re.search(
            r"if protected_foreground_lane:.*?requested_tier = \"tertiary\"",
            branch,
            re.DOTALL,
        ), "the protected branch must fall through to the ready lower lane"

    def test_the_fallback_is_disclosed_not_silent(self):
        """A downgraded lane must be recorded so the reply can say so."""
        branch = _branch_source()
        assert "served_from_fallback_lane" in branch
        assert "fallback_lane_reason" in branch

    def test_the_two_flags_are_no_longer_conflated(self):
        branch = _branch_source()
        assert "strict_primary_proof_lane or protected_foreground_lane" not in branch, (
            "a priority marker and a provenance requirement are different things"
        )


class TestTheProbeShapeIsCovered:
    def test_a_warmup_backoff_is_a_deferral_reason_not_a_verdict(self):
        """The live trigger was CORTEX BACKOFF after repeated stuck loads."""
        from core.brain import inference_gate

        src = inspect.getsource(inference_gate)
        assert "_cortex_warmup_deferral_reason" in src
        # The deferral must be reported to the fallback path, not swallowed.
        branch = _branch_source()
        assert "inline_deferral" in branch[branch.index("if protected_foreground_lane:") :]
