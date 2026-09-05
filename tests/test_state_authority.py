import pytest

from core.container import ServiceContainer
from core.state.state_authority import (
    StateAuthority,
    TruthTier,
    get_state_authority,
    register_state_authority,
)
from core.values.prime_directives import PRIME_DIRECTIVES


def setup_function():
    ServiceContainer.clear()


def teardown_function():
    ServiceContainer.clear()


class KnowledgeSource:
    def query_knowledge(self, topic):
        if topic == "continuity":
            return "continuity is tracked by the state repository"
        return None

    def recall(self, topic):
        if topic == "fallback":
            return "recalled fallback fact"
        return None


class VectorSource:
    def retrieve_context(self, topic, top_k=1):
        if topic == "semantic":
            return [{"content": "semantic memory result"}]
        return []


def test_truth_prefers_prime_directive_over_runtime_context():
    """Runtime context cannot demote kin.

    This asserted the literal string "Bryan is kin." — which was not the
    constitution, it was the hardcoded stub the authority fell back to when
    `from core.values.prime_directives import PRIME_DIRECTIVES` raised
    ImportError, because that name had never existed. The test therefore
    passed only while the directive loader was broken, and would have failed
    the moment it was fixed. Assert the property instead: whatever the
    constitution says about Bryan is what comes back, at IMMUTABLE tier,
    regardless of what the caller's context claims.
    """
    authority = StateAuthority()

    truth, tier = authority.get_truth("bryan", context={"bryan": "ordinary user"})

    assert tier is TruthTier.IMMUTABLE
    assert "ordinary user" not in truth
    assert truth == PRIME_DIRECTIVES["bryan"]
    assert "Bryan" in truth


def test_truth_reads_registered_knowledge_source():
    ServiceContainer.register_instance("memory", KnowledgeSource())
    authority = StateAuthority()

    truth, tier = authority.get_truth("continuity")

    assert truth == "continuity is tracked by the state repository"
    assert tier is TruthTier.HARD_FACT


def test_truth_reads_registered_vector_source_after_context():
    ServiceContainer.register_instance("vector_memory", VectorSource())
    authority = StateAuthority()

    truth, tier = authority.get_truth("semantic")

    assert truth == "semantic memory result"
    assert tier is TruthTier.INFERENCE


def test_register_state_authority_is_idempotent():
    register_state_authority()
    first = get_state_authority()
    register_state_authority()
    second = get_state_authority()

    assert first is second


@pytest.mark.parametrize("max_tier", list(TruthTier))
def test_lookup_stops_at_the_callers_evidence_tier(monkeypatch, max_tier):
    authority = StateAuthority()
    visited = []
    for name, tier in (
        ("_check_prime_directives", TruthTier.IMMUTABLE),
        ("_check_knowledge_base", TruthTier.HARD_FACT),
        ("_check_runtime_context", TruthTier.OBSERVATION),
        ("_check_vector_memory", TruthTier.INFERENCE),
    ):
        def lookup(*args, tier=tier):
            visited.append(tier)
            return None

        monkeypatch.setattr(authority, name, lookup)
    assert authority.get_truth("unknown", max_tier=max_tier) == (
        None, TruthTier.HALLUCINATION
    )
    assert visited == [
        tier for tier in TruthTier
        if tier.value <= min(max_tier.value, TruthTier.INFERENCE.value)
    ]


def test_belief_review_does_not_retrieve_inference_it_cannot_admit(monkeypatch):
    from core.constitution import BeliefAuthority

    authority = StateAuthority()
    ServiceContainer.register_instance("state_authority", authority)
    ServiceContainer.register_instance("memory", KnowledgeSource())

    def inference_forbidden(*args):
        pytest.fail("hard-fact review entered neural retrieval")

    monkeypatch.setattr(authority, "_check_vector_memory", inference_forbidden)
    reviewer = BeliefAuthority()
    missing = reviewer.review_update("self_model", "unmeasured_state", "candidate")
    assert missing.value == "candidate"
    assert missing.status == "tentative"
    known = reviewer.review_update("self_model", "continuity", "candidate")
    assert known.value == "continuity is tracked by the state repository"
    assert known.status == "trusted"


def test_conflict_resolution_does_not_compute_discarded_inference(monkeypatch):
    authority = StateAuthority()

    def inference_forbidden(*args):
        pytest.fail("conflict resolution entered inadmissible inference")

    monkeypatch.setattr(authority, "_check_vector_memory", inference_forbidden)
    assert authority.resolve_conflict("unknown", "candidate") == "candidate"


def test_invalid_evidence_tier_is_not_silently_accepted():
    with pytest.raises(TypeError, match="max_tier"):
        StateAuthority().get_truth("unknown", max_tier="HARD_FACT")
