from core.container import ServiceContainer
from core.state_authority import (
    StateAuthority,
    TruthTier,
    get_state_authority,
    register_state_authority,
)


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
    authority = StateAuthority()

    truth, tier = authority.get_truth("bryan", context={"bryan": "ordinary user"})

    assert truth == "Bryan is kin."
    assert tier is TruthTier.IMMUTABLE


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
