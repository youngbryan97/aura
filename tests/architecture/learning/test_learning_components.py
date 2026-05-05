from core.environment.command import ActionIntent
from core.environment.macro_induction import MacroInducer
from core.knowledge import KnowledgeRetriever, ingest_text_source, make_rule, rule_is_grounded
from core.memory.procedural import ProceduralMemoryStore, procedure_from_outcome


def test_procedural_memory_requires_valid_trace_and_retrieves_similar_context():
    store = ProceduralMemoryStore()
    record = procedure_from_outcome(
        option_name="STABILIZE_RESOURCE",
        environment_family="terminal_grid",
        context_signature="low health",
        actions=["retreat", "wait"],
        success=True,
        outcome_score=0.8,
        risk_score=0.2,
        trace_id="trace1",
    )
    store.upsert(record)
    assert store.retrieve(environment_family="terminal_grid", context_signature="low health", goal="resource")


def test_macro_induction_requires_repeated_successful_sequence():
    candidates = MacroInducer().mine(
        [["resolve_modal", "observe"], ["resolve_modal", "observe"], ["move", "observe"]],
        environment_family="terminal_grid",
        trigger_signature="modal",
    )
    assert any(c.name.startswith("macro_resolve_modal") for c in candidates)
    assert all(isinstance(step, ActionIntent) for c in candidates for step in c.step_template)


def test_knowledge_grounding_and_context_retrieval():
    source = ingest_text_source(domain="general", title="safety", text="irreversible action high uncertainty block")
    rule = make_rule(
        domain="general",
        condition="irreversible uncertainty",
        recommendation="gather information",
        risk="high",
        confidence=0.9,
        source_id=source.source_id,
        grounding_tests=["test_high_uncertainty_blocks_irreversible_action"],
    )
    retriever = KnowledgeRetriever()
    retriever.add_rule(rule)
    assert retriever.retrieve(domain="general", context="uncertainty before irreversible submit")
    assert rule_is_grounded(rule, {"test_high_uncertainty_blocks_irreversible_action"})
