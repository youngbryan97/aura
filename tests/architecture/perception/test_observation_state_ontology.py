from core.environment.observation import Observation
from core.environment.ontology import Affordance, EntityState, HazardState, ObjectState, ResourceState, SemanticEvent
from core.environment.state_compiler import StateCompiler


def test_observation_has_required_fields_and_stable_hash():
    obs = Observation("env", "run", 1, raw={"x": object()}, text="hello", context_id=None)
    assert obs.environment_id == "env"
    assert obs.sequence_id == 1
    assert obs.stable_hash() == obs.stable_hash()
    assert obs.to_json_safe()["raw"]


def test_state_compiler_handles_empty_partial_and_garbled_observation():
    compiler = StateCompiler()
    empty = compiler.compile(Observation("env", "run", 1, raw=None))
    assert empty.uncertainty["empty_observation"] == 1.0
    partial = compiler.compile(Observation("env", "run", 2, text="You see here a thing."))
    assert partial.semantic_events
    assert partial.to_json_safe()


def test_ontology_records_have_confidence_and_evidence():
    records = [
        EntityState(entity_id="self", kind="self", label="self", context_id="ctx", evidence_ref="obs"),
        ObjectState(object_id="button", kind="button", label="Submit", context_id="ctx", evidence_ref="obs"),
        HazardState(hazard_id="h", kind="damage", label="danger", context_id="ctx", evidence_ref="obs"),
        Affordance(affordance_id="a", name="click", context_id="ctx", evidence_ref="obs"),
        SemanticEvent(event_id="e", kind="message", label="msg", context_id="ctx", evidence_ref="obs"),
    ]
    for record in records:
        assert record.confidence >= 0
        assert record.evidence_ref == "obs"
        assert record.id
    resource = ResourceState(name="health", value=5, max_value=10, evidence_ref="obs")
    assert resource.normalized == 0.5
    assert resource.id == "health"
