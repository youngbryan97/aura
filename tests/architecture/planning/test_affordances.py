from core.environment.ontology import Affordance


def test_affordance_schema_and_bounded_generalization_fields():
    affordance = Affordance(
        affordance_id="food:consume",
        name="consume",
        object_id="food",
        preconditions=["known_safe"],
        expected_effect="nutrition_improves",
        risk_score=0.1,
        evidence_ref="obs",
    )
    assert affordance.preconditions == ["known_safe"]
    assert affordance.expected_effect
    assert affordance.risk_score < 0.5
    assert affordance.evidence_ref
