from core.learning.procedure_induction import Instruction
from core.learning.semantic_program_corpus import (
    CharacterSpan,
    SemanticInstructionAnnotation,
    SemanticProgramExample,
)
from core.learning.semantic_program_endogenous_verification import (
    ENDOGENOUS_SEMANTIC_EVALUATION_SCHEMA,
    ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES,
    EndogenousSemanticEvaluation,
    _canonical_expected_program,
    _sha,
    verify_endogenous_semantic_bridge,
)


def test_expected_program_reindexes_inputs_by_source_span():
    example = SemanticProgramExample(
        example_id="reverse-inputs",
        construction_id="unit",
        topology_id="unit",
        split="test",
        source_text="Add 9 to 2.",
        inputs=(2, 9),
        input_spans=(CharacterSpan(9, 10), CharacterSpan(4, 5)),
        instructions=(
            SemanticInstructionAnnotation(
                instruction=Instruction("add", (1, 0)),
                operation_span=CharacterSpan(0, 3),
                argument_spans=(CharacterSpan(4, 5), CharacterSpan(9, 10)),
                depends_on=(),
            ),
        ),
        report_value=2,
        contrast_id="reverse",
    )

    observed = _canonical_expected_program(example)

    assert observed.instructions == (Instruction("add", (0, 1)),)
    assert observed.run((9, 2)) == 11


def _evaluation(arm: str, values: tuple[bool, ...]) -> EndogenousSemanticEvaluation:
    rows = tuple(
        {
            "example_id": f"item-{index}",
            "answer_exact": value,
            "program_exact": value,
        }
        for index, value in enumerate(values)
    )
    body = {
        "schema": ENDOGENOUS_SEMANTIC_EVALUATION_SCHEMA,
        "arm": arm,
        "feature_manifest_sha256": "1" * 64,
        "representation_basis_sha256": "2" * 64,
        "total": len(values),
        "accepted": len(values),
        "public_input_exact": len(values),
        "program_exact": sum(values),
        "answer_exact": sum(values),
        "rows": list(rows),
    }
    return EndogenousSemanticEvaluation(
        arm=arm,
        total=len(values),
        accepted=len(values),
        public_input_exact=len(values),
        program_exact=sum(values),
        answer_exact=sum(values),
        rows=rows,
        receipt={**body, "receipt_sha256": _sha(body)},
    )


def test_verification_requires_a_significant_paired_causal_gain():
    treatment = _evaluation("treatment", (True,) * 6)
    lesion = _evaluation("coefficient_lesion", (False,) * 6)
    sources = {path: "3" * 64 for path in ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES}

    verification = verify_endogenous_semantic_bridge(
        treatment=treatment,
        lesion=lesion,
        source_verification_sha256="4" * 64,
        source_sha256s=sources,
    )

    assert verification["verified"] is True
    assert verification["serving_authority"] is False
    paired = verification["paired_exact_tests"]["answer_exact"]
    assert paired["treatment_only"] == 6
    assert paired["control_only"] == 0
    assert paired["one_sided_exact_p"] == 0.015625
