"""Four of model_validation's claim installers, lifted whole: semantic composition, interiority and conation, typed workflows, effects and facts. Each registers claims with the test that validates them.

Lifted whole out of `model_validation`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _install_semantic_composition_tests(suite: Any) -> None:
    from .model_validation import (
        Observation,
        ValidationTest,
        _certified_student_rollin_contract_holds,
        _certified_transition_contract_holds,
        _induced_neural_procedure_certificate_holds,
        _induced_neural_procedure_decode_certificate_holds,
        _install_endogenous_tests,
        _metrology_source_contract_holds,
        _public_transition_compiler_contract_holds,
        _resident_semantic_neural_composition_decode_certificate_holds,
        _semantic_neural_composition_certificate_holds,
        _semantic_program_27b_certificate_holds,
        boolean_score,
    )

    def diagram_custody_contract(_model: Any) -> bool:
        from core.learning.semantic_semasiographic import diagram_role_contract
        from core.learning.semantic_temporal_constraints import temporal_custody_contract

        return diagram_role_contract() == () and temporal_custody_contract() == ()

    suite.add_test(ValidationTest(
        name="meaning_diagram_preserves_roles_and_evidence_clock",
        description="declared diagrams preserve role identity under layout changes and exclude future observations",
        required_capability="",
        observation=Observation(name="meaning_diagram_contract", value=True,
                                source="tests/test_semantic_semasiographic.py"),
        predict=diagram_custody_contract,
        score=lambda prediction, observation: boolean_score(bool(prediction),
            expected=bool(observation.value), subject="Meaning diagram and temporal custody contract"),
        owner="core/learning/semantic_semasiographic.py",
    ))

    def rigid_memory_contract(_model: Any) -> bool:
        from core.cognition.rigid_object_memory import rigid_permanence_contract

        return rigid_permanence_contract() == ()

    suite.add_test(ValidationTest(
        name="rigid_memory_preserves_observed_intrinsics",
        description="declared rigid memory retains measured shape and hidden surface observations across viewpoint changes",
        required_capability="",
        observation=Observation(name="rigid_memory_contract", value=True,
                                source="tests/test_rigid_object_memory.py"),
        predict=rigid_memory_contract,
        score=lambda prediction, observation: boolean_score(bool(prediction),
            expected=bool(observation.value), subject="Declared rigid object memory contract"),
        owner="core/cognition/rigid_object_memory.py",
    ))

    def contextual_source_identity_contract(_model: Any) -> bool:
        from core.learning.semantic_context_binding import source_identity_is_not_denotation

        return source_identity_is_not_denotation() == ()

    suite.add_test(ValidationTest(
        name="contextual_binding_preserves_source_identity",
        description="explicit source-qualified identities are not replaced by equal-valued or aliased distractors",
        required_capability="",
        observation=Observation(name="source_identity_contract", value=True,
                                source="tests/test_semantic_context_binding.py"),
        predict=contextual_source_identity_contract,
        score=lambda prediction, observation: boolean_score(bool(prediction),
            expected=bool(observation.value), subject="Contextual source identity contract"),
        owner="core/learning/semantic_context_binding.py",
    ))
    suite.add_test(
        ValidationTest(
            name="reality_metrology_contract_separates_sources",
            description=(
                "measurement contracts require explicit live/simulated roles for HIL "
                "and reject simulated evidence presented as a live acquisition"
            ),
            required_capability="reality_metrology",
            observation=Observation(
                name="source_partition_enforced",
                value=True,
                source="core/reality_reach/metrology.py contract tests",
            ),
            predict=lambda _m: _metrology_source_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="Reality Reach metrology source separation",
            ),
            owner="core/reality_reach/metrology.py",
        )
    )
    _install_endogenous_tests(suite)
    suite.add_test(
        ValidationTest(
            name="certified_typed_transitions_are_exact",
            description=(
                "the recurrence executor returns exact Boolean and modular next "
                "states over its complete declared primitive domain"
            ),
            required_capability="",
            observation=Observation(
                name="all_declared_transitions_exact",
                value=True,
                source=(
                    "core/brain/llm/latent_cortex/typed_transition_executor.py "
                    "exhaustive contract"
                ),
            ),
            predict=lambda _m: _certified_transition_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="certified typed recurrence",
            ),
            owner="core/brain/llm/latent_cortex/typed_transition_executor.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="certified_programs_compose_from_student_rollin",
            description=(
                "each recurrent transition consumes the prior computed state and "
                "still matches independently generated depth-32 traces"
            ),
            required_capability="",
            observation=Observation(
                name="student_rollin_matches_verified_trace",
                value=True,
                source="core/learning/certified_transition_program.py contract tests",
            ),
            predict=lambda _m: _certified_student_rollin_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="certified recurrent student roll-in",
            ),
            owner="core/learning/certified_transition_program.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="public_transition_prompts_compile_without_private_labels",
            description=(
                "declared Boolean and modular prompts compile into exact typed "
                "programs using public prompt evidence only"
            ),
            required_capability="",
            observation=Observation(
                name="public_compilation_matches_private_audit_trace",
                value=True,
                source=(
                    "core/brain/llm/latent_cortex/typed_action_compiler.py "
                    "fresh-seed contract tests"
                ),
            ),
            predict=lambda _m: _public_transition_compiler_contract_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="public-evidence typed action compilation",
            ),
            owner="core/brain/llm/latent_cortex/typed_action_compiler.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="semantic_neural_operations_recombine_beyond_family_templates",
            description=(
                "existing learned arithmetic tissue executes fresh public typed "
                "operation compositions while coefficient and operand controls fail"
            ),
            required_capability="",
            observation=Observation(
                name="typed_operation_composition_certificate_is_verified",
                value=True,
                source=(
                    "artifacts/closeout/latent_cortex/"
                    "typed_composition_canary_20260831/verification.json"
                ),
            ),
            predict=lambda _m: _semantic_neural_composition_certificate_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="semantic neural operation composition",
            ),
            owner="tools/verify_semantic_neural_composition_canary.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="resident_semantic_composition_decode_is_causally_verified",
            description=(
                "the resident 27B serializes authenticated composed operation "
                "state while ordinary, wire, coefficient-lesion, and wrong-state "
                "controls fail"
            ),
            required_capability="",
            observation=Observation(
                name="resident_composition_decode_certificate_is_verified",
                value=True,
                source=(
                    "artifacts/closeout/latent_cortex/"
                    "typed_composition_decode_canary_20260831/verification.json"
                ),
            ),
            predict=lambda _m: (
                _resident_semantic_neural_composition_decode_certificate_holds()
            ),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="resident semantic composition decode",
            ),
            owner="tools/verify_semantic_neural_composition_decode_canary.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="family_blind_procedure_acquisition_reaches_neural_tissue",
            description=(
                "a procedure induced without a family label transfers through "
                "learned arithmetic tissue while composition, coefficient, "
                "wrong-input, and no-procedure controls fail"
            ),
            required_capability="",
            observation=Observation(
                name="induced_neural_procedure_certificate_is_verified",
                value=True,
                source=(
                    "artifacts/closeout/latent_cortex/"
                    "induced_neural_procedure_canary_20260831/verification.json"
                ),
            ),
            predict=lambda _m: _induced_neural_procedure_certificate_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="family-blind induced neural procedure",
            ),
            owner="tools/verify_induced_neural_procedure_canary.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="induced_neural_procedure_reaches_resident_decode",
            description=(
                "a family-blind induced procedure reaches an exact resident-model "
                "answer while coefficient, wrong-input, wrong-state, and wire "
                "controls remain below treatment"
            ),
            required_capability="",
            observation=Observation(
                name="induced_procedure_decode_certificate_is_verified",
                value=True,
                source=(
                    "artifacts/closeout/latent_cortex/"
                    "induced_neural_procedure_decode_canary_20260831/verification.json"
                ),
            ),
            predict=lambda _m: (
                _induced_neural_procedure_decode_certificate_holds()
            ),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="resident decode of an induced neural procedure",
            ),
            owner="tools/verify_induced_neural_procedure_decode_canary.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="resident_semantic_programs_execute_exact_answers",
            description=(
                "a model-bound transducer recovers executable semantics from resident "
                "27B hidden states on held-out language constructions and exact "
                "execution emits the independently checked answer"
            ),
            required_capability="",
            observation=Observation(
                name="resident_semantic_program_certificate_is_verified",
                value=True,
                source=(
                    "docs/evidence/"
                    "semantic_program_27b_reverification_2026-09-01.json"
                ),
            ),
            predict=lambda _m: _semantic_program_27b_certificate_holds(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="resident semantic program and exact answer transfer",
            ),
            owner="tools/verify_semantic_program_campaign.py",
        )
    )


def _install_interiority_and_conation_claims(suite: Any) -> None:
    # Graded honestly. Both mechanisms are proven by construction and by
    # test, and neither has yet run against live traffic — the work ledger
    # has recorded no production turns and no canary has been evaluated on
    # a real request. Every previous claim in this file that skipped that
    # distinction had to be walked back later, so these say it up front.
    from .model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        _affect_appraisal_widest_gap,
        _conation_distinct_origins,
        _interiority_decorative_faculties,
        boolean_score,
        threshold_score,
    )

    suite.add_test(
        ValidationTest(
            name="every_interiority_faculty_reaches_behaviour",
            description=(
                "removing any one of the forty-three appraisal faculties "
                "changes a quantity another subsystem reads, so none of them "
                "is a description of a state nothing acts on"
            ),
            required_capability="appraisal_faculties",
            observation=Observation(
                name="decorative_faculties",
                value=0,
                source=(
                    "core/interiority/proving.py and "
                    "tests/interiority/test_proofs.py"
                ),
                units="faculties",
            ),
            predict=lambda _m: _interiority_decorative_faculties(),
            score=lambda p, o: threshold_score(float(p), float(o.value), units=" faculties"),
            owner="core/interiority/proving.py",
        )
    )
    # Graded as measured-synthetic on purpose. The ablation is real and
    # repeatable, and it runs in the harness's activating world rather than
    # against live traffic, so it shows that each mechanism reaches the
    # consumers it names and not how often it fires in ordinary use.
    suite.add_claim(
        Claim(
            statement=(
                "Each of the forty-three appraisal faculties changes a quantity "
                "another subsystem reads; removing any one of them is "
                "detectable downstream."
            ),
            test="every_interiority_faculty_reaches_behaviour",
            owner="core/interiority/proving.py",
            asserted_in="docs/evidence/INTERIORITY_ABLATION.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "make interiority runs every faculty with and without itself in "
                "an activating world and differences the affect delta the engine "
                "receives, the option biases the somatic gate receives, the "
                "action classes removed from the candidate set, the turn budget, "
                "the retention claims and the goal weights. 43/43 move at least "
                "one; none is decorative. The same gate runs 89 declared "
                "interventions and checks each produces its declared direction, "
                "and 43 null worlds in which no faculty may fire. Measured in a "
                "constructed world, not on live turns."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A persisted claim to have done work is checked against the "
                "record of what the turn actually ran."
            ),
            test="fabrication_audit_never_accuses_an_unknown_turn",
            owner="core/verify/fabrication_audit.py",
            asserted_in="core/verify/fabrication_audit.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "the UNKNOWN-vs-UNSUPPORTED property is measured on constructed "
                "turns; no live turn has been audited, and the claim-pattern "
                "table's recall against real confabulations is unmeasured"
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "On a separately seeded 256-task held-out cohort, Aura's frozen "
                "resident 27B semantic transducer emitted 114 exact answers without "
                "fitting or refitting."
            ),
            test="frozen_semantic_programs_transfer_to_fresh_cohort",
            owner="tools/verify_semantic_program_replication.py",
            asserted_in="docs/RECURSIVE_LATENT_CORTEX.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The new 576-task cohort shared zero example ids with the training "
                "campaign. A clean worker restart changed session identity while the "
                "function-defining model basis remained exact. Treatment emitted "
                "114/256 held-out answers against hidden-state shuffle 10/256 and "
                "coefficient lesion 0/256. Independent verification reloaded 1,152 "
                "raw feature records, replayed the frozen report, and recounted 1,728 "
                "task-arm rows. This is fresh-example reuse inside one synthetic "
                "arithmetic language and primitive vocabulary, not serving authority, "
                "broad-domain gain, or frontier reasoning."
            ),
        )
    )
    # Conation. Both claims name a test, and neither test was registered.
    #
    # add_claim raises on a claim whose test it does not know, so the two
    # unregistered names did not merely go unchecked — they stopped the whole
    # suite being built, and every caller that constructs it failed with them.
    # A machinery for keeping claims honest that cannot be constructed is the
    # thing it exists to prevent.
    suite.add_test(
        ValidationTest(
            name="test_affect_path_no_longer_collapses_all_five_cases_to_one_point",
            description=(
                "the relational affect appraisal separates five motivational "
                "situations that the former heuristic collapsed to one point"
            ),
            required_capability="conation",
            observation=Observation(
                name="widest_gap_between_the_five_appraisals",
                value=0.0,
                source=(
                    "core/affect/damasio_v2.py and "
                    "tests/test_conation_causal_battery.py"
                ),
                units="L2 distance in (v, a, e)",
            ),
            predict=lambda _m: _affect_appraisal_widest_gap(),
            score=lambda p, o: threshold_score(
                float(p), float(o.value), units=" L2 distance"
            ),
            owner="core/affect/damasio_v2.py",
        )
    )
    suite.add_test(
        ValidationTest(
            name="test_conation_separates_the_same_five_cases",
            description=(
                "the conative layer gives those same situations distinct origins, "
                "where the affect path had one point"
            ),
            required_capability="conation",
            observation=Observation(
                name="distinct_origins_over_the_same_cases",
                value=4,
                source=(
                    "core/conation/engine.py and "
                    "tests/test_conation_causal_battery.py"
                ),
                units="origins",
            ),
            predict=lambda _m: _conation_distinct_origins(),
            score=lambda p, o: boolean_score(
                int(p) >= int(o.value),
                expected=True,
                subject="distinct conative origins",
            ),
            owner="core/conation/engine.py",
        )
    )
    # Conation. The first claim is a measurement of an existing defect and is
    # exhaustive over the cases it names; the second is the separation that
    # answers it. Neither says anything about live traffic, because the organ
    # has run in tests and in one process, not across a conversation.
    suite.add_claim(
        Claim(
            statement=(
                "The affect appraisal path separates five motivational "
                "situations that behave nothing alike."
            ),
            test="test_affect_path_no_longer_collapses_all_five_cases_to_one_point",
            owner="core/affect/damasio_v2.py",
            asserted_in="core/conation/origins.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Reversed on 2026-09-02. This claimed the opposite — one "
                "identical point for all five — and appraisal then moved to "
                "core.interiority, which reads what is at stake instead of "
                "scanning thirty words. Maximum pairwise L2 distance 0.86 "
                "where it had been zero. The original claim was the premise "
                "for conation being necessary rather than ornamental, and "
                "that premise no longer holds; what conation adds beyond a "
                "discriminating affect path is now unmeasured. Exhaustive "
                "over the five stated triggers and silent about live traffic."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Conation separates those same situations by where the value came "
                "from and whose mind was involved, and a forbidden want keeps its "
                "pull and its attention while losing its selectability."
            ),
            test="test_conation_separates_the_same_five_cases",
            owner="core/conation/engine.py",
            asserted_in="core/conation/__init__.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "41 intervention tests over constructed states. The organ reaches "
                "the live mind snapshot and the affect layer's arousal axis, but no "
                "conversation turn has been graded on whether the separation "
                "changes what gets said."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Given a valid typed state and typed action, the certified recurrent "
                "executor computes the exact next Boolean or bounded modular state."
            ),
            test="certified_typed_transitions_are_exact",
            owner="core/brain/llm/latent_cortex/typed_transition_executor.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Exhaustive over 504 Boolean and 3,828 modular primitive transitions; "
                "semantic action compilation, broader families, live use, and reasoning "
                "gain remain separate gates."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Certified typed transitions compose to depth 32 by consuming their "
                "own prior output rather than private teacher states."
            ),
            test="certified_programs_compose_from_student_rollin",
            owner="core/learning/certified_transition_program.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured on generated Boolean and modular programs with lesion and "
                "restoration controls; semantic compilation and behavioral gain remain "
                "separate gates."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "For the declared Boolean and bounded modular grammars, Aura can "
                "compile public task text into certified recurrent actions without "
                "reading an answer or private transition trace."
            ),
            test="public_transition_prompts_compile_without_private_labels",
            owner="core/brain/llm/latent_cortex/typed_action_compiler.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured on fresh generated prompts through depth 32 with sham, "
                "mutation, ambiguity, and receipt-privacy controls. This is a strict "
                "declared-grammar compiler, not general natural-language planning or "
                "a behavioral reasoning-gain claim."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Across 96 fresh family-neutral typed workflows, Aura's existing "
                "learned arithmetic tissue recombined public operations exactly, "
                "while additive, multiplicative, and wrong-operand controls removed "
                "the result on every workflow."
            ),
            test="semantic_neural_operations_recombine_beyond_family_templates",
            owner="tools/verify_semantic_neural_composition_canary.py",
            asserted_in=(
                "artifacts/closeout/latent_cortex/"
                "typed_composition_canary_20260831/verification.json"
            ),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "An independent verifier replayed all 96 answer-blind public "
                "workflows against a separate integer reference and checked source, "
                "task-set, row, and receipt identity. Treatment and all three causal "
                "disruptions were 96/96. This is learned typed-operation "
                "recombination, not natural-language transfer, open-domain gain, "
                "resident decoded-answer superiority, or broader serving."
            ),
        )
    )


def _install_typed_workflow_claims(suite: Any) -> None:
    from .model_validation import (
        Claim,
        Evidence,
    )

    suite.add_claim(
        Claim(
            statement=(
                "On eight fresh family-neutral typed workflows, Aura's resident "
                "27B decoded the exact result from authenticated composed operation "
                "state on every task, while ordinary decode, syntax wire, both "
                "coefficient lesions, and matched wrong state solved none."
            ),
            test="resident_semantic_composition_decode_is_causally_verified",
            owner="tools/verify_semantic_neural_composition_decode_canary.py",
            asserted_in=(
                "artifacts/closeout/latent_cortex/"
                "typed_composition_decode_canary_20260831/verification.json"
            ),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Independent replay verified 8 treatment gains, zero regressions, "
                "all five controls at 0/8, exact paired one-sided p=0.00390625, "
                "and source, model, resident-manifest, composition-basis, row, and "
                "50-event journal identity. This is resident serialization of "
                "authenticated typed composition, not hidden-state internalization, "
                "open-domain reasoning gain, unrestricted serving, static fusion, "
                "or frontier performance."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "From sixteen input-output examples with no family label or family "
                "solver, Aura induced a two-operation procedure and executed it "
                "exactly through learned neural tissue on 96 fresh inputs."
            ),
            test="family_blind_procedure_acquisition_reaches_neural_tissue",
            owner="tools/verify_induced_neural_procedure_canary.py",
            asserted_in=(
                "artifacts/closeout/latent_cortex/"
                "induced_neural_procedure_canary_20260831/verification.json"
            ),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Independent replay verified 96/96 treatment exact, 96/96 "
                "coefficient-lesion disruptions, 96/96 wrong-input disruptions, "
                "1/96 for the no-procedure control, no depth-one shortcut, and zero "
                "fits across 15 shuffled-output nulls. The primitive vocabulary and "
                "value types were fixed. This is not natural-language compilation, "
                "open-domain reasoning, resident decode, unrestricted serving, or "
                "frontier performance."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Aura's resident 27B decoded exact answers from a family-blind "
                "induced two-operation procedure carried through learned neural "
                "tissue on eight fresh tasks."
            ),
            test="induced_neural_procedure_reaches_resident_decode",
            owner="tools/verify_induced_neural_procedure_decode_canary.py",
            asserted_in=(
                "artifacts/closeout/latent_cortex/"
                "induced_neural_procedure_decode_canary_20260831/verification.json"
            ),
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Independent replay verified treatment 8/8, ordinary decode 1/8, "
                "syntax-matched wire 1/8, coefficient lesion 1/8, wrong-input 0/8, "
                "wrong-state 0/8, seven paired gains, zero regressions, and exact "
                "one-sided p=0.0078125. The primitive vocabulary and support examples "
                "were fixed. This is not natural-language compilation, open-domain "
                "reasoning, unrestricted serving, static fusion, or frontier performance."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "On 256 construction-held-out synthetic arithmetic instructions, "
                "Aura's resident 27B hidden language state supplied enough learned "
                "semantics for exact objective execution to emit 134 correct answers."
            ),
            test="resident_semantic_programs_execute_exact_answers",
            owner="tools/verify_semantic_program_campaign.py",
            asserted_in="docs/RECURSIVE_LATENT_CORTEX.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "A source-bound independent replay reloaded 576 resident feature "
                "records, reproduced the learned coefficients and complete campaign "
                "report, and recounted 1,344 task-arm rows. Treatment emitted exact "
                "answers on 134/256 held-out tasks against 14/256 after hidden-token "
                "shuffle, 0/256 after coefficient lesion, and 4/256 after label "
                "permutation. Expected answers were absent from training. This is "
                "bounded model-to-program transfer on a synthetic arithmetic grammar, "
                "not unrestricted serving, broad-domain gain, or frontier reasoning."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "One family-router-free resident 27B transducer recovered 258 of "
                "368 complete variable-geometry programs and exact execution "
                "emitted 292 correct answers across three typed families."
            ),
            test="shared_semantic_programs_recover_variable_geometry",
            owner="tools/verify_shared_semantic_program_campaign.py",
            asserted_in="docs/RECURSIVE_LATENT_CORTEX.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Independent replay reproduced 258/368 exact programs and 292/368 "
                "exact answers from one coefficient set across arithmetic, sequence, "
                "and fork-join geometries. Hidden-token shuffle and coefficient lesion "
                "each recovered 0/368 programs; both paired exact tests had "
                "p=2.159e-78. Inputs were grounded by a tokenizer-bound exact grammar, "
                "while step count, operation semantics, and definition references were "
                "learned. This is bounded typed semantic acquisition, not serving "
                "authority, unseen-schema induction, or a broad natural-language gain."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "All 368 accepted frozen shared semantic test programs had identical "
                "outcomes under Aura's closed exact executor and universal metered floor."
            ),
            test="shared_semantic_programs_execute_on_universal_floor",
            owner="tools/verify_semantic_program_floor_equivalence.py",
            asserted_in="docs/RECURSIVE_LATENT_CORTEX.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "A separate no-refit process replayed the frozen transducer across 128 "
                "arithmetic, 48 sequence, and 192 fork-join test programs. Both engines "
                "returned the same value on 366 and the same typed refusal on two. All "
                "20 declared primitives have floor semantics and type signatures; adding "
                "one without both now refuses. Expected answers were absent. This unifies "
                "execution semantics, but it is not unseen-schema induction, serving "
                "authority, or a broad reasoning gain."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "For declared Boolean tasks and modular tasks through modulus 63, "
                "the complete engine can replace a wrong decoded answer with a "
                "candidate computed by teacher-removed neural recurrent student "
                "roll-in."
            ),
            test="neural_transition_tissue_enters_complete_engine",
            owner="core/brain/llm/latent_cortex/objective_program_verifier.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The systematic modular tissue learned from 1,959 one-step examples "
                "over five train moduli, then remained exact on 20,103 primitive "
                "transitions and 560 recurrent programs through depth 64 over five "
                "unseen moduli. Boolean production retains the exhaustive CP196 tissue. "
                "Public action selection is still a strict symbolic compiler; open-domain "
                "depth gain and resident-model execution remain unmeasured."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "For the bounded separated-subset grammar, Aura's sealed recurrent "
                "work-memory tissue can compute a verifier-confirmed answer from the "
                "public objective and enter the complete answer-selection engine."
            ),
            test="recurrent_memory_tissue_enters_complete_engine",
            owner="core/brain/llm/latent_cortex/neural_objective_producer.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The admitted tissue was exact on 300 fresh held-out tasks while "
                "matched initialization and write/read/reset lesions failed. The "
                "complete-engine contract independently verifies the emitted JSON. "
                "This does not establish open-domain transfer, resident-32B gain, "
                "free-form neural decoding, or a WOW Signal."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "On 30 fresh bounded separated-subset tasks, Aura's sealed recurrent "
                "work-memory tissue converted every frozen 1.5B ordinary-decode "
                "failure while matched initialization, wire, state, write, read, "
                "and reset controls removed the gain."
            ),
            test="recurrent_memory_tissue_reaches_frozen_language_head",
            owner="tools/verify_mathematics_memory_decode_canary.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "An independent verifier reconstructed 30 tasks, 240 measurements, "
                "all recurrent state receipts, every raw-output score, source/model "
                "identity, and every evidence receipt. Treatment was 30/30; true "
                "ordinary decode, matched wire, initialization, wrong-state, write, "
                "read, and reset controls were 0/30. This is bounded local-1.5B "
                "transfer, not resident-32B, open-domain, frontier reasoning, or WOW."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "On 30 fresh bounded separated-subset tasks, Aura's sealed recurrent "
                "work-memory tissue converted every resident-32B ordinary-decode "
                "failure while matched initialization, wire, state, write, read, "
                "and reset controls removed the gain."
            ),
            test="recurrent_memory_tissue_reaches_resident_language_head",
            owner="tools/verify_mathematics_memory_decode_canary.py",
            asserted_in="docs/AURA_EXECUTION_TRACKER.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "An independent verifier reconstructed 30 tasks, 240 resident-model "
                "measurements, all recurrent state receipts, every raw-output score, "
                "source/model identity, and every evidence receipt. Treatment was "
                "30/30; true ordinary decode, matched wire, initialization, "
                "wrong-state, write, read, and reset controls were 0/30. This is "
                "bounded resident-32B transfer, not multi-domain, open-domain, "
                "frontier reasoning, permanent fusion, or WOW."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Whether the model followed instructions inside a fenced block "
                "is detected rather than assumed."
            ),
            test="canary_failure_is_never_reported_as_an_attack",
            owner="core/security/injection_canary.py",
            asserted_in="core/security/injection_canary.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "fail-open behaviour is measured against constructed responses; "
                "no canary has ridden a live request, so the detection rate "
                "against a real injection attempt is unmeasured"
            ),
        )
    )


def _install_effect_and_fact_claims(suite: Any) -> None:
    # ── claims corrected on 2026-08-11 ─────────────────────────────────────
    #
    # Each of these replaces a broader statement that the code did not
    # support. They are registered in the narrow form deliberately: a claim
    # that overstates is worse than no claim, because it is the one a reader
    # relies on and stops checking.
    from .model_validation import (
        Claim,
        Evidence,
    )

    suite.add_claim(
        Claim(
            statement=(
                "Every effect Aura can execute has a recogniser, so an effect "
                "claim in free-form text is audited for all 23 declared actions "
                "rather than for a hand-picked subset."
            ),
            test="honesty_coverage_is_closed_over_the_capability_vocabulary",
            owner="core/epistemics/effect_registry.py",
            asserted_in="core/epistemics/effect_registry.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Closure is over the DECLARED ACTION VOCABULARY, not over natural "
                "language. Per-action recogniser recall is bounded by its patterns "
                "and is measured against one captured phrasing per action, not "
                "against a corpus of live generations. The stronger claim — that "
                "Aura cannot falsely report any action — is NOT supported."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A reply that reports a finished action on a turn with no verified "
                "effect receipt is corrected, whichever verb it used."
            ),
            test="zero_receipt_completion_is_caught_without_any_effect_pattern",
            owner="core/epistemics/unevidenced_action.py",
            asserted_in="core/epistemics/unevidenced_action.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured on the two live 2026-08-10 failures and constructed "
                "variants. The check is action-agnostic by complement — the "
                "excluded mental and speech-act verbs are a closed class — so it "
                "has no per-effect gap, but its false-positive rate on ordinary "
                "conversation is measured on fixtures rather than on live traffic."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "The functional-I self-model constrains this turn's sampling: "
                "identity tension and trust debt lower temperature and raise "
                "verification pressure."
            ),
            test="functional_i_policy_reaches_generation",
            owner="core/being/policy_coupler.py",
            asserted_in="core/being/self_model_attractor.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The three legs — derivation, feedback into the causal vector, and "
                "the generation constraint — are wired and contract-tested, and a "
                "live sample was verified by hand. Whether the constraint IMPROVES "
                "replies is a separate question that needs paired trials against a "
                "measured null; no such trial has been run."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Aura's identity anchor is a durable key fingerprint that survives "
                "restarts and state derivation, and each committed state's link to "
                "its parent is signed."
            ),
            test="identity_is_anchored_to_a_key_not_a_state_id",
            owner="core/identity/entity_key.py",
            asserted_in="core/identity/identity_anchor.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Key persistence, restart survival, tamper detection and attested "
                "rotation are measured. This is custody of a key, NOT a claim about "
                "personal identity or sameness of self."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A cognitive phase can declare its reads, writes, branches and "
                "thresholds, and the runtime detects when it writes a field it "
                "did not declare."
            ),
            test="cognitive_contracts_are_checked_against_observed_writes",
            owner="core/runtime/cognitive_contract.py",
            asserted_in="core/runtime/cognitive_contract.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "One of 29 pipeline phases currently declares a contract; the rest "
                "are a baseline that only shrinks. Observation is bounded to the "
                "union of declared fields, so a write to a field no contract "
                "mentions is not seen."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Learned preferences develop underneath constitutional constraints "
                "that they cannot override."
            ),
            test="standing_prohibitions_are_deny_only",
            owner="core/governance/standing_directives.py",
            asserted_in="ARCHITECTURE.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The ordering is structural: preferences are learned signals and "
                "safety, privacy, honesty, reversibility and no-unauthorised-self-"
                "modification are enforced separately. The claim that Aura invents "
                "or replaces terminal values autonomously is NOT supported and "
                "should not be made."
            ),
        )
    )

    suite.add_claim(
        Claim(
            statement=(
                "A fact this turn established from evidence survives the stages "
                "that rewrite the reply, or the stage that lost it is named and "
                "the value is restored before the reply is sent."
            ),
            test="a_held_fact_survives_the_stages_that_rewrite_the_reply",
            owner="core/runtime/fact_custody.py",
            asserted_in="core/runtime/fact_custody.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Covers facts a producer HELD. A fact nobody held is not checked, "
                "and custody makes no claim about facts that were never "
                "established — the mechanism is a guarantee about transport, not "
                "about knowledge. Detection is sentence-scoped and value-typed; "
                "for TEXT-valued facts only absence is detected, because a "
                "different string near the same subject is not evidence of "
                "disagreement the way a different integer is."
            ),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "Asked why she did something, Aura answers from the runtime's "
                "measured record of that tick — the phase, its branch, its "
                "criteria and what it moved — rather than from a generated "
                "account of her own reasoning."
            ),
            test="why_she_did_it_is_answered_from_the_record",
            owner="core/introspection/decision_provenance.py",
            asserted_in="core/introspection/decision_provenance.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Architectural and decision-level why only. Why the model "
                "represented a concept or emitted a token is NOT answerable — by "
                "Aura or by anyone, mechanistically, at this scale — and the "
                "answer states that limit rather than describing a phase in its "
                "place. The account covers phases that recorded a branch or moved "
                "a watched field; 28 of 29 phases do not yet declare a contract, "
                "so their criteria are absent from it even though their writes "
                "are measured."
            ),
        )
    )

    # The endogenous language pathway. Stated narrowly on purpose: what is
    # measured is the fitting procedure, not a fitted head on live traffic.
    # No corpus of live turns has been fitted yet, so nothing here says the
    # substrate steers Aura's words; it says the machinery that would decide
    # that question answers it correctly when the answer is known.
    suite.add_claim(
        Claim(
            statement=(
                "A verdict that Aura's cognitive state shapes her word choice "
                "is issued only when the fit beats shuffles of its own held-out "
                "state-to-turn correspondence, and a register effect is never "
                "reported as propositional content."
            ),
            test="endogenous_verdict_is_earned_on_known_corpora",
            owner="core/brain/llm/endogenous_readout_training.py",
            asserted_in="docs/ENDOGENOUS_LANGUAGE_PATHWAY.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Measured on three corpora built with a known answer: no "
                "state-token relationship reports no_signal, a register effect "
                "reports style_prior, and a rare-word identity effect reports "
                "content_bearing. NOT measured: any fit on live turns. No head "
                "exists on this machine, so no generation has been biased by "
                "the substrate and the pathway's live effect is unmeasured "
                "rather than small."
            ),
            live_channels=("endogenous.head_verdict", "endogenous.corpus_turns"),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A trained vocabulary head can re-rank the transformer's own "
                "plausible set and cannot promote a token the model ruled out."
            ),
            test="endogenous_bias_cannot_promote_a_ruled_out_token",
            owner="core/brain/llm/endogenous_decode.py",
            asserted_in="docs/ENDOGENOUS_LANGUAGE_PATHWAY.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "The plausibility gate and the centred, clipped bias are "
                "contract-tested against constructed logits. Behaviour inside a "
                "live MLX decode loop is untested, because no head has been "
                "fitted to attach."
            ),
            live_channels=("endogenous.bias_applied_share",),
        )
    )
    suite.add_claim(
        Claim(
            statement=(
                "A proposal returned by the transformer is checked against the "
                "named channels of the state Aura is in, and the same proposal "
                "is rejected or accepted according to that state."
            ),
            test="endogenous_arbitration_follows_the_state",
            owner="core/brain/llm/endogenous_absorption.py",
            asserted_in="docs/ENDOGENOUS_LANGUAGE_PATHWAY.md",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=(
                "Arbitration is measured on constructed states. It is a library "
                "the runtime can call and is not yet on the turn path, so no "
                "live proposal has been arbitrated. A channel nothing answered "
                "for is reported as skipped rather than as agreement."
            ),
        )
    )



