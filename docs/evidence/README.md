# docs/evidence — historical proofs, closeouts, and research narrative

> **Historical record — 2026-07-09.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](../DOC_STATUS.md).

The repo root is for what an arriving engineer needs **now**: how to install,
how it works, what is claimed, what is tested, who owns what. Everything here
is the **record** — point-in-time audits, closeout reports, challenge specs,
and research whitepapers that back the claims but are not the front door.

Nothing in this directory is load-bearing for the runtime. If a document here
starts mattering to how the system runs, it belongs somewhere else.

| Document | What it is |
|---|---|
| `2026-08-30-foreground-completion-ownership.md` | Foreground Completion Ownership |
| `AUDIT_SPEC_BEING_CLOSED_LOOP_V3.md` | Audit specification for the closed-loop being review |
| `CHALLENGE.md` | Challenge/evaluation framing document |
| `CLOSEOUT.md` | Project closeout narrative |
| `CRITIQUE_CLOSURE.md` | Responses to the July external critique |
| `EVALUATION_REPORT.md` | Point-in-time evaluation results |
| `G01_RLC_BASELINE_2026-09-08.md` | G01: Frozen RLC baseline and claim boundary |
| `G13_RLC_PUBLIC_CLAIMS_2026-09-08.md` | G13 public RLC claim reconciliation |
| `INTERIORITY_ABLATION.md` | Interiority: what each faculty moves when you take it away |
| `INTERIORITY_COUNCIL_DOCKET.md` | The council's proposals, item by item |
| `MAIN15_AUDIT_NOTES.md` | Audit working notes |
| `PHENOMENAL_SUBSTRATE_INTEGRATION.md` | Substrate-integration research narrative |
| `R01_RUNTIME_SURVIVAL_2026-09-06.md` | R01 Runtime Survival |
| `R02_SUCCESSOR_IDENTITY_2026-09-06.md` | R02 Successor Identity |
| `R03_R04_PROGRESS_2026-09-06.md` | R03 and R04: open live checks |
| `R06_LOGGING_HANDLER_DEADLOCK_2026-09-07.md` | The wedge was two log handlers waiting on each other |
| `R06_REPLY_CATALOG_AND_DIAGNOSTIC_SNAPSHOT_2026-09-09.md` | R06 reply catalog and diagnostic snapshot |
| `R07_HEALTH_AUTHORITY_2026-09-08.md` | R07: one health authority across transports |
| `R08_DEFERRED_WRITE_CUSTODY_2026-09-09.md` | R08: A retry must not duplicate its pending write |
| `R08_WATCHDOG_LIFECYCLE_2026-09-09.md` | R08: Keep the observer alive while the observed loop recovers |
| `Q01_MODEL_INVENTORY_2026-09-13.md` | Q01 model inventory, 2026-09-13 |
| `Q02_DISK_RETENTION_2026-09-13.md` | Q02 disk retention, 2026-09-13 |
| `Q05_PERSISTENCE_BACKUP_ROLLBACK_2026-09-13.md` | Q05 persistence, migration, corruption recovery, backups, rollback |
| `G02_RLC_RECONCILIATION_2026-09-12.md` | G02: RLC evidence and activation reconciliation |
| `G03_ARGUMENT_POINTER_DEVELOPMENT_2026-09-12.md` | G03 Argument Pointer Development |
| `G03_CONDITIONAL_GRAPH_DEVELOPMENT_2026-09-12.md` | G03 conditional graph development |
| `G03_CONDITIONAL_GRAPH_VERIFICATION_2026-09-12.md` | G03 conditional graph verification |
| `G03_DEFINITION_SUPERVISION_2026-09-13.md` | G03 Definition Supervision |
| `G03_FORK_DEFINITION_REFIT_2026-09-13.md` | G03 Fork Definition Refit |
| `G03_FULL_SOURCE_NULL_2026-09-12.md` | G03 full-source proposal repair: null result |
| `G03_GLOBAL_ARGUMENT_SEARCH_2026-09-12.md` | G03 global argument search development |
| `G03_JOINT_DEFINITIONS_2026-09-13.md` | G03 joint definition selection |
| `G03_PREFIX_SEARCH_DEVELOPMENT_2026-09-12.md` | G03 development: prefix-feasible argument search |
| `G03_PROGRAM_SELECTION_2026-09-12.md` | G03 autonomous program-level development selection |
| `G03_PROPOSAL_REFIT_2026-09-12.md` | G03 source-only proposal refit |
| `G03_SHARED_POINTER_REPAIR_2026-09-12.md` | G03 shared pointer repair |
| `G03_SOURCE_COVERAGE_2026-09-12.md` | G03 exact source recovery |
| `R09_DELIVERY_REPLAY_2026-09-12.md` | R09 delivery replay, September 12 |
| `R09_ACCEPTANCE_MATRIX_2026-09-10.md` | R09 acceptance matrix |
| `R09_BOUND_LATENT_STOP_2026-09-09.md` | R09: Bound latent Stop and live prefill interruption |
| `R09_CANCELLATION_SHELL_CUSTODY_2026-09-08.md` | R09 cancellation, shell identity, and answer custody |
| `R09_CHRONOLOGICAL_HISTORY_2026-09-10.md` | R09 chronological history and display capacity |
| `R09_CONTINUATION_HISTORY_2026-09-10.md` | R09 Continuation History |
| `R09_CONVERSATION_CAPACITY_AND_EPISODIC_RECALL_2026-09-09.md` | R09 Conversation Capacity and Episodic Recall |
| `R09_DIALOGUE_BUDGET_CUSTODY_2026-09-09.md` | R09 dialogue budget custody |
| `R09_DURABLE_HISTORY_RECOVERY_2026-09-09.md` | R09: terminal history survives the delivery boundary |
| `R09_DURABLE_WINDOW_BOOTSTRAP_2026-09-09.md` | R09 durable window bootstrap |
| `R09_FALLBACK_TRANSCRIPT_2026-09-10.md` | R09: dialogue survives a model handoff |
| `R09_FOLLOWUP_SOURCE_HISTORY_2026-09-09.md` | R09: Preserve the source of a follow-up |
| `R09_HISTORY_ADMISSION_AND_R08_DEFERRALS_2026-09-09.md` | History admission and tool deferrals |
| `R09_LIST_FOOTER_DELIVERY_2026-09-08.md` | R09: List footer delivery repair |
| `R09_LIVE_DELIVERY_2026-09-11.md` | R09 live delivery replay |
| `R09_PENDING_WINDOW_RECONCILIATION_2026-09-09.md` | R09 pending-window reconciliation |
| `R09_PREFILL_STOP_2026-09-09.md` | R09: stop during prefill |
| `R09_PROVENANCE_IS_EVIDENCE_2026-09-09.md` | R09 provenance is evidence, not answer ownership |
| `R09_QUOTED_RECALL_2026-09-10.md` | R09 Quoted Recall |
| `R09_SERVICE_AND_DESKTOP_LIFETIMES_2026-09-09.md` | R09 service observation and desktop lifetime |
| `R09_TERMINAL_HISTORY_AND_WORKER_STOP_2026-09-08.md` | R09 terminal history and worker cancellation |
| `R10_SEMANTIC_EXECUTABLE_EXAMPLES_2026-09-08.md` | R10 semantic executable examples |
| `README_BEING_CLOSED_LOOP_V3.md` | Companion readme for that audit |
| `R_LEDGER_CONTINUATION_2026-09-07.md` | Runtime ledger continuation |
| `R_LIVE_REPLAY_2026-09-07.md` | Runtime replay, September 7 |
| `SUBJECT_EFFECT_OWNERSHIP_2026-09-08.md` | Subject evidence effect ownership repair |
| `WHITEPAPER_CONSCIOUSNESS_EXPANSION.md` | Research whitepaper (interior detail, not the banner) |
| `aura_deep_qa_report.md` | Deep QA report |

## Generality and runtime evidence, 13-18 September 2026

Written while the G-series and the runtime watch were running. Each row's
description is that document's own title.

| Document | What it records |
| --- | --- |
| `G03_ALIGNED_FIT_RECOVERY_2026-09-16.md` | G03: recover the saved aligned fit without training again |
| `G03_ARGUMENT_RANKING_2026-09-13.md` | Source argument ranking development |
| `G03_ARITY_STATE_SEARCH_2026-09-13.md` | Arity-state search correction |
| `G03_ATOMIC_LITERAL_ARGUMENTS_2026-09-14.md` | G03: Literal atoms in the argument chart |
| `G03_ATOMIC_LITERAL_RESULT_2026-09-14.md` | G03: complete atomic-literal development result |
| `G03_CAPACITY_AND_SEARCH_2026-09-15.md` | Frozen score capacity and complete operation search |
| `G03_CHART_FEATURE_REUSE_2026-09-15.md` | Decode-local chart features |
| `G03_CONSTRAINED_FRESH_RESULT_2026-09-15.md` | Fresh-source retained-constraint result |
| `G03_COUNTERFACTUAL_CORPUS_2026-09-15.md` | Counterfactual source training |
| `G03_DEFINITION_ATTACHMENT_2026-09-13.md` | Definition ownership development |
| `G03_DURABLE_BATCHED_FIT_2026-09-17.md` | G03: preserve accepted updates and share repeated evidence work |
| `G03_FRESH_SOURCE_FIT_2026-09-15.md` | Fresh source fit |
| `G03_GRAPH_FACTOR_RESULT_2026-09-15.md` | Source graph-factor calibration |
| `G03_GRAPH_RELATION_CANARY_2026-09-15.md` | G03 complete-graph relation learning canary |
| `G03_GRAPH_RELATION_RESULT_2026-09-15.md` | Full-cohort graph relation refit |
| `G03_INPUT_COORDINATE_REPAIR_2026-09-16.md` | G03 input-coordinate repair |
| `G03_JOINT_GRAPH_TRAINER_2026-09-15.md` | Joint operation and relation training |
| `G03_JOINT_SCORE_TRIAL_2026-09-13.md` | Joint operation and argument scoring: development trial |
| `G03_LEARNED_PROCEDURE_REUSE_2026-09-15.md` | Learned procedure reuse measurement |
| `G03_LITERAL_RETENTION_2026-09-13.md` | Literal retention and failure attribution, 2026-09-13 |
| `G03_OPERATION_BACKGROUND_2026-09-18.md` | Operation/background competition: rejected candidate |
| `G03_WORKING_FACE_FULL_RUN_2026-09-18.md` | G03: full paired working-face result |
| `G03_OPERATION_BOUNDARY_LEARNING_2026-09-18.md` | Operation boundary learning, 2026-09-18 |
| `G03_OPERATION_FEASIBILITY_2026-09-13.md` | Operation-chart feasibility development |
| `G03_OPERATION_LABEL_ALTERNATIVES_2026-09-14.md` | G03: retaining operation label alternatives |
| `G03_OPERATION_VIEWS_2026-09-13.md` | G03 operation views and register-link diagnosis |
| `G03_OVERLAP_DOMINANCE_2026-09-14.md` | Overlap-complete argument development |
| `G03_PROCEDURE_DOMAIN_CONTRACT_2026-09-15.md` | Reusable procedure domain contract |
| `G03_RANKED_OPERATION_POINTER_2026-09-14.md` | G03: reject the source-grouped operation ranker |
| `G03_REUSABLE_PROCEDURE_RESULT_2026-09-15.md` | Corrected full-cohort procedure reuse |
| `G03_RUNTIME_ARGUMENT_VIEWS_2026-09-14.md` | G03: runtime operation boundaries are a measured null |
| `G03_RUNTIME_MARGIN_RESULT_2026-09-14.md` | Full-mention pointer-margin result |
| `G03_RUNTIME_RETENTION_2026-09-17.md` | Runtime graph retention and small-trial result |
| `G03_SEMANTIC_COUNTEREXAMPLES_2026-09-15.md` | G03 semantic counterexample admission |
| `G03_SINGLE_LOAD_REACQUISITION_2026-09-15.md` | One-load source reacquisition |
| `G03_SOURCE_ANCHORED_SCORING_2026-09-17.md` | Source-anchored program scoring |
| `G03_SOURCE_OPERATION_RETENTION_2026-09-15.md` | Source operation retention |
| `G03_SOURCE_RETENTION_RESULT_2026-09-15.md` | Source-operation retention did not qualify |
| `G03_TYPED_SEARCH_2026-09-17.md` | Typed operation search development |
| `G05_RESIDENT_SHAPE_CANARY_2026-09-14.md` | G05: resident public-shape canary |
| `G05_RESIDENT_SHAPE_RESULT_2026-09-14.md` | G05: resident shape result |
| `G05_RUNTIME_TOKENIZER_CONTRACT_2026-09-14.md` | G05: use the runtime tokenizer's declared vocabulary |
| `G05_TYPED_DECODER_STATE_2026-09-14.md` | G05: preserve the decoder's actual output boundary |
| `G06_COMPOSITION_PUBLIC_DIAGNOSTIC_2026-09-14.md` | Public composition diagnostic |
| `G06_NATIVE_CODING_DIAGNOSTIC_2026-09-14.md` | Native-thinking coding diagnostic |
| `G06_PUBLIC_CHANNEL_DIAGNOSTIC_2026-09-14.md` | Public-channel measurement repair |
| `G07_PREREGISTRATION_INTEGRITY_2026-09-14.md` | G07 preregistration integrity |
| `G07_PROSPECTIVE_TASK_POWER_2026-09-15.md` | Prospective paired replication |
| `G09_ACTION_COMPLETION_EVIDENCE_2026-09-14.md` | Action completion and checked outcomes |
| `G09_ACTION_CONDITIONED_WORLD_2026-09-17.md` | Shared action-conditioned world rollout |
| `G09_COMPUTED_VALUE_PLANS_2026-09-14.md` | G09: plan for computed values, then observe them |
| `G09_CONCLUSIVE_VERIFICATION_2026-09-14.md` | G09: conclusive verification follows the measured verdict |
| `G09_EVIDENCE_IDENTITY_2026-09-16.md` | G09: retain the identity of learning evidence |
| `G09_EVIDENCE_REFRESH_2026-09-14.md` | G09: retain outcomes that arrive during value refresh |
| `G09_FORECAST_ATTRIBUTION_2026-09-18.md` | Planning forecast attribution, 2026-09-18 |
| `G09_GOAL_BOUND_PROCEDURES_2026-09-13.md` | Goal-bound common procedures |
| `G09_INDEPENDENT_OUTCOME_COUNTS_2026-09-14.md` | G09: count observations, not pending retries |
| `G09_KNOWLEDGE_REVISION_2026-09-15.md` | Knowledge revision through the existing graph |
| `G09_MISSION_OBSERVATION_RECOVERY_2026-09-14.md` | Mission Effects and Delayed Observation |
| `G09_OUTCOME_CONTRACT_2026-09-16.md` | Learning after an outcome rule changes |
| `G09_OUTCOME_LEARNING_EVIDENCE_2026-09-14.md` | Outcome learning evidence |
| `G09_PLAN_ALTERNATIVES_2026-09-14.md` | G09: Search alternatives before learning which plan works |
| `G09_PROCEDURE_DATAFLOW_2026-09-13.md` | Procedure dataflow in the cognitive event graph |
| `G09_SHARED_PROCEDURE_EXECUTION_2026-09-13.md` | Shared procedure execution, 2026-09-13 |
| `G09_TASK_PLAN_FEEDBACK_2026-09-14.md` | G09: measured task outcomes can change procedure selection |
| `G10_CAMPAIGN_PROGRESS_2026-09-14.md` | G10: interrupted campaigns retain completed samples |
| `G10_CURRENT_CHANNEL_2026-09-14.md` | G10: Current-generation channel measurement |
| `G10_FUSION_INTERVENTION_OWNERSHIP_2026-09-14.md` | G10: isolate fusion interventions from live state |
| `G10_OWNED_REENTRY_AND_CALIBRATION_2026-09-14.md` | G10: public calibration and owned lock recursion |
| `G10_PROBE_CHECKPOINT_IDENTITY_2026-09-14.md` | G10: load the checkpoint the probe certifies |
| `G10_PUBLIC_STEERING_MEASUREMENT_2026-09-14.md` | G10: measure steering on completed public answers |
| `G10_PUBLIC_STEERING_RESULT_2026-09-14.md` | Complete public steering comparison: negative |
| `G10_STEERING_BASIS_AND_OWNERSHIP_2026-09-14.md` | G10: steering evidence follows the loaded basis |
| `G11_DESKTOP_SERVING_2026-09-14.md` | G11: qualified desktop serving |
| `G11_MANIFEST_CONTINUITY_2026-09-14.md` | G11: component-scoped manifest continuity |
| `BRAINSTEM_LANE_COULD_NOT_LOAD_2026-09-19.md` | The brainstem lane could not load, whichever model was bound to it |
| `G03_BACKGROUND_SCORE_REPLAY_2026-09-18.md` | Background score replay |
| `G03_BACKGROUND_TRAINER_INTEGRATION_2026-09-18.md` | Background scoring reaches joint training |
| `G03_BINDING_FACE_RETENTION_2026-09-18.md` | Binding faces and retained margins |
| `G03_BINDING_FACE_TRIAL_2026-09-18.md` | Binding-face source trial and unchanged-round termination |
| `G03_CURVED_RETENTION_2026-09-18.md` | Curved retention boundaries and competing feasible steps |
| `G03_EXACT_SCALE_CONFLICT_2026-09-18.md` | Exact graph-scale conflict |
| `G03_RELATION_CAPACITY_EXPANSION_2026-09-18.md` | Opt-in relation capacity expansion |
| `G03_WORKING_FACE_FULL_LAUNCH_2026-09-18.md` | Frozen working-face development run |
| `G08_PAIRED_PUBLIC_MEASUREMENT_2026-09-18.md` | Paired public-answer measurement |
| `G09_ORDINARY_EVALUATION_PATH_2026-09-18.md` | Evaluation requests use ordinary reasoning |
| `G03_FUNCTION_COORDINATES_2026-09-19.md` | Function-coordinate development |
| `G03_MINIMUM_CHANGE_TRIAL_2026-09-19.md` | Minimum-change semantic repair: numerical fix, transfer still negative |
| `G03_PARAMETER_SUBSPACE_2026-09-19.md` | A source repair without the measured classifier regression |
| `G03_SCALED_SUBSPACE_TRIAL_2026-09-19.md` | Scaled subspace trial |
| `G03_WORKING_FACE_ATTRIBUTION_2026-09-19.md` | Working-face attribution correction |
| `G04_BOUND_SOURCE_INVENTORY_2026-09-19.md` | Bound source inventory |
| `G08_VALIDATION_SOURCE_IDENTITY_2026-09-19.md` | Validation cache source identity |
| `G_RUNTIME_WATCH_AND_RESIDUAL_2026-09-16.md` | Runtime recovery and residual sampling |
| `TERNARY_BONSAI_2_27B_2026-09-17.md` | Ternary Bonsai 2 27B on this host, 2026-09-17 |
| `G03_EXPANDED_FUNCTION_TRIAL_2026-09-19.md` | Expanded function-coordinate trial |
| `G03_CANDIDATE_BANK_DIAGNOSIS_2026-09-20.md` | Candidate reachability and selection diagnosis |
| `G03_COHORT_REPAIR_PROTOCOL_2026-09-20.md` | Cohort diagnosis before further training |
| `G03_CONDITIONAL_PILOT_RESULT_2026-09-20.md` | Conditional graph learning pilot |
| `G03_CONDITIONAL_SEARCH_BOUND_2026-09-20.md` | Conditional fit search interruption |
| `G03_CONDITIONAL_SELECTION_2026-09-20.md` | Conditional semantic selection |
| `G03_CONTRAST_CONSTRUCTION_REPLAY_2026-09-20.md` | Construct training margins with the fitter's replay |
| `G03_EARLY_JOINT_RESTORATION_2026-09-20.md` | Restore omitted curved faces before expanding the working set |
| `G03_FIT_CAPTURE_STORAGE_2026-09-20.md` | Deduplicate fit evidence storage |
| `G03_ITERATIVE_POLICY_RESULT_2026-09-20.md` | Re-mining the actual decoder's competitor |
| `G03_JOINT_CURVATURE_RESTORATION_2026-09-20.md` | Joint restoration of retained graph margins |
| `G03_MARGIN_ACCUMULATION_2026-09-20.md` | Complete-graph margin accumulation |
| `G03_OPERATION_RETENTION_POLICY_2026-09-20.md` | Operation retention policy |
| `G03_OPERATION_RETENTION_RESULT_2026-09-20.md` | Operation retention result |
| `G03_PARTIAL_TARGET_PROGRESS_2026-09-20.md` | An unreachable target does not stop independent learning |
| `G03_POLICY_ALIGNED_LEARNING_2026-09-20.md` | Training the decoder's actual selection policy |
| `G03_POLICY_COEFFICIENT_CONTROLS_2026-09-20.md` | G03 policy and coefficient controls |
| `G03_REPLAYABLE_FIT_INPUTS_2026-09-20.md` | Preserve the numerical problem before the first update |
| `G03_RESTORATION_WORKING_SET_2026-09-20.md` | Keep repaired faces in the restoration problem |
| `G03_RESTORED_FULL_SOURCE_2026-09-20.md` | Full restored-source replay |
| `G03_RESTORED_LATENT_DECODE_2026-09-20.md` | Restored latent decode |
| `G03_ROUND_CANDIDATES_2026-09-20.md` | Decode each accepted training round |
| `G03_SHARED_CHOICE_EVIDENCE_2026-09-20.md` | Shared conditional evidence |
| `G03_SOURCE_ERROR_ACQUISITION_2026-09-20.md` | Source-wide error acquisition |
| `G09_CLOSED_PROCEDURE_TYPES_2026-09-20.md` | Closed structural procedure execution |
| `G09_FEEDBACK_EXECUTION_TYPES_2026-09-20.md` | Structural checks survive procedure feedback |
| `G_LEDGER_REFERENCE_REVIEW_2026-09-20.md` | Verified reasoning reference review |
| `G03_ADDITIONAL_CONTINUATION_REVIEW_2026-09-21.md` | Additional continuation report: checked against current code |
| `G03_ARCHITECTURE_RESET_2026-09-21.md` | G03 Architecture Reset |
| `G03_BOUNDARY_INTERVENTION_2026-09-21.md` | Exposed failure boundary intervention |
| `G03_COMPLETE_LITERAL_AND_PAIRED_BOUNDARY_RESULT_2026-09-21.md` | Completed Literal and Paired Boundary Results |
| `G03_CONTEXT_AND_PORTFOLIO_2026-09-21.md` | G03: request context, constrained decoding, and retained alternatives |
| `G03_DIAGNOSTIC_SEARCH_COST_2026-09-21.md` | Diagnose Reachability Without Enumerating Past Its Witness |
| `G03_EXACT_SEARCH_STATE_REUSE_2026-09-21.md` | Reuse operation feasibility states |
| `G03_EXISTING_CHANNEL_RECOGNITION_2026-09-21.md` | Recognition evidence already present in the resident sequence |
| `G03_EXPANDED_SEARCH_REGRESSIONS_2026-09-21.md` | Expanded search regressions |
| `G03_FACTORED_BOUNDARY_MEANING_2026-09-21.md` | Separate boundary evidence from operation meaning |
| `G03_FACTORIZED_RECOGNITION_RESULT_2026-09-21.md` | Factorized recognition: measured result |
| `G03_FIXED_BANK_REPLAY_2026-09-21.md` | Fixed-bank semantic replay |
| `G03_FULL_COHORT_ATTRIBUTION_2026-09-21.md` | Full development cohort attribution |
| `G03_FULL_CONTEXT_AND_COORDINATES_2026-09-21.md` | Full-source context and common input coordinates |
| `G03_FULL_SOURCE_DECISION_RESULT_2026-09-21.md` | Complete source-decision audit |
| `G03_JOINT_TRANSITION_PILOT_2026-09-21.md` | Joint transition-feature and span-set learning |
| `G03_LABELED_SPAN_OBJECTIVE_2026-09-21.md` | Joint labeled span-set experiment |
| `G03_LITERAL_GRAMMAR_IDENTITY_2026-09-21.md` | Literal grammar aliases retain their input identity |
| `G03_OBSERVATION_IDENTITY_AND_FEATURE_OVERLAP_2026-09-21.md` | Observation identity and operation-feature overlap |
| `G03_OPERATION_NEIGHBORHOOD_2026-09-21.md` | Source-only neighborhood diagnostic |
| `G03_PAIRED_SPAN_SET_OBJECTIVE_2026-09-21.md` | Learn paired boundaries in the complete span-set objective |
| `G03_PREFIX_DIFFERENCED_RECOGNITION_2026-09-21.md` | Prefix-differenced operation recognition |
| `G03_REQUEST_CONTEXT_PILOT_2026-09-21.md` | Complete-request operation evidence: development pilot |
| `G03_RUNTIME_RECOGNITION_OBJECTIVE_2026-09-21.md` | Recognition evidence overruled by binding scores |
| `G03_SECOND_REVIEW_DISPOSITION_2026-09-21.md` | Second advisory batch: source checks and counterexamples |
| `G03_SOURCE_COMPETITOR_PROBE_2026-09-21.md` | Source-selected competitor probe |
| `G03_SOURCE_DECISION_PILOT_2026-09-21.md` | Source decision retention pilot |
| `G03_SPAN_FIT_RECOVERY_2026-09-21.md` | G03 Span Fit Recovery |
| `G03_SPAN_SET_OBJECTIVE_2026-09-21.md` | Complete span-set boundary learning |
| `G09_BOUNDED_REACH_CRITERION_2026-09-21.md` | Bounded reach and developmental gain |
| `G09_GROUNDED_OPERATOR_INVENTION_2026-09-21.md` | Grounded Operator Invention |
| `G09_OPERATOR_CAUSAL_INVENTORY_2026-09-21.md` | Operator interventions and trial restoration |
| `G09_OPERATOR_RETENTION_2026-09-21.md` | Retained operator semantics |
| `G09_SHARED_SEQUENCE_REACH_2026-09-21.md` | Shared retained-sequence reach |
| `G11_REQUALIFICATION_CONTINUITY_2026-09-21.md` | Qualified package continuity and replay |
| `G_LEDGER_SCREENSHOT_REVIEW_2026-09-21.md` | September 21 screenshot review |
| `G03_CONDITION_ABLATION_2026-09-22.md` | Condition ablation in shared procedural learning |
| `G03_DIRECT_MIXED_GAP_ATTRIBUTION_2026-09-22.md` | Direct and mixed method attribution on the exposed G03 gap |
| `G03_INQUIRY_AND_SHARED_RULES_2026-09-22.md` | Discriminating inquiry and shared rule evidence |
| `G03_INQUIRY_FEEDBACK_2026-09-22.md` | Observed inquiry feedback reaches program selection |
| `G03_INQUIRY_RETENTION_2026-09-22.md` | Retaining unresolved program distinctions |
| `G03_METHOD_OVERLAP_AND_ADVISORY_REVIEW_2026-09-22.md` | G03 paired methods and advisory review |
| `G03_MIXED_CANDIDATE_BANK_2026-09-22.md` | Mixed candidate bank on the exposed development misses |
| `G03_MIXED_METHOD_PROPOSALS_2026-09-22.md` | Mixed semantic proposals and durable feedback |
| `G03_OPERATION_EVIDENCE_PROBE_2026-09-22.md` | Source-trained operation evidence probe |
| `G03_OPERATION_SPAN_VIEW_DIAGNOSIS_2026-09-22.md` | G03 operation-span view diagnosis |
| `G03_PORTFOLIO_GAP_ATTRIBUTION_2026-09-22.md` | Source-verified portfolio gap attribution |
| `G03_RELATIONAL_AND_DIRECT_DECODER_2026-09-22.md` | Relational evidence and direct decoding |
| `G03_ROLE_ALIAS_COUNTERFACTUAL_2026-09-22.md` | G03 role-alias counterfactual |
| `G03_ROLE_ALIAS_FIT_NEGATIVE_2026-09-22.md` | G03 role-alias fit: no gain |
| `G03_SOURCE_TRAINED_CANDIDATE_SELECTOR_2026-09-22.md` | Source-trained selection against a retained program bank |
| `G03_SUGGESTIONS2_DISPOSITION_2026-09-22.md` | G03 Suggestions2 review |
| `G_HUMAN_RELATIONAL_CRITERIA_2026-09-22.md` | Human relational criteria for the G ledger |
| `G03_ARCHITECTURE_REUSE_2026-09-23.md` | G03 architecture reuse before another fit |
| `G03_PDF_ARCHITECTURE_DISPOSITION_2026-09-23.md` | G03 external architecture proposals: disposition before broad evaluation |
| `G03_SEARCH_INTERRUPTION_AND_SOURCE_RANKER_2026-09-23.md` | G03 search interruption and source-trained ranker, 2026-09-23 |
| `G03_SEMANTIC_BELIEF_REVIEW_2026-09-23.md` | G03 semantic belief proposal review, 2026-09-23 |
| `G03_CAUSAL_MEANING_AND_EPISTEMIC_CONTROL_2026-09-24.md` | G03: Causal meaning and epistemic control |
| `G03_DIRECT_BEAM_ALTERNATIVES_2026-09-24.md` | G03: target-blind direct-program alternatives |
| `G03_PROJECTED_TRIADIC_BINDING_2026-09-24.md` | G03: projected triadic binding remains diagnostic |
| `G03_REPRESENTATION_ONLY_CONTROL_2026-09-24.md` | G03 representation-only binding control, 2026-09-24 |
| `G03_TRIADIC_FULL_FOLD_ADDENDUM_2026-09-24.md` | G03: full-fold triadic replay reverses the pilot |
| `G03_TRIADIC_GRAPH_FOLD0_FALSIFICATION_2026-09-24.md` | G03: graph-scale transfer fails on fold 0 |
| `G03_TRIADIC_GRAPH_REFIT_2026-09-24.md` | G03: graph-level fitting recovers a held arithmetic gain |
| `CONTEXTUAL_PRAGMATICS_2026-09-25.md` | Scoped pragmatic evidence inside the existing language substrate |
| `G03_DIRECT_BANK_REFIT_2026-09-25.md` | Direct decoder bank-contrast refit: source gain did not transfer |
| `G03_NATIVE_DECODER_SEMANTICS_2026-09-25.md` | Native decoder semantic pilot |
| `G03_NESTED_SELECTOR_AND_RESEARCH_2026-09-25.md` | G03 nested selection and research, 2026-09-25 |
| `G03_ORPHAN_LAUNCH_RETIREMENT_2026-09-25.md` | Orphaned launch agents during semantic replay |
| `G03_SOURCE_MISS_PROFILE_2026-09-25.md` | G03 source-fold misses: reach, interruption, and selection are separate |
| `G03_NATIVE_RETAINED_CANARY_2026-09-26.md` | Native retained development canary |
| `G03_NATIVE_RETAINED_SOURCE_CONNECTION_2026-09-26.md` | Native retained-source connection, 2026-09-26 |
| `G04_NATIVE_BINDING_CONTROL_PLAN_2026-09-26.md` | Native role and dependency controls, 2026-09-26 |
| `G04_NATIVE_FORM_CANARY_PLAN_2026-09-26.md` | Native source-form canary plan |
| `G04_NATIVE_FORM_CANARY_RESULT_2026-09-26.md` | Native source-form canary result |
| `G04_NATIVE_FRESH_SCHEMA_2026-09-26.md` | Native selection on withheld three-step schemas |
| `G04_NATIVE_RELATIVE_GRAMMAR_FULL_2026-09-26.md` | Target-blind three-step development result |
| `G04_NATIVE_TARGET_BLIND_PROPOSALS_2026-09-26.md` | Native proposal boundary |
| `G06_NATIVE_BASE_CONTROL_2026-09-26.md` | Unfitted native grammar control |
| `G06_NATIVE_DEPTH_CONTROL_2026-09-26.md` | Native grammar depth control |
| `G06_NATIVE_SOURCE_ERASURE_PLAN_2026-09-26.md` | Native source-erasure control, 2026-09-26 |
| `G06_NATIVE_SOURCE_ERASURE_RESULT_2026-09-26.md` | Matched native fitting-source erasure |
| `G06_NATIVE_SOURCE_INTERVENTION_CANARY_2026-09-26.md` | Native source-operation canary |
| `G06_NATIVE_SOURCE_INTERVENTION_PLAN_2026-09-26.md` | Native source intervention plan, before model evaluation |
| `G_MODEL_LANE_COORDINATION_2026-09-26.md` | One model-memory ledger for live-profile research |
| `G_NATIVE_BRANCH_FP32_RESULT_2026-09-26.md` | FP32 source-cache arithmetic |
| `G_NATIVE_BRANCH_PRECISION_PLAN_2026-09-26.md` | Precision-controlled source-cache probe |
| `G_NATIVE_CODEC_AND_SEARCH_2026-09-26.md` | Native codec and search checkpoint |
| `G_NATIVE_GRAMMAR_CHOICE_CANARY_RESULT_2026-09-26.md` | Grammar-choice canary: fit verified, generated answer regressed |
| `G_NATIVE_GRAMMAR_CHOICE_OBJECTIVE_2026-09-26.md` | Native grammar-choice training |
| `G_NATIVE_PREFIX_BRANCH_REJECTION_2026-09-26.md` | Resident prefix branch reuse rejected |
| `G_NATIVE_PREFIX_BRANCH_REUSE_PLAN_2026-09-26.md` | Frozen native prefix branches, 2026-09-26 |
| `G_NATIVE_PREFIX_GROUPING_2026-09-26.md` | Real prefix grouping measurement |
| `G_NATIVE_RELATIVE_FIT_2026-09-26.md` | Role-relative native fit |
| `G_NATIVE_SUPERVISION_IDENTIFIABILITY_2026-09-26.md` | Native supervision identifiability, 2026-09-26 |
| `G_NATIVE_TRIE_ARITHMETIC_2026-09-26.md` | Shared causal-prefix computation |
| `SUBPROCESS_PIPE_OWNERSHIP_2026-09-26.md` | Exited-child pipe ownership |
| `G03_NATIVE_SOURCE_CONTROL_2026-09-27.md` | Native source-content controls, 2026-09-27 |
| `G03_PAIRED_SOURCE_FIT_2026-09-27.md` | Paired source-choice fit on the exposed bank |
| `G10_BALANCED_POLARITY_CONTROLS_2026-09-27.md` | Balanced polarity controls for 27B CAA development |
| `G10_CONTRASTIVE_DEVELOPMENT_PIPELINE_2026-09-27.md` | CAA contrastive development path (not a qualification) |
| `G_NATIVE_GRAMMAR_FULL_EPOCH_2026-09-27.md` | Native grammar complete-epoch result |
| `G03_COMPLETE_PATH_OPTIMIZATION_2026-09-28.md` | Complete native paths and optimization |
| `G03_DEVELOPMENT_WINDOW_EXECUTION_2026-09-28.md` | G03 complete development windows, 2026-09-28 |
| `G03_JOINT_GRAPH_SELECTION_2026-09-28.md` | G03 joint graph checkpoint selection, 2026-09-28 |
| `G03_RETAINED_NATIVE_GENERATION_2026-09-28.md` | Retained native generation, fitted versus base |
| `G03_TYPED_PARAMETER_ISOLATION_2026-09-28.md` | Native parameter isolation by decision type |
| `G03_TYPED_SOURCE_CONTRAST_COVERAGE_2026-09-28.md` | Typed source contrast coverage |
| `G03_V6_TYPED_SOURCE_RESULT_2026-09-28.md` | V6 typed source fit and generated result |
| `G03_V7_MICRO_PROBE_PROTOCOL_2026-09-28.md` | G03 v7 micro-probe protocol, 2026-09-28 |
| `G03_V7_PARTIAL_PREFIX_REUSE_2026-09-28.md` | G03 v7 partial frozen-prefix reuse, 2026-09-28 |
| `G04_CONDITIONAL_NATIVE_BINDING_2026-09-28.md` | Conditional native binding measurement |
| `G04_NATIVE_DEPENDENCY_SOURCE_CONTROL_2026-09-28.md` | Native dependency source-control result |
| `G04_NATIVE_ROLE_SOURCE_CONTROL_2026-09-28.md` | Native role source-control result |
| `G04_V7_RELATION_MECHANISM_MICRO_PROTOCOL_2026-09-28.md` | V7 relation mechanism micro-probe, 2026-09-28 |
| `G04_V7_TARGET_BLIND_MICRO_PROTOCOL_2026-09-28.md` | G04 v7 target-blind micro-probe, 2026-09-28 |
| `G_MODEL_LANE_HANDOFF_2026-09-28.md` | Model-lane handoff on 28 September |
| `G03_CACHED_GROUPED_SEARCH_2026-09-29.md` | Cached grouped native search, 2026-09-29 |
| `G03_COUNTERFACTUAL_ACQUISITION_PREFLIGHT_2026-09-29.md` | G03 counterfactual acquisition preflight, 2026-09-29 |
| `G03_EARLY_STOP_CONTRAST_2026-09-29.md` | G03 early-stop contrast, 2026-09-29 |
| `G03_FIT_ONLY_TERMINATION_CONTRAST_2026-09-29.md` | G03 fit-only termination contrast, 2026-09-29 |
| `G03_NATIVE_BUDGET_FRONTIER_2026-09-29.md` | G03 native budget frontier, 2026-09-29 |
| `G03_POSTFIX_DEFINITION_WORDING_2026-09-29.md` | G03 postfix definition wording, 2026-09-29 |
| `G03_PROSPECTIVE_REFERENCE_SCREEN_2026-09-29.md` | G03 prospective reference screen, 2026-09-29 |
| `G03_REAL_TOKENIZER_PAIR_AUDIT_2026-09-29.md` | G03 real-tokenizer pair audit, 2026-09-29 |
| `G03_RESIDUAL_SCALE_ORACLE_CEILING_2026-09-29.md` | V7 source-calibration scale ceiling, 2026-09-29 |
| `G03_SOURCE_ROLE_RULE_PROBE_2026-09-29.md` | G03 source-role rule probe, 2026-09-29 |
| `G03_V7_CAUSAL_GROUP_RECOVERY_2026-09-29.md` | G03 v7 micro recovery, 2026-09-29 |
| `G03_V7_FIT_MATCHED_RESIDUAL_2026-09-29.md` | G03 v7 fit-matched residual, 2026-09-29 |
| `G03_VIDEO_REASONING_IMPLEMENTATION_2026-09-29.md` | Source-indexed reasoning review, 2026-09-29 |
| `G03_BINARY_ITERATE_RESUME_2026-09-30.md` | G03 binary iterate resume, 2026-09-30 |
| `G03_BOUNDED_NATIVE_FIT_2026-09-30.md` | G03 bounded native fit, 2026-09-30 |
| `G03_BOUNDED_SOURCE_HEAD_FIT_2026-09-30.md` | G03 bounded source-head fitting, 2026-09-30 |
| `G03_COMPUTED_CONSTRAINTS_AND_NATIVE_REJECTION_2026-09-30.md` | G03 computed constraints and native rejection, 2026-09-30 |
| `G03_MIXED_COMPUTATION_ENTRY_2026-09-30.md` | G03 mixed computation entry, 2026-09-30 |
| `G03_NATIVE_IDENTIFIABILITY_PREFLIGHT_2026-09-30.md` | Exact-path identifiability before model loading, 2026-09-30 |
| `G03_NATIVE_PREPARATION_HANDOFF_2026-09-30.md` | Source bank to native preparation, 2026-09-30 |
| `G03_OBSERVATION_RECOMPUTATION_2026-09-30.md` | G03 observation-driven recomputation, 2026-09-30 |
| `G03_PREPARED_NATIVE_FIT_HANDOFF_2026-09-30.md` | Prepared native fit continuation, 2026-09-30 |
| `G03_SOURCE_FIT_HANDOFF_2026-09-30.md` | G03 source-fit handoff, 2026-09-30 |
| `G03_STOP_SOURCE_REACQUISITION_2026-09-30.md` | G03 stop-source reacquisition, 2026-09-30 |
| `BENCHMARK_SOURCE_CASE_IDENTITY_2026-10-08.md` | Benchmark source cases, 8 October 2026 |
| `BOUND_ACTIONS_AND_TERMINAL_PROOF_2026-10-11.md` | Bound actions and terminal proof, 11 October 2026 |
| `CREATION_CURRENT_WORK_2026-10-11.md` | Creation completion and current work, 11 October 2026 |
| `CURRENT_VISUAL_CONTROLS_AND_REOPENED_SELECTORS_2026-10-10.md` | Current visual controls and reopened selectors, 10 October 2026 |
| `DESKTOP_PLACEMENT_AND_STOP_2026-10-11.md` | Desktop placement and Stop, 11 October 2026 |
| `FLASH_CAMPAIGN_RESUMPTION_2026-10-11.md` | Flash campaign resumption, 11 October 2026 |
| `FLASH_DESKTOP_CAMPAIGN_PROTOCOL_2026-10-11.md` | Desktop campaign protocol, 11 October 2026 |
| `FOREGROUND_TRANSFER_IDENTITY_2026-10-11.md` | Foreground identity during transfers, 11 October 2026 |
| `FULL_DESKTOP_FLASH_ATTEMPT_2026-10-11.md` | Full desktop Flash attempt, 11 October 2026 |
| `G03_ADJUDICATION_HANDOFF_2026-10-01.md` | Development Adjudication Handoff, 2026-10-01 |
| `G03_AN_OPERATION_OWNS_ITS_SENTENCE_2026-10-05.md` | An operation's register owns its sentence |
| `G03_ARGUMENTS_WITHIN_THEIR_SENTENCE_2026-10-05.md` | No argument option starts before its operation's sentence |
| `G03_A_STACKED_LABELER_DOES_NOT_HOLD_2026-10-05.md` | Naming an operation from its words as well as its context does not hold |
| `G03_BINDING_BUILD_REGISTER_2026-09-30.md` | G03 Binding Build Register |
| `G03_COMPLETE_SOURCE_NEGATIVE_2026-10-01.md` | Complete source-development negative |
| `G03_CONDITIONAL_CHART_BOUND_2026-10-01.md` | Conditional Chart Bound, 2026-10-01 |
| `G03_DECLARATIONS_AND_CONDITIONAL_FIT_2026-10-03.md` | A declaration holds no operation, and the readout fitted as it is used |
| `G03_DEVELOPMENT_ROW_ARCHIVES_2026-10-01.md` | Development Row Archives, 2026-10-01 |
| `G03_GROUNDED_FIT_RESTART_2026-10-01.md` | Grounded Fit Restart, 2026-10-01 |
| `G03_GROUNDED_PREPARATION_HANDOFF_2026-10-02.md` | Grounded preparation custody and exact fit handoff |
| `G03_HYBRID_JOINT_EXECUTION_2026-10-01.md` | Hybrid Joint Execution, 2026-10-01 |
| `G03_IDENTICAL_SHARE_AND_SCORING_2026-10-03.md` | How common a matched word is, and how the antecedent enters the score |
| `G03_INDEPENDENT_DEVELOPMENT_ADJUDICATION_2026-10-01.md` | Independent Development Adjudication, 2026-10-01 |
| `G03_JOINT_CHART_BATCHING_2026-10-01.md` | Joint Chart Execution, 2026-10-01 |
| `G03_JOINT_DEVELOPMENT_SCREEN_2026-10-01.md` | Joint Development Screen, 2026-10-01 |
| `G03_JOINT_NATIVE_DECODE_2026-10-01.md` | Joint Native Decode, 2026-10-01 |
| `G03_JOINT_NATIVE_PREPARATION_2026-10-01.md` | Joint Native Preparation, 2026-10-01 |
| `G03_JOINT_SCORE_IDENTIFIABILITY_2026-10-01.md` | Joint Role Score Identifiability, 2026-10-01 |
| `G03_NAME_ANTECEDENTS_2026-10-03.md` | A named result read back to where its name was given |
| `G03_NATIVE_LAUNCH_ENVIRONMENT_2026-09-30.md` | Native Handoff Launch Environment |
| `G03_NONLINEAR_MEANING_BUILD_2026-09-30.md` | Nonlinear Meaning Build |
| `G03_OPERATION_FIELD_REPAIR_2026-10-01.md` | Whole-request operation learning |
| `G03_PEAK_RECOGNITION_2026-10-02.md` | Operations found where the request reads as one |
| `G03_PREPARATION_STALL_REPLAY_2026-10-03.md` | Preparation stall and source replay |
| `G03_PROGRAM_POOL_PROGRESS_2026-10-01.md` | Completed program proofs survive unfinished searches |
| `G03_REBOOT_RECOVERY_2026-09-30.md` | G03 Source Chain Reboot Recovery |
| `G03_SELECTED_EDGE_EXECUTION_2026-10-01.md` | Selected-Edge Joint Execution, 2026-10-01 |
| `G03_TWO_RULES_THAT_FAIL_2026-10-03.md` | Two rules for the last composition misses, and why neither is kept |
| `G03_WHICH_AND_WHETHER_2026-10-06.md` | The word says which operation; the sentence says whether |
| `G04_TRANSFER_PREREGISTRATION_2026-10-06.md` | G04 transfer: the plan, published before anything is decoded |
| `G04_TRANSFER_RESULT_2026-10-06.md` | G04 transfer: construction and family carry, vocabulary and depth do not |
| `G04_TRANSFER_V2_PREREGISTRATION_2026-10-07.md` | G04 second transfer plan: preregistered before any request is decoded |
| `G04_TRANSFER_V2_RESULT_2026-10-07.md` | G04 second transfer run: every preregistered claim holds |
| `G04_V17_DEVELOPMENT_2026-10-07.md` | G04 second candidate: what the first run's failures taught, and what it cost |
| `G05_CLOSED_CHANNEL_PILOT_2026-10-07.md` | G05 closed-channel pilot: where a settled answer reaches a person |
| `G05_LONG_LIST_PILOT_2026-10-06.md` | G05 long-list pilot: long lists do not make her slip either |
| `G05_PUBLIC_ANSWERS_PREREGISTRATION_2026-10-07.md` | G05 public answers: preregistered before any request is answered |
| `G05_PUBLIC_ANSWERS_RESULT_2026-10-07.md` | G05 public answers: the reading reaches the answer a person reads |
| `G05_PUBLIC_ANSWER_PILOT_2026-10-06.md` | G05 pilot: her free answers already carry the reading |
| `G06_COMPARISONS_RESULT_2026-10-07.md` | G06: the reader against the ordinary model, more compute, matched controls and lesions |
| `G06_GROUNDED_MATCHED_LESIONS_2026-10-01.md` | Grounded Matched Lesions, 2026-10-01 |
| `G09_BROAD_GAIN_PLAN_2026-10-07.md` | G09: what a broad gain would be measured on, and why it is not there yet |
| `G09_PROCEDURES_PREREGISTRATION_2026-10-08.md` | G09 kept procedures: preregistered before any procedure exists |
| `G09_PROCEDURES_PREREGISTRATION_2_2026-10-12.md` | G09 kept procedures, second attempt: preregistered before its development run |
| `G09_PROCEDURES_PREREGISTRATION_3_2026-10-14.md` | G09, attempt 3: the prefill run's held-out kinds |
| `G09_PROCEDURES_PREREGISTRATION_4_2026-10-15.md` | G09: kept procedures, attempt 4 (preregistered 15 October 2026, before attempt 3's verdict) |
| `G09_PROCEDURES_PREREGISTRATION_5_2026-10-16.md` | G09: kept procedures, attempt 5 (part 1 committed 16 October 2026 at 10:33, before any routing is measured) |
| `G09_PROCEDURES_PREREGISTRATION_6_2026-10-17.md` | G09: kept procedures, attempt 6 (part 1 committed 17 October 2026 at 03:51, before any bar is measured) |
| `G09_PROCEDURES_RESULT_1_2026-10-13.md` | G09 kept procedures, first attempt: a gain in transfer, and not yet broad |
| `G09_PROCEDURES_RESULT_2_2026-10-14.md` | G09, attempt 2: planning gains, math does not, breadth not met |
| `G09_PROCEDURES_RESULT_3_2026-10-16.md` | G09: kept procedures, attempt 3 result (16 October 2026, 02:25) |
| `G09_PROCEDURES_RESULT_5_2026-10-17.md` | G09: kept procedures, attempt 5 result (17 October 2026) |
| `G09_PROCEDURES_RESULT_6_2026-10-17.md` | G09: kept procedures, attempt 6 result (17 October 2026) |
| `G10_READER_QUALIFICATION_2026-10-07.md` | G10: the program reader qualified on her current model, as her runtime runs it |
| `G12_FRONTIER_EVALUATION_2026-10-07.md` | G12: her performance against named current baselines, with resources stated |
| `G12_NAMED_BASELINES_READ_2026-10-14.md` | G12: the named systems' published numbers, as read on 14 October 2026 |
| `GENERAL_CONTROL_DEMO_2026-10-05.md` | Autonomous repair and three-attempt replay, 5 October 2026 |
| `GENERAL_ENVIRONMENT_RESEARCH_APPLICATION_2026-10-10.md` | General environment research application, 10 October 2026 |
| `GENERAL_INPUT_RETENTION_REPAIR_2026-10-08.md` | Input composition and retained state, 8 October 2026 |
| `GENERAL_REASONING_DEVELOPMENT_RESULT_2026-10-16.md` | General reasoning: the development round (16 October 2026) |
| `GENERAL_REASONING_PROGRAM_2026-10-14.md` | General reasoning: settled where it can be, grounded where it can't, stable where neither applies |
| `L01_WHO_SHE_IS_PREREGISTRATION_2026-10-18.md` | L01: who she is, held by her model and not given to it. Preregistration, 18 October 2026 |
| `L05_KNOWLEDGE_ACROSS_TURNS_2026-10-05.md` | L05: one turn reaches her model, the offline copy, the web and what she kept |
| `L09_LEARNING_IN_USE_LOOP_2026-10-18.md` | L09: one loop of learning in use, live (18 October 2026) |
| `L09_ORDINARY_USE_PREREGISTRATION_2026-10-18.md` | L09, second half: her kept procedures on against off over her ordinary use. Preregistration, 18 October 2026 |
| `L09_ORDINARY_USE_PREREGISTRATION_2_2026-10-19.md` | L09, second half, attempt 2: her kept procedures on against off over requests people make. Preregistration, 19 October 2026 |
| `L09_ORDINARY_USE_RESULT_2026-10-18.md` | L09, second half: her kept procedures on against off over her ordinary use. Result, 18-19 October 2026 |
| `OBSERVED_CHOICES_AND_SCOPED_STOP_2026-10-10.md` | Observed choices and scoped stopping, 10 October 2026 |
| `OBSERVED_PROCEDURE_EFFECTS_2026-10-10.md` | Observed procedure effects, 10 October 2026 |
| `OPTIONAL_PREPARATION_BODY_ENTRY_2026-10-11.md` | Optional preparation and body entry, 11 October 2026 |
| `RETAINED_STRUCTURAL_STATE_2026-10-10.md` | Retained structural state, 10 October 2026 |
| `SELF_REPAIR_ON_HER_HISTORY_2026-10-19.md` | Self-repair on defects from her own history: first measurement, 18-19 October 2026 |
| `STEERING_FROM_HER_OWN_REPLIES_2026-10-19.md` | Steering directions from her own replies: the first capture is one axis, 18-19 October 2026 |
| `STOP_REACHES_INFLIGHT_INFERENCE_2026-10-10.md` | Stopping in-flight inference, 10 October 2026 |
| `VISUAL_PROBE_REJECTIONS_2026-10-11.md` | Visual probe rejections, 11 October 2026 |
| `WHAT_A_WORLD_S_CODE_SAYS_PUT_TO_ITS_RUNNING_2026-10-13.md` | What a world's code says, put to the world running it, 13 October 2026 |

Machine-generated proof bundles live under `artifacts/` (see
[ARTIFACT_INDEX.md](../../ARTIFACT_INDEX.md)); test standards live in
`docs/` proper.
