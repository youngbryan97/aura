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
| `G_RUNTIME_WATCH_AND_RESIDUAL_2026-09-16.md` | Runtime recovery and residual sampling |
| `TERNARY_BONSAI_2_27B_2026-09-17.md` | Ternary Bonsai 2 27B on this host, 2026-09-17 |

Machine-generated proof bundles live under `artifacts/` (see
[ARTIFACT_INDEX.md](../../ARTIFACT_INDEX.md)); test standards live in
`docs/` proper.
