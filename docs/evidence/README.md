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

Machine-generated proof bundles live under `artifacts/` (see
[ARTIFACT_INDEX.md](../../ARTIFACT_INDEX.md)); test standards live in
`docs/` proper.
