# Aura 1.0 master work queue

Created 2026-09-06. Owner: Codex, coordinating with Bryan and Claude.

This is the index of the continuing pass, not a replacement for detailed
ledgers or historical evidence. All inherited unresolved items remain in scope.
It is not yet a certified exhaustive inventory: reconciliation below is work,
not an assumption. Newly discovered defects are added before they are deferred.
No percentage is meaningful until that inventory is reconciled.

## Execution contract

- Commits and pushes save progress; proceed to the next ready item afterward.
- Close an item only with source identity, tests, and required runtime evidence.
- Record owner, dependency, next action, and evidence when an item is blocked.
- Reuse existing substrates and repair causal mechanisms, not prompts or labels.
- Separate implementation, offline tests, live behavior, and scientific claims.
- Preserve historical verdicts. Never migrate evidence by renaming a model.
- Platform limits can suspend execution; continuation does not bypass them.
- Unknown research outcomes are not deadline commitments or guaranteed wins.

## 0. Reconcile the complete inventory

Working evidence: [inherited reconciliation](AURA_INHERITED_RECONCILIATION.md).
All 66 unchecked occurrences are mapped; prose/table review and full evidence
reconciliation remain open. This does not close I01-I03.

- [ ] I01 Read and map every unresolved entry in the inherited ledgers below.
- [ ] I02 Reconcile stale checkboxes against commits, tests, and live receipts.
- [ ] I03 Deduplicate by mechanism; retain links to every original obligation.
- [ ] I04 Incorporate all advisory-review items: adopted, superseded, rejected
  with rationale, or still open. Do not claim all were read without evidence.
- [ ] I05 Inventory every shipped capability, integration, UI control, and model.
- [ ] I06 Link each release requirement to an executable acceptance check.
- [ ] I07 Record current closeout audit, semantic status, and release rubric.

Inherited ledgers (every unresolved child item is included, not just headings):

- [AURA_EXECUTION_TRACKER.md](AURA_EXECUTION_TRACKER.md)
- [GENERALITY_TODO.md](GENERALITY_TODO.md)
- [AGI_GAUNTLET_TRACKER.md](AGI_GAUNTLET_TRACKER.md)
- [worktodo/TODO.md](worktodo/TODO.md)
- [gap_atlas/TODO.md](gap_atlas/TODO.md)
- [CONNECTOME_TODO.md](CONNECTOME_TODO.md)
- [LEARNED_LANGUAGE_INTERPRETATION_TODO.md](LEARNED_LANGUAGE_INTERPRETATION_TODO.md)
- [RECURSIVE_ENDOGENOUS_EXPANSION_TODO.md](RECURSIVE_ENDOGENOUS_EXPANSION_TODO.md)
- [AUTONOMOUS_DEVELOPMENTAL_AGENCY_TODO.md](AUTONOMOUS_DEVELOPMENTAL_AGENCY_TODO.md)
- [evidence/CLOSEOUT.md](evidence/CLOSEOUT.md)

## 1. Restore and validate the live runtime

- [x] R01 Repair reboot successor survival and detached-launch startup exit.
  Closed 2026-09-06: e82ccce56 isolates the runtime session from shell
  process-group cleanup. Detached PID 94572 reached readiness after its
  launcher exited; API reboot replaced it with PID 96426, which reached
  kernel_ready and became the sole port-8000 listener. Real-process isolation
  tests and reboot contracts pass. Evidence: [R01 receipt](evidence/R01_RUNTIME_SURVIVAL_2026-09-06.md).
- [ ] R02 Prove exactly one replacement process, expected revision, model,
  preserved state, and stable readiness after launch/restart/update.
- [ ] R03 Live-validate native-thinking public sentence grace (d472d2268).
- [ ] R04 Live-validate progress-aware owner cleanup (2aefb6f46); audit other
  eviction paths and cross-client ownership, not only the patched function.
- [ ] R05 Resolve empty latent answers without exposing private reasoning.
- [ ] R06 Repair event-loop blocking: filesystem writes, fsync under locks,
  knowledge operations, learning callbacks, and scheduler contention.
- [ ] R07 Reconcile health probe expiry, false readiness, and actual failures.
- [ ] R08 Resolve neural-feed warnings individually by cause; distinguish
  unrun evidence, missing telemetry, real failure, and historical observations.
- [ ] R09 Verify complete streaming, durable reconnect, one final answer per
  turn, cancellation, follow-up semantics, and multi-turn context retention.
- [ ] R10 Verify executable examples semantically, not merely process exit zero.
- [ ] R11 Measure prefill, decode, tool, retrieval, and queue latency separately;
  remove waste without degrading reasoning or arbitrarily cancelling work.

## 2. General RLC reasoning: the scientific critical path

- [ ] G01 Freeze a current baseline and exact mechanism/claim boundary.
- [ ] G02 Reconcile existing bounded 1.5B/32B/27B evidence and live activation.
- [ ] G03 Close learned semantic binding/composition failures on development
  tasks using the existing language and computational substrates.
- [ ] G04 Demonstrate held-out construction, vocabulary, depth, and family
  transfer; separate neural computation from executable-system assistance.
- [ ] G05 Translate internal gains into correct freely decoded public answers.
- [ ] G06 Compare ordinary model, equal-compute alternatives, matched controls,
  and causal lesions; account for selection, retries, and regressions.
- [ ] G07 Preregister powered fresh-task/seed replication and stopping rules.
- [ ] G08 Independently verify artifacts, uncertainty, contamination controls,
  and cross-domain outcomes. Negative/inconclusive results remain such.
- [ ] G09 Establish broad reasoning gain; bounded synthetic success is not this.
- [ ] G10 Qualify runtime materialization or fusion on the current model;
  verify geometry, identity, rollback, canaries, and no-regression behavior.
- [ ] G11 Prove the qualified mechanism actually serves eligible live requests.
- [ ] G12 Evaluate frontier performance against named current baselines on
  independent broad tasks, with resource and tool access reported fairly.
- [ ] G13 Update public RLC documentation to precisely match measured claims.

## 3. Unified learning, knowledge, and agency

- [ ] L01 Verify persona fusion/continuity and regenerate incompatible steering.
- [ ] L02 Close experience-to-dataset-to-training-to-evaluation-to-publication
  loops; reconcile marker formats, quality metadata, and active model identity.
- [ ] L03 Use independent correctness/preference evidence, not fluency proxies,
  for learning promotion; measure retention, forgetting, and regression.
- [ ] L04 Integrate reusable learned abstractions across RLC and Claude's
  generality/improvement work; test actual consumers and transfer.
- [ ] L05 Verify unified access to model knowledge, offline Wikipedia, web,
  retained memory, provenance, freshness, and uncertainty across turns.
- [ ] L06 Close action outcome, selfhood reinforcement, prediction resolution,
  and CRSM feedback edges where advertised; measure what actually updates.
- [ ] L07 Verify skill invention, installation, execution, longitudinal benefit,
  recovery, and rollback on unseen tasks rather than installation counts.
- [ ] L08 Audit affect, self-state, phi, and consciousness-related claims with
  causal controls; configured priors are not measured subjective experience.

## 4. User-facing capability acceptance

- [ ] U01 Natural conversation: direct answers, conversational repair, memory,
  preferences, evidence-grounded self-description, and no canned dead ends.
- [ ] U02 Voice: startup, hearing, turn-taking, interruption, speech generation,
  device changes, recovery, and measured end-to-end latency.
- [ ] U03 Vision/camera: permission, detailed perception, freshness, source
  attribution, and graceful device failure without invented observations.
- [ ] U04 Companion mode: bubble, hide/clear, notifications, restrained chat,
  proactive observation, and continuity with full mode.
- [ ] U05 Screen/browser understanding and interaction; exclude incognito,
  honor permissions and hidden state, and verify requested highlighting.
- [ ] U06 General desktop actions, app controls, wallpaper/media selection,
  file creation/download/organization, and undo where applicable.
- [ ] U07 Research, coding, mathematical execution, planning, and multi-step
  tasks through the same unified runtime users invoke.
- [ ] U08 Exercise every capability from I05 with varied phrasing, follow-ups,
  failures, cancellation, and recovery; track every observed defect.
- [ ] U09 UI accessibility, responsive layout, truthful progress, and polish.

## 5. Reliability, security, and release

- [ ] Q01 Local model inventory and retired-provider removal; verify all model
  identity, tokenizer, geometry, context, and allocation consumers.
- [ ] Q02 Disk retention: bound logs/exports/checkpoints/caches while preserving
  live dependencies, unique source work, state, and scientific evidence.
- [ ] Q03 Memory/process lifetime, cache ownership, leaks, pressure recovery,
  shutdown/restart, sleep/wake, and single-resident ownership.
- [ ] Q04 Scoped tool authority, privacy, prompt-injection boundaries, sandbox,
  secret handling, and fail-safe behavior without suppressing correct work.
- [ ] Q05 Persistence, migration, corruption recovery, backups, and rollback.
- [ ] Q06 Close all inherited architecture/governance/security debt entries.
- [ ] Q07 Refresh semantic ledger near code freeze; reconcile changed or
  superseded items in batches, then complete all remaining review coverage.
- [ ] Q08 Run focused, smoke, chunked full-suite, lint, compile, layering,
  governance, production, enterprise, documentation, and release gates.
- [ ] Q09 Resolve order-dependent tests; no isolated pass erases a batch fail.
- [ ] Q10 Run source-matched multi-hour soak only after short gates pass;
  inspect latency, growth, errors, capability retention, and recovery.
- [ ] Q11 Validate installation/update/uninstall and ordinary desktop launch.
- [ ] Q12 Publish reproducible evidence, release notes, known limitations,
  supported capabilities, and recovery instructions.
- [ ] Q13 Final 1.0 adjudication: all required child items closed with current
  evidence, no concealed blockers, and explicit unresolved research boundaries.

## Current evidence, not closure

Public-boundary/health-preview fix d472d2268 and progress-owner fix 2aefb6f46
are pushed. Focused tests and smoke passed; live validation remains open.
The 2026-09-06 queue replay's latent episode failed after 520.6 seconds with
4736 generated tokens and no accepted answer. An alternate response followed.
This does not establish RLC success, broad reasoning gain, or frontier parity.
