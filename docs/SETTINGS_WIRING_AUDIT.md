# Runtime Settings Control-Plane Audit

**Status (2026-07-15): CP87 pushed at `747bceea`; exact signed-app proof in progress.**

The desktop settings surface no longer treats `localStorage` as runtime or
governance authority. Runtime settings are hydrated from an authenticated API,
validated against one core schema, committed as one compare-and-swap
transaction, read by their actual owners, and acknowledged back to the control
plane. Browser-local storage remains only for non-authoritative presentation
preferences and compatibility migration.

This document records what the implementation proves, what each setting owns,
and what remains open. A rendered control, persisted value, or passing mock is
not sufficient evidence that a setting works.

## Canonical Architecture

1. `core/runtime/settings_schema.py` owns schema version 2, defaults, types,
   ranges, enums, owner names, apply modes, strict patch validation, and legacy
   migration. Boolean and numeric values are not silently coerced or clamped.
2. `core/runtime/settings_control_plane.py` owns the durable state envelope,
   revision, atomic patch/reset/rollback, retained history, mutation receipt
   chain, owner-application receipt chain, integrity verification, and
   subscriber dispatch.
3. `interface/routes/settings.py` exposes the authenticated internal API. File
   I/O and `fsync` work run off the event loop. The API requires an expected
   revision for every mutation and returns structured conflict or validation
   outcomes.
4. `core/runtime/runtime_settings.py` is the layering-clean read boundary for
   core owners. It accepts versioned state and legacy flat maps, validates data,
   and refreshes on file identity, nanosecond mtime, or size changes without
   importing the interface layer.
5. `interface/static/aura.js` hydrates backend truth before enabling runtime
   controls, sends atomic CAS patches with request IDs, rejects stale or
   superseded responses, reports owner application state, and uses browser
   microphone capture only as a fallback when canonical server capture is not
   available.

## Transaction and Recovery Contract

The source implementation provides:

- atomic multi-key patches with strict schema validation before any write;
- compare-and-swap revisions and HTTP 409 conflict responses;
- request idempotency, including detection of request-ID reuse with different
  content and explicit reporting when a replay has been superseded;
- thread and interprocess serialization;
- crash-safe durable replacement and deterministic recovery when an audit
  receipt lands before the state envelope;
- a retained 16-revision rollback window;
- legacy flat-map migration, including boolean proactive messaging values;
- hash-linked mutation receipts and a separate hash-linked owner-application
  journal;
- state/audit/application integrity checks, unknown-key reporting, and
  unapplied or unacknowledged revision reporting;
- serialized subscriber dispatch that re-reads durable truth and marks stale
  same-key work as superseded instead of applying an older value.
- full mutation-chain verification before version-2 values reach core owners;
  corruption, incompatible state, permission loss, or post-read deletion
  activates conservative autonomy, approval, privacy, permission, cloud, and
  self-modification overrides until valid state returns. A never-created file
  remains the distinct first-boot case and uses documented defaults.

The SHA-256 chains detect corruption, truncation, inconsistent recovery, and
ordinary tampering where the attacker does not rewrite every linked artifact.
They are **not a keyed signature or hostile-local-admin security boundary**. A
principal with permission to rewrite the state and both journals can recompute
the chains. Managed signing, key custody, and independent export verification
remain under `ENTERPRISE-CONTROL-001` and `SECURITY-001`.

## API Contract

| Method | Route | Contract |
| :-- | :-- | :-- |
| `GET` | `/api/settings` | schema, revision, values, integrity, and owner-application state |
| `PATCH` | `/api/settings` | atomic `{expected_revision, request_id, changes}` transaction |
| `POST` | `/api/settings/reset` | reset one schema section through CAS |
| `POST` | `/api/settings/rollback` | restore one retained revision as a new mutation |
| `GET` | `/api/settings/integrity` | verify state, mutation chain, and application chain |
| `POST` | `/api/settings/application-ack` | append owner application evidence for changed keys |
| `POST` | `/api/settings/auth/fresh` | authorize one exact, expiring action challenge |
| `POST` | `/api/settings/auth/revoke` | cancel one unconsumed challenge after abandonment or transport failure |

All routes require the existing internal authentication dependency. Settings
state is not writable through unauthenticated browser storage.

## Action and Approval Semantics

`autonomy.actions_enabled` is now a protected, default-true compatibility
field rather than a mutable kill switch. Patch and rollback attempts to set it
false are rejected, and legacy false state is advanced through an audited
reconciliation transaction. Runtime admission preserves the agency invariant
even when the settings file is absent, corrupt, or carries a pre-invariant
value. Emergency containment remains explicit through `safety.safe_mode` and
`autonomy.level=paused`; consequential effects still traverse Constitution,
Unified Will, Conscience, standing authority, substrate authority,
ExecutiveCore, capability tokens, and effect receipts.

`governance.approval_mode` is an **additional confirmation overlay**, not a
replacement for Constitution, Unified Will, Conscience, standing authority,
substrate authority, ExecutiveCore, capability tokens, or effect receipts.

- `none`: no additional settings-layer prompt;
- `destructive`: prompt for critical risk or destructive effect scopes;
- `all`: prompt for every governed tool or environment action.

A prompt issues a random challenge bound to the canonical tool/action name,
arguments digest, authenticated source, risk, and effect scope. The pending
challenge expires after 300 seconds. Authorization expires after 60 seconds and
is atomically consumable once by the exact same action. Changed arguments,
source, risk, scope, expiry, a second consumer, or a process restart cannot
reuse it. The desktop retries the same request only after confirmation and does
not duplicate the visible user turn. Confirmation requests are bounded, the
dialog contains keyboard focus, and abandonment or transport failure revokes
the unconsumed challenge. Outstanding challenges are intentionally
process-local; durable cross-restart action authorization is not claimed.

## Current Setting Classification

| Setting | Classification | Enforced owner/effect |
| :-- | :-- | :-- |
| `safety.safe_mode` | wired | live safe-mode bridge and boot posture |
| `autonomy.level` | wired | `paused` drives the same restricted runtime posture |
| `autonomy.actions_enabled` | protected invariant | immutable true compatibility field, audited legacy repair, and source-independent agency admission |
| `governance.approval_mode` | wired | exact one-time confirmation overlay |
| `autonomy.proactive_messaging` | wired | `never/minimal/balanced/frequent` have distinct daily, interval, and idle policies; counters reset daily; critical alerts bypass ordinary quotas |
| `autonomy.self_modification` | wired | growth-ladder admission posture |
| `learning.auto_enrichment_enabled` | wired | model extraction and persistence admission |
| `learning.reflection_enabled` | wired | automatic conversation reflection, lesson, and online-learning persistence; it does not disable Aura's internal thought |
| `model.local_path` | wired | primary local model-path override when valid |
| `model.deep_path` | wired | deep-solver model-path override when valid |
| `model.cloud_fallback_enabled` | wired | authoritative off-box model-routing permission |
| `voice.input_enabled` | wired | microphone capture loop gate |
| `voice.output_enabled` | wired | synthesis and streaming output gate |
| `voice.output_rate` | wired | bounded speech-rate multiplier |
| `permissions.camera` | wired | camera capture entry points; macOS TCC remains independent |
| `permissions.screen` | wired | screenshot/screen-sensor entry points; macOS TCC remains independent |
| `memory.retention_days` | wired | sovereign-pruner recency horizon |
| `privacy.mode` | partial | world-action isolation/posting policy is enforced; complete telemetry and perception redaction is still open |
| `dev.developer_mode` | wired | diagnostic trace-route admission |
| `notify.enabled` | wired | desktop notification emission |
| `notify.quiet_hours_start/end` | wired | local-time quiet window, including midnight wrap |
| `theme.mode` | frontend-only | desktop presentation |
| `theme.reduced_motion` | frontend-only | animation/motion presentation |
| `onboarding.completed` | frontend-only | the first-run wizard sets it when it finishes; the desktop reads it to decide whether to offer setup |
| `voice.auto_listen` | wired | live canonical server-capture owner; browser capture is a non-duplicating fallback |
| `permissions.files_workspace` | known dead/open | central workspace file-gateway enforcement is not implemented |
| `memory.review_window` | known dead/open | no canonical age-windowed narrative consolidation owner exists yet |
| `dev.diagnostics_enabled` | known dead/open | a distinct optional boot diagnostic must be separated from load-bearing health owners |

`tests/test_settings_no_dead_controls.py` requires every schema key to remain in
exactly one classification bucket and prevents the known-dead set from growing
silently.

## Source Evidence and Remaining Live Proof

The pushed CP87 checkpoint and the current voice-authority follow-up have
deterministic coverage for strict validation,
CAS conflicts, idempotent replay, concurrent writers, crash recovery, rollback,
migration, state and journal tampering, stale dispatch suppression, owner
acknowledgements, safe-mode bridging, direct-user preservation, forged-context
rejection, exact one-use confirmation, environment-action coverage, desktop
approval propagation, learning gates, malformed model/input handling,
proactive cadence, daily reset, critical quota bypass, and failed-delivery
retention. The expanded bounded settings/governance/voice/boot slice passes
`532/532`; the focused desktop confirmation and Conscience contracts remain
covered by that integrated source campaign.

Exact-app testing found and retained one important contradiction rather than
crediting a shallow receipt: revision 5 reported `voice.input_enabled=false`
and `voice.auto_listen=false`, while the same signed process reported
`microphone_enabled=true`, `auto_listen=true`, `listening=true`, and continued
emitting live RMS samples. The source follow-up removes launch-environment
override authority, initializes the one canonical voice engine from verified
settings, applies input/auto-listen/output changes at that resident owner, and
requires the owner receipt to reflect actual start/stop/interruption state.
Exact rebuilt-app reproof remains mandatory.

The following are still required before `RUNTIME-SETTINGS-001` can close:

1. Build and install the signed app from the exact pushed CP87 commit.
2. Prove GET/PATCH/readback, stale-revision conflict, restart persistence,
   reset, rollback, and integrity from the installed desktop shell.
3. Toggle each enabled runtime control and observe its real owner effect, owner
   acknowledgement, and recovery path in the desktop, terminal, and Neural
   stream.
4. Prove `all`, `destructive`, and `none` with an exact action challenge,
   cancellation, expiry, changed-argument rejection, one-use consumption, and
   successful downstream governance/effect closure.
5. Prove the two frontend-only theme controls in the real shell and keep the three
   known-dead controls absent or visibly unavailable until implemented.
6. Run accessibility/keyboard/focus and responsive-layout proof for settings
   conflicts, owner failures, and the confirmation modal.

Source-green is not installed-app green. This audit must be updated with exact
build provenance and retained live artifacts before the task status changes to
complete.
