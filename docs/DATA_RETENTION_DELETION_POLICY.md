# Aura Data Retention and Deletion Policy

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

Aura keeps four things: what she needs for continuity, safety receipts,
diagnostics, and whatever you asked her to remember.

These are retention requirements for the stores named below, especially
`ContinuousExperienceStream` in `core/consciousness/continuous_experience.py`.
They are not an exhaustive inventory or a demonstrated claim that no other
data is collected. Experiment artifacts, caches, exports and external-service
records need their own handling, as [the data card](../DATA_CARD.md) explains.

## Retention Classes

- Private experience frames: default 24 hours.
- Standard experience frames: default 30 days.
- Conversation exports: retained until the user deletes them.
- Audit and governance receipts: retained for incident reconstruction unless
  the owner explicitly purges the local Aura home directory.
- Diagnostics bundles: operator-created artifacts; delete after incident close.

## Deletion Requirements

- Privacy routes under `interface/routes/privacy.py` are the API surface for
  camera/microphone/privacy state.
- Continuous experience deletion uses
  `ContinuousExperienceStream.delete_privacy_tier(...)` or
  `ContinuousExperienceStream.delete_where(...)`.
- Retention is enforced by `ContinuousExperienceStream.enforce_retention(...)`.
- Deletes rebuild the hash chain so replay validation remains honest after
  intentional removal.

## Redaction Requirements

- Private frames export as hashes, not raw summaries.
- Source references are removed from private redacted exports.
- Secrets and credentials must never be written to memory; release gates run
  `tools/security_scan.py`.
