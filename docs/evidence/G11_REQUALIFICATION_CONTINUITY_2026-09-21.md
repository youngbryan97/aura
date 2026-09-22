# Qualified package continuity and replay

The activation builder rejected a manifest update that the serving verifier
already accepts: signed changes confined to steering, which the qualified
CPU executor does not consume. The builder now uses the existing component
continuity verifier. It retains the originally measured manifest identity;
it does not relabel historical evidence or authorize changed dependent tissue.
The materialization CLI accepts an explicit owner-private verification-key
path, as the serving checker does. No key is copied into an artifact.

Two additional signed-fixture tests cover the successful independent update
and rejection of a changed recurrence component. Together with the existing
manifest and generation-upgrade tests, 49 tests pass.

After rebasing onto current main, the original qualified candidate identity
is unchanged: `733882c8bde72f6ec34044ead6596c747c0364af2bd922351d4c1848edc67f77`.
The earlier ingress source-drift alarm does not reproduce on this checkout.
The installed-package checker reports active with signed steering continuity.
No replacement of the operational activation was necessary or performed.

A fresh CPU replay through foreground ingress and LatentCortexService uses
seed 2026092161. All 120 tasks pass; all 120 targeted lesions disrupt the
answer, all 120 backend receipts match, and both integration call counts are
120. Unsupported language is refused. Maximum measured latency is 80.972 ms.
This exercises production functions in a verification process, not the live
desktop or model decode. It is not new comparative reasoning-gain evidence.

Artifacts are in
`artifacts/migration/27b/recovery/requalification-20260921-context-checkpoint/`:

- `candidate.json`: unchanged candidate identity above.
- `runtime-verification.json`: receipt
  `96d8d1782774db4aab3477c2ba89578f80fe231346a79ccc5e25302d6e04cb93`.
- `activation.json`: new qualification receipt
  `6f36ffb6f4221942f6d7f14d5ca07571109afdbcbcfa939b8cefc80881d209a7`.

Smoke passes 164 tests with one source-drift diagnostic skipped because no
source drift exists. Lint, compile, governance and layering pass. The G03
experimental context recognizer remains separate and has no serving authority.
