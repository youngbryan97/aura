# R09: terminal history survives the delivery boundary

Date: 2026-09-09. R09 remains open.

## Repair

The delivery journal could seal an answer before its separate transcript write
finished. A process exit in that interval lost the in-memory exchange capture.
The journal now stores that capture in the same SQLite transaction as the
terminal response. Retention excludes undrained captures. Replay and the existing
supervised memory worker drain them through conversation persistence, and only
a committed transcript retires a matching obligation.

The live transcript accepts duplicate delivery of the same exchange and bytes
without appending another entry or notifying listeners twice. Changed bytes
still require the existing compare-and-swap correction path. A history-store
failure does not hide a sealed public answer. Cancellation keeps its terminal
history but does not enqueue its abandoned draft for learning.

## Checks

- 95 focused custody, journal, cancellation, session-isolation and persistence
  tests passed in 22.55 seconds, including the real-store crash-window case.
- All 10 recovery cases then passed in 5.33 seconds, including real SQLite
  transcript/outbox replay after an injected lost acknowledgement.
- Smoke: 164 passed, one skipped in 86.31 seconds.
- Lint, compile, governance-lint and layering passed (37 existing layering
  exceptions, none added).
- Injected history insert failure rolls back the terminal seal. Corrupt capture
  JSON/hash, mismatched acknowledgement, journal reopen and retention are tested.

## Live observations on 354f6c444

The supported desktop reboot produced PID 29866, with expected and actual source
revision 354f6c4443f91ba771d570edb83076bbda9d447e. That runtime does not yet contain
this journal repair.

At 01:51:51 the Stop exercise entered `latent_reason`. Its cancellation reply
was delivered once and survived page reload. The caller unconditionally recycled
the worker after requesting soft cancellation, even though the worker returned
`soft_cancelled`. This is a separate remaining lifecycle defect.

At 01:55:44 a five-point database crash-recovery request delivered five items and
the complete requested footer, `Recovery explanation complete.` Page reload
preserved it without another generation. This proves the list-footer repair on
this turn, not every streaming condition or factual claim in the answer.

At 01:59:17 the referential follow-up asked which step handles uncommitted work.
The dispatch log carried only three messages: two system messages and the new
question. The five-step answer was absent. The first decode produced no public
answer; a retry eventually answered in terms of workflow compensation rather
than the database example. Referential continuity and R05 remain open.

Neural-feed observations retained for R08: autonomous planning reported
`No JSON array in response`; background self-description reported zero energy
despite live vitals around 74%; a deep-narrative repetition rejection purged its
cache; the phi estimator reported its state-summary fallback. These observations
are not closed by the chat custody tests.
