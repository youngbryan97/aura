# Runtime graph retention and small-trial result

The completed 500-row development comparison selects the incumbent. The
incumbent scores 472 exact and 472 structurally equivalent; joint decoding
before fitting scores 459 and 459; the refit scores 441 and 444. Relative to
its own fit-start model, the refit gains 9 equivalent programs and loses 24.
These exposed development rows do not establish fresh transfer.

The final report is retained at
`~/.aura/rlc-evidence/semantic-aligned-multiround-20260916/validation.json`.
Its SHA-256 is
`dfb1728dd52a5d7ffd44ac9bd696b5bba4c9131b07a7a65533e8a2bd26e2487a`.
The detached evaluator completed in 2192.041541 seconds with exit code zero.
That establishes completion, not improvement.

## Mechanism change

Training previously retained argument competitors within the annotated
operation chart. It could change operation scoring without protecting
correct autonomous decodes against other charts. The learner now obtains
operation charts from the same builder used by runtime decoding, finds
semantically distinct competitors using the existing universal floor, and
retains their score inequalities. Equivalent programs are positive evidence;
finite agreement alone is not an equivalence proof. Incomplete searches
remain explicitly incomplete.

The new small-trial tool separates acquisition, fitting, and replay. It does
not export a candidate or grant serving authority. Training acquisition can
scan a larger training-only pool and select witnessed failures alongside
retention controls. Validation never enters the fit.

## First small trial

Eight training and eight validation rows were selected by source hash and
geometry. Training stayed 8/8 equivalent; validation stayed 6/8. All 223
retained constraints were already satisfied, so no update was accepted.
There were no gains or regressions. The tool correctly declined a larger
development run on this evidence.

Receipt:
`~/.aura/rlc-evidence/semantic-runtime-retention-small-20260917/trial.json`,
content receipt
`4a83130d51e80d27ef1c40414b9d07027f53f4371b14e51a2cbb73bd61515a24`.

This exposed an acquisition limitation: an all-correct training sample
cannot test error repair. The next small trial must contain witnessed
training failures without selecting or fitting validation failures.

G03 remains open. No candidate is promoted.

## Training acquisition follow-up

The next bounded run scanned 48 training examples. All 48 were equivalent,
so acquisition again found no witnessed training error. The selected eight
training examples remained correct, validation remained 6/8, and the 223
constraints required no update. This does not justify repeating the full
500-row evaluation or fitting validation answers. The next diagnostic is
construction coverage and representation on the missed validation cases.

Follow-up receipt:
`~/.aura/rlc-evidence/semantic-runtime-retention-acquisition-20260917/trial.json`,
content receipt
`1c422718367537817c32a10460d5304a0bfc00516476027b48cc82b6e796fdba`.

Verification: 133 focused tests passed across graph retention, small-trial
acquisition, world-model rollout and existing world-model/planner suites.
Smoke passed 164 tests with one skip. Lint, compile, governance and layering
passed. These are component gates and do not close the G ledger.
