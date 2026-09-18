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

## Pairwise likelihood development trial

An opt-in likelihood objective continues learning from correct but uncertain
training comparisons. Every proposed update still checks all retained
inequalities at the stored float32 precision. The original squared-deficit
objective remains the default.

Eight training rows remained equivalent. The same eight exposed validation
rows improved from six equivalent programs to seven, with no loss among
previously correct cases. The remaining row refused decoding with
`typed_argument_chart_empty`. This is a measured completion failure, not an
unknown diagnostic. The v1 trial reporter classified it as `unmeasured`;
v2 separates decoder refusals from unknown semantic verification and no
longer calls annotated-graph feasibility runtime target reachability.

The run accepted ten steps over 223 constraints. Pairwise loss decreased from
0.0005533504109949285 to 0.00005139420917746016. No retained positive inequality
regressed. These training inequalities do not guarantee held-out chart
coverage or correctness.

Original immutable receipt:
`~/.aura/rlc-evidence/semantic-runtime-retention-likelihood-20260917/trial.json`,
content receipt
`8eb6c6d96ef229fd8106205258db98e0f7d68ae714b27cea3741d8f8c3a5b9d9`.

Focused verification: 39 objective, checkpoint and trial tests passed. A
separate 53-test run covered floor observation reuse, counterexamples,
runtime retention and the small-trial reporter. No serving activation,
fresh transfer or G03 closure follows from this trial.

## Reusing floor computations during search

Counterexample search now shares bounded, invocation-local observations
across competing operation charts. Keys include the exact program, typed
inputs and fuel allowance. Only reference-checked values and undefined-domain
observations are retained. Execution errors are retried. Returned evidence
cannot mutate cached receipts. Finite agreement still does not establish
equivalence.

The focused tests compare cached and uncached witness receipts, different
input and fuel identities, bounded eviction, domain failures and mutation
isolation. The diagnostic replay's first training example reused 65
observations and performed seven fresh executions. That is avoided duplicate
work, not a measured end-to-end speedup or a general search-completeness claim.

## Refusal localization

The train-only fit was reproduced with the same final loss. Its non-serving
candidate and component-restoration observations are retained under
`~/.aura/rlc-evidence/semantic-likelihood-diagnosis-20260917/`.
The cataphoric source `041682cce3dc244504989f04531772a51e939aac8cda24230057db847e6de5f4`
has a sequence input, but the fitted classifier's sixteen selected charts
contain only `mul` and `idiv`. Restoring only the parent operation classifier
removes the refusal. Restoring only relation or argument heads does not.
The restored classifier still gives an incorrect program; this intervention
localizes refusal, not the full interpretation error.

Two retained operation labels still refuse. Four labels produce a program
whose result is 4 where the source program produces 5. Increasing label count
alone therefore does not solve the semantic failure. No variant is promoted.

The next small trial retains source operation labels from 48 training rows
while mining full graph competitors from eight. It reuses the source-label
constraints already implemented by the compiler. This separates cheap
classifier retention from expensive graph search and does not fit validation.

That trial completed with 723 inequalities satisfied, training 8/8 and
validation 6/8. There were no gains, and the cataphoric decode still refused.
The broader source-label retention did not solve the failure. Receipt:
`~/.aura/rlc-evidence/semantic-wide-operation-retention-20260917/trial.json`,
content receipt
`e7cf55399e659737b8d24364b3bff7039aec2f9a0cc8c667046968ebc256cb56`.
This negative result does not authorize a larger trial or serving change.
