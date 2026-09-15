# Action completion and checked outcomes

The production goal planner discarded compute, reasoning and reach outcomes.
Each action returned `None`, and the fluid executor's `always_true` predicate
counted it as verified success. An unsupported computation or denied fetch
could therefore complete a goal. A missing verifier also trusted dispatch.

`StepActionResult` now carries completion and an optional existing
`VerificationResult`. Explicit failures survive the planner/executor boundary.
Useful answers without a conclusive check can still complete, but do not add
verified progress. The reasoning amplifier carries the selected answer's
verdict separately from aggregate checking activity. A failed candidate cannot
lend its verdict to a different, unchecked winner.

Unavailable effect observation retries only the observer. A completed action
is not repeated while its required observation is unavailable; goal pursuit
returns a pending-observation receipt without invoking the replanner. This is
an in-process contract, not a claim of durable fluid-executor resumption. The
separate mission path already owns its persisted observation state.

The replanner now targets failed work rather than earlier unchecked success.
It cannot remove a required step or mark it optional to make a plan finish.
Explicitly optional nonterminal steps remain removable. The requirement is
preserved through both option generation and application of legacy repairs.

The first regression pass was 121 tests. After the required-step repair and
the empty-recombination verdict test, 70 focused tests passed. Those tests
include a real exact computation, unavailable observers, denied fetches,
selected-answer verdicts, pursuit outcomes and required-step preservation.
The registered invariant is `agency.action_completion_is_not_verification`.

This change repairs execution and evidence flow used by goal pursuit. It does
not establish broad reasoning gain, independent task assessment, or live
qualification. G09 remains open.
