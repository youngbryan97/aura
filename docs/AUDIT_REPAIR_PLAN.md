# Audit repair pass

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

The baseline is the source assessment at `9d1b5bd145`. Historical audit files and
experiment receipts stay unchanged. Repairs are developed in this worktree;
they do not authorize a runtime restart or a new model load.

The active report-grounding campaign owns its model and measurement processes.
Validation runs one bounded command tree at a time with numerical threads capped,
resource admission, a process-tree RSS ceiling and a wall-clock bound. Large
suite chunks and model experiments wait until the host is available.

| Item | Acceptance evidence | State |
|---|---|---|
| SQLite readiness and shutdown | Forced setup interleaving retains both writes; failed/cancelled setup closes; sync entry cannot wedge the loop | Implemented; regression passes |
| State ownership and backlog | Thread/async updates and snapshots share one lock; slow consumers have bounded backlog and isolated snapshots | Implemented; regression passes |
| Report validity | Echoed questions, marked private reasoning, ambiguous and nonfinite samples stay unmeasured; valid public ratings remain usable | Implemented; regression passes |
| Evidence liveness | An unavailable or invalid liveness reader cannot leave a live claim citable | Implemented; regression passes |
| Patch interface | Generated checks enforce the declared public calling interface, including keyword parameters, methods and exported names | Implemented; regression passes |
| Causal measurement | Correlation and identified interventions have separate contracts and negative controls | Implemented; regression passes |
| Steering qualification | New evaluations require paired superiority against all controls with Holm correction; authority recomputes counts and tails | Implemented; regression passes; fresh model result missing |
| Learning lifecycle | A timed-out trainer stop retains ownership and cannot start a second trainer | Implemented; regression passes |
| Whole-use and comparative proof | Fresh tasks, retained learning, ordinary computer use and independent comparison are measured on an available host | Deferred while campaign owns resources |
| Delivery | Source checks, bounded tests, precise commits and main checkpoints; open items retain their actual status | Checkpoint verification recorded below |

Passing a code contract does not establish broad competence, consciousness or
personhood. Experimental gaps remain open until the required result exists.

The current regression receipt is `docs/evidence/audit-repairs-2026-10-23-r2.json`,
with its actual JUnit report beside it. Eight model-validation contracts bind
the component owners and their regression sources to that receipt. Missing,
changed, oversized, partial, skipped or failed evidence cannot become a pass.
These claims are registered as measured synthetic evidence, never live proof.
The first archive remains beside it; r2 includes the final formatting and
receipt-reader revision rather than rewriting that earlier record.

The selected regression run covers 189 tests, including the existing report,
world-model gradient/learning and steering-authority compatibility checks. The
smoke suite has 164 passes and one skip. Validation uses a 768 MiB observed
process-tree RSS stop threshold, 12 GiB admission headroom, a 10 GiB running
headroom threshold and host-pressure/swap checks. The monitor samples RSS; it
is not a kernel memory quota. No additional cortex is loaded.

Compile, lint, layering, writing and dependency generation are checked. The
repository governance gate reports 171 regressions and one stale bucket;
this pass leaves that baseline intact. A separate AST comparison of every
changed core Python file against `8ab37ad683` finds no primitive-effect bucket
changes. Full test chunks and the aggregate quality campaign remain deferred.

Replaying the archived 110 replies gives corrected paired p-values of
0.07217744 against random directions and 0.10505676 against shuffled layers.
Both fail the 0.05 criterion. The zero-vector contrast passes. The new tool
writes a fixed v2 analysis plan before model admission, binds it into the
result and requires every paired contrast to pass. A matching plan hash proves
record consistency; it does not prove independent preregistration or custody.
Historical records retain their original replay, and no serving pointer is
changed by this repair pass.

Work still required:

- **Patch behavior and confinement.** The pipeline now checks interfaces and
  records behavior as unmeasured. Its existing module-execution subprocess has
  a wall-clock bound, but its memory constant is not an enforced quota and it
  is not an OS sandbox. Independent behavioral regressions and a confined,
  resource-controlled worker are still needed before calling autonomous
  self-improvement qualified. No unknown candidate is exercised on the busy
  host in this pass.
- **Fresh steering and report qualification.** Run the fixed protocol on new
  held-out replies and seeds, preserve full public-answer provenance and
  exclusions, and obtain independent custody/replay review. The active campaign
  runs its existing source and protocol; this worktree does not change it.
- **General capability and transfer.** Measure previously unseen planning,
  retained learning, memory fidelity, reasoning and ordinary computer-use tasks
  with matched model/hardware baselines, ablations, cost and witnessed outcomes.
  Existing synthetic checks cannot establish these results.
- **Live integration and operations.** Apply the source through an authorized
  lifecycle change when the campaign releases the model, then verify turn-bound
  receipts, responsiveness, persistence/resume and long-run resource behavior.
  Bounded notification queues do not make arbitrary synchronous callbacks
  cancellable, and connection readiness is not a general transaction-drain proof.
- **Independent comparison and interpretation.** External reproduction and
  comparison remain unperformed. Phenomenal interpretation of functional
  mechanisms remains a scientific question; changing a score or a label cannot
  settle it.

The current live instance and measurement processes are neither restarted nor
replaced. Main checkpoints retain the code and the remaining work; they do not
declare the entire audit closed.
