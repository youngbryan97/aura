# The Frontier-General Arc — breaking the intelligence ceiling honestly

**Mandate (Bryan, 2026-07-14):** "Let's break through that ceiling. Let's make
her frontier-general. We did it with Φ. Let's do it here too."

Same discipline as the Φ arc: name what is truly impossible, reframe to what
the goal actually requires, build the strongest honest machinery, and make
the claim falsifiable with a measured trend line.

## The wall, stated exactly

Making the resident cortex's *weights* match a frontier model's raw pretrained
capability is off the table — the pretraining compute gap is orders of
magnitude, LoRA does not close it, and self-training on unverifiable outputs
collapses models. That wall is real and stays real.

## The reframe

The goal was never frontier weights. It is **frontier-general Aura** — the
organism, not the cortex. The field has proven systems beat weights three
ways, and she already owns the seeds of each:

1. **Test-time compute scaling** (the o1/R1 lesson): reasoning quality
   scales with inference compute spent per problem. EXISTS in embryo:
   `core/brain/reasoning_amplifier_v2.py` already does
   problem-model → N attempts → verify → search/repair → judge → receipt,
   with demand-based budget allocation coupled to Φ/surprise
   (`_resolve_compute_budget`). Wired live via `reasoning_strategies`.

2. **The verifier boundary is movable** — the deep lever. "Frontier-
   competitive on *verifiable* reasoning" treats verifiability as fixed; it
   is not. Factuality → retrieval+citation (CitationEngine exists). Agency →
   outcome ledgers (exists). Prediction → **the universal verifier**: reality
   grades every forecast (`expectation_feedback` exists as the seam). Taste →
   rubric ensembles + debate (courtroom exists). The ceiling rises exactly
   as fast as verification expands — and that rate becomes a number.

3. **Compounding that cannot collapse**: verified reasoning traces distilled
   into procedural memory (reusable strategies indexed by problem shape,
   injected at inference — zero collapse risk), periodically distilled INTO
   weights via the proven CRSM→LoRA loop. R1 showed verifiable-reward
   training transfers to general reasoning; her loop can compound it.

## Operational definition (the falsifiable goal)

"Frontier-general" := parity with a named frontier model on a broad,
contamination-resistant, regularly refreshed benchmark battery (reasoning,
knowledge, coding, agency, writing, calibration) at matched per-task budget.
Tracked through a checked-in **gap-to-frontier evidence index** (the Φ-report
pattern: `artifacts/frontier_gap/latest.json`). V5 stores complete outputs and
receipts in atomically publish-once, content-addressed local blobs and keeps a
bounded hash-chained index. This is local tamper evidence, not an externally
witnessed append-only transparency log. Endpoint movement is reported
separately from a trend claim. A closing
trend additionally requires at least five matched independently challenged
runs, unique held-out challenges, a minimum effect, a confidence interval
excluding zero, a significance threshold, and consecutive non-worsening
results.

This is the program target, not the scope of the current battery. The current
v5 instrument covers four deterministic classes only: integer arithmetic,
short-chain ordering, restricted single-function Python, and fixed
short-answer facts. It is a diagnostic of those classes and of the evidence
pipeline. It cannot establish broad reasoning, knowledge, coding, agency,
writing, calibration, or general frontier parity. Expanding to the operational
definition requires fresh held-out suites across all six named domains,
contamination controls, repeated matched-budget runs, and independent review.

## Build phases (each: functional, live, governed, tested, pushed)

**P1 — Verifier Foundry** (the ceiling-mover; genuinely new):
`core/brain/verifiers/foundry.py`. Extends the existing registry
(`core/brain/verifiers/registry.py`, 6 domain engines, binary hard-gate + soft mean)
with: per-verifier **measured reliability** (score every verdict against
later ground truth / spot audits; Brier-style ledger), reliability-weighted
verdict folding, an **admission gate** — a domain may enter the self-training
loop only when its verifier's measured reliability clears threshold — and
three new verifier classes: PredictionResolutionVerifier (via
expectation_feedback; reality as ground truth), OutcomeLedgerVerifier
(long-horizon agency), RubricEnsembleVerifier (weak-verifier ensembles for
open-ended quality). Governed writes; tamper-evident reliability ledger.

**P2 — Procedural Memory Compiler**: post-task reflection distills
VERIFIED traces into strategy playbooks (`reasoning_memory` today stores
failure modes; this adds success schemata indexed by problem features),
injected into the amplifier's problem-model stage; periodic batch
distillation of the best playbooks into LoRA via the CRSM loop.
Compounding without collapse.

**P3 — Gap-to-Frontier telemetry**: build a versioned measurement system with
fresh-generation templates, sealed evaluation, independently trusted signed
reference evidence, content-addressed candidate model and source windows,
full per-item receipts, typed ledgers, and artifact pinning. The current v5
four-class diagnostic implements these admission mechanics only. P3 remains open
until the battery covers the six-domain operational definition and repeated
independently verified live runs establish a trend.

**P4 — Curriculum fusion**: FDE + curiosity engine propose edge-of-
competence tasks; foundry-admitted verifiers resolve them; verified
solutions feed P2 playbooks and CRSM training sets. The Absolute-Zero
pattern, gated by P1 admission so unverifiable domains cannot pollute
weights.

**P5 — (env-gated) Teacher distillation lane**: when reach/cloud is
enabled, harvest frontier-teacher traces on HER real task distribution,
filter through the foundry, distill. Honestly labeled imported capability,
compounded permanently into her own weights on her own curriculum.

## Honest expected shape

Not "her weights become frontier." Rather: the organism, spending time and
verifiers wisely, closes the measured general-capability gap class by class
— fastest where verification is strong, slower as the foundry admits new
domains — with the trend line checked in. That is the strongest honest
formulation, and unlike the slogan, it can actually be won.

## Session log

- 2026-07-14: arc chartered; recon complete (amplifier v2 budget seam,
  registry shape, eval arena, CRSM entry points identified). Next: P1.
- 2026-07-14 (later): **P1 SHIPPED** (e1b462bc). Foundry live: reliability
  ledger (Wilson-pessimistic, false-pass-bounded), registry folding
  reliability-weighted (hard gate untouched), training pipe gated by
  domain_admitted() with revocable seed admissions, booted in aura_main,
  21 known-answer tests. First live contact caught a real leak: the code
  engine reports checked=True headless when it cannot actually execute —
  foundry measured false_pass_ub=0.879 in two grades. Follow-up: fix code
  engine checked-semantics (report checked=False when execution
  unavailable). Next: P2 Procedural Memory Compiler, P3 gap-to-frontier
  telemetry (needs new verifier classes: PredictionResolution/Outcome-
  Ledger/RubricEnsemble to start moving the boundary beyond the seeds).
- 2026-07-14 (later): foundry finding CLOSED (bd356c5b). Assert-bearing
  blocks now execute in the symbolic sandbox; unexecutable claims demote a
  passing static verdict to checked=False. Live regrade: code engine 4/4
  correct, zero false passes (was the 0.879 poison case). The
  measure→catch→fix→remeasure loop worked end-to-end on its first cycle —
  the foundry is functioning as the arc's immune system.
- 2026-07-15: P3 evidence semantics first corrected before accepting its first
  capability result. A deterministic pipeline control had been recorded in
  the same shape as an Aura-model score and its runs were eligible to enter
  the capability trend. The intermediate v2 artifact used neutral candidate fields,
  isolates synthetic controls in a claim-ineligible ledger, keeps the real
  Aura capability ledger separate, records per-run execution failures, and
  can fail closed with `--require-capability-evidence`. Running the tool while
  the desktop owns the model lane remains control-only until an authenticated
  resident-amplifier proof channel exists; no desktop-resident capability
  result is claimed yet. The measurement call is also read-only through the
  amplifier and verifier registry: it cannot update exact-answer caches,
  playbooks, preference pairs, reasoning episodes, self-improvement captures,
  or Foundry weights while later battery items are still being scored.
- 2026-07-15 (v3 closeout): independent review found a second scientific
  defect in v2: substring graders and unnamed hard-coded "reference anchors"
  could manufacture both correctness and a frontier gap. V3 removes embedded
  references completely. Math, short-answer reasoning, and facts use exact
  normalized truth; code is parsed through a restricted AST and executed over
  hidden cases without disclosing expected outputs. A frontier gap is `null`
  until a named model/source/date artifact matches the exact battery seed,
  class coverage, budget, and per-class run receipts. Capability eligibility
  is decided only after every item completes with a non-empty answer and an
  amplifier verifier receipt, and only on a clean full-commit/full-tree source
  identity. Failed real-model runs are retained in a rejected-attempt ledger;
  they can never enter either the capability or control trend. Historical v2
  anchor gaps are quarantined to `null` during migration.
- 2026-07-15 (v4 trust-chain closeout): a second independent review found
  that v3 references were still self-attested, candidate evidence was not
  bound to exact model material, procedural-memory retrieval could leak prior
  answers, stored capability summaries could be fabricated, and the tiny
  battery was described too broadly. V4 requires an explicit Ed25519 trust
  root and verifies a signed battery manifest, model identity, per-item answer
  and execution receipts, exact budget, unique response IDs, and reproducible
  grader outcomes. Candidate admission now requires unchanged clean source
  before and after the run, a commit-to-tree check, unchanged full-file model
  manifests before and after inference, complete verified item receipts, and
  sealed evaluation with cache and playbook retrieval disabled. Capability
  ledgers retain the complete evidence snapshot and revalidate it on restore;
  summary-only or altered history is rejected. The checked-in artifact pins
  every execution component to its current content. No Aura-model run has yet
  passed this v4 contract, and the instrument is explicitly only a four-class
  deterministic diagnostic. Broad frontier-general capability remains open.
- 2026-07-15 (v5 protocol remediation, source candidate): adversarial review
  established that v4 still let the coordinator manufacture generation
  metadata, did not pin hidden truth or grader implementations, declared
  matched budgets rather than measuring them, imported measurement code before
  source capture, conflated generic verifier prose with correctness, normalized
  signed payloads, silently replaced corrupt ledgers, overstated endpoint
  movement as a trend, repeated effective samples, and embedded unbounded
  evidence snapshots. V5 moves capability admission behind separately pinned
  Ed25519 actors: challenge issuer/reference evaluator, generation worker,
  correctness verifier, run coordinator, and repository release authority.
  The exact externally supplied pin set is itself hashed into the reference and
  candidate run envelopes. Reusing a signer identity or public key across any
  trust role is rejected even when every individual signature is valid.
  A commit/reveal challenge
  is issued only after candidate source/runtime and reference runtime identities
  freeze. Every generation receipt binds the unpredictable run nonce, unique
  item, prompt/output hashes, exact commit-bound source identity, effective
  runtime, stable candidate-model window, exact decoding parameters,
  tokenizer-measured token use, candidate count, generation calls, worker wall
  time, and zero tool/network/cache use. A separate coordinator observation
  records the actual worker process and IPC wall time under the hard deadline;
  the run signer binds every worker receipt and every supervisor observation.
  The signed task
  specification commits to hidden answers/cases and grader source digests and
  names a verifier whose key, implementation, and release are independently
  pinned. Correctness receipts are separate from execution receipts, so exact
  short answers do not need generic long-form verifier prose.

  The CLI is now a standard-library-only launcher until it verifies a clean
  commit and creates a fresh permission-hardened clone; measurement imports
  occur only in that child, internal symlinks must remain inside the checkout,
  and source identity is revalidated before and after the run. Trusted source
  identity additionally requires a signed repository release attestation and
  ancestry proof. The effective runtime records exact worker/Python executable,
  library-lock, platform, model, tokenizer, adapter, steering, and prompt-template
  identities. Reference evidence now carries its complete stable model window,
  not only a claimed runtime digest. Challenges have bounded lifetimes and must
  be fresh at candidate admission. In-process MLX execution remains an explicitly
  unattested diagnostic. Claim-bearing execution requires a fresh signed worker
  plus independent verifier and run-signer commands. Malformed or unknown-schema
  ledgers fail closed and are preserved to a quarantine file instead of
  overwritten. V5 evidence is stored in bounded hash-chained indexes over
  atomically published content-addressed blobs, including full rejected outputs.
  Artifact updates use an interprocess lock and exact-head compare-and-swap so
  concurrent valid runs cannot erase one another. This does not replace future
  external transparency witnessing or independent replication. The
  checked-in `latest.json` remains historical v4 until a clean post-commit v5
  control run regenerates it. No v5 Aura-model result or frontier trend is
  claimed by this source change.
