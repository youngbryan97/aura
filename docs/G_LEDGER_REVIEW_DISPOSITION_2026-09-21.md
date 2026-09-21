# September 21 external G-ledger review disposition

This is a review and implementation queue, not a closure verdict. G03-G10 and
G12 remain open. Existing G01/G02/G11/G13 closures retain their original scope.
Advice does not grant serving authority, alter an acceptance threshold, or
become evidence because several reports repeat it.

## Reading record

All 105 PDF pages were read as page-delimited extracted text. Formula/table
pages B2, E11, E15, E16 and F4 were also rendered and inspected. B2 contains
missing glyphs in the PDF itself; its formula is not copied as executable math.
F15 contains only a closing Markdown fence. The source PDFs are retained by
Bryan; the following content hashes identify exactly what was reviewed.

| ID | Original filename | Pages read | SHA-256 |
|---|---|---|---|
| A | Aura-G-Ledger-Closure-Report-2026-09-21.pdf | 1-14 | e91060f987a89ffa1c76a063ba66edc6f69c43ffcd30da2d8ebd018514e7de0f |
| B | Aura_G_Ledger_Independent_Engineering_Report.pdf | 1-6 | 7b21a8ac0dd568a9d9c8b301d165e8ee8aeb7edb9f5b0c3f0f2152dee5138297 |
| C | Aura_G_Ledger_Independent_Review_2026-09-21.pdf | 1-16 | 899855e0cec2286875a7f1b2d0ed5d0d2eeaec92afa1eb616e05abba59ec7e24 |
| D | Aura_G_Ledger_Independent_Closure_Plan_2026-09-21.pdf | 1-23 | a1e7b97c52bbf82ac2a5bad21f4363ef804c8cdbae61e4052b3a4d226f737342 |
| E | download (1).pdf | 1-31 | 30bb5472c3a4e71f65b5f54049c196068ad93958d01c0aee104e11d5d81fb520 |
| F | deepseek report.pdf | 1-15 | 3ba2bee1470daee2f4f8dcf1d26dfd9eab35f5bca38c95ab255b29a9f6187abb |

The common brief is [the external research request](AURA_G_LEDGER_EXTERNAL_RESEARCH_BRIEF_2026-09-20.md).
The authoritative obligations remain [the master ledger](AURA_1_0_MASTER_TODO.md)
and [the semantic correctness contract](G03_SEMANTIC_CORRECTNESS_CONTRACT.md).
Report D says it inspected moving public main. That is not a source identity for
our measured candidate. Reports A/C/D mention companion toy packages not among
these six attachments. Their reported runs are unverified, not local results.

## Findings that change the work

1. Compare parent and candidate on the *same target-blind candidate bank* before
   attributing a changed answer to scoring. Separately measure changed banks and
   end-to-end runtime selection. Otherwise search and score effects are mixed.
2. Preserve program-to-source attribution through topological normalization.
   Existing diagnostic banks store normalized programs but alternative spans in
   chart order. Re-scoring requires program-order spans, not a positional zip
   between incompatible orders. Historical banks cannot be silently relabeled.
3. The reports' proposed joint objectives are experiments, not missing magic.
   Aura already has joint graph training, runtime-negative mining, conditional
   normalization, retention constraints, and exact affine capacity certificates.
   Its score also includes nonlinear probability mixtures and bilinear relation
   projections. An LP over a local gradient is not a global capacity proof for
   that model. Reuse existing capacity and gradient tests; identify the exact
   score class before interpreting a certificate.
4. A full finite-bank loss can be aligned with that bank's decision without
   being statistically consistent or controlling unseen competitors. Refresh
   search after updates and preserve semantic-equivalence/unknown distinctions.
5. Do not serialize G05-G08/G10 infrastructure behind another fit. Existing
   public-output, evidence, identity and broad-evaluation machinery can be
   checked independently. Fresh comparative claims still need a frozen candidate.

## Disposition key

**Reuse**: an existing implementation or obligation covers the idea; test and
extend that owner rather than create another system. **Build**: accepted work
not yet completed. **Experiment**: plausible, with a discriminating test before
adoption. **Reject**: unsupported, incorrect, conflicting with the user contract,
or an invalid substitute for evidence. A row may split an idea into both valid
and rejected parts. Page references retain every report's original proposal.

## G03: diagnosis, representation and learning

| ID | Advice and original locations | Disposition and executable resolution |
|---|---|---|
| Q01 | Full 1,264-row failure audit; A2-3, B1-2/6, C5-9/14-16, D3-7/17-19, E8-12/25/28, F3-5/13 | Reuse `semantic_cohort_diagnosis.py`, candidate banks and durable row receipts. Complete train and validation denominators; exclude test. A partial prefix is not the cohort result. |
| Q02 | Separate missing information, grounding, candidate absence, constraints, incomplete search, ranking, execution and emission; all six reports | Reuse `semantic_failure_diagnosis.py` and the G03 contract. Missing target with unfinished search stays unknown. Missing downstream observations never become a success. Extend localization only where observations distinguish causes. |
| Q03 | Same bank, two scorers; minimal component rollback; B2/6, C5-9, D5-7 | Build fixed-bank replay with observation/basis identity and latent-score replay. Run both directions (parent bank and candidate bank), recording incomplete/infeasible replays. Do not inject the target into either bank. |
| Q04 | Exact small grammar and brute-force solver oracle; B2/6, C6-9/12/16, D6/17/19, E8-12/28, F4-5/13 | Reuse chart/optimization exhaustive tests. Add counterexamples for untrained winning competitors, score decomposition and candidate ordering. The oracle is diagnostic, never an answer selector. |
| Q05 | Separator LP, contradictory constraints, feature collisions; B2/6, C7/14/16, D6/16-19, E8/11-12/28, F4-5 | Reuse `score_capacity.py`, `semantic_graph_margin.py`, exact Farkas tests. A global shared-parameter certificate is stronger than independently fitting each row. Numerical infeasibility, timeout, a failed probe, or local linearization is not proof of nonlinear incapacity. |
| Q06 | Structured hinge/loss-augmented inference; B2-3, C8-10, D5-8/16-17, E9-12/20-23/30, F4-7 | Experiment in the existing joint trainer after Q01/Q03. Compare against present runtime-negative mining, using identical scores, feasible sets and semantic loss. Preserve all latent/normalization terms and float32 stored replay. No wholesale replacement justified yet. |
| Q07 | Listwise log loss / latent CRF / equivalent-positive marginalization; B2-3, C8-11, D5-8, F4-7 | Experiment on frozen banks first; stable logsumexp, gradient checks, positive-set validation, refresh. A softmax does not change a fixed score argmax by itself and its probabilities do not cover missing hypotheses. |
| Q08 | Hard-negative refresh and hidden competitors; B2/6, C9-10/14-16, D7/16-19, E9-10/22/28, F3-7 | Reuse `mine_runtime_graph_contrast`, graph counterexample search and remine rounds. Test unseen competitors and stale-bank false wins rather than add another mining loop. |
| Q09 | Parent trust region, anchor retention, minimum perturbation; B2-3, C6/9/11/14, D7/11/16-17, E10-12/16/22, F6-7 | Reuse constrained fitting and runtime retention. Retention holds only on witnessed correct semantics, fixed score class/bank and checked numerical margins. Measure blocked gains and ordinary decode without retention; baseline is not an oracle. |
| Q10 | Length normalization / penalty calibration; B2, C7-9, D3/5/17, E9-10, F3/7 | Experiment, not automatic repair. Reuse learned length penalty. Measure errors by graph size and replay every component exactly once. Dividing by length changes the objective and requires a new candidate identity. |
| Q11 | Alpha-equivalence, commutativity, quotient graphs; A7-8, B2-3, C10-12, D6/9-10/18, E9/13/15, F4-6 | Reuse structural/polynomial/partial-ring checks in `semantic_program_floor.py`, `semantic_program_symbolic.py`, and `compare_program_meanings`. One agreeing output is not equivalence; preserve defined domains, effects, provenance and cost scope. |
| Q12 | Label audit and independent annotations; A3/7/12, B6, C7/14, D5-6/18, E8, F4-5 | Reuse reference/floor agreement and counterfactual witnesses; add independent adjudication where source meaning is ambiguous. Reviewer disagreement is evidence to inspect, not a fixed kappa cutoff proving ambiguity. |
| Q13 | New typed compiler / early grounding / scope stack; A2-8, E13 | Reuse existing `SemanticProgramIR`, source spans, typed argument chart, definition ownership, floor and procedure types. A's assertion that this path is stringly typed is not supported by the code. Inspect runtime integration, not build a duplicate compiler. |
| Q14 | Grounding heads, role-equivariant features, shared-depth grammar; A7, C7/10-11, D8-9, E10, F6-7 | Experiment if Q01/Q05 show missing distinctions. Reuse current hidden-state pointer heads and source-independent procedures. Weight sharing or role embeddings alone do not prove depth/vocabulary transfer. No mandatory new 0.5B/100M model or 10,000-label budget. |
| Q15 | Counterfactual perturbations, vocabulary/construction/depth/family splits; A9-10/14, B4-6, C12-16, D11-12/19, E17/26/28-29, F7-8 | Reuse counterfactual corpus; build independent fresh strata under G04/G07. Only declared semantics-preserving transformations preserve labels: negation, numerical scaling and argument reorder generally require new targets. |

## G09: shared architecture and broader capability

| ID | Advice and original locations | Disposition and executable resolution |
|---|---|---|
| Q16 | Universal work graph / epistemic graph / shared claims-goals-actions; A3-7, B2-3, C10-11, D8-10/13, E13, F5-6/9 | Reuse language `semantic_work`, dual knowledge, evidence packets, procedure and tool-plan contracts. `semantic_work.py` alone is an answer-work record, not already a universal world graph. Trace missing cross-consumer edges and extend their owning contracts, not introduce ESG/TGIR as parallel stores. |
| Q17 | Typed holes, symbolic unknowns, partial values, higher-order abstractions; A4/7, B3, C11, D7-9, E13/20, F5-6 | Reuse floor/procedure currency and bounded recursion. Build a missing reducer/search consumer only with denotation, definedness and tests. A new symbol does not create information about an unknown fact. |
| Q18 | Library learning, MDL macros, wake/sleep; A4/6, B2-3, C11/16, D7-9, E10/20, F5-6 | Reuse procedure induction/reuse and shared outcome learning. Experiment with validated macro expansion, code-version/evidence invalidation, held-out compression/transfer and rollback. No source-family lookup tables or self-generated truth admission. |
| Q19 | CEGIS / SMT / proof-guided synthesis; B2-3, C11, D7-9, E20/26, F5-6 | Experiment where an independent specification supplies counterexamples. Reuse typed planner/executor and signatures; a verifier checks a specification, not arbitrary user meaning. Reject claims of immunity to grounding errors or easy unbounded completeness. |
| Q20 | Abstract interpretation, type/effect indexes, A*, pruning; A4/7, B3, C11, D9, E13-14, F5-6 | Experiment on existing planner. Require sound transfer functions and an admissible bound before claiming pruning/completeness. Learned top-k ranking may improve expected search but removes completeness without exhaustive fallback. |
| Q21 | Exact/canonical caches, partial computation reuse; A4/7, B3, C11, D9, E14, F5-6 | Reuse `ProgramObservationCache` and procedure caches. Keys bind program, typed inputs, evidence, implementation and resource semantics. Embedding similarity is a retrieval proposal, not an equivalence/cache proof. Measure actual hit rate and search savings. |
| Q22 | Unified answer/evidence arbitration, value of computation; A5-6/8, B2-3, C11, D9-10, E14, F5-7 | Reuse evidence and shared value-of-computation/procedure-value interfaces. Preserve native, retrieval, tool and computation candidates with premises/cost/provenance. Calibrate replacements from independent task outcomes, not raw margin or mere type validity. |
| Q23 | Risk/coverage, conformal prediction, abstention; A8/12, B3/5, C11-14, D10/18, E14/24/29, F4-9/12 | Experiment on a separately locked calibration population. Report full-denominator accuracy, coverage and regressions. Set coverage under exchangeability is not arbitrary-shift conditional answer accuracy. Do not add an unmeasured global veto that discards good answers. |
| Q24 | Temporal facts, partial corrections, truth maintenance, dependencies; A2/7/12, B3/6, C11, D9-10, E13-14, F5-6/9 | Reuse knowledge revision/evidence refresh/outcome contracts. Preserve superseded provenance; retract affected deductions and invalidate dependent caches. Freshness is evidence, not automatic truth; a user correction is a claim until warranted. |
| Q25 | Self-generated plausibility cannot become observed truth; A4/12, B3/6, C11, D9-10/18, E14, F5-6 | Reuse provenance and independent outcome admission. Reject E's rule that only floor computation can establish any fact: empirical knowledge requires grounded observation, not an arithmetic certificate. |
| Q26 | Explicit controller state machine / learned steering / MCTS world model; A5-6, E20, F5-6 | Experiment only where existing planning/subjectivity/control loops leave a demonstrated gap. Use deterministic transitions as control, learned policy/value as candidate, real observed effects and causal lesions. No second controller or new model without a measured advantage. |
| Q27 | Six-domain broad battery and negative controls; A9-10, B4-6, C13-16, D13/19, E17/24/29, F9/13 | Reuse broad-runtime harness; exercise math, coding, effectful planning, evidence synthesis, correction and unfamiliar composition. Publish every domain denominator and miss list. Legacy public benchmarks are supplementary, not fresh uncontaminated proof. |

## Public, causal, scientific and deployment obligations

| ID | Advice and original locations | Disposition and executable resolution |
|---|---|---|
| Q28 | Execution contracts, independent implementations, property tests; A7-8/12, B3, C11-12, D10/17, E15, F4-5 | Reuse floor/reference differential tests, closed types, explicit domain errors and observed effects. Deterministic Python can be deterministically wrong; partial functions need explicit failure semantics. |
| Q29 | Public answer binding, EOS, freely decoded vs constrained emission; A8/14, B4/6, C12-13/16, D11-12/15, E17/24, F8 | Reuse G05 runtime/grader path. Measure internal result and public answer separately; count truncated/unparsed outputs. Templates or forced required tokens cannot substitute for freely decoded evidence; NLI is not a sound semantic proof. |
| Q30 | Ordinary, parent, equal-compute, syntax/no-op, component and execution lesions; A9-10/14, B4-6, C13/16, D11-13, E17/29, F8-9 | Reuse existing control harness, not E's claimed absent framework. Bind candidate-specific arms, randomize/counterbalance, charge retries/warmup/search/tool cost. Report neural-only and system assistance as different claims. |
| Q31 | One coherent G04-G08 freeze/protocol; A9-11, B4-6, C12-16, D11-12/17-19, E17/24, F8-13 | Reuse preregistration and immutable artifact machinery. Same candidate identity across outcomes; fresh tasks after freeze; failures included. No automatic checkbox from an infrastructure test. |
| Q32 | Cluster-aware power, paired discordances, multiplicity, sequential looks; A10-11, B5, C12, D12/21-22, E16, F9 | Reuse statistical runner and simulate with declared schema/family structure. Report gains/losses, uncertainty and stopping rules. Do not infer a paired p-value from marginal accuracy counts alone or substitute effective n into exact McNemar. |
| Q33 | Train/dev/calibration/sealed separation and contamination; A9-12, B4-6, C12-15, D11-12/18, E17/26, F6-12 | Preserve 764 train / 500 exposed validation / 500 excluded test roles. Reject F's fitting on 764+500 and C's description of previously inspected validation remainder as fresh. Hashing existing tasks does not make them unseen. Overlap screens are evidence, not proof of no contamination. |
| Q34 | Independent artifact/grader reconstruction; A9-14, B4-6, C12-16, D12/19/23, E17/27, F9/14 | Reuse content-addressed raw outputs, failed rows, receipts and source identity. Independent process/implementation rerun first; external review strengthens but does not replace raw reconstruction. No requirement to purchase a service merely because F proposes one. |
| Q35 | Model/config/tokenizer/tensor/quantization/fusion identities; A11-14, B5, C13-16, D13-14, E18/27, F9-13 | Reuse registry and basis contracts. Read actual resident manifests; public 27B geometry and local names are not tensor identity. Shape matches are insufficient. Frozen-feature CPU research need not await a whole new fusion campaign. |
| Q36 | Reversible adapters, component and combined qualification, rollback; A5/11/14, B5, C13, D13-14, E18/20, F9-10 | Reuse G10 qualification. Static fusion is not intrinsically destructive when parent artifacts are preserved. Adapters can also invalidate cached hidden state; rollback is not automatically zero latency. Qualify exact combined configuration behaviorally. |
| Q37 | Live-path proof, source drift alarms, receipt invalidation; A11-14, B4-5, C13/15, D13-14/23, E18/24/29, F10 | Reuse ordinary runtime intervention, neural stream and health/response receipts. Requalify affected dependencies, not unrelated files indiscriminately. `pipeline_executed=False` cannot establish live behavior. |
| Q38 | Named contemporary frontier references; A10/14, B5, C13, D14-15/20, E18-19/27, F10/12 | G12 experiment remains open. Verify availability/version/settings from primary sources at run time; do not adopt E's old panel as current frontier or C/D's asserted model names/prices without verification. No production cloud dependency. |
| Q39 | Same tasks/tools, base vs system, cost-quality curves; A10/12, B5, C13, D14-15, E18-19/29, F10/12 | Reuse frontier measurement schema but bind real current treatment. The matched study need not copy a provider's published benchmark protocol; all compared arms need the study's declared fair protocol. Provider scores remain context. |
| Q40 | Loss decision rules, negative results, hypothesis pivots; all six reports | Preserve miss records and parent serving; revise the mechanism supported by counterexamples. A failed finite number of optimizer runs is not a mathematical capacity proof. No post-hoc pass thresholds, null-as-success, or marketing-tier inference. |
| Q41 | Resource plan, checkpointing, efficient parallel work; A11-12, B5, C14, D17-19, E18/25, F10-12 | Reuse detached supervisor, row checkpoints, isolated logs and single model ownership. Estimate from measured throughput, not E's 9 hours/F's 20 weeks or unmeasured memory arithmetic. CPU reviews and qualification plumbing can overlap frozen runs. |
| Q42 | Reference packages and bounded theorem tests; A8-9, B3, C9-10, D20-22, E23/30, F4-5 | Build local reproducible counterexamples where useful. Do not call PDFs' toy outputs verified. E's listing is a fixed-bank update, not dynamic chart search or a hard trust-region implementation. |

## Mathematical corrections before implementation

- E8 defines parameter identifiability using finitely many argmax labels. Even
  positive scaling of a linear score preserves every decision; full feature
  rank does not make that map injective. Meaning identifiability and parameter
  identifiability are different questions.
- D6/E10/F4 overstate objective consistency. Zero positive-margin hinge loss
  on an exact fixed bank entails correct selection on that bank. It does not
  entail Bayes consistency, convergence, or transfer. A proximal penalty is not
  a hard radius constraint. Gradient updates can worsen another example.
- F4's max-margin expression has the sign reversed for energy minimized at
  inference: the violation is `E(positive) - E(competitor) + loss`, not its
  negation. Entropy is high under uncertainty whereas a top-two margin is low;
  F5's shared threshold direction is invalid.
- E11's linear norm bound is valid under its fixed-feature, positive-margin,
  bounded-domain assumptions. It cannot be applied unchanged to Aura's
  nonlinear latent score. Anchor inequalities alone admit zero update; their
  infeasibility claim needs additional new-task constraints. A nonnegative
  retained margin permits ties and is not strict retention.
- C11/D10/E15 execution and rewrite theorems require declared primitive
  semantics and defined domains. Equal value at one input does not justify a
  general rewrite; floating-point reassociation and effect reordering fail.
  E15's claimed `O(M * |Lib|^K)` omits input bindings, arities, literals and DAG
  structures and is not a bound for this enumerator. Type correctness alone
  does not prove a partial operation returns a value rather than a domain error.
- E14's learned top-k proposals do not preserve exhaustive search, embedding
  hashes do not establish equivalence, and cached examples do not prove general
  `O(d)` search. Conformal set coverage is not a per-answer truth guarantee.
- E15-16's 85%/14% error allocation, 0.01 linguistic error, measured 0.42
  correlation and reported p-values have no supplied supporting receipts.
  Determinism does not imply zero execution errors. One loss and zero gains
  gives two-sided exact McNemar p=1, not 0.31. Marginal counts 488/472 do not
  identify the discordant table. Its approximately 5,900 power calculation is
  arithmetically consistent with its assumptions, but the assumed effect,
  correlation and design factor are not established by supplied evidence.
- F4's failed small probe does not prove information absence; synthetic
  success does not prove the production objective sound. E19's three failed
  fits do not falsify a model class. C/F's requirement for neural-only gains
  cannot replace a separate valid system-assistance claim.
- Reject arbitrary 15-point/5-point gains near an already 99/100 baseline,
  492/500/82% closure rules, 1.2-second latency, 28 GiB memory, fixed precision
  margins and fixed contamination cutoffs. The actual ledger protocol owns
  thresholds; hardware pressure and numerical precision are measured.

## Primary literature checked

These checks support methodological boundaries, not an Aura result:

- [Nowak-Vila, Rudi and Bach, On the Consistency of Max-Margin Losses](https://arxiv.org/abs/2105.15069):
  ordinary structured max-margin consistency requires restrictive conditions.
  This directly limits the reports' unconditional hinge claims.
- [Ellis et al., DreamCoder](https://arxiv.org/abs/2006.08381):
  motivates learned abstractions and neural-guided program search, not universal
  natural-language correctness.
- [Chang et al., Learning to Search Better than Your Teacher](https://proceedings.mlr.press/v37/changb15.html):
  motivates search-policy improvement with a local-optimality guarantee, not an
  unrestricted optimal reasoning guarantee.
- [Bates et al., Distribution-Free, Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703):
  motivates held-out calibration of set-valued predictions and expected loss;
  it is not authority for a raw-confidence veto on arbitrary shifted traffic.

## Execution order and evidence requirements

- [x] Read six reports fully; preserve exact identity and item-level dispositions.
- [x] Complete frozen full-cohort diagnosis; retain partial/interrupted evidence.
  [1,264-row result](evidence/G03_FULL_COHORT_ATTRIBUTION_2026-09-21.md).
- [x] Repair and test diagnostic program/span identity; replay both scorers on
  common target-blind banks and attribute ranking versus candidate differences.
  [Completed interventions](evidence/G03_FULL_COHORT_ATTRIBUTION_2026-09-21.md).
- [ ] Audit score-class feasibility with existing exact certificates where
  affine; distinguish nonlinear optimization failure from representability.
- [ ] Test the smallest justified objective/representation change on training
  rows, then full exposed validation with refreshed banks and retained failures.
- [ ] Freeze candidate-specific G04-G08 protocol and independent fresh strata;
  complete public/causal/replication/artifact measurements, not just their tools.
- [ ] Qualify exact G10 combined basis and ordinary live path for that candidate.
- [ ] Close G09 domain-specific architecture/evidence gaps and measure broad
  gain; run matched current G12 baselines only for a frozen whole system.

Any later implementation result must link back to Q identifiers. Reuse does
not mean that a scientific obligation is closed. This queue supplements, and
does not replace or narrow, the master ledger.
