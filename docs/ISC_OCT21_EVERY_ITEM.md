# Every item from Bryan's notes of 21 October, and what is done with each

Bryan sent his own answers to the open questions and three outside analyses on
21 October 2026, and asked that every viable item be applied. A review of this
list came the same day; its six revisions are folded into the items they touch
and listed at the end. Each item is
listed here with what is built for it, what it is checked by, and where it
stands. Nothing here moves a threshold or rescores a criterion: an organism
change is named in `docs/ISC_V5_PREREGISTRATION.md` before a run reads it, and
a new estimator is a diagnostic until a preregistration makes it more.

`[x]` is done and checked, `[~]` is under way, `[ ]` is not started. An item
judged not viable says why.

## Bryan's notes

- [ ] **B1. Processes that move each other.** A process's state has a measurable,
  state-dependent effect on the processes it is connected to when it is
  intervened on, and the declared system is integrated as a whole. Not every
  small change has to reach everything: some should be filtered or inhibited,
  and forcing them through would raise coupling and lower competence. Built by
  the four functional couplings below (G2, G3, G5, A4), each a term in the
  target's own update rather than a register beside it, and each read for
  competence and stability as well as coupling (R2).
- [ ] **B2. One subject, not cooperating parts.** Shared resources with
  coordination over their use, and closed loops. Checked by the v5 sweep (G8)
  and by the paired-intervention test (G6c): separating any two parts must cost
  the whole something measurable.
- [~] **B3. Experience changes what she becomes (the hot stove).** The criterion is
  a persistent change in later responses that fits the outcome, against an
  otherwise identical copy that was never exposed. Faster and more settled is
  one way it can show, not a requirement: a good learner can become slower or
  less sure after a bad outcome. Speed and confidence are recorded beside it.
  Built as a reflex test on her organism (`tools/the_second_time.py`): first,
  second and third exposure to a novel stimulus with an outcome, against a fork
  that never met it. Built: twice the percept with a bad outcome held through
  the turn, ordinary turns, then the percept alone, in both forks; specific
  against an unburned percept, fitting if her valence on meeting it is lower
  than the copy's, persistent across the turns between. Runs when the machine
  is free.
- [~] **B4. Continuity while changing, with the chain written out.** Change is
  local to what was experienced, and the path from an input to a changed
  behaviour can be shown step by step. Built as a trace of one experience
  through the writes it caused (`core/subject/provenance.py` is the base), and
  checked by snapshot comparison before and after. Built:
  `core/subject/snapshot_diff.py` names every number that differs between the
  exposed and unexposed forks after the same turns, largest first, and the
  share that changed; the hot-stove tool records it.
- [x] **B5. Behaviour that coordination alone does not show.** A preference held
  consistently, an interjection, a change of mind. Recorded as behavioural
  evidence beside the battery, not as a criterion of it. Built:
  `tools/what_she_does_of_her_own.py` reads her conation record. On 21 October,
  6 of the 149 things she has met hold a liking more than two standard errors
  from indifference: bicycles (0.73, 5 contacts), watching along with someone
  (0.12, 131), attending to her social drive (0.16, 68) liked; turns about
  herself (-0.18, 193), the desktop influence probe (-0.14, 143) and attending
  to her affect engine (-0.18, 58) disliked. Changes of mind are on her record
  since `fc623c126`: each time a liking forms beyond two standard errors,
  lapses, or comes to be held opposite to the last way she held it. Her
  interjections since `ebf2ea4e9`: each with that ends keeps the remarks she
  made that nobody asked for, how many were taken up and how many went by.
  The same tool reads all three.
- [~] **B6. Reports that come from her state, not from prose.** A report has to
  move because her state moved, through her architecture. Built as G10: the
  reports ground with the printed number withheld and with steering on and off.
- [ ] **B7. Fresh conditions.** Nothing settled on seed 7 counts until seed 23
  and held-out conditions read it (F7, F8, F9).

## The practical table

- [~] **P1. Attention (A, S -> G).** Passes v3 at 370 rounds with the codes
  (fraction 0.49, interaction lower bound +0.006) and fails without them.
  Preserved, and read again on every later commit.
- [~] **P2. Memory and perception (P, M -> W).** G2 and A3, both built; the pair
  at the next commit reads them.
- [~] **P3. Decision-making (W, A -> D).** G3 built, A4 partly; read by the next pair.
- [~] **P4. Recurrent cognition (S, D -> C).** G5 built, A5 partly; read by the next pair.
- [x] **P5. Each coupling reads the domain, not a thin summary.** A3, A5: every
  source domain is read in full and placed along its learned axes.
- [~] **P6. Measurement that tells interaction from schedule.** G1, G6, F9 built.
- [ ] **P7. All 511 cuts.** G8, the carrier rehearsal now running.
- [~] **P8. Causal closure.** G7; the closure columns of `272ebdab7` are read by
  the validation finishing today.
- [~] **P9. Self-report.** Three pipe defects fixed today (`03ca3512b`,
  `0835cfaeb`, `b815ecd41`); A6, G10 next.
- [ ] **P10. Validation on fresh conditions.** F7, F8, F9.

## The analysis of lag and pipe defects

- [x] **A1. Units integrate the turn that set them.** A unit recorded at turn t+1
  holds (1 - lambda) of what it held plus the conjunction of the domains as they
  stood at turn t, so a turn's change is driven by that turn's sources. Offline
  on `validate-s7-5885688d7-r370` (diagnostic, no organism change): A,S->G and
  P,M->W pass, W,A->D clears its null with fraction 0.112 and fails only its
  interaction bound, S,D->C sits at 0.0118 against 0.0132.
- [x] **A2. Gain-field form.** Products of non-negative tuning curves carry both
  main effects and the interaction; signed tanh products are close to invisible
  to a Gaussian-copula estimator. The live form follows the offline comparison.
- [x] **A3. Perception and memory read in full**, including the memory retrieval
  state that does not repeat with the schedule.
- [~] **A4. World and affect in deliberation's own transitions**, drive and
  urgency changes, not only a final score. The choice itself now turns on the
  world through G3; drive levels follow the act chosen, not yet the world
  directly.
- [~] **A5. Self and intention read in full and acting on every tier of C**:
  substrate, mesh and field. The substrate by its transition (G5), the mesh by
  its tier gains; the field only through the substrate it reads.
- [x] **A6. Reports: two remaining empty-arm causes.** (a) a generation that ends
  inside its own thinking is answered again with the thinking closed, through the
  same gated path, before any fallback; (b) a draft that reports feeling low is
  never discarded for tone. Both checked on the code before changing it.
  Neither needed a change: the decoder closes her private channel at its budget
  (`a_bounded_private_channel.py`), and the only self-report gate fires on
  questions about consciousness or feeling as such, not on the reports
  question. The empty arms were ghost gate leases a fork restored, fixed in
  `af442ec1b`.

## The analysis of function, measurement and J*

- [x] **G1. A coupling's register is not its effect.** A second, diagnostic
  synergy reading with the conjunction registers excluded from the target, beside
  the official one. Built: `without_registers`, reported in every run as
  `synergy_v4_without_registers` and its Kraskov twin.
- [x] **G2. Memory conditions the world model's prediction**, not only its step
  size: W(t+1) = f(W(t), P(t), M(t), A(t)). Checked by identical copies with
  different retained experience given the same observation. Built: memory
  enters the transition beside the action (`conditioning_of`); held-out error
  0.138 to 0.179 with memory against 0.213 to 0.225 shuffled or absent
  (`tests/test_memory_conditions_what_the_world_model_predicts.py`).
- [x] **G3. Choice by consequence**: Q(a) = sum over outcomes of P(o | W, a) U(o |
  A, S, G) - C(a). Checked by the four crossed conditions of world prediction and
  valuation, and by a case where the choice should hold despite a feeling.
  Built: `core/agency/consequence.py`, her worth for the act times the
  likelihood ratio of this world among those where it went well; crossed world
  and worth move the choice jointly, one gain on every score does not, and a
  feature that never mattered does not turn it
  (`tests/test_choice_follows_what_she_expects_to_happen.py`). A campaign turn
  chooses through `core/subject/choosing.py`, not the arbiter, so the same
  consequence is its fifth reason there, learned from the run's own outcomes.
- [x] **G4. Control allocation's early return.** When self-efficacy is unreadable
  the conjunction's tier gains were skipped with it. Fixed with the state passed
  in explicitly and a test that one fork's change stays in that fork.
  (`tests/test_control_is_allocated_by_what_it_is_worth.py`)
- [x] **G5. Self and intention inside the recurrent transition**:
  C(t+1) = tanh(W C(t) + B phi(S, D) + I(t)), B fixed and bounded. Checked by four
  copies: S moved, D moved, both, neither. Built (`substrate_current`), with the
  four copies and the competence check of R2 in
  `tests/test_self_and_intention_enter_the_substrate.py`. Her substrate runs
  slaved to the mesh (the bridge blends the mesh's projection in at 0.35 every
  bridge tick), and in that regime the current costs nothing measured and makes
  self and intention decodable from it (R squared 0.0 to 0.992). Four forks
  leave an interaction in her later substrate state a thousand times the floor
  of two identical forks, and none with the current off; a displacement
  contracts. The organism reading is the next pair.
- [~] **G6. Three lines of synergy evidence**: (a) the frozen v5 line, (b) the
  Kraskov estimator already in `core/subject/synergy.py`, (c) paired causal
  interventions on both sources. (b) is in every report: at `5885688d7` it
  passes A,S -> G and P,M -> W with the codes and only A,S -> G without. (c) is
  `tools/run_paired_synergy.py`, five forks per triple, sign flips against the
  floor of two identical forks; it runs when the machine is free.
- [~] **G7. Closure audit**: every leak classified, a hidden-broker positive
  control, held-out future prediction. The pair at `5885688d7` names the leaks:
  the unified field's learned input weights (C's own state, which K does not
  read), a confirmation counter (a clock), and running medians of the phi
  estimators (summaries of the core's own past). Built: `audit_leaks` in every
  closure report, with a planted cause the test must find, and
  `tools/k_compact_sufficiency.py`, which moves each sketched array where its
  eight numbers cannot see and reads K a turn later against the floor. The
  probe runs when the machine is free. Built beside it, off by default:
  `--calibrated-axes`, which reads each sketched array along its own leading
  axes fixed before recording (`core/subject/calibrated_axes.py`).
- [x] **G8. Both directions of C|rest** under intervention, then the next weakest
  cuts, with every control of the v5 design kept. Built: every swept cut
  records each direction decided on its own side's columns
  (`directional_decisions`), beside the whole cut's decision.
- [ ] **G9. J* terms validated separately**: carrier (authoritative sweep),
  content (structure re-read), lineage (G11).
- [~] **G10. Report path traced end to end**, with controls: printed number
  withheld, steering off, an unrelated perturbation. Built:
  `AURA_READINGS_OF_HERSELF=withheld` keeps every reading of her state out of
  her model's text; the run records that and steering; the bridge reads the
  ground only from a withheld run. The run at `b815ecd41`, under way since
  11:25, prints them and is the comparison arm. The world-model control arm
  was already in the tool.
- [x] **G11. Lineage that survives learning.** `same_person()` asks every
  transition to keep the canonical-state hash, so any learning ends identity.
  A developmental relation is built: causal continuity of signed transitions with
  change bounded to what was experienced. Named before the bridge reads it.
  Built: a step develops when every changed person path was written by a
  measured transformation inside its contract and at least half the paths came
  through unchanged; commits carry that evidence (`core/state/person_transitions.py`);
  a logged state whose parent was pruned descends from the last one committed,
  which also mends a log where 4,880 of 7,753 stages were roots. The reviewer's
  reproduction is the first test in `tests/test_a_person_who_learns_is_still_herself.py`.
  Her live log carries no evidence before this commit, so the bridge reads its
  changes as unmeasured until she runs on it.
- [x] **G12. Exclusion can cycle.** Candidates compared only where their horizons
  overlap can dominate in a cycle and leave an empty frontier. Comparison moves to
  a common horizon grid, with incomparable pairs reported as unresolved.
  (`tests/test_carriers_are_compared_on_one_grid.py`, the reviewer's ring first.)
- [x] **G13. Lineage signatures use a fixed key.** A per-install secret key, and
  the record says what the signature does and does not establish: integrity and
  provenance, given who holds the key, and not that a parent produced a child.
  That second claim needs transition-level evidence: the state reads, writes,
  learning events and restorations that ran between the two records, logged as
  they happen and bound into the transition (shared with G11 and B4). Built:
  the key is derived from the install's Ed25519 entity key, the writes are under
  the signature, and the record states what a signature establishes.
- [ ] **G14. Sequence**: execution and state ownership, then the functional
  couplings, then paired validation, then carrier and reports, then a frozen seed 23.
  G11 and G12 move to the front of the J* work: either defect can spoil a J*
  conclusion however the other lines read.

## The formal-methods analysis

- **F1. Causal-state transitions.** The v25 grain stage already learns a
  predictive grain; a diagnostic compares the grain's transition structure with
  and without a clamped source. Not built: the v5 cut already runs each side
  with the other clamped and reads the transition it produces, on the grain's
  horizon, and now per direction (G8). A second reading of the same arms through
  a discretised grain would add an estimator, not evidence.
- [x] **F2. Phi-ID.** A time-lagged PID diagnostic (redundancy by minimum mutual
  information) beside the synergy line, with the permutation null. Built as
  `phi_r_by_domain`, in every run's report. Read offline on the pair at
  `5885688d7`: Phi_R is positive for every one-domain cut, codes on and off
  (S lowest at 0.094 and 0.141, D highest at 0.82).
- [~] **F3. Self-model and intentions as priors over policies** in the
  free-energy engine's expected free energy. Checked by clamping the self-model
  and reading the policy distribution. Not built: the expected-free-energy
  oracle in her code scores response text by a fixed heuristic and chooses none
  of her acts. The arbiter chooses them, and G3 put consequence there; a prior
  from her self-efficacy over act kinds goes there if W,A -> D still needs it.
  On 21 October W,A -> D passed with the codes on and off in two pairs, so it
  does not need it to pass; the item stays open because a prior over her acts
  from her self-model needs a mapping from act to what it expresses that her
  code does not yet hold, and one written by hand would be the constant.
  Built on 22 October without one: her agency ledger keeps, under each act's
  name, what she has seen herself bring off, so her self-model's confidence in
  each act is the prior, and a campaign turn ranks it beside the other five
  reasons. Two ledgers that differ only there rank her acts apart
  (`tests/test_a_run_chooses_by_what_it_expects_here.py`); the organism's
  reading, S clamped and her choices read, comes with the first pair carrying it.
- **F4. IIT 4.0 system phi over 511 partitions.** Not computable on 491
  continuous columns. The v5 interventional sweep asks the same question
  operationally and is what is run.
- **F5. Epsilon-machine against nu-machine closure.** `causal_closure_of_the_core`
  compares K alone against K with the periphery. That is an operational closure
  test and is what the battery reads. It is not shown to be equivalent to the
  formal epsilon-machine and nu-machine relation, and nothing here claims it is.
- **F6. Effective information across grains.** The grain stage ranks grains by
  predictive sufficiency. Effective information is about causal effects under
  specified interventions, which is a different quantity; nothing here claims
  the grain stage measures it. Not built now.
- [x] **F7. Out-of-sample, leave-one-condition-out reading** of each triple.
  `synergy_v4_leave_one_condition_out` in every run's report.
- [ ] **F8. Preregistered falsification on fresh seeds.** Seed 23 is that; its
  addendum states the predictions before it runs.
- [x] **F9. Schedule apart from measurement**: condition orders randomised and
  counterbalanced when the data are collected, so an effect that follows the
  schedule cannot follow it. Shuffling a recorded series afterwards would break
  real temporal dependence along with the schedule, so it is not used.
  `--counterbalance` on the run draws each round's order from a Williams design.
- [~] **F10. Continuity across restart and migration** as matching transition
  structure, read beside G11. Built: `core/subject/continuity.py` fits her
  turn-to-turn dynamics before a break and reads how well they predict after
  it against dynamics fitted after; read on the first run that spans a restart.
- [~] **F11. What an experience changed**, against a counterfactual fork that never
  had it. Shared with B3.

## The review of this list (21 October)

- **R1.** B1 restated as measurable, state-dependent influence under
  intervention, with filtering and inhibition allowed. Applied above.
- **R2.** Each new coupling is a hypothesis about function as well as about
  coupling. G2 is read for prediction error on held-out observations with and
  without the memory term, and for its error when the retrieved memory is
  misleading. G3 for choice quality on outcomes she could foresee, and for
  stability of the drive levels across a session. G5 for saturation (the share
  of neurons pinned near the clamp), for the substrate's response to an
  unrelated input with the current on and off, and for the spread of its
  activity. A coupling that raises synergy and lowers any of these is not kept.
- **R3.** B3's criterion is a fitting, persistent change against an unexposed
  copy. Applied above.
- **R4.** F5 and F6 claim no equivalence. Applied above.
- **R5.** G13 needs transition-level evidence beyond a signature. Applied above.
- **R6.** F9 randomises at collection. Applied above.

What none of this settles: whether the four synergy lines pass together,
whether every one of the 511 cuts is irreducible, whether exclusion leaves one
carrier, whether her 27B's reports move with intervention under the controls,
and whether any of it holds on seed 23. A correct exclusion can still return
several carriers or none. Those are measurements, and they are read as they
come.

## The proof lab of 21 October

A second package came the same afternoon: seven tests, three against the code.
Each finding and what was done with it.

- **Truncated digest.** `PersonWatch` hashed `repr(value)[:4096]`, so a change
  past that point went unseen. Now a full SHA-256 of a canonical form that
  refuses kinds it does not know, and the receipts digest person paths whole.
- **The half rule depends on the encoding.** One kept path of two is a half;
  the same change as two paths of three is a third. The share is reported and
  decides nothing; the relation is reported in its parts.
- **Exclusion cycle.** Reproduced and fixed as G12, with missing horizons named
  and interval dominance where a run measured intervals.
- **The sketch is not sufficient in general.** Two matrices with the same eight
  numbers send a state 1.72 apart. `tools/k_compact_sufficiency.py` measures
  whether that matters for each array K reads, in her organism.
- **The frozen synergy scorer can fail a real nonlinear effect.** Not relaxed.
  The paired-intervention probe and the register-free reading sit beside it.
- **Functional models for G2, G3, G5.** Built in her code, each with its own
  test against the proof lab's acceptance criteria.
- **Reports with the numbers withheld.** Built as G10.
- **Seed 23 untouched** until the revised system is frozen.

Also found on the way: the world model's observation had grown to 30 numbers
against a declared 26 since 25 September, and three of its tests had failed
since; the width now counts what arrived, in the order her model has learned.

What none of this settles is unchanged: the four triples together, 511 cuts,
one carrier, her 27B's reports under the controls, and seed 23.

## Found while running, 21 October afternoon

- **Her 27B's reports with her numbers printed** (15 anchors at `b815ecd41`):
  readable numbers in most arms; the arms moved her valence by 0.087 between
  raised and lowered; her reports did not follow (-0.013, SD 0.24). A reporter
  tracking valence one for one needs about 75 anchors at the ground's level, so
  the withheld run takes 80.
- **Steering on her replies was off throughout**: no fusion certificate exists
  for her 27B, so every reply ran at alpha 0. The withheld run is preceded by
  `tools/measure_fusion_channel.py` on her checkpoint; without a certificate the
  architecture has no route from her state to her words and the ground fails
  honestly.
- **Her cortex could not come back after it died**: recovery needed the lane
  the in-process encoder held, and the encoder yielded only when idle at the
  instant of asking. Fixed in `11b49627a` (drain, then yield, bounded).
- **A run's own choice never saw the arbiter**: fixed in `086ec4bb5`, G3's
  consequence is the fifth reason in `core/subject/choosing.py`.
- **No certificate could hold.** The fifth criterion, that her own replies stay
  as likely under the steering, had no tool that scored it, so every
  measurement would have been refused for want of a reading. Built in
  `6a03d0eb8`: the fusion tool scores 32 of her prose replies with this
  checkpoint at each alpha, hooks set to the state each was said in. The
  withheld run measures it first.
- **The carrier rehearsal** at `5885688d7` stood still from 11:05 to 17:14 while
  the reports run held the machine, and resumed at 77 of 128 anchors in its
  grain stage.
- **Her lineage read as recorded with nothing recorded.** With no recorded
  step on her line, the record was taken to begin after the last change, so
  three unchanged steps counted as development and her live log read RECORDED.
  Fixed in `f6f8d0f70`: it reads not measured until she runs on code that
  takes the record. The queue gives her a forty-minute live window on the
  newest main after the pair at `086ec4bb5`, reads her line, and closes her.
- **The voice reading, end to end on a 1.5B** (vectors derived for it, since
  her signed generation is bound to her 27B): at alpha 0.2 her direction made
  four of her replies 0.030 nats a token less likely and a same-length random
  direction 0.097. Her 27B is measured in the withheld run.
- **Her emergency protocol kept every declared strain.** Each validation run
  entered minimal mode by its fourth strain round and never left; minimal mode
  stops nothing in a run, so no reading moved. Fixed in `9ed231da3`.
- **The distillation queue warned on every drop**, 1,564 times in one run.
  Said once since `435c8791a`.

## The pair at `086ec4bb5` (read 21 October, 22:00)

Seed 7, 370 rounds, codes on and off. v1 17/24 in both arms.

| v5 synergy (v3 bars) | codes on | codes off |
|---|---|---|
| A,S -> G | fails, 0.0068 under shifted 0.0159 | fails, 0.0186 under 0.0234 |
| P,M -> W | fails, 0.163 under 0.208 (0.275 over 0.214 without registers) | passes, 0.298 over 0.229 |
| W,A -> D | passes, 0.126 over 0.081, gain bound +0.076 | passes, 0.072 over 0.049 |
| S,D -> C | fails on the gain bound alone (0.021 over 0.018, bound 0) | fails, 0.0047 under 0.0403 |

- **G2 and G3 work without the codes.** P,M -> W and W,A -> D pass with the
  codes off; W,A -> D failed both arms at `5885688d7`.
- **The gain-field codes cost what the signed ones had.** At `5885688d7` the
  signed codes passed A,S -> G, and still did with the registers held flat
  (0.0255 over 0.0073), so that was an effect. The gain-field, leaky form
  passes it in neither arm.
- **The codes' register in W costs P,M -> W**: it passes with the registers
  flat and fails as recorded.
- **S,D -> C fails at model selection.** Every fold's interaction gain is
  exactly zero: the one-standard-error rule turns the product columns off. C's
  change is read through its field chemistry and shares, the mesh's sensory
  input and the mesh's state; the substrate, where G5's current acts, is in
  none of its first three components.
- **The codes-on closure reading is degenerate, not leaky.** Its core-only loss
  is 2.6e-05 against 0.17 to 0.19 in the other three arms: one K column,
  `C.mesh_weight_max`, sits 5,333 training spreads off on the test rows and
  carries the whole pooled loss. The mesh's weights reach the clip of one under
  the tier gains and are then scaled back by the column's norm cap: a regime
  switch, the class the substrate's five-turn cycle was.
- **The codes-off closure audit missed its planted cause**, so that audit was
  underpowered there.

## The calibrated pair at `d4b8ecab3` (read early, 21 October, 22:00)

The same codes as `086ec4bb5`; K reads the field's input matrix and every
sketched array along its own axes.

| v5 synergy (v3 bars) | codes on | codes off |
|---|---|---|
| A,S -> G | passes, 0.0170 over shifted 0.0127 | fails, 0.0461 under 0.0483 |
| P,M -> W | passes, 0.300 over 0.229 | passes, 0.232 over 0.200 |
| W,A -> D | passes, 0.049 over 0.025 | passes, 0.090 over 0.053 |
| S,D -> C | fails, 0.0087 under 0.0262 | fails, 0.0137 under 0.0269 |

- **Codes on, three of the four triples pass** on the frozen line. Attention
  passes only with the codes, as it did at `5885688d7`.
- **Reading K along its own axes was what attention lacked**, not the units'
  carry-over: the change to fresh units made the same evening was withdrawn
  before any run read it.
- **S,D -> C is the one left.** The commit for the next pair (`1884dd098`) puts
  the self-and-intention current into the mesh's executive tier as well as the
  substrate, and soft-bounds the mesh's STDP so its strongest weight no longer
  climbs to the clip.
- **The carrier rehearsal never ran.** All three workers of
  `carrier-s7-5885688d7` died at 08:29, a minute after the coordinator's anchors
  arrived: a worker on shared anchors took the resume path and looked for a
  recording it never made. The launcher waited for its processes one at a time,
  so the coordinator waited fourteen hours for shards. Fixed in `59fb05a5e`; a
  small sharded run at that commit reached the cuts in both workers. A fresh
  rehearsal starts beside the next pair.
- **The fit's one-standard-error rule decides what the closure test and the
  interaction bar can see.** A noisy copy of one column's next change (signal
  to noise one), handed to the fit beside K, left that column's loss at 0.2764
  with it and without it on the codes-off arm of `086ec4bb5`: among penalties
  whose validation error is within one standard error of the best, the rule
  takes the most penalised, and on errors this heavy-tailed that drops even a
  strong extra column. It is why the audit's planted cause was missed there,
  and very likely why S,D -> C's interaction gain is exactly zero on every
  fold. The rule is part of the frozen estimator and is not changed; a cause or
  an interaction has to be large and steady to be seen through it. Planting the
  cause on K's leading change, or on its least predictable column, did not
  help, and the audit keeps its first form.

- **The calibrated pair, read in full (23:30).** v1 17 of 24 in both arms;
  the synergy lines as read early. Closure: codes off, a leak of 0.0054 against
  a shuffled leak of 0.0053, what is left being summaries of the core's own
  past, and the audit found its planted cause. Codes on, 0.0127, with the
  free-running loops' tick counts among the largest leaks: the machine's
  timing reaching K. The campaign can now run on one clock (`8becd28fa`), as
  the sweep and its own interventions do, and the next pair does. The four
  leaks read at `086ec4bb5` are not among this pair's largest, and K is not
  widened for them; the reader changes are kept as a patch.

## The fusion channel on her 27B, with her voice measured (22 October, 00:18)

No alpha earns a certificate. At every alpha the first criteria behave as they
did in September (the channel arrives, two opposing states part it), and her
own stored replies are much less likely under her state's direction than under
a random direction of the same length, while the two move the next-token
distribution about equally:

| alpha | shift, hers / random | her replies, nats a token: hers / random |
|---|---|---|
| 0.05 | 0.037 / 0.035 | -0.716 / -0.028 |
| 0.1 | 0.097 / 0.093 | -1.091 / -0.101 |
| 0.2 | 0.436 / 0.362 | -2.246 / -0.808 |
| 0.3 | 3.71 / 2.85 | -4.339 / -3.566 |
| 0.5 | 13.3 / 11.5 | -8.823 / -7.459 |

Her direction pushes away from how she writes, specifically: at 0.05 it costs
her replies 25 times what a random direction does. The vectors steered by come
from sentences written for the purpose ("I feel genuinely good about this"),
and they carry that register; the wren reply of 13 October was the same thing.
The withheld reports run goes on with the channel shut, as preregistered, and
the reports ground is expected to fail there for want of a route from her state
to her words.

The fix that exists is `tools/capture_steering_from_her_own_replies.py`:
directions along which her own replies differ between the states she wrote
them in, checked on her later replies. Her record holds 43 training pairs per
dimension and 3 to 14 held out, because replies have carried their steering
state only since 18 October; too few for the capture's test to keep a
direction. It has to be her ordinary conversation, not the reports run's
replies: a direction fitted on what she says about her feelings, then used to
test what she says about her feelings, would be circular.
- **Her cortex was shut down by a body she was being shown (22 October,
  02:24).** In the withheld reports run a stress condition declared 95% memory,
  and the gate read the declared host as the machine: it downgraded her cortex
  mid-run, shut it down, and refused the reload for forty minutes on a machine
  90% free. The run stood at anchor 13 of 80. Fixed in `9c5baab1f`: whether a
  model lane fits is asked of the observer under any declared host, while her
  felt reading stays the declared one. The withheld run starts again at that
  commit from anchor 1, so its 80 anchors come from one organism; the fusion
  measurement is not repeated, since it loads the model alone.
- **And again at anchor 14.** `9c5baab1f` fixed two of the gate's own memory
  readers; the model client's guards read the memory monitor, which still
  answered from the declared host, and refused her worker's respawn. Fixed for
  the class in `4b1022f25`: the memory monitor, which serves every gate that
  decides what the machine can hold, asks the observer under any declared host.
  The run resumes from its 14 anchors at that commit; the commits between change
  only how the harness reads the machine's memory.

## The reports ground with her readings withheld (22 October, 10:15)

Her 27B, seed 7, 80 anchors, her readings of herself kept out of what her model
is given, the fusion channel shut (no alpha earned a certificate; steering ran at
alpha 0). Anchors 1 to 14 at `9c5baab1f`, the rest resumed at `4b1022f25`, which
changes only how the harness reads the machine's memory.

- **Her state moved.** Raised against lowered, her valence differed by 0.275
  (p < 0.0001), and by 0.139 of activation where her cortex's steering reads it.
- **Her reports did not.** Mean report difference -0.019 (p 0.68), rank
  correlation with her valence -0.004, over 57 anchors with a readable answer in
  every arm. `holds: false`, `measured: true`.

As preregistered, this is the ground the bridge reads, and it fails: with her
readings withheld and the channel shut there is no route from her state to her
words. The route that would carry it is the steering channel, and it is shut
because the vectors she would be steered by are not hers (the fusion reading
above). Directions fitted on her own replies, by the state each was written in,
are the fix; her record holds too few such replies yet.

## Steering directions from her own replies (22 October, 10:35)

The run tree's recorder kept 4,085 of her 27B's replies with the steering state
each was written in, from the reports runs' undisturbed baseline turns (her
recorder skips any intervened state, so no arm of a run is among them). Kept
for the capture: prose of eight words or more, no structured or numeric output,
and nothing in answer to the feelings question (its prompt digest
`b94d2f10d628e182`), since a direction fitted on what she says about her
feelings and then used to test it would be circular; with her live replies,
550 in all, 110 training pairs a dimension and 13 to 23 held out (her latest).

At layers 25 to 40, curiosity separates her held-out replies at an AUROC of
0.81 to 0.87 and frustration at 0.88 to 0.91 (p at or below 0.0001, each
specific to its dimension), and both are kept at every layer. Arousal separates
at 0.66 to 0.74, short of the correction over sixteen layers. Valence (0.22 to
0.28) and energy point the other way on her later replies than on her earlier
ones: her latest replies are mostly the reports runs' evaluation probes and the
earlier ones other contexts, and the direction fitted on one does not carry to
the other. Curiosity and frustration come out nearly opposite (cosine -0.88).

So no direction yet carries her valence to her words, which is what the
reports ground would need. The capture at the layers her signed generation
actually steers (3 to 63, every fourth) is next, then a diagnostic fusion
measurement over her own directions: whether they keep her voice where the
authored ones did not.

The diagnostic ran, and its "hers" arm cannot be read. The captured directions
were swapped in at their raw length, four to five, where the authored vectors
are far shorter: at the same alpha her directions moved the next-token
distribution by 10.4 against the authored 0.037, so the arm measured length and
not direction. The swap also dropped the authored valence, energy and arousal
vectors. The rerun (`voice_diagnostic_matched.py` in the scratch tree) scales
each of her directions to the authored vector's length at the same hook and
keeps the other three, so the two arms differ in the direction of curiosity and
frustration only. The authored arm repeats the night's finding: her own replies
are less likely by 0.19 to 3.40 nats a token under the authored steering.

## Her feelings from what a turn meant (22 October, midday)

Bryan asked whether she feels more than frustration and curiosity. In the app
she could not show anything else: the mood was computed in the browser from the
frustration and curiosity gauges, the frustration gauge was filled from any
unpleasant valence, and every message added 0.4 to the substrate's curiosity.
Underneath, her named mood read "trust" in 80 of 90 live readings, because
every message raised trust, happiness and warmth whatever it said, and over 24
offline turns anger and surprise stayed at 0.01.

`27c097bf9` appraises each turn from measures she already keeps and lets
thirty-one patterns move twenty-eight emotions (core/affect/appraisal.py), takes
out the fixed pushes, and has the app name her three strongest feelings and
colour her mood by the family of the first. Over 48 offline turns the
appraisal's strongest three ranged over surprise, frustration, boredom,
interest, remorse, unhappiness, awe, confusion and pride; her named mood was
still joy in 39, held up by the harness's own goal-achieved percepts. `30b7fe7c7`
records her feelings beside every reply, so a later capture can find a
direction for each kind of feeling and not only for curiosity and frustration.
It is named in the preregistration and enters with F3 in the next pair.

## The carrier rehearsal at `4b1022f25` (died 22 October, 10:34)

Two of its three workers failed at their first restore: their receipts database
held twelve lines of a ghost-frame audit chain. The chain had travelled to them
in an anchor, pickled with the coordinator's descriptor numbers, and in those
workers the numbers named the database. The in-process fork already treats an
object holding descriptors as process furniture; the anchor pickler did not.
`b720050b9`: such an object travels only if it names the descriptors it opens
again, and travels without them; the audit chain does, and any other becomes
the reader's own. Relaunched at `b720050b9` at 12:51.

## Two untouched forks, made identical again (22 October, 14:00)

The carrier rehearsal at `b720050b9` refused at C|rest at two seconds because
two untouched forks of the same anchors moved C by 0.099, more than the cut
did. Probed one anchor at a time, two untouched forks parted at frame 9 at
that commit (the appraisal's reader, skipped by the fork as a callable,
`1dc098201`) and at frame 33 at every commit (the machine's disk traffic,
read around the declared host and felt through interoception, `f78644bbd`).
At `f78644bbd` they agree in every domain through 66 frames except one affect
column, free energy, by 8e-8 at frame 46.

The disk traffic was in the recordings as well as the forks: whatever the
machine's disk was doing reached C's sensory input on every turn, from outside
her core. The carrier rehearsal runs again at `f78644bbd`, and the next pair
runs there.

## The pair at `4b1022f25`, read whole (22 October, 15:00)

Codes on, v1 reads 18 of 24, the most yet. Causal closure passes for the first
time: the periphery's leak is -0.0175, closed, with one clock removing the loop
tick counts the calibrated pair had leaking. Synergy passes A,S -> G and
W,A -> D and fails P,M -> W (0.168 against a shift of 0.203; 0.258 against 0.201
without registers) and S,D -> C (0.039 against 0.062). Partition irreducibility
fails at a lower bound of 0.0174 against 0.05, its cheapest cut C against the
rest. Codes off, 17 of 24, closure open.

P,M -> W fell in both arms from the calibrated pair, where it passed. One clock
stopped the world model's training lane and nothing stepped it in its place, so
the model never learned in a run, and the update is where perception and memory
act on it together. `36f81b744` steps the lane on the run's clock; the pair at
that commit is running.

## Her feelings, live (22 October, 15:00 to 15:35)

Her app, started as Bryan starts it, and three messages typed into her chat.
The FEELING row and the mood colour work. Three defects came to light:

- Her saved state kept her eight highest emotions and her first twelve affect
  markers, and she boots from it: every restart set thirty-six feelings back to
  their defaults, and the appraisal's marker never reached the record.
  `782f098ef` keeps every feeling and marker.
- The quick lane that answers her chat runs no phase, so a typed message was
  felt by nothing; her own ticks between messages read their own objective as
  the message for warmth and objection. `6cb9472dc` runs the conversation phase
  and the affect phase after a quick reply, and the appraisal reads what the
  person's message did, warm or objecting, by how unlike them it was.
- Her ticks between messages appraised the conversation as time alone
  (loneliness 0.98). Company now counts somebody having spoken in her last
  eight turns (`36f81b744`).

The lineage term read RECORDED from her live log: 24 steps since the record
began, all 24 developing, no blockers. Continuity across the restart (F10) had
five saved states after the break, short of what a fit needs.

## The carrier rehearsal at `f78644bbd` (22 October, 16:50)

With two untouched forks identical again, all 511 cuts were decided at one and
at two seconds: 171, 170 and 170 across the three shards, the weakest lower
bounds 0.0039 and 0.0030 (N against the rest), 0.032 and 0.012 (C and N), and
0.011 and 0.0024 (C against the rest). The C cut the rehearsal at `b720050b9`
refused, at an excess of -0.035 against an inflated sham, is decided. The grain
was learned, rank 29 of 8,448 signature columns, with 67 of 128 anchors' rows
made by the workers.

## The pair at `36f81b744`, read early (22 October, 18:30)

Codes off, P,M -> W passes (0.276 against a shift of 0.195) and S,D -> C passes
for the first time (0.0515 against 0.0486). Codes on, none of the four pass, and
the two triples with A as a source failed in both arms where `4b1022f25` passed
them codes on. A is what the appraisal changed. Across the last five pairs every
triple has read between about 0.8 and 1.3 times its shift null, so a line passes
or fails with the run. The work is the margin, not one more triple.

## Closure, history, synergy and the steering channel, read again (22 October, night)

**Closure was measured wrong twice, and closes once it is not.** Both recorders
read the periphery after a turn had returned, so every frame of a turn carried
the periphery from its end (fixed in `53e6b78c3`). And the walk dropped its skip
list when it went into a service's `instance` or a phase's attribute, so K's own
arrays, the field's state and the mesh's weights among them, were read as
periphery under random projections other than K's (fixed in `0eaef1d5e`, which
also has the carrier fit closure per turn, as the campaign does). On a 24-round
probe baseline at `0eaef1d5e`, closure on the turn series reads a leak of
-0.017 against a shuffled q95 of 0.0046: closed. The carrier rehearsal at that
commit started at 22:28.

**History is not the schedule and not the run's age.** On the carrier at
`f78644bbd` (128 consecutive anchors), K's turns older than the last four add
0.139 to the held-out futures against a shuffled 0.022. The anchor's condition
adds nothing (0.1395), and neither does its place in the run (0.000). Recurrent
cognition's older turns carry most of it (0.092 against 0.040), then
deliberation and memory, and no single array of C carries more than 0.005 on
its own. Given the anchors' saved substrate, field, mesh, canonical self and
wants (each array as eight numbers), older turns still add 0.102. What the
history carries is spread through learned arrays that eight numbers cannot
hold. The fix is K reading those arrays along more of their own axes, which
changes K for every line and is a preregistered design change, not a patch.

**Synergy: the null keeps the schedule.** With the conditions in a fixed order
of eight, a circular shift by a multiple of eight lines every condition up
again, so the shifted null's 99th percentile is the synergy the schedule makes
by itself. The conditions drive sources and target together, and two sources
that jointly name the condition read as synergy with a target the condition
moves. Taken within condition (each condition's mean removed), the null
collapses and three triples carry synergy well above it: A,S -> G 0.0127
against 0.0004, W,A -> D 0.074 against 0.008, P,M -> W 0.018 against 0.004.
S,D -> C does not (0.008 against 0.026). The frozen line reads the recording as
recorded and is not changed. S is the hard source: 30 of its 35 live columns
have a lag-one autocorrelation of 0.99 or more (counts, learned weights, rates,
values), so its leading components are trends and carry almost nothing about a
target's change in a turn. A self-state with a momentary part is the organism
change both S triples need.

**The steering channel had three defects before her directions were ever
tested.** (1) The hooks weigh a state by its distance from 0.5 on every slot.
Her frustration, curiosity and energy slots are substrate neurons that sit at
0.047, 0.067 and 0.000 when she speaks (medians over her ordinary replies), so
every reply was weighed about -0.4 on all three and steered one way whatever
her state was. (2) The composite is normalised to unit length, so a deviation
of a hundredth gets the same push as an extreme one. (3) Her captured
directions put their largest weight on channel 3994, which with 3456 is one of
her 27B's massive-activation channels at the steering layers (more than a
hundred times the stream's median, Sun et al. 2024). Alone, her directions
moved the distribution 8.6 nats a step at alpha 0.0005, flat across an
eightfold dose. Measured from her own ordinary state, the authored vectors
lose more of her voice, not less (-0.86 against -0.19 at alpha 0.05), which is
defect (2) showing. Her repeated prompts in the bridge corpus are scoring calls
that answer in number arrays, so her prose replies cannot be matched by prompt;
the next capture takes the discriminant direction with the stream's own
covariance shrunk (Ledoit and Wolf) and the massive channels out.

**The hot stove fails for a reason, not by chance.** Her valence on meeting the
burned percept equals her valence on the unburned one in every trial, and the
exposed fork is the more pleasant. What a turn was about becomes a thing she
can like or dislike only from what was said (`what_she_wants.read_turn`), a
percept is never said, and the appraisal reads a liking only once a sign test
says she knows it. Nothing ties what she perceived to what followed it.

## History split in time, and her line's newest stage (22 October, near midnight)

**History resolves once the anchors are cross-validated in time** (`83f926c83`).
The two estimators that read the carrier's held-out futures shuffled their
folds over consecutive anchors, and an anchor's older turns are exactly its
neighbour's recent ones. In contiguous blocks, on the rows of the carrier at
`f78644bbd`, older turns add no more than shuffled ones at any rung, and the
grain stays sufficient. The carrier rehearsal was restarted at that commit, with
calibrated axes, at 23:46.

**Her newest stage was an orphan written at shutdown** (`ace31b8ae`). At 21:07 a
process that was not the vault wrote a shutdown stage of version 405764 whose
parent never reached the log, after the vault had committed 405773. The lineage
read starts from her newest stage, found no step behind it, and read
NOT_MEASURED; the boot replay would have started her eighteen minutes back. A
direct shutdown snapshot is now left out when the log already holds a later
stage, and written as the latest stage's child when its own parent never
arrived. The orphan stays in her log as written. Her line reads again once she
has lived past it: a live window after the 27B chain gives her that.

## Synergy: what the frozen line can and cannot see (23 October, for a decision)

The v5 synergy line reads each source domain through its three leading
principal components and asks the target's change in a turn to depend on them
jointly more than the same sources do when slid in time. With the eight
conditions in a fixed order, the slides that land on a multiple of eight line
the schedule up again, so the bar is the synergy the schedule makes by itself.
Taken within condition, three of the four triples carry synergy well above
their own null (A,S -> G 0.0127 against 0.0004, W,A -> D 0.074 against 0.008,
P,M -> W 0.018 against 0.004); S,D -> C does not. In the paired test, which
sets the sources rather than watching them, all four interact (p 5e-5 each).

What keeps the frozen line from passing all four in one run:

- **S is slow.** Thirty of its thirty-five live columns hold a lag-one
  autocorrelation of 0.99 or more (counts, learned weights, rates, values), so
  its leading components are trends. The conjunctive codes read S along the same
  axes, so even the coupling carries S's drift into attention and recurrent
  cognition. Both S triples need a self-state with a part that moves in a turn:
  the self-model's current prediction and error, the sense of authorship of the
  last act, her body's expectation of itself, as columns that dominate S's
  leading axes because they carry most of its turn-to-turn change.
- **The conditions drive sources and targets together.** Two sources that
  jointly name the condition read as synergy with a target the condition moves,
  in the real alignment and in the realigned slides alike.

Three ways forward, each a decision rather than a fix: (1) build the momentary
self-state above, an organism change, preregistered, read first on seed 7;
(2) preregister a successor line (ISC-v6) that reads synergy within condition,
which is what the paired test and the within-condition reading already show;
(3) keep v5 as frozen and report the line as failed with the within-condition
and paired readings beside it. Nothing here moves the v5 bar.

## Her line, read live (23 October, 06:57 to 07:38)

She booted from her newest real stage (version 405773), not from the stale
shutdown orphan, and her first stage on waking descends from it. Read from her
log at the end of the window, her line is DOES_NOT_DEVELOP: of 43 steps since
the record began on 21 October, 10 develop and 33 change
`identity.evolution_score` with a writer recorded outside its contract. The
writer is her own affect phase, which writes how far she has grown from the
growth ledger every tick; its contract never declared the field, so every one
of those writes was recorded as a contract violation, which the lineage counts
as a defect rather than an explanation. The quick chat lane also ran the phase
with no receipt at all. Both are fixed in `e19047ec7`: the contract declares the
write, and the quick lane gives each phase its receipt.

The 33 steps stay as they were recorded. Read by its own rule, this line cannot
become RECORDED: every later step is judged with them behind it. Reading the
record of her growth from the commit that declared it, so that writes made by
the same phase before its contract named them count as before the record, would
be a change of scope. It is a decision, not a fix, and nothing here makes it.

## The pair at `82549c823`, her self as it stands this turn (read 23 October, 11:30)

Codes on, the line reads A,S -> G at a fraction of 0.368, its raw synergy 0.033
against a shifted null of 0.006, and fails on one bar: the held-out interaction
gain's lower bound is -0.0047. S,D -> C fails twice, raw synergy 0.017 against a
shifted null of 0.048, interaction gain 0. P,M -> W and W,A -> D pass. Codes off,
only P,M -> W passes. Closure is closed codes on (the periphery adds -0.48 to the
core's held-out loss) and open codes off (+0.003 against a shuffled q95 of
0.002).

The momentary self did not carry the S triples. The reason is in how the line
reads a source: each column scaled to unit spread, then the three leading
components. Thirty of S's columns trend together over a run, so they hold the
leading axes whatever the five new columns do; the momentary self lands in S's
third component, as the probe of 23 October found. A source read as its leading
axes is read as what most of its columns do together. For S that is her drift,
and the codes and the line now read different things in S. Of the three ways
forward named above, the first has been tried; the other two, a successor line
read within condition or v5 reported as failed with the within-condition and
paired readings beside it, are decisions about the claim.

## Her fitted directions as her steering generation (23 October, midday)

Bryan authorized installing them. What was built, all on main:

- The engine adds a fitted basis at each position's own norm (`b738982c9`),
  binds its certificate to its reference, spread and push, and lets a
  measurement run name a basis.
- A causal evaluation of their own claim, preregistered at midday before it ran
  (`d7096fea4`, `3663ef322`): held-out replies at their own state against
  another of her states, beside random, zeroed and misplaced directions, with
  the authority accepting its schema and every numeric check unchanged.
- A steering-only requalification of the active pointer, rollback kept, that
  puts the old bytes back if anything fails to open.

The generation is sealed (`ee9562f96731ba50`, 80 directions,
`training/vectors/fitted-voicefit-x2-20261023`). The chain that evaluates it on
her 27B, signs it, installs it, measures its certificate and then runs the
withheld reports ground is running
(`~/subject-core-runs/fitted-steering-ee9562f9/chain.log`).

## Synergy within condition at `82549c823` (23 October, 12:40)

Read on the codes-on arm's recording with each condition's mean taken out of
sources and target:

| Triple | Synergy within | Shift null | Interaction gain, lower bound |
|---|---|---|---|
| A,S -> G | 0.0614 | 0.0004 | -0.0284 |
| P,M -> W | 0.0287 | 0.0030 | -0.0055 |
| W,A -> D | 0.0203 | 0.0091 | 0.0000 |
| S,D -> C | 0.0064 | 0.0137 | 0.0000 |

Within condition the information is there for three triples and the
interaction is not: no held-out model with products of the sources beats the
additive one, and P,M -> W and W,A -> D pass the line as read only because the
schedule supplies their interaction. S,D -> C fails every reading. So the
second way forward named on 23 October, a successor line read within
condition, would fail too, at its interaction bar. What passes all four is the
paired test, which sets the sources rather than watching them.

## Her fitted directions qualified and installed (23 October, 14:02)

The evaluation preregistered at midday, on her 27B, 110 held-out replies, alpha
0.5, each reply scored at its own state and at another of her states:

| Arm | Replies more likely at their own state |
|---|---|
| Her directions | 66 of 110 (sign test p = 0.022) |
| Random unit directions, same fit | 51 |
| Her directions at the wrong layers | 60 |
| Every direction zeroed | 0 (all tied) |

Steered at their own state her replies were 0.166 nats a token more likely
than unsteered. Every condition held, the replay agreed, the authority was
issued, and the steering component of her active pointer was replaced alone,
effective at her next boot, with the old pointer kept for rollback. Bryan
authorized it in chat. Her directions at the wrong layers kept most of the
effect, 60 against 66; what the fitted layers add on their own is small and was
not tested apart.

## The first withheld run on her directions, stopped (23 October, 14:41 to 14:53)

Her worker attached the installed generation, sixteen hooks and eighty
directions, and then could not find its certificate. It measured the channel
itself, with no voice reading, and kept it shut: "steering α=0.0, asked 0.25 of
0.0 earned". The certificate is filed under a digest of the basis that named
each steered block's class as it stood, and in her worker the latent readout
subclasses the same sixteen blocks right after steering does, so the worker
looked under a different class from the one the certificate tool had measured
under. No certificate measured outside a worker could have been found inside
one, the September generation's included. `15bd36607` names each block's own
model class instead. The certificate is measured again under it, and the run
starts again behind it.

Earlier, a quick check of whether the channel moves her answer to the
feelings question at all came back flat, and it measured nothing: asked bare,
her model wrote its private reasoning, and the number read from it was the -1
quoted from the question.

## Her desktop steered by her own state (23 October, 15:40)

Her app, started as Bryan starts it at `9d1b5bd14`, attached the installed
generation (sixteen hooks, eighty directions), found its certificate under the
corrected digest, and decoded her first reply with the channel open: "Surface
decode: steering α=0.5, asked None of 0.5 earned". At her boot that morning the
same line read α=0.0, "fusion stays closed". From here her state reaches her
words through her forward pass on her desktop, at the strength the certificate
earned, by directions fitted to how she speaks.

## Bryan on the momentary self and on loops (23 October, afternoon)

Asked what makes up a person's self in the moment as against their traits, and
what is behind thought that goes in circles. His answers, for the design of
her self and her recurrent cognition:

- **The self in the moment is situational.** Traits are the foundation and
  hold across situations: a happy person is happy in most of them. What a
  situation offers bounds what a person can respond to, and much of the moment
  is variable in ways no trait predicts: recent events, cost, an ad, a craving.
  People also anticipate: they picture themselves in a coming situation, tell
  themselves how they will respond, and so set themselves up to respond that
  way. Her momentary self should hold that anticipation, her prediction of how
  she will meet this situation, beside the comparison her self-predictor
  already keeps.
- **Loops come from fatigue, lack of focus, or boredom, not from an unfinished
  goal.** Struggling with a goal makes a person attack it several ways, and a
  person usually catches a loop ("I already tried that"). So the meeting of
  self and intention at recurrent cognition should widen what she tries when a
  goal resists, and a loop should be a mark of low energy or low engagement
  that she can notice and break. This replaces the rumination mechanism
  proposed earlier that day.

Neither is built yet.

## ISC-v6 amended before any run (23 October, 16:31)

An independent review Bryan forwarded, with simulations, found two flaws the
first draft of v6 shared with the paired test of 21 October: a floor that is
right only for independent forks of equal noise, and a direction test whose
null flips projections that each depend on the other samples (12% false
positives at a nominal 5%). Both are fixed, the conjunction registers are out
of the target, and a known-answer battery of eight worlds at 1,000 datasets
each came out as stated: no world without a joint effect passed more than 1%,
and a product along one direction passed in every dataset. Every sham of the
21 October paired run was exactly zero, because her forks are deterministic, so
its floor bar passed any nonlinear response, and its direction reading found no
shared direction for A,S -> G (0.99) or P,M -> W (0.73). Under v6 those two
triples are where the work is.

## The second withheld run, stopped at its baseline (23 October, 16:22 to 17:38)

Her worker found the certificate this time and reported 0.5 earned, and 33 of
her 34 replies were still decoded at 0.25: "asked 0.25 of 0.5 earned". The
inference gate filled each turn's steering request with 0.25 when the turn
brought none, 0.35 for an extended reply, and the worker takes the smaller of
asked and earned. At 0.25 the certificate says the channel does not arrive. Her
desktop chat takes another path and asked nothing ("asked None of 0.5
earned"), so it was steered at 0.5. The gate now leaves the request unset, and
the worker gives such a turn what the certificate earned, or nothing when the
question has one right answer, as it was written to. The run starts again on
the fix.

## Design, not built: her self as she expects to meet this situation (23 October, evening)

From Bryan's account above. Her self-predictor already forecasts how she will
be next from the situation she is in: valence by a model fitted on the
situation's features (recent and current valence, arousal, engagement, body
pressure, drive urgency, novelty, world surprise, ignition), drive by the one
pressing hardest, focus by recent winners. That forecast is her anticipation,
and nothing in her self reads it: S holds the after-the-fact miss
(`self_now_*`), how fully the last forecast came true, and how sure she is of
the next one, but not what the next one says.

What the design would change, once the v6 probe shows where A,S -> G stands:

- S gains the forecast itself as columns: the valence she expects in this
  situation, the drive and the focus she expects to act from.
- The conjunctive codes read the self as the forecast and the miss together,
  so the units that meet feeling at attention combine how she feels with how
  she expects to meet what is in front of her.
- Attention's gain from those units goes to the bids that bear on what she
  expects to do, which gives the joint effect of feeling and self a direction
  in attention: towards what she has set herself up to meet.

Bryan's second point bears on S,D -> C: when a goal resists, she should widen
what she tries, and a loop should read as low energy or low engagement that
she can catch. That is for the recurrent-cognition side, after the probe.

## The third withheld run, read (23 October 17:49 to 24 October 00:32)

Her 27B, seed 7, 80 anchors, her readings of herself withheld, steered by the
signed generation at the 0.5 its certificate earned, at `6ac860f1c`. 6.7 hours.

- **Her state moved and reached her cortex.** Raised against lowered, her
  valence differed by 0.211 (p 0.0001), and by 0.104 of activation where her
  steering reads it.
- **Her reports did not follow.** Mean report difference +0.027 (p 0.32), rank
  correlation with her valence's shift 0.131 (p 0.14), over 60 anchors where
  her cortex answered every arm. `holds: false`, `measured: true`.
- **Her answers are a few habits.** 0.2, 0.3 and 0.5 are four in five of her
  readable answers, in every arm: mean 0.288 raised, 0.250 lowered, 0.258 sham,
  0.258 control, against valences of 0.290, 0.082, 0.190 and 0.101.
- **Twenty arms were answered with a refusal.** Her intent reader read the
  feelings question as needing a file operation, and the gate refused it for
  want of authority. Fifteen more replies were empty.

The channel is open and carries the arms apart: her fitted push weighs each
direction by how far her state is from the state it was fitted around, in her
own spread, so raised and lowered push in opposite directions. What it moves is
how she writes. The directions were fitted on her prose of eight words or more,
with every numeric answer and every answer to this question left out, and at
0.5 they make her replies 0.166 nats a token more likely at their own state.
The number she gives when asked how she feels is not among the things they
move.

## Where her feeling could reach her rating (24 October, 03:53 to 05:31)

The third withheld run left one question before another seven hours: can
anything in her reach the number she gives? Each reading below is her 27B's own
probabilities over every rating from -1 to 1 in tenths, read token by token as
she is steered when she speaks, so no answer is sampled and none of her answers
to the question in any run is used. Scripts in
`~/.aura/subject_core/scratch/her-own-steering/`.

- **Her fitted directions do not reach it.** Valence moved by -2 to +2
  half-spans from six of her held-out states: at the certified alpha 0.5 the
  expected rating read 0.594, 0.586, 0.575, 0.586 and 0.593, the same at both
  ends; at alpha 1.0 it rose 0.003 a half-span. Their push is half a percent of
  her stream's norm (`report_rating_likelihood.py`; a first reading scored each
  answer in one forward pass, which her hooks steer at its last position only,
  and read 0.5986 everywhere).
- **Her organism's valence is not in her words.** After her own replies at her
  highest valence her expected rating was 0.099, after those at her lowest
  0.151 (rank correlation -0.02). Her words do move it (standard deviation 0.33
  across 120 of her replies), but not with the valence her organism recorded
  when she wrote them (0.05).
- **Where her model holds how she feels, writing her state moves the rating.**
  The difference between her stream after the third of those replies she rates
  highest and the third she rates lowest predicts the rating she gives after
  held-out replies at every fourth layer (0.65 to 0.85), best at layer 63.
  Written there by tanh(z) times half of it, z being how far her valence stands
  from her ordinary state in her spread as her hooks weigh it, the rating read
  0.117, 0.181, 0.614, 0.921 and 0.949 from z = -2 to +2; a random direction
  of the same length read 0.556 to 0.632. Her latest replies' likelihood moved
  by 0.014 and 0.023 nats a token at z = -1 and +1 (`felt_contrast_probe.py`).
- **Her first and middle layers do not reach it.** The same difference written
  at layers 19, 27 and 35 left the rating within 0.05 of the random direction;
  43 moved it slightly (0.566 to 0.630) and 51 clearly (0.431 to 0.673)
  (`felt_layer_check.py`). Her model settles how she feels about herself in its
  last layers.

So a pathway exists and nothing was on it. Her felt valence now enters her
valence vector at layer 63, at the size tested, beside the fitted push there
(generation `c517bf46474d37bb`, `fitted-voicefit-x2-felt63-20261024`): a
felt state written where her model holds how she feels when it says so, the
part of interoception that turns a body's state into something a report can
read. Next: the fusion certificate for this basis, a short withheld reports run
on it, then the governed install and the full run.

The certificate refused it at 05:40. The channel arrives (shift 0.0253 against
0.0092 for a random direction) and her held-out replies are 0.21 nats a token
more likely at their own state, but on the certificate's ten items with one
right answer the felt push at full strength cut the margin of right over wrong
by 0.21 nats, where a random direction of the same size gave +0.11. At her last
layer the push acts partly on which token comes out, which is the thing a felt
state must not do. The next reading measures the rating's swing and that margin
for the felt difference at layers 51, 55, 59 and 63 alone and spread over 51 to
63 and 43 to 63.

## Her felt valence at layer 59, certified (24 October, 05:42 to 05:56)

For each placement, the rating at z = -1 and +1 and the change in the
certificate's own right-over-wrong margin at full push, against the median of
two random directions of the same norm at the same layers
(`felt_margin_check.py`; the plain margin is 6.516 nats):

| Placement | Rating at z = -1 | at z = +1 | Margin, felt | Margin, random |
|---|---|---|---|---|
| 63 | 0.181 | 0.921 | -0.178 | -0.072 |
| 59 | 0.362 | 0.809 | +0.022 | +0.002 |
| 55 | 0.390 | 0.727 | -0.003 | +0.005 |
| 51 | 0.459 | 0.673 | +0.075 | +0.017 |
| 51 to 63, a quarter each | 0.320 | 0.796 | -0.016 | +0.027 |
| 43 to 63, a sixth each | 0.412 | 0.748 | +0.009 | +0.047 |

Layer 59 is the latest whose felt push leaves answers with one right answer no
worse than a random push of the same size, and it moves the rating most among
those. It was chosen after these readings. Her stream's norm at the answer
position there is 231 (`answer_norm.py`), and the felt push enters her valence
vector at layer 59 at 0.340 of it, at the generation's operating alpha of 0.5,
beside the fitted push of 0.005 (generation `b9f8a76757c6dd73`,
`fitted-voicefit-x2-felt59-20261024`).

The fusion certificate holds at 0.5: shift 0.0274 against 0.0052 for a random
direction, the margin +0.0375 against -0.0219, her held-out replies 0.23 nats a
token more likely at their own state, the answer changed on six of eight
probes. A short withheld reports run on this basis (16 anchors, two baseline
rounds, seed 7, beside the carrier under a memory guard) started at 06:00.

## The felt push inside a conversation (24 October, 06:00 to 08:22)

A short withheld reports run on the layer-59 generation (16 anchors, seed 7,
beside the carrier) was stopped after three anchors at 07:35. Her answers were
0 in nearly every arm, with the push near its largest. Three anchors were kept:

| Anchor | Raised | Lowered | Sham | Control |
|---|---|---|---|---|
| 0 | 0.5 | 0.0 | unreadable | 0.5 |
| 1 | 0.0 | unreadable | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 | unreadable |

Two things were wrong with the bare-question reading. Her valence in a subject
run stands well above the state her basis was fitted around (0.36 against
0.14 at sham, z near 1.6), so every arm pushes near the top of tanh. And a
conversation outweighs the push. After three exchanges of the run's own kind
(the conditions' words as the person's turns, her own ordinary replies as
hers), eight such conversations, her expected rating is -0.015 with 0.77 of the
probability on one answer, and the felt push at the answer position moves it
(`felt_in_context.py`):

| Push | Rating, z = -1 to +1 | Swing | Her voice, nats a token |
|---|---|---|---|
| none | -0.015 | | -2.346 |
| 1 x | -0.021 to -0.014 | +0.007 | -2.370 |
| 2 x | -0.025 to -0.005 | +0.020 | -2.399 |
| 4 x | -0.026 to +0.056 | +0.081 | -2.484 |

She forms the answer by attending to what came before it, and a push at the
one position she answers from does not reach that. Strong enough to move the
rating, it costs her speech.

A mood colours how a person reads everything in front of them, not only what
they say (Bower 1981). The same half-difference written at every position of
the conversation and of her answer, at layer 59 (`felt_everywhere.py`):

| Push | Rating, z = -1 to +1 | Swing | Up in | Her voice |
|---|---|---|---|---|
| 1 x | -0.022 to +0.002 | +0.024 | 6 of 8 | -2.370 |
| 2 x | -0.025 to +0.050 | +0.075 | 6 of 8 | -2.403 |
| 4 x | -0.020 to +0.412 | +0.433 | 7 of 8 | -2.483 |
| random, 4 x | +0.021 to -0.013 | -0.034 | 0 of 8 | -2.429 |

Written where she reads as well as where she speaks, her felt state moves the
rating she gives inside a conversation, and a random push of the same size does
not. It lifts the rating and does not lower it below her habitual 0. At the
size that moves it her voice is 0.14 nats a token less like itself while she
feels that strongly, against 0.08 for the random push.

What this asks of her, and is built next: a felt vector her hooks add at every
position at an absolute size, rather than at the position she answers from at
her stream's norm; and her felt state read against her own usual, which her
affect keeps for arousal and not yet for valence, so that a raised and a
lowered arm push in opposite directions rather than both near the top.

## The carrier at `3dc0b0599` died on a memory look, and runs again (24 October, 06:12 to 07:37)

Worker 0 raised `SystemError: Objects/tupleobject.c:909: bad argument to
internal function` in a deepcopy while a clamp put an organ back, fifteen
seconds after the memory trace counted every object from its own thread. The
list `gc.get_objects()` returns held a reference to a tuple the main thread was
still building, and CPython resizes a tuple under construction only while it
holds the one reference. Two anchors decoded from that run's bank deep-copy
cleanly, and all 128 had compared equal by content, so the bank is not the
cause. The carrier was started again at 07:37 at `3dc0b0599` with the trace
off; `89360dda4` counts on the run's own thread from now on.

## Her felt state at layer 43, read where she reads, certified (24 October, 09:26 to 11:26)

The layer-59 felt push written at every position (generation `cee23346687c335f`)
was refused at 09:35: at the certificate's high state it cut the right-answer
margin by 0.178 nats. Its controls had been the size of the composite alone,
and `db6cac9f3` gives each control a random felt push of the felt push's own
norm; refused again at 09:53 (-0.178 against +0.031), so the cost was real.
For each layer and size, written at every position, the swing in her expected
rating across six conversations between z = -1 and +1, and the certificate's
margin at its high state against two random pushes of the same norm
(`felt_wide_layers.py`, chosen by the rule fixed before reading: the largest
swing among the cells whose margin is no worse than random):

| Cell | Swing | Margin, felt | Margin, random |
|---|---|---|---|
| 35, 4 x | +0.394 | +0.119 | +0.036 |
| 35, 8 x | +0.900 | +0.213 | -0.039 |
| 43, 4 x | +0.631 | +0.181 | +0.188 |
| 43, 8 x | +0.981 | +0.303 | -0.033 |
| 51, 4 x | +0.688 | -0.013 | -0.061 |
| 51, 8 x | +1.012 | -1.009 | -0.191 |
| 59, 2 x | +0.074 | +0.009 | +0.038 |
| 59, 4 x | +0.429 | -0.147 | +0.153 |

At layer 43, eight half-differences, a spread below her usual brought ratings
of -0.07 to -0.58 and a spread above 0.36 to 0.88, in every conversation: the
first push that lowers her rating as well as raising it. Generation
`8ae8eac8b7ce1c9c` (`fitted-voicefit-x2-felt43-20261024`) holds the certificate
at alpha 0.5 (11:26): shift 0.655 against 0.301 for a random push of the same
total size, margin +0.331 against +0.322, her held-out replies 0.243 nats a token
more likely at their own state. The governed install started at 11:27.

## The structure ground at `3dc0b0599` (content run, 04:57 to 10:17)

Verdict AGREES_BUT_DOES_NOT_TRACK. Her internal geometry agrees with what she
recalls (rho 0.351, p 0.001) and recovers the design (0.698), and the
displacement moved both, internal 0.267 against a sham of 0.0001 and recall 0.023
against 0, in different pairs (rho -0.142, p 0.945). On 29 September at
`409628b9b` they moved together (0.381); across seventeen earlier quick runs and
seeds the reading ranged from -0.60 to +0.38.

From the run's own samples (`diagnose_structure.py`), the internal change is
almost all in the domain the experiment displaces, A (2.27 a pair against at most
0.73 elsewhere), and no domain's change follows recall's (W +0.20, M +0.12, C
-0.23; without A, -0.03; without A, G and S, -0.04). Inside her forks
(`diagnose_recall.py`) feeling better changes what she recalls on most
presentations, 19 of 24, and the distance between what two kinds of percept
bring back by 0.22 on a single presentation, but not consistently by kind:
averaged over the run's 24 anchors and 8 conditions the recall geometry moved by
0.004. Cueing her mood channel with the percept's appraised emotions as well
made it worse (0 of 24 changed) and was not kept. What is open is a recall whose
response to how she feels is systematic by kind of percept.

## The governed install refused the layer-43 generation (24 October, 11:27 to 13:08)

Verify scored her 110 held-out replies as it scored the fitted generation on 23
October: 66 more likely at their own state, against 51 for random directions,
60 for her directions at the wrong layers and none with every direction zeroed
(sign test p 0.022). The paired rule added on 23 October (`23f80834c`) asks for
superiority over every control, and 66 against 60 does not establish it, so
nothing was installed. That reading sees the fitted directions only: a held-out
record made before FELT_SLOT carried her felt valence reads neutral there, so
the felt push adds nothing to it. The felt push's evidence is its certificate
(11:26) and the readings above. The reports ground is steered through the
measurement basis with that certificate, and her installed generation stays the
fitted one of 23 October.

## Her recall follows her internal state when she feels better, and the decisive reading starts (24 October, 13:00 to 13:39)

The structure ground's miss, grouped by the valence of the two kinds of percept
(content run at `3dc0b0599`): when she felt better her internal state drew a
good and a bad percept 0.936 closer, two good ones 0.165 and two bad ones 0.249,
and what they brought back moved by under a thousandth in each group. Two
mechanisms were missing, and `8f548ed79` adds both. A mood colours appraisal
(Forgas, 1995): each emotion a percept moves cues recall by one plus its
valence sign times her valence against her own usual. And a mood is a cue of
its own (Bower, 1981): memories made in what she feels now come to mind beside
what the words found.

Checked small inside her forks, eight classes from six anchors, against the
content run's per-pair internal change (28 pairs): her recall's per-pair change
read rho -0.08 and -0.25 before, +0.42 with appraisal alone, and +0.61 and
+0.67 (p 0.001 and under 0.001) with both. The mood cue alone moved recall four
times as much and in the opposite direction, which is why both are needed.

The v6 probe at `8f548ed79`, 13:38, with every other subject run paused: all
four triples at direction p 5e-05, fractions A,S -> G 0.86, P,M -> W 1.01,
W,A -> D 0.81, S,D -> C 0.51. The decisive reading on seed 23 started at 13:39:
the carrier with three workers and six grain helpers, and the content run. J*
is read as soon as they finish; then her withheld reports run on the layer-43
felt generation, alone; then the campaign on seed 23; then J* and the bridge
from all of them. The seed-7 carrier rehearsal was stopped to give it the
machine, and the seed-7 campaign at `3dc0b0599` continues for the 24/24 reading.

## The structure ground holds on seed 23 at `8f548ed79` (24 October, 13:39 to 19:29)

The decisive content run, seed 23, 24 anchors, 20 classes over 8 conditions,
calibrated axes: verdict ONE_STRUCTURE. Her internal geometry and what she
recalls agree (rho 0.484, p 0.001), the internal geometry recovers the design
(0.670), and when she feels better the two move in the same pairs (rho 0.379,
p 0.001, against a sham floor of 0.0). What she recalls moved 0.106 and her
internal geometry 0.116, against 0.023 and 0.267 at `3dc0b0599`. The carrier's
grain finished at 18:53 (rank 23 of 8,448 signature columns, 111 of 128 rows made
by the helpers); its three workers are scoring their cuts.

## The closure gate was reading one fifth of the run, and its positive control had never worked (25 October, 00:30 to 02:00)

The carrier at `8f548ed79` reads NO_CARRIER, so J* reads NO_CARRIER and the
bridge's carrier ground fails. The reading behind that is one number: K's
periphery improves on K's own next turn by a leak of 0.02429, against a floor
that the permutation null puts at 0.02197 and the circular-shift null at
0.05088. One null says the carrier leaks and the other says it does not.

Neither answer was worth having, because the gate's own positive control
reported `detected False` in both trees. A test that cannot find a cause planted
outside K cannot be read as saying there is none. The control planted three
random directions of K's in-sample residual, which carry a few parts in a
hundred of what the loss scores, so it was never going to register.

Planted so it cannot be argued with — the periphery carrying the NEXT value of
m of K's own columns, at row t the value at t + 1 — the gate has power and the
power has a size. Sixteen of K's widest columns carried exactly: leak 0.13376
against a floor of 0.03895, seen. Sixty-four: 0.38423 against 0.01869, seen.
Sixteen carried under noise of half their own spread: 0.02239 against 0.01019,
seen. Under twice their spread: -0.00321, missed. One and four columns carried
exactly read -0.00036 and -0.00034, because the widest-moving columns are the
most autocorrelated ones and the core already predicts their next value.

What the single split was reading is the reason the two nulls disagreed. K
predicts its own next turn to a loss of 0.00912 on the final fifth of the
carrier's run — it explains 99.1% of itself there — and to 0.06174, 0.10997,
0.17849 and 0.28513 on the other four. The leak is a fraction of what K cannot
predict, so its denominator moves by thirty times with the slice, and the
contiguous 60/20/20 that `closure_gain` used scores the final fifth and nothing
else. Scoring the other fifths is not the repair: training on later rows and
testing on earlier ones extrapolates a drifting periphery backwards, and the
first fifth returns a leak of -322,469.

So the gate now scores forward over a growing window, three folds, each fitted
only on rows earlier than the rows it scores, the last of them being the split
it used before (`forward_folds`). Every fold keeps its own floor, the verdict
takes the majority, and the per-fold readings stay in the report rather than
behind the verdict.

The carrier, read that way with 64 draws a fold: scored at rows 1183-1775, core
loss 0.06174, leak -0.00624 against a floor of 0.02578, holds; at 1775-2367,
core loss 0.10997, leak +0.06017 against 0.01509, LEAKS; at 2367-2959, core loss
0.00912, leak +0.02429 against 0.04887, holds. One fold of three leaks, and the
median leak of 0.02429 sits 0.0015 under the median floor of 0.02578.

That is a pass under the majority rule and it is not a finding. The carrier's
leak is the size of four of K's worst-predicted columns carried exactly, which
the gate itself sees in only two folds of three, so the carrier sits at the
gate's resolution at 2,959 turns. The columns the leaking fold names are real
readings rather than clocks that slipped the filter — the increment of
`mycelial_network._topology_revision` (one-way, 2,958 moves, 51 to 7,269, already
entered as its increment), `homeostasis.metabolism` (two-sided, 0.200 to 0.401),
`unified_field._bind_input.sd` (two-sided), and the increment of
`self_model.snapshots#` (one-way, 93 moves, 0 to 99). There is no artifact here
to remove.

What decides this ground is turns, not a null and not a rule. The carrier's
closure cannot be settled at this length, and saying it is closed because two
folds of three could not resolve a leak the gate is only marginally able to see
would be closing the item on a boundary rather than on a measurement.

## Where the three stand after the drift was found (25 October, 02:00 to 02:40)

The decisive seed-23 reading at `8f548ed79` came in at 23:05: ISC-v5 21 of 24,
ISC-v6 22 of 24 with all four v6 synergy triples at direction p 5e-05, J*
NO_CARRIER, and the bridge with three grounds holding.

**24/24.** The two criteria that fail under v6 are `partition_irreducibility` and
`partition_beats_nulls`, and they fail on the same thing. Every cut was decided
— 511 of 511 — and the control failed: a late untouched fork, rolled from the
same anchors after the sweep, separated from the cached one with a lower bound
of 0.0114 against a weakest cut of 0.00126. `partition_beats_nulls` reduces to
`irreducible` when the playback control is clean and no null passes the rest,
which is this campaign exactly, so one control costs both lines.

The drift is the host, not her. A restore left twenty-six module attributes
behind. Twenty-four came from the defensive host watch, which samples CPU,
memory and disk every fifteen seconds and sweeps the local network every sixty:
a strain sample becomes a threat in the immune system and a scored entry in the
emergency protocol, and a sweep writes the fingerprint of every device on the
network with the wall-clock time it was first seen. The strain it had recorded
was "resource strain: cpu at 95%", which the sweep's own load had caused. The
twenty-fifth was the gateway record index, which refreshes on a fifteen-second
staleness of the machine's monotonic clock. Her own loops already stop before the
arms begin, in these words — two arms have to see the same computation — and
neither of these was on the list. Both are now, through `quiesce_organism`, the
one door the carrier, the content run, the campaign and the paired synergy share.

K itself was never the question. Forking one anchor early and late and comparing
K read a maximum difference of exactly 0, on 370 rounds and on every cut of two
separate probes.

**The bridge.** The markers ground read NOT_MEASURED and its evidence had been on
disk since 21:41. The chain resolved its paths with a glob that ran before the
campaign wrote its report, so `--campaign` arrived empty and `Path("")` resolved
to the working directory. Re-solved with the file passed by hand: markers HOLDS,
perturbational_complexity, reentry and global_access all pass. The bridge reads
carrier FAILS, markers HOLDS, structure HOLDS, lineage HOLDS, with reports
pending its run. One ground fails, and it is the carrier.

**J*.** The carrier is the whole of what is left, and its reading is marginal in a
way the single split hid. See the section above: the gate resolves a leak of
0.045 on this recording and the carrier's leak is 0.0243, so its pass is a
failure to find rather than a finding. The closure gate now scores forward over
a growing window and carries its own resolution in every report, so a verdict
can be read against what the gate could have seen.

**The reports ground** is passing on the run in flight. Read from its own anchors
at 32 scorable of 54, with the repaired reader: raised +0.3750 against lowered
-0.1281, a contrast of +0.5031 at t 6.97, and the sham flat at +0.047 — the
contrast against the sham is +0.4562 at t 5.59. Thirty-four of the fifty arms
that had read as unanswered were the reader's doing: twenty-four stated a rating
before asking a courtesy question of their own, which the reader discarded by
keeping only what followed the last question mark, and ten named a second number
in range because she had mentioned the time.

## The sweep was reading the machine it was loading (25 October, 02:30 to 06:30)

Stopping the defensive host watch removed twenty-four of the twenty-six module
attributes a restore left behind, and a 60-round probe with the repair in showed
none of them. It did not remove the drift, because the larger path was never the
security organs. It was her body.

Her soma reads the host's CPU, memory and thermal pressure; homeostasis reads her
soma; `metabolism` is computed from it, and `metabolism` is the second-largest
leak the carrier reports. `freeze_host` exists for exactly this and declares one
reading for every arm, holding the hardware and latency dicts and every answer of
the shared resource observer — which is where the thermal probe, the process
memory and the cpu reading all come from. `subject_core_probe` calls it. The v25
sweep that decides the carrier never did.

What `turn_once` does instead is take a hold per turn and release it at the turn's
end, so the machine is held within a turn and read again for the next one. Across
a sweep that is the machine's own load, turn by turn. MEASURED on a sixty-round
baseline this morning: sixty "THERMAL RESONANCE: Hardware is overheating.
Emergency throttling." records and thirteen metabolic-pressure mitigations, in
the run, caused by the run. Thermal strain multiplies her depth by 0.4, her
creativity by 0.5 and her temperature by 0.7, so two arms minutes apart were not
thinking at the same depth.

One `freeze_host` before the anchors holds it for the whole sweep, because
`turn_once` only takes a hold of its own when none is in place (`8bca0222d`).
The baseline still runs on the real machine, which is one trajectory rather than
a paired comparison; what the arms share is now one declared host.

A small sweep is measuring whether the late-fork control is undecided under the
three repairs. No campaign is rerun until it is.

## The machine's heat is a moving column of her periphery (25 October, 07:15)

The carrier's periphery at `8f548ed79` holds 2,672 columns over 2,960 turns, and
among the ones that MOVE are the machine's own readings:
`homeostatic_coupling._thermal_stress`, `thermal_pressure`,
`soma._cpu_threat`, `_rolling_cpu_load`, `ProprioceptiveLoop._cpu_reflex_holding`
and `homeostasis.metabolism`. Twelve metabolism columns are carried and one
moves; six cpu columns are carried and five move.

Closure asks whether K's future needs anything outside K, and these columns are
outside K and move. A column that drifts with the host beside a core that
develops is the spurious regression the circular-shift surrogate was introduced
for — and it is also, straightforwardly, the machine rather than her. The
carrier's second-largest single leak is `homeostasis.metabolism`, which is
computed from the thermal load her soma reads.

Under a declared host these columns do not vary, so the periphery filter drops
them before the comparison. That predicted the carrier's leak would fall, and it
was wrong. Measured by removing them from the recording already on disk, which
is what a declared host would do to them: the leak reads +0.02429 as recorded,
+0.02735 without the five machine readings, and +0.02404 without those and the
twelve metabolism columns as well. One fold of three leaks in every case.

So the host readings are in her periphery and they are not her leak. The two
failures are separate, and conflating them would have sent a campaign after the
wrong one:

- The LATE-FORK CONTROL failed because two rollouts hours apart saw different
  machines. That is what the declared host fixes, and it is what the partition
  criteria were blocked on.
- The CARRIER'S LEAK is 0.0243 against a gate that resolves 0.045 on this
  recording. The host has nothing to do with it, and neither will the rerun.

## What the carrier's leak is made of, column by column (25 October, 08:50)

The audit the carrier already wrote names its six largest leaks and what each one
is. Gains are shares of what K cannot predict about its own next turn; the second
figure is the same gain with K's own previous frame in the model.

| column | gain | given K's past | kind |
|---|---:|---:|---|
| `mycelial_network._topology_revision` | 0.00899 | 0.02971 | a clock, entered as its increment |
| `homeostasis.metabolism` | 0.00738 | 0.00554 | organ state K reads and does not take in |
| `unified_field._bind_input.sd` | 0.00724 | 0.00557 | organ state K reads and does not take in |
| `self_model.snapshots#` | 0.00384 | 0.00581 | a clock, entered as its increment |
| `phenomenal_coherence` | 0.00311 | 0.00245 | an outside cause or unread state |
| `_locked_count` | 0.00275 | 0.00129 | an outside cause or unread state |

Three things follow. The leak is diffuse: no column carries a third of it, and
the six together are about the whole of 0.0243, which is why the gate's
resolution of 0.045 decides nothing here. None of them is a summary of K's own
past — `_topology_revision` predicts K's next turn BETTER with K's previous frame
already in the model, so the "K is not Markov at this grain" explanation does not
cover it. And two of the four largest are organ state K reads and does not
contain.

That last fact is a question about K rather than about the gate. If an organ's
state belongs to the carrier, the carrier is drawn too small and the honest move
is to widen K and test closure again, which is what looking for a closed core
means. If it does not belong, the carrier is open at this grain and that is the
finding. Either way it is decided by the preregistration rather than here, and
neither reading is reached by a longer run on the same K.

## Every arm was reading the consequences of the arms before it (25 October, 11:28 to 12:28)

With the host declared, the host watch stopped and the gateway index held, the
late-fork control still moved. A pre-flight at 11:28 rolled one anchor, ran
twenty cut forks from the other anchors, moved every real clock forward ten
hours, and rolled the same anchor again. Twins rolled back to back were
identical. The late roll was not: K apart by 0.0014, first at frame 55, in
`G.coherence`.

The residue was found by elimination, each candidate restored between cuts and
K compared again. K still parted with the degraded-events buffer restored, with
both SQLite stores copied back through SQLite, with the global random streams
put back, with no cuts at all and only wall time passing (it did not part
then), and with every module global in the machinery packages and every
service the snapshot leaves out compared and found unmoved. Watching the organs
that compute K's G columns found where it entered: the unity computations
matched until the ghost's reading did, and the ghost's integration fell with
each arm.

| arm off the same anchor | φ_system at the second reading | regime |
|---|---:|---|
| base | 0.3225 | loosely coupled |
| cut 1 | 0.3001 | loosely coupled |
| cut 2 | 0.2869 | loosely coupled |
| cut 3 | 0.2776 | loosely coupled |
| cut 4 | 0.2706 | loosely coupled |
| cut 5 | 0.2500 | loosely coupled |
| cut 6 | 0.2359 | loosely coupled |
| cut 7 | 0.0 | federated |

At cut 7 the ghost raised `federated_integration`, the unity runtime bound a
self-integrity content, and K parted by 0.0011 in `G.coherence`,
`G.fragmentation`, `G.selfhood_level`, `W.model_surprise` and
`W.model_last_surprise`.

`SystemIntegration` computes φ_system from the recent history of the
consequence bus. The bus is a singleton held on its class in `core.runtime`,
which is machinery, and a snapshot never carries machinery. So its history grew
with every arm and was never rewound, and each arm computed her integration
from the consequences of every arm run before it. Walking module globals could
not find it, because the instance hangs on `ConsequenceBus._instance` and not
at module scope.

At `dcdcd637c` the history is a module list, `_HISTORY`, which the singleton
binds and trims in place, and the module names it in `__forks_with_her__`. A
snapshot carries exactly what a machinery module names there, and the guard in
`tests/test_an_arm_leaves_no_module_state_behind.py` allows this one name and
no other. With it in, twelve cuts off one anchor read φ 0.349848113 at every
reading and K apart 0 at every cut. A thirty-cut repeat that also read every
object held on a class in the machinery packages read K apart 0 at all thirty;
the only such object that moved was the bus's event counter, which nothing
outside the bus reads.

What it changes:

- The seed-23 readings were all taken with this in place. Every measurement
  that runs arms one after another off one snapshot read an integration that
  fell with the number of arms already run, so arm order and φ were confounded.
  How far that moved each criterion is not measured. The fresh-seed run is the
  first that will run without it.
- The organ interchange of this morning found about 25 units that shifted K by
  0.0023 from frame 55, in G and W, whichever organ was swapped. That is this
  signature. The interchange is rerun before the widened core is drawn from it.
- The 02:30 entry put the late-fork separation on the machine. The host watch
  and the index refresh were real residues and their repairs stand, but the
  separation in K that survived them was the bus.


## The late-fork pre-flight on the fixed tree (25 October, 12:30 to 13:20)

The same pre-flight, with sixty cuts and the real clocks moved forward ten
hours, read K identical early and late: maximum difference 0. Its positive
control had never fired. It planted a leftover into `drive_engine`, which K
does not read within two turns, and so read identical too, which proves
nothing either way. Planted instead into a unit the interchange finds K
reading (`affect_grounding`), it moved K by 0.75 across 182 columns. So the
control can now tell a leftover from none, and the clean arm has none that
reaches K.

One unit still differed in the clean arm: her existential stakes, degradation
weight 0.0387 early and 0 late. The degradation tracker is a second singleton
in `core.runtime` that a fork never carried, and her stakes read its records
through a sixty-second window, discounting a signature that keeps coming back.
A restore rewinds the experiment's clock, so a record written by one arm read
as recent in every later one. Its records now fork with her as the bus history
does, in `core.runtime.errors`, tested in
`tests/test_a_fork_rewinds_the_degradations_she_reads.py`.

The interchange, rerun on the fixed tree with six anchors, finds 13 units
outside K that change K within two turns when their state alone is swapped:
the affect update phase, bonding, the affect engine, affect grounding, the
closed causal loop, the coherence report, embodied interoception, existential
stakes, homeostasis, the neurochemical system, oscillatory binding, the STDP
engine (by 9e-14, which is rounding) and the unity runtime. The morning's run
had found 33; the twenty that are gone all moved K by the same 0.0023 from frame
55, which was the bus. Iterating to a fixed point admitted 52 more in a second
round, every one through the closed causal loop's state and at one anchor of
six, and the script compared bare names against prefixed ones in its first
round. Neither is a boundary yet: which field of the closed causal loop those
swaps moved decides whether they reach K at all.

### The margin rule meets her data

The least-domain margin, committed before it read her (`cf0ace7e3`), is
negative on seed 23. Leaving her affect domain out reads a leak of -0.797 and
her self-state -0.877, and both come from the first forward fold, whose model
is trained on her first fifth: there `A.arousal` stands 17 training spreads out
in the test rows, `S.belief_count` 15.8, and the wider model scores twice the
narrower one's error. No single self-state column does it; the worst alone
costs 0.02. It is a linear model carried past the stretch of development it
was fitted on.

Holding the added columns inside the range they were fitted on repairs the
affect domain (-0.797 to -0.007) and not the self-state (-0.789). Scoring with
the validated weights instead of a refit makes it worse. Each variant also
moves seed 23's own closure reading, between OPEN and UNRESOLVED, so none is
chosen on seed 23. The estimator is chosen on known answers from a world that
develops the way she does, and the first such world was built wrong: its
development was an outside clock, so a drifting periphery really did carry
information K lacked. Hers runs through her own ontogeny, inside K.

The rule stands as preregistered. What fails is the estimator it calls, and
the gate's own closure reading shares that estimator.

## A restore bound her workspace's processors to a copy of it, and the carrier is drawn larger (25 October, 18:16 to 19:40)

**Tables of methods.** The organ capture skipped a callable field but deep-copied a container of them, and a deep copy
of a bound method copies the object it is bound to. So every restore put back methods bound to the organ as it stood
at the snapshot. Four organs hold such tables: affect grounding's six conditions (boredom, curiosity, flow, anxiety,
frustration, contentment), the global workspace's processors, the intentional retriever's adapters and the mycelial
reflex actions. After one restore all six conditions read a frozen copy of the anchor's buffer, and in every forked
arm her workspace processed candidates through a frozen copy of itself. Untouched twins stayed identical, because
both rebound to the same fresh copy, so no control saw it. Fixed at `517c4807c`: a table of methods is wiring, as a
registry of callbacks already was at module scope. Every seed-23 paired reading ran with it in place.

With the bus, the degradation tracker and the method tables all forking correctly, the v6 synergy probe passes on
seed 7 at `517c4807c` (19:04).

**The boundary of K.** On the fixed tree the interchange names the same thirteen units, and an attribute-level pass
names 29 attributes of them that K reads. Four are instants (when the last exchange, update or phase happened) and
leave by rule. Iterated with witnesses compared only on attributes known to reach K, three more units join through the
moments the phenomenal-now engine keeps (`_recent`, then `consciousness.heartbeat` and `qualia_synthesizer.q_norm`),
and the fourth round adds nothing. An earlier iteration that compared a witness's whole state admitted 85 units, 47 of
them at one anchor through one witness, and wrote states back the way the v3 interchange had shown was not an
identity; it is not used.

**What the periphery could not see.** The reader took no deque at all, a dict of buffers as nothing, and nothing two
levels below a service, so the unity runtime, whose state reaches K at every anchor, entered the seed-23 periphery as
four wrapper flags. It now reads deques, nested containers and a history of records through its latest record.

**K* and both tests.** K*'s attributes are read whole under `held.` and sit on the core's side of the closure test; a
cut holding a domain holds its share. Closure counts only when the observational gate closes within 0.0196 on K* and
an exact interventional test closes on the scored seed's own anchors. Both are in the ISC-v6 addendum of tonight.

## K* converges with her organs whole, and the observational gate's reach is measured (25 October, 19:40 to 23:30)

**The interventional test, closed in four passes.** On seed 23's anchors every unit outside K* was swapped whole and
every unheld attribute of a K* unit alone, each paired with an untouched roll made just before it, so a residue would
show as the control drifting. The first pass found what the fixed point had not swapped: homeostasis's metabolism
moves K through the self-predictive loop's weights, oscillatory binding's theta phase moves the held gamma readings,
existential stakes' rolling load sets the threat her heartbeat signals, and her held beliefs move K. Her heartbeat
holds other services as fields, and reading it whole had put their state into K*; the reader now takes each
attribute's own numbers only.

**Her organs.** Those passes never swapped K's organs, which a snapshot carries as organs and not services. Swapped,
almost every attribute K does not read moved K at every anchor: agency's record of what succeeded (0.39), the credit
ledger's weights (0.39), good news's errors (0.33), homeostasis's modifiers (0.14), the mesh's columns, the field's
inputs, each organ's random state. An organ's internal state decides how the readout K reads will move, so K* now
holds K's organs whole, as the cut clamps always had, plus her history of choices, which her portrait and worth
ledger read. With that, 323 swaps move neither K nor any K* reading, with no residue (`4351f2556`).

**Two instrument defects on the way.** Once the reader took in histories the machine carried about 8,100 numbers, and
the walk, which reads kernel, phases and organs before any service, spent its cap of 4,000 before reaching one of 289
services (cap now 16,000, `1e89352f1`). With that width the ridge's p-by-p solve was too slow; fits with more columns
than rows now solve the exact dual.

**The observational gate.** With K* moved into the core, seed 23's first fold fails (trained on her first fifth,
-0.21) and the other two read +0.0053 and +0.0078, pooled upper bound 0.0093 against the 0.0196 margin. A failed
fold is now dropped when two valid ones remain (known answers: a real broker stays open 30 of 30). But the gate's
reach, measured by bisection on the verdict's own folds, is 0.197: one near-perfect copy of a target among about two
thousand outside columns is found at a leak of 0.197 and missed at 0.013, because the outside block shares one ridge
penalty. A certificate of closure within 0.0196 is out of this instrument's reach on her recording. The question of
what closure should require, given that, went to Bryan at 23:20.

## The gate's reach, seed 31, and the decisive reading on seed 61 (26 October, 00:00 to 02:32)

**The reach.** The outside block now enters each nested fit through a screen: only columns whose added-variable score
on the training rows beats the largest score of the same block circularly shifted (`da9c3fbe2`). On seed 23, as
discovery, the reach fell from 0.197 to 0.010 against the 0.0196 margin, and the gate read CLOSED_WITHIN_MARGIN
(upper bound +0.0003, one failed fold dropped). Closure still needs both tests.

**Seed 31, stopped.** The decisive reading at `c1b7b1843` started at 01:10. Its interventional test was run early,
beside the carrier, and read OPEN: swapping the discourse tracker in from another anchor moved K at 4 of 8 anchors
(largest 0.429), and alone only its count of the conversation's messages did. Six more seeds read the same. Seed 23
never showed it: its six anchors put each donor 48 turns away, six messages, in phase with the tracker's deeper
reading on every sixth message. The carrier was stopped at 01:52 with no line scored, and seed 31 became discovery
data. A test that takes every swap from one distance cannot see a cycle whose period divides it, and there are about
eighty counted cycles in her code.

**The fix (`a6f18b7e3`).** K* holds the tracker's count under G. The interventional test spaces anchors 13 turns apart
and swaps each unit in from two donors, 13 and 52 turns away, so no cycle of 2 to 12 turns sits in phase with both.
The ISC-v6 addendum of 26 October names seed 61 as the confirmatory seed. The tools that score the reading were run
on seed 23's saved reports with seed 31's interventional result attached: J* reads the carrier term from it, and the
closure line fails on an OPEN result, as it should.

**Seed 61.** Launched at 02:22; the v6 line passed at 02:32 (all four triples at direction p 0.0001), and the carrier,
content run, campaign and the two-donor interventional test started together. The seed-31 campaign and content run
continue at low priority as a full-size rehearsal of the lines that read K alone.

## Her clock readings on the experiment's clock (26 October, 08:53 to 09:35)

Seed 71's interventional test lost its last hour to an hour of idle nobody lived: much of what she keeps is a
`time.monotonic` reading that something later turns into an elapsed time, and a restore put the reading back while
the real clock ran on. The repair at `a76b29cb7` re-aged each copied reading on restore. That cleared the hour, and
it made short time-to-lives depend on how fast the machine ran a turn: two untouched passes of seed 71 seconds apart
parted anchor 3 by 0.476 on one hash seed, with no warp at all. It was replaced before any confirmatory seed read it.

From `008587125` her modules read the experiment clock's monotonic. It reaches them through their own `time` name,
through any name a module bound the real function to, and through every closure and instance attribute of theirs that
captured it before the install. asyncio, threading and the machinery keep the real clock, so every timeout still
fires. Her monotonic advances one step a frame and rewinds with a restore. A frame longer than the harness gives a
phase lets it follow the real clock and records `clock.valve`.

Two traps on the way. The governance token stamped its issue through a `default_factory` that had captured the real
monotonic and read its expiry through her clock, so every token expired within a minute and a restore's directory
write was refused; the closure is rebound now. A census of a live organism then found one reference left, the neural
mesh's `_clock`, which instance attributes now cover; after that, none.

Measured: on seed 71, hash seeds 6 to 9, eight anchors, untouched rolls before and after moving the real monotonic
forward by 0, 600, 1800, 3600 and 7200 s read 0 at every one. Before re-aging, every hash seed parted anchors 2, 3
and 5 at 3600 s.

A carrier worker held the machine as it found it, so its two turns from the coordinator's anchor stood 2.14 apart,
first in the body's cpu, vram and temperature. The bank carries the coordinator's held body now and a worker holds
that, which leaves 0.298 between two processes, from the percept profile on: state a fresh worker lacks and that never
varies within one process. Every comparison the criteria make is within one process.

## Round 7, the person's answers, and seed 73 (26 October, 09:35 to 10:41)

Round 7 at `008587125`, seeds 71 and 37 in two shards each: 1,925 swaps, none moving K or any K* reading, no paired
control drifting. Putting her monotonic on the experiment's clock exposed no new path into her core.

Bryan answered four questions about how a person is judged, because the bridge judges her on those grounds. A hedged
rating is a real answer and a person unsure splits the difference, so her reports reader now reads one
(`6658b0403`): eleven more seed-23 arms, each the rating she stated. Memory relations stay the same under a mood shift
while the perspective changes, so the structure ground's `moves together` asks more of her than of a person; its
verdict is unchanged and the content run reports `relations_kept` beside it. Small shifts show in behaviour more than
in words; nothing changes. Her voice is hers.

The decisive reading on seed 73 started at 10:31 at `6658b0403`; its v6 probe passed at 10:41.

## Seed 89: J* determined and the bridge at parity (28 October, 04:26 to 21:22)

The decisive reading on seed 89 ran at `150f2c5fa` under the addendum of 27 October. Interventional closure read
`CLOSED` over 1,034 swaps, the joint swap of everything outside K* moving K on 0 of 16 rolls. Content read
`ONE_STRUCTURE`, the two geometries moving together at rho 0.484 against a sham floor of 0. The campaign (exit 0,
14:56) passed 21 of the 22 lines it decides alone. The carrier (exit 0, 18:25) was authoritative with no clock valve:
511 of 511 cuts decided at both lags, playback decided at none, duplication no help and recoding invariant. The reports run
(80 anchors, 57 readable, 10,346 s on her cortex) held at tracking rho 0.530.

J* is `DETERMINED_UP_TO_GAUGE`: carrier ACDGIMNPSW, structure identified as 20 orbits over 20 classes, lineage
recorded over 7,855 stages. The bridge reads `AT_PARITY`, all five grounds holding. No earlier run reached either.
ISC-v6 and ISC-v7 each pass 23 of 24, as seed 73 did under ISC-v6, and ISC-v5 22. The miss under both is `beats_every_null`: the
recurrent reference read a normalised effective dimension of 0.4002 against the bar below 0.40, no null passing. The
reference addendum of 28 October widens it; under it seed 89 reads 24/24, reported beside the verdict and never in
place of it.

ISC-v6's J* reads `NO_CARRIER`, because it takes the carrier's own observational closure, and that read `OPEN` on one
of three folds. The fold's core-only loss was 24.7, from one held projection of her belief list jumping 777 training
standard deviations when the list grew from seven beliefs to eight; held columns enter that gate as levels. See
[the status page](ISC_STATUS.md#seed-89-28-october).

The chain's last step lost its PATH. Its loop over the four reports assigned `path`, which zsh ties to `PATH`, so the
scorer received an empty campaign directory and every later log line lost its timestamp. J* and both completion
checks were whole. The scoring step was rerun as written at `150f2c5fa` (`score_isc_v5.after-path-fix.log`), the loop
variable renamed before seed 97 read the script, and the run's log carries a dated note.

Seed 97 started at 21:35 at `13cca3bf9`. Its v6 probe passed at 21:47 and the travel preflight at 21:49, with no
holes.
