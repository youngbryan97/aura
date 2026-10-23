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
