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
- [~] **B5. Behaviour that coordination alone does not show.** A preference held
  consistently, an interjection, a change of mind. Recorded as behavioural
  evidence beside the battery, not as a criterion of it. Built:
  `tools/what_she_does_of_her_own.py` reads her conation record. On 21 October,
  6 of the 149 things she has met hold a liking more than two standard errors
  from indifference: bicycles (0.73, 5 contacts), watching along with someone
  (0.12, 131), attending to her social drive (0.16, 68) liked; turns about
  herself (-0.18, 193), the desktop influence probe (-0.14, 143) and attending
  to her affect engine (-0.18, 58) disliked. Interjections and changes of mind
  have no record to read yet.
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
- [ ] **F3. Self-model and intentions as priors over policies** in the
  free-energy engine's expected free energy. Checked by clamping the self-model
  and reading the policy distribution. Not built: the expected-free-energy
  oracle in her code scores response text by a fixed heuristic and chooses none
  of her acts. The arbiter chooses them, and G3 put consequence there; a prior
  from her self-efficacy over act kinds goes there if W,A -> D still needs it.
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
