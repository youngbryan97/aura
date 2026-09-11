# The ISC completion list

Every checkbox in `Aura_ISC_Completion_Master_List.pdf`, one row each, with
its status read off this repository rather than remembered. An item counts as
done only when the checks recorded beside it in
`config/isc_completion_evidence.json` all pass, so a note that has gone stale
reverts to open by itself.

```bash
.venv/bin/python tools/isc_completion_status.py          # rewrite this file
.venv/bin/python tools/isc_completion_status.py --check  # fail if it is out of date
```

**480 done, 0 blocked, 0 not applicable, 423 open, of 903.**

Newest run with a report: `run_019` — 14/24 criteria, commit `cc4e5f1a60c6`.


## Phase 0 — Freeze what “passing” means

- [x] `P0.1` Keep thresholds centralized and immutable during a campaign. — `THRESHOLDS` in core/subject/battery.py, hashed into the campaign fingerprint
- [x] `P0.2` Keep ISC an AND, not a weighted score. — `Verdict.isc` is `all(...)`
- [x] `P0.3` Treat missing evidence as failure. — a criterion whose measurement did not run reads as failed
- [x] `P0.4` Run multiple full repetitions instead of reporting the best one; subject-core-frozen now exists. — `make subject-core-frozen`
- [x] `P0.5` Pick the exact candidate commit and freeze it. — every run records the commit it ran on, the hash of the tree that decides the answer, and whether that tree was dirty
- [x] `P0.6` Record commit SHA and tree SHA in every artifact. — commit, tree hash and dirty flag in every campaign block
- [x] `P0.7` Freeze every configuration value used by the run. — every frozen value hashed into the fingerprint
- [x] `P0.8` Hash the configuration. — every frozen value hashed into the fingerprint
- [x] `P0.9` Freeze the state schema. — the state schema's width and hash are in the frozen block
- [x] `P0.10` Freeze null architectures. — the null architectures are named in the frozen block
- [x] `P0.11` Freeze intervention size delta. — displacement size, ceiling, turns per arm and substrate step are frozen
- [x] `P0.12` Freeze intervention timing and lag horizon. — displacement size, ceiling, turns per arm and substrate step are frozen
- [x] `P0.13` Freeze all random seeds before seeing results. — the seed is recorded and enters the run fingerprint
- [x] `P0.14` Freeze the eight evaluation conditions. — the eight conditions are in the frozen block
- [x] `P0.15` Freeze the model version for the full-cortex experiment. — which mind answered is recorded with its module, its checksum and whether it decodes deterministically
- [x] `P0.16` Do not move thresholds because Aura narrowly misses one. — no threshold has moved; the contested line is reported failed beside its argument
- [ ] `P0.17` Any methodological change after seeing a result begins a new evaluation campaign. — the fingerprint is a hash over every frozen value, so a methodological change produces a different campaign by construction and the scorecard refuses to read across two
- [x] `P0.18` Keep all old failed runs permanently available. — a run directory is never reused, so a run that came out badly is still there
- [x] `P0.19` Generate human-readable scorecards from raw artifacts instead of manual transcription. — the scorecard is generated from the reports
- [x] `P0.20` Never overwrite prior runs; retain run_001, run_002, and so on. — `next_run_directory` never returns a name that already holds a report
- [x] `P0.21` Archive recordings, intervention arms, edge tables, null draws, lesion arms, rescue arms, and logs for every run. — the recording, every intervention arm, the edge table, the null draws and the lesion arms are written beside the report, with a hash for each

## Phase 1 — Rerun current Aura before adding more architecture

- [ ] `P1.1` Run make subject-core-frozen on a frozen current head.

**Required work:**

- [ ] `P1.2` Stop using 11/13/12 as the current score after that rerun.
- [ ] `P1.3` Record all three complete reports.
- [ ] `P1.4` Recompute stable-pass, unstable, and stable-fail criteria.
- [ ] `P1.5` Recompute the entire retained-edge graph.
- [ ] `P1.6` Identify the new largest strongly connected component.
- [ ] `P1.7` Identify the new minimum-Phi cut.
- [ ] `P1.8` Identify weakest outgoing coupling for every domain.
- [ ] `P1.9` Identify weakest incoming coupling for every domain.
- [ ] `P1.10` Recompute per-source perturbational spread.
- [ ] `P1.11` Recompute all four fixed synergy triples.
- [ ] `P1.12` Recompute per-condition graphs.
- [ ] `P1.13` Recompute lesion and rescue. Everything that follows should be driven by this new evidence rather than by the obsolete graph.

## Phase 2 — Complete the causal graph


**Perception (P)**

- [ ] `P2.1` P -> W: perception changes the learned world model. — P->W is a retained edge in the newest run
- [x] `P2.2` P -> A: perceptual content changes appraisal/affect. — P->A is a retained edge in the newest run
- [x] `P2.3` P -> G: salient perception changes attention competition. — P->G is a retained edge in the newest run
- [x] `P2.4` P -> C: sensory content changes recurrent cognition/substrate state. — P->C is a retained edge in the newest run
- [ ] `P2.5` P -> M: perception changes active/episodic memory. — P->M is a retained edge in the newest run
- [ ] `P2.6` P -> D: perception can alter intention/planning. — P->D is a retained edge in the newest run
- [x] `P2.7` Demonstrate a return route to perception through action/environment: D/S -> Act -> E -> P. — she acts on her own intention, the action writes the filesystem, and what the filesystem then holds is read back in through the senses in the same call
- [x] `P2.8` The return must be genuinely environmental, not a harness directly copying deliberation into perception. — the percept carries what was read off the disk — a line count, a room count, what a room holds — and never the objective or the intention text
- [x] `P2.9` Generalize the current file-action reafference probe beyond one scratch-file pathway. Interoception/body (I) — four action kinds over three different filesystem structures, chosen by what won attention and then by which drive is most depleted; one of them can fail, and fails for a reason of her own making
- [x] `P2.10` I -> A: sustained load/body state changes affect. — I->A is a retained edge in the newest run
- [x] `P2.11` I -> G: bodily pressure can compete for attention. — I->G is a retained edge in the newest run
- [x] `P2.12` I -> C: body state changes cognitive depth, temperature, focus, or recurrent dynamics. — I->C is a retained edge in the newest run
- [x] `P2.13` I -> D: resource state changes planning/action selection. — I->D is a retained edge in the newest run
- [x] `P2.14` Establish at least one legitimate route back into I. — the effort ledger is the route back into the body: five places in cognition report what they spent, the proprioceptive loop drains the ledger at the top of the turn, and exertion is what it computes
- [x] `P2.15` Cognition/action must be capable of changing some sensed bodily/computational state. — acting spends effort through the intention loop, and the interoception domain reads that spend as its own column
- [x] `P2.16` The return cannot be a test harness simply writing a new temperature or load number. A legitimate return loop could be: D→chosen computational workload →host/resource state →I. If I remains purely exogenous, it is difficult to justify including it inside a fully recurrent intrinsic core. Affect/conation (A) — the I displacement moves a share of what she has already spent through the ledger rather than writing a temperature; the readout the harness could have written is derived from the ledger at the top of every turn and would not have survived one
- [x] `P2.17` A -> G: feelings genuinely affect attention. — A->G is a retained edge in the newest run
- [ ] `P2.18` A -> D: affect changes motivation/intention. — A->D is a retained edge in the newest run
- [x] `P2.19` A -> C: affect changes cognitive dynamics. — A->C is a retained edge in the newest run
- [ ] `P2.20` A -> N: affectively significant experience changes development where justified. — A->N is a retained edge in the newest run
- [x] `P2.21` G -> A: broadcast/ignition changes later affect. — G->A is a retained edge in the newest run
- [x] `P2.22` W -> A: prediction error/surprise changes affect. — W->A is a retained edge in the newest run
- [x] `P2.23` I -> A: bodily state changes affect. — I->A is a retained edge in the newest run
- [x] `P2.24` C -> A: continuous recurrent/substrate state returns into canonical affect. — C->A is a retained edge in the newest run
- [x] `P2.25` Prove these as retained intervention edges rather than source-level wiring. Global workspace/attention (G) — every retained edge is a displaced arm against two sham arms, kept on a corrected q and a replication count rather than on the existence of a call between two modules
- [x] `P2.26` At least 3 heterogeneous outgoing retained causal edges on every successful replicate. — the workspace reaches three or more different domains on the newest run
- [x] `P2.27` Preferably preserve/establish G -> C. — G->C is a retained edge in the newest run
- [x] `P2.28` G -> S. — G->S is a retained edge in the newest run
- [x] `P2.29` G -> M. — G->M is a retained edge in the newest run
- [x] `P2.30` G -> A. — G->A is a retained edge in the newest run
- [x] `P2.31` G -> D. — G->D is a retained edge in the newest run
- [x] `P2.32` Multiple domains must return into G. — two or more domains reach the workspace on the newest run
- [ ] `P2.33` G must not become the mandatory hidden broker for the entire architecture. — the graph survives the removal of any one node, so no single domain brokers it; and the architecture where one does is a null the battery separates by closure
- [x] `P2.34` Registered processors only count if destination-domain state actually changes. Recurrent cognition/substrate (C) — an edge is scored on the destination domain's own columns, and a column whose reader failed is recorded as a miss rather than as a zero
- [x] `P2.35` G -> C: broadcast changes substrate/recurrent state. — G->C is a retained edge in the newest run
- [x] `P2.36` P -> C: perception enters continuous cognition. — P->C is a retained edge in the newest run
- [x] `P2.37` A -> C: affect changes recurrent dynamics. — A->C is a retained edge in the newest run
- [x] `P2.38` W -> C: prediction/world-state changes recurrent processing. — W->C is a retained edge in the newest run
- [x] `P2.39` C -> G: substrate activity/volatility changes attention. — C->G is a retained edge in the newest run
- [x] `P2.40` C -> A: recurrent state changes affect. — C->A is a retained edge in the newest run
- [x] `P2.41` C -> S: recurrent processing affects self-prediction/self-model where justified. — C->S is a retained edge in the newest run
- [ ] `P2.42` C -> D: recurrent computation changes deliberation. — C->D is a retained edge in the newest run
- [x] `P2.43` Verify the new substrate-volatility workspace bid earns a real retained edge. Self-state (S) — C->G is a retained edge in the newest run
- [x] `P2.44` Preserve S -> G. — S->G is a retained edge in the newest run
- [ ] `P2.45` Preserve S -> D/action. — S->D is a retained edge in the newest run
- [x] `P2.46` Preserve authorship-sensitive updates into S. — the two arms of the ownership experiment differ only in who is named as the author, and the self model is handed that attribution through the agency ledger
- [ ] `P2.47` Preserve M -> S. — M->S is a retained edge in the newest run
- [x] `P2.48` Add or retain additional independent incoming paths so S is not connected only through G/M. — the self is reached by something other than the workspace and memory
- [x] `P2.49` Add or retain independent outgoing paths that do not all require G. — the self reaches something other than the workspace
- [x] `P2.50` Generalize self-causation beyond one file task. Active memory (M) — ownership is measured over four action kinds rather than one, and it has to clear its own floor on more than one of them
- [x] `P2.51` G -> M: attended material changes active memory. — G->M is a retained edge in the newest run
- [ ] `P2.52` P -> M: perception enters memory. — P->M is a retained edge in the newest run
- [ ] `P2.53` S -> M: self-state changes retrieval/consolidation where appropriate. — S->M is a retained edge in the newest run
- [ ] `P2.54` N -> M: developmental state changes retrieval policy. — N->M is a retained edge in the newest run
- [x] `P2.55` M -> G: recalled material can win attention. — M->G is a retained edge in the newest run
- [ ] `P2.56` M -> S: memory changes the self. — M->S is a retained edge in the newest run
- [ ] `P2.57` M -> W: recalled context contributes to world inference. — M->W is a retained edge in the newest run
- [ ] `P2.58` M -> D: relevant memory changes planning. — M->D is a retained edge in the newest run
- [x] `P2.59` Ensure current M intervention affects the exact retrieved content that consumers use. World model (W) — the memory displacement moves the retrieved items themselves, which is what the consumers read
- [ ] `P2.60` P -> W: actual observations update W. — P->W is a retained edge in the newest run
- [ ] `P2.61` Action consequences update W. — D->W is a retained edge in the newest run — what she did changes what she predicts
- [ ] `P2.62` M -> W: remembered evidence helps construct/predict the world. — M->W is a retained edge in the newest run
- [x] `P2.63` W -> G: surprise/prediction error changes attention. — W->G is a retained edge in the newest run
- [x] `P2.64` W -> A: prediction error changes affect/free energy. — W->A is a retained edge in the newest run
- [ ] `P2.65` W -> D: predictions affect plans. — W->D is a retained edge in the newest run
- [x] `P2.66` W -> C: predictive discrepancies can alter ongoing cognition. — W->C is a retained edge in the newest run
- [x] `P2.67` Demonstrate multiple return routes rather than one brokered channel. Deliberation/intention (D) — two or more domains reach the world model on the newest run
- [ ] `P2.68` S -> D. — S->D is a retained edge in the newest run
- [ ] `P2.69` W -> D. — W->D is a retained edge in the newest run
- [ ] `P2.70` A -> D. — A->D is a retained edge in the newest run
- [x] `P2.71` G -> D. — G->D is a retained edge in the newest run
- [ ] `P2.72` M -> D. — M->D is a retained edge in the newest run
- [ ] `P2.73` D -> G: goals/urgency enter attention. — D->G is a retained edge in the newest run
- [ ] `P2.74` D -> W: chosen action produces evidence that changes the world model. — D->W is a retained edge in the newest run
- [ ] `P2.75` D -> P: through real action/environment/reafference. — D->P is a retained edge in the newest run
- [ ] `P2.76` D -> S: actions and commitments feed future self-state. — D->S is a retained edge in the newest run
- [x] `P2.77` D must not be merely a text field containing a goal. Ontogenetic/developmental state (N) — the deliberation domain is eighteen numbers — budgets, urgencies, the initiative count, what the will deferred — and displacing them changes what she does
- [x] `P2.78` Retain fast-to-slow causal influence. — fast domains reach slow ones on the newest run
- [x] `P2.79` Retain slow-to-fast causal influence. — slow domains reach fast ones on the newest run
- [x] `P2.80` A/C/G/P -> N through meaningful experience. — development is reached by more than one of the domains that carry experience
- [x] `P2.81` N -> A: novelty/development changes affect. — N->A is a retained edge in the newest run
- [x] `P2.82` N -> G: developmental novelty changes attention. — N->G is a retained edge in the newest run
- [ ] `P2.83` N -> M: development changes retrieval breadth/policy. — N->M is a retained edge in the newest run
- [ ] `P2.84` N -> D: accumulated development alters priorities/decisions. — N->D is a retained edge in the newest run
- [x] `P2.85` N must not be a terminal accumulator. — development reaches something; it is not the end of the line
- [x] `P2.86` N must not feed only one other node. — development reaches two or more domains
- [x] `P2.87` Specifically target information escaping N, C, I, and D if their attenuation remains high in the new run. — each of the four domains whose information was trapped now has an outgoing retained edge; the clamp for the layer that runs fastest and the dose matched to each domain's own variation are what freed them

## Phase 3 — Make the graph robust, not merely connected


**κ( G) ≥2**

- [ ] `P3.1` Full 10-node SCC. — all ten domains in one strongly connected component
- [ ] `P3.2` Remove P; remaining 9 are still an SCC. — removing P leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.3` Remove I; remaining 9 are still an SCC. — removing I leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.4` Remove A; remaining 9 are still an SCC. — removing A leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.5` Remove G; remaining 9 are still an SCC. — removing G leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.6` Remove C; remaining 9 are still an SCC. — removing C leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.7` Remove S; remaining 9 are still an SCC. — removing S leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.8` Remove M; remaining 9 are still an SCC. — removing M leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.9` Remove W; remaining 9 are still an SCC. — removing W leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.10` Remove D; remaining 9 are still an SCC. — removing D leaves the other nine strongly connected: vertex connectivity is at least two
- [ ] `P3.11` Remove N; remaining 9 are still an SCC. Do not solve this by creating a star centered on G or by duplicating one broker under two names. Look for genuine alternate recurrent loops such as: P -> W -> D -> P I -> A -> C -> I M -> W -> D -> S -> M A -> N -> M -> S -> A These are examples of desired topology, not mandated literal pathways. — removing N leaves the other nine strongly connected: vertex connectivity is at least two

## Phase 4 — Give every domain multiple recurrent lives

- [ ] `P4.1` P belongs to at least 2 distinct directed cycles. — P lies on at least two directed cycles
- [x] `P4.2` I belongs to at least 2. — I lies on at least two directed cycles
- [x] `P4.3` A belongs to at least 2. — A lies on at least two directed cycles
- [x] `P4.4` G belongs to at least 2. — G lies on at least two directed cycles
- [x] `P4.5` C belongs to at least 2. — C lies on at least two directed cycles
- [x] `P4.6` S belongs to at least 2. — S lies on at least two directed cycles
- [x] `P4.7` M belongs to at least 2. — M lies on at least two directed cycles
- [x] `P4.8` W belongs to at least 2. — W lies on at least two directed cycles
- [x] `P4.9` D belongs to at least 2. — D lies on at least two directed cycles
- [x] `P4.10` N belongs to at least 2. — N lies on at least two directed cycles
- [ ] `P4.11` P has at least one return path P -> X -> Y -> P. — P returns to itself through at least two other domains
- [x] `P4.12` I has one. — I returns to itself through at least two other domains
- [x] `P4.13` A has one. — A returns to itself through at least two other domains
- [x] `P4.14` G has one. — G returns to itself through at least two other domains
- [x] `P4.15` C has one. — C returns to itself through at least two other domains
- [x] `P4.16` S has one. — S returns to itself through at least two other domains
- [x] `P4.17` M has one. — M returns to itself through at least two other domains
- [x] `P4.18` W has one. — W returns to itself through at least two other domains
- [x] `P4.19` D has one. — D returns to itself through at least two other domains
- [x] `P4.20` N has one. A two-node ping-pong such as A <-> G is not enough for the explicit reentry bar. — N returns to itself through at least two other domains

## Phase 5 — Stabilize global access

- [x] `P5.1` G produces at least 3 outgoing retained causal edges. — at least three retained edges out of the workspace
- [x] `P5.2` Each meets q < 0.01. — each of them under the preregistered q
- [x] `P5.3` Each has effect >= 0.30. — each of them at or above the preregistered effect
- [x] `P5.4` Each replicates in at least 3 conditions. — each of them in at least three conditions
- [ ] `P5.5` This holds on every replicate run, not only 2 of 3. — the workspace's own tally says which sources reached attention, run by run, so a criterion holding on one run and not the next is visible
- [x] `P5.6` G’s consumers modify fields that the destination-domain reader actually reads. — a broadcast consumer writes a field the destination domain's reader reads, and the displacement test says which
- [x] `P5.7` Changes survive long enough to affect later state instead of being overwritten downstream. — what a consumer writes lands where the readout is derived from, and a gate fails when something starts overwriting it again
- [x] `P5.8` Consumer failures are visible. Resolve the affect-to-drive semantic mismatch A current workspace consumer can strip affect_ from a winner and treat the remainder as a drive name. That can turn an emotion such as affect_joy into joy, while motivation budgets use a different vocabulary such as social, curiosity, rest, integrity, and energy. Most emotions therefore do not correspond to a real drive budget. — a consumer that raised is recorded as a degradation rather than swallowed
- [ ] `P5.9` Define an explicit, theoretically justified affect-to-drive mapping; or
- [ ] `P5.10` stop treating affect_* as drive names and only credit genuine drive_* candidates.
- [ ] `P5.11` Add causal tests showing attended affect changes later deliberation if that is the intended mechanism.
- [ ] `P5.12` Do not let string-prefix stripping define psychology.

## Phase 6 — Raise perturbational spread to at least 0.60

- [ ] `P6.1` Increase legitimate long-range causal propagation.
- [ ] `P6.2` Do not do it by broadcasting one scalar everywhere.
- [ ] `P6.3` Preserve structured perturbational complexity.
- [x] `P6.4` Compute per-source spread and rank weak domains. — spread is reported per source and the weak ones are visible
- [ ] `P6.5` Improve lowest-spread sources first.
- [ ] `P6.6` Ensure I, C, N, and D perturbations escape those domains if they remain weak.
- [ ] `P6.7` Extend legitimate return routes.
- [ ] `P6.8` Test whether the 2-turn horizon is long enough for slow intended mechanisms.
- [ ] `P6.9` If horizon changes, preregister it before rerunning.
- [ ] `P6.10` Prefer multiple preregistered horizons over choosing the favorable one afterward.
- [ ] `P6.11` Verify effects persist beyond one instantaneous phase. — each edge records the lag its effect peaked at, so an effect at the horizon is visible
- [ ] `P6.12` Verify they do not arise entirely from one common broker. Do not solve spread by turning Aura into an all-to-all bus. — no single domain brokers the graph: vertex connectivity of at least two means removing any one leaves the rest connected

## Phase 7 — Make minimum partition irreducibility convincingly positive


**Required work:**

- [ ] `P7.1` Rerun after current causal fixes.
- [x] `P7.2` Identify the current minimum cut. — the cheapest cut is named in every run
- [ ] `P7.3` Determine which side predicts itself too independently.
- [ ] `P7.4` Determine what cross-cut information the intact predictor is failing to exploit.
- [ ] `P7.5` Strengthen real cross-cut dependencies.
- [ ] `P7.6` Repeat after the minimum cut shifts.
- [x] `P7.7` Continue until the cheapest cut, not merely average/dearest cuts, exceeds 0.05. — the cheapest cut, not the average, clears the bar
- [ ] `P7.8` Ensure no single domain is mostly self-contained.
- [ ] `P7.9` Ensure no large subgroup is mostly self-contained.
- [ ] `P7.10` Ensure cross-partition signals affect transition dynamics, not just snapshots.
- [ ] `P7.11` Increase recording length enough to reduce variance of the minimum.
- [x] `P7.12` Estimate confidence intervals or bootstrap uncertainty. — the score carries its own uncertainty, read off folds it was not chosen on
- [ ] `P7.13` Prefer a positive lower confidence bound. — and its lower bound is above zero
- [ ] `P7.14` For a truly strong result, prefer the lower bound itself to clear 0.05. — with the lower bound itself clearing the bar
- [ ] `P7.15` Repeat across seeds.
- [ ] `P7.16` Repeat across machines.
- [ ] `P7.17` Compare against matched surrogate floors.
- [ ] `P7.18` Use many more surrogate draws for authoritative inference.
- [ ] `P7.19` Use the same estimator on Aura and all nulls.
- [ ] `P7.20` Do not tune PCA/ridge/folds after seeing Aura’s direction without beginning a new preregistered campaign. The conceptual requirement is: Every attempted way of dividing Aura into two predictive machines must make the future harder to predict.

## Phase 8 — Beat matched nulls on irreducibility

- [ ] `P8.1` Real Phi exceeds replay surrogate. — irreducibility above the replay null
- [x] `P8.2` Real Phi exceeds time-shuffled surrogate. — irreducibility above the time_shuffle null
- [x] `P8.3` Real Phi exceeds star architecture. — irreducibility above the star null
- [ ] `P8.4` Real Phi exceeds stateful hub architecture. — irreducibility above the hub null
- [x] `P8.5` Real Phi exceeds one-way architecture. — irreducibility above the one_way null
- [x] `P8.6` Real Phi exceeds prompt-only architecture. — irreducibility above the prompt_only null
- [x] `P8.7` Real Phi exceeds frozen-slow architecture. — irreducibility above the frozen_slow null
- [x] `P8.8` The recurrent positive reference still passes. — the recurrent reference passes the conjunction the nulls fail, so the instrument can say yes
- [ ] `P8.9` Increase replay-surrogate draws substantially.
- [ ] `P8.10` Increase shuffle draws substantially.
- [x] `P8.11` Report distributions, not one or a few point estimates. — every draw is kept, not one point estimate
- [x] `P8.12` Report percentile and confidence interval. — the quantile a null is read at is recorded with it
- [ ] `P8.13` Add multiple random instantiations of each synthetic architecture. — several instantiations of each architecture, so a null is a distribution rather than one draw of a weight matrix
- [x] `P8.14` Match dimensionality, noise, persistence, and coupling strength as closely as practical. — the architectures are built with matched widths, decay, noise and coupling strength, and displaced to a matched dose over the same horizon

## Phase 9 — Make the null suite match the strength of the full equation

- [ ] `P9.1` Either rename the current criterion honestly to something like partition_null_separation, or

**or**

- [x] `P9.2` Preferably run the complete relevant metric suite on every null. — irreducibility, the graph measures and causal closure all run on every null, and the conjunction is what decides
- [x] `P9.3` Calculate null distributions for recurrence/topology. — the graph measures run on every architecture
- [ ] `P9.4` Differentiation.
- [x] `P9.5` Phi. — irreducibility runs on every null
- [ ] `P9.6` perturbational complexity/spread.
- [ ] `P9.7` synergy.
- [ ] `P9.8` intrinsic persistence.
- [x] `P9.9` causal closure where applicable. — causal closure runs on every architecture, with the broker read as the periphery
- [x] `P9.10` Evaluate the 24-part conjunction on each null. — each null is put through the conjunction rather than one line of it
- [ ] `P9.11` Require ISC(N)=0 for every null N. — no null passes
- [ ] `P9.12` Require the positive recurrent reference to pass the criteria it is meant to demonstrate. — the reference is held to the conjunction, not to one line of it
- [x] `P9.13` Add a hidden-broker null where the broker is deliberately outside K. — a broker outside K that remembers nearly all its own past; the graph cannot tell it from a mind and causal closure can
- [x] `P9.14` Add a high-dimensional independent-noise null. — ten domains of independent noise at the same width, decay and noise, with nothing crossing
- [ ] `P9.15` Add a common-clock/common-driver null.
- [ ] `P9.16` Add a random recurrent-network null.
- [ ] `P9.17` Add a memory-only recurrence null.
- [ ] `P9.18` Add a self-label-without-self-causation null.
- [ ] `P9.19` Add a pure all-to-all broadcast null.
- [ ] `P9.20` Add an LLM/prompt-broker null.

## Phase 10 — Hit the formal differentiation threshold honestly


**Required work:**

- [x] `P10.1` At least 40% of live dimensions contribute effectively. — the criterion is two-sided: many effective dimensions, no component holding half the variance, and the ratio below the bar rather than above it — the one-sided reading is passed by the least mind-like nulls and reported beside it on every run
- [x] `P10.2` Do not achieve this through independent noise. — independent noise scores at the top of the one-sided reading and is what the two-sided one excludes; the independent null measures it
- [x] `P10.3` Do not add filler state variables just to inflate dimension. — a column that never moves is named and excluded from the live width, so a filler variable cannot inflate the dimension
- [ ] `P10.4` Increase genuinely distinct cognitive modes.
- [ ] `P10.5` Increase content-sensitive variation in P/M/W/S.
- [ ] `P10.6` Let different conditions recruit meaningfully different configurations.
- [x] `P10.7` Avoid one ubiquitous latent/arousal factor dominating every domain. — no single component holds half the variance, which is the dominance limit under the bar
- [x] `P10.8` Remove invalid or truly redundant measurement columns only when scientifically justified. — columns are removed for failing the schema's own rule — nothing reads them, or they are another domain's column under this one's name — and the reason is written beside each removal
- [ ] `P10.9` Do not remove correlated columns simply because they hurt the score. Resolve the known specification problem The current project already found that the formal D_eff/D >= 0.4 bar can reward degenerate controls while a healthy recurrent reference scores much lower. Strong integration naturally shares variance, which can reduce participation ratio. Therefore there are two honest options: If the goal is a literal ISC-v1 24/24:
- [ ] `P10.10` Aura must genuinely reach >= 0.40 anyway. If the goal is the best scientific criterion:
- [ ] `P10.11` Keep ISC-v1 recorded as failed. — every run is kept with its own campaign fingerprint and its criteria, and the generated table in the document lists them all
- [ ] `P10.12` Preregister ISC-v2.
- [ ] `P10.13` Use a better differentiation measure.
- [x] `P10.14` Never retroactively change v1’s threshold. Do not add noise until the number turns green. — the thresholds are hashed into the fingerprint, so moving one starts a different campaign and the scorecard refuses to read across them

## Phase 11 — Fix measurement geometry for content


**Required work:**

- [x] `P11.1` Finite categories: use one-hot/effect coding. — `_one_hot` over declared vocabularies; `_ACTORS` and the unpredictable dimensions
- [x] `P11.2` Free text: use a fixed preregistered content representation. — `_content_buckets` projects every token onto every coordinate
- [ ] `P11.3` Consider stable semantic embeddings from a frozen model where justified.
- [x] `P11.4` Or use a fixed-dimensional lexical sketch whose distance has an explicit interpretation. — `_content_buckets` projects every token onto every coordinate
- [x] `P11.5` Represent memory content multidimensionally. — active memory carries a content profile and a real recency
- [x] `P11.6` Represent objectives multidimensionally. — the objective is a profile, not a hash
- [x] `P11.7` Represent attention focus appropriately. — attention focus is a profile
- [x] `P11.8` Represent world facts appropriately. — world facts are a profile over what they say
- [x] `P11.9` Represent self beliefs appropriately. — self beliefs are a profile
- [x] `P11.10` Treat actor/source labels as categorical, not scalar hash magnitude. — `_one_hot` over declared vocabularies; `_ACTORS` and the unpredictable dimensions
- [x] `P11.11` Recompute baseline scales after the representation change. — the scales are recomputed from the recording each run, so a schema change cannot leave a stale unit behind
- [x] `P11.12` Treat this as a new evaluation campaign because state geometry changed. This can materially affect Phi, differentiation, synergy, and edge effect sizes. — the schema hash is in the fingerprint, so the change begins a new campaign

## Phase 12 — Pass all four fixed synergy tests


**A + S -> G**

- [ ] `P12.1` Attention depends jointly on what Aura feels and what that state means relative to herself.
- [ ] `P12.2` Preserve this result if it remains a pass after rerun. P + M -> W
- [ ] `P12.3` World interpretation combines present observation with remembered context.
- [x] `P12.4` It must not be merely W=f(P)+g(M). — a second measurement asks the question a different way: whether the products of the two sources improve held-out prediction over the two side by side
- [x] `P12.5` There must be meaningful interaction f(P,M). — and the interaction term has to be there for the triple to mean anything
- [ ] `P12.6` Memory should disambiguate perception.
- [ ] `P12.7` Perception should update the meaning of recalled context. W + A -> D
- [ ] `P12.8` Deliberation uses both predicted world state and affect/value state.
- [ ] `P12.9` Preserve this if it remains a pass. S + D -> C
- [ ] `P12.10` Recurrent cognition depends on the combination of current self-state and current intention.
- [ ] `P12.11` The two signals should not merely add independently.
- [ ] `P12.12` Candidate mechanisms include identity-conditioned planning, self-consistency constraints, conflict/commitment computation, and goal relevance conditioned on self/value state. Measurement hardening:
- [x] `P12.13` Increase null draws beyond the current exploratory count. — a thousand shifts, not two hundred: the ninety-ninth percentile of two hundred draws is the second-largest of them
- [x] `P12.14` Report confidence intervals. — with the bar's own spread and the margin over it beside each triple
- [ ] `P12.15` Retain the independent held-out interaction-gain check.
- [ ] `P12.16` For the strongest claim, require both information-theoretic synergy and positive held-out interaction gain.
- [ ] `P12.17` Never change the four triples after seeing which ones pass.

## Phase 13 — Stabilize lesion deficit

- [ ] `P13.1` Increase lesion sample size.
- [ ] `P13.2` Increase lesion rounds.
- [ ] `P13.3` Increase lesion intervention trials.
- [x] `P13.4` Do not use a dramatically weaker lesion experiment than the main edge experiment. — the lesion arms are read at the partition that was cut, which is the comparison the main edge experiment makes within a column
- [ ] `P13.5` Measure more than the first three conditions.
- [ ] `P13.6` Use sufficient horizon.
- [ ] `P13.7` Preregister lesion power analysis.
- [ ] `P13.8` Intact/cut differences must exceed normal measurement noise.
- [x] `P13.9` Require Phi_do to fall. — irreducibility falls when the partition is cut
- [x] `P13.10` Require perturbational spread to fall. — perturbational spread falls
- [x] `P13.11` Require synergy to fall. — synergy falls
- [ ] `P13.12` Repeat across seeds.
- [ ] `P13.13` Repeat across run order.
- [ ] `P13.14` Rule out ordinary temporal drift.

## Phase 14 — Implement the lesion the equation actually specifies


**To satisfy the equation literally:**

- [ ] `P14.1` Build a canonical cross-domain channel registry/intervention layer.
- [ ] `P14.2` Identify real state-transfer channels crossing A* | B*.
- [x] `P14.3` Disable those channels without freezing either side internally. — each side runs with the other held at the cut, so no information crosses and neither side is frozen inside
- [x] `P14.4` Preserve A’s internal dynamics. — the free side keeps its own pipeline
- [x] `P14.5` Preserve B’s internal dynamics. — and so does the other, in its own arm
- [ ] `P14.6` Remove only E(A*,B*) and E(B*,A*).
- [ ] `P14.7` Verify no alternate harness bypass remains.
- [ ] `P14.8` Verify severed channels are actually inactive.
- [x] `P14.9` Measure the cut system. — the cut system is measured from the two arms composed column-wise
- [ ] `P14.10` Restore exactly those channels.
- [ ] `P14.11` Measure rescue.
- [x] `P14.12` Keep the existing node clamp as a separate useful ablation, but do not equate it with the formal partition lesion. This is one of the highest-priority methodological corrections in the program. — the node clamp is kept and reported under its own name

## Phase 15 — Make rescue real and reliable


**intact -> lesion deficit -> restore -> recovery**

- [ ] `P15.1` Lesion produces a reliable deficit first.
- [x] `P15.2` Restore only what was cut. — the rescue restores exactly the channels the lesion cut
- [x] `P15.3` Phi rises again. — irreducibility rises again
- [x] `P15.4` Spread rises again. — spread rises again
- [ ] `P15.5` Synergy rises again. — synergy rises again
- [ ] `P15.6` Recovery is larger than normal drift.
- [ ] `P15.7` Prefer rescued confidence intervals that overlap intact values.
- [x] `P15.8` Report recovery fraction: R= Mrescue−Mcut Mintact−Mcut . — the recovery fraction is recorded beside the arms
- [ ] `P15.9` Do not accept a trivial improvement as successful rescue merely because rescued > cut.
- [ ] `P15.10` Define recovery tolerance before the experiment.
- [ ] `P15.11` Allow a preregistered washout/recovery period if the architecture has memory.
- [ ] `P15.12` Show that releasing a lesion does not leave stale state indefinitely.
- [ ] `P15.13` Counterbalance lesion ordering in auxiliary validation.
- [ ] `P15.14` Show repeatable lesion-rescue cycles.
- [ ] `P15.15` Rescue must occur in the same continuing individual, not by replacing her with a fresh baseline instance.

## Phase 16 — Natural-condition replication

- [ ] `P16.1` At least three individual conditions produce a full 10-domain SCC. — at least three single conditions are strongly connected on their own
- [x] `P16.2` The result cannot be an SCC formed only by unioning edges across conditions. — each condition's graph is built from its own trials, never from the union
- [ ] `P16.3` Prefer more than three conditions.
- [x] `P16.4` Report which edges recur across contexts. — each edge records the conditions it replicated in
- [x] `P16.5` No special consciousness-test workload. — the eight conditions are ordinary work, declared in advance and hashed into the campaign
- [ ] `P16.6` Conversation alone should not carry integration.
- [ ] `P16.7` Tool use alone should not carry integration.
- [ ] `P16.8` Stress alone should not carry integration.
- [x] `P16.9` Include idle if the theory says integration persists during idle. — idle is one of the eight
- [x] `P16.10` Preserve meaningful differences among contexts. Settle the specification mismatch If the stronger mathematical specification requires every condition to satisfy the core criteria, while battery.py only requires SCC in three, choose one normative definition before the definitive campaign. — the conditions differ: their graphs are not the same graph
- [ ] `P16.11` Decide whether the formal equation or current battery semantics are authoritative.
- [ ] `P16.12` If the equation wins, require all intended conditions.
- [ ] `P16.13` If v1 wins, explicitly document that natural replication means at least 3 conditions.
- [ ] `P16.14` Do not leave two competing definitions of “pass”.

## Phase 17 — Fix intervention-arm state isolation

- [x] `P17.1` Snapshot/restore self-prediction. — `ORGAN_FIELDS` is read off the `Organs` declaration
- [x] `P17.2` Snapshot/restore the efference comparator. — `ORGAN_FIELDS` is read off the `Organs` declaration
- [x] `P17.3` Prefer deriving snapshot requirements from the organ dataclass/protocol instead of another hand-maintained list. — `ORGAN_FIELDS` is read off the `Organs` declaration
- [x] `P17.4` Add a regression test: every stateful organ read by K must be forked between arms. — tests/test_a_fork_leaves_nothing_behind.py
- [x] `P17.5` Snapshot all mutable fields that can affect future K. — the fork copies every field an organ carries, by value, two levels deep
- [x] `P17.6` Verify restored organ state equals the pre-arm snapshot within declared tolerance. This is the same class of bug that previously caused the harness to read self-prediction/comparator as absent despite those organs existing. Do not repeat it in the fork layer. — tests/test_a_fork_leaves_nothing_behind.py

## Phase 18 — Snapshot the intention system

- [ ] `P18.1` Make IntentionLoop state forkable.
- [x] `P18.2` Or use transaction rollback per arm. — the intention database is rolled back to the snapshot's rows per arm
- [ ] `P18.3` Or create an isolated database/state copy per arm.
- [x] `P18.4` Prevent perturbed-arm intentions from leaking into sham. — the intention database is rolled back to the snapshot's rows per arm
- [x] `P18.5` Prevent efficacy history from leaking. — the intention database is rolled back to the snapshot's rows per arm
- [x] `P18.6` Prevent action receipts from leaking. — the intention database is rolled back to the snapshot's rows per arm
- [x] `P18.7` Prevent comparator/efference records from leaking. — the comparator and the self-prediction organ are forked by name
- [ ] `P18.8` Test intervention-arm order reversal.

## Phase 19 — Snapshot the external scratch world

- [x] `P19.1` Fork/reset filesystem state between arms. — the scratch tree is captured byte for byte and put back per arm
- [x] `P19.2` Each arm begins from byte-identical world state. — the scratch tree is captured byte for byte and put back per arm
- [x] `P19.3` Action consequences remain inside the arm that caused them. — the scratch tree is captured byte for byte and put back per arm
- [x] `P19.4` A later sham cannot inherit a file produced by a prior intervention arm. — the scratch tree is captured byte for byte and put back per arm
- [x] `P19.5` inode/mtime differences cannot become uncontrolled cues unless explicitly modeled. — modification times are put back with the bytes
- [x] `P19.6` Action success comes from reading the arm’s own environment. — the probe reads its outcome off the filesystem it just wrote
- [ ] `P19.7` Prefer a transactional sandbox/environment snapshot. The strongest experiment forks: ( Kt,Et) , not only K_t.

## Phase 20 — Audit every mutable singleton for cross-arm leakage


**Perform an exhaustive fork-isolation audit across:**

- [x] `P20.1` StateRepository. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.2` ServiceContainer singleton state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.3` learned world-model state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.4` self-model state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.5` workspace state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.6` substrate state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.7` free-energy state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.8` ontogeny state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.9` agency ledger. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.10` self-prediction. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.11` efference comparator. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.12` intention records. — intention rows are rolled back per arm
- [ ] `P20.13` memory stores.
- [ ] `P20.14` background-task state.
- [x] `P20.15` RNG state. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.16` caches that affect cognition. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.17` filesystem/database state. — the scratch world is forked with the state
- [x] `P20.18` global module-level mutable variables. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [x] `P20.19` class-level counters. — one test: after an arm runs and the snapshot is restored, every K feature and every peripheral feature is back
- [ ] `P20.20` subprocess state if involved. Add one decisive regression test: After one intervention arm runs and its snapshot is restored, every measured K feature and every sampled peripheral-state feature must match the pre-arm snapshot within declared tolerance before the next arm starts. If not, the causal experiment is contaminated. Long-term, process-isolated arms from a serialized snapshot may be safer than indefinitely extending manual deep-copy logic.

## Phase 21 — Control randomness completely

- [x] `P21.1` Python random. — `random.getstate` carried and restored
- [x] `P21.2` NumPy RNG. — numpy's global state carried and restored
- [x] `P21.3` service-local RNGs. — service-local generators are deep-copied with the service
- [x] `P21.4` substrate RNG. — torch's generator, which the substrate draws its integration noise from
- [x] `P21.5` model sampling RNG. — the run installs a deterministic mind and records which one answered, so decoding cannot differ between arms
- [x] `P21.6` search/planning RNG. — search and planning draw from the process stream: no generator in the tree is built without a seed
- [x] `P21.7` randomized tie resolution. — the workspace breaks a tie on fatigue and then on the tick, never on arrival order, and its generator is seeded
- [x] `P21.8` file/environment randomness. — the scratch world an arm acted in is restored with the snapshot, and the machine the run ran on is recorded rather than assumed
- [x] `P21.9` randomized arm ordering remains recorded. — the order the three arms run in is drawn from the seeded generator and recorded with the trial
- [x] `P21.10` matched arms use the same stochastic stream where scientifically appropriate. — both arms of a trial draw the same stream: the process generators are snapshotted and every private generator now comes from them
- [x] `P21.11` independent replicate runs use distinct preregistered seeds. Sham-vs-sham divergence should quantify whatever nondeterminism remains. — the frozen campaign runs three declared seeds

## Phase 22 — Replace contaminating wall-clock dependence with an experiment clock

- [x] `P22.1` Introduce a deterministic experiment clock where possible. — core/subject/clock.py, installed for the run and rewound by a restore
- [x] `P22.2` Phases using elapsed time receive identical dt across matched arms. — `time.time` is the experiment's clock, so every elapsed interval is the same in both arms
- [x] `P22.3` Timestamps do not create cognitive differences by accident. — the fork also rewinds every wall-clock instant it can reach, by value
- [x] `P22.4` Recency advances identically. — `time.time` is the experiment's clock, so every elapsed interval is the same in both arms
- [x] `P22.5` Drive decay advances identically. — `time.time` is the experiment's clock, so every elapsed interval is the same in both arms
- [x] `P22.6` Memory recency advances identically. — `time.time` is the experiment's clock, so every elapsed interval is the same in both arms
- [x] `P22.7` Workspace fatigue/recency uses controlled progression. — `time.time` is the experiment's clock, so every elapsed interval is the same in both arms
- [x] `P22.8` Real wall clock remains only where it is genuinely the environmental variable under study. — `time.monotonic` is untouched, and the harness times itself on the real clock
- [x] `P22.9` Compare virtual-clock and ordinary runtime behavior to establish equivalence. — the step is measured from real frames before the clock is installed

## Phase 23 — Make every free-running cognitive loop deterministically stepable

- [x] `P23.1` NeuralMesh deterministic step. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.2` Neurochemical layer. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.3` EmbodiedInteroception. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.4` OscillatoryBinding. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.5` UnifiedField. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.6` SubstrateEvolution. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.7` ConsciousnessBridge. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.8` ClosedCausalLoop prediction. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.9` StreamOfBeing. — core/subject/steppable.py calls each layer's own loop body on a count
- [x] `P23.10` LiquidConsciousness/substrate. — the substrate takes its whole loop body on a count: dynamics and psych settling every iteration, the recurrent self-model every fifth, plasticity every hundredth
- [x] `P23.11` heartbeat. — the heartbeat is called once a turn, which is its own declared one hertz against a turn worth one second
- [x] `P23.12` every newly discovered free-running cognitive loop. — a layer with no entry point is named rather than skipped, and the check runs over whatever the organism brought up
- [x] `P23.13` Desktop runtime timers call these exact functions. — the harness calls the same methods the loops call; nothing here is a second implementation
- [x] `P23.14` Battery calls these exact functions in fixed sequence/count. — the harness calls the same methods the loops call; nothing here is a second implementation
- [x] `P23.15` No separate test-only implementation of cognition. — each layer steps at its own declared rate against the experiment's clock
- [x] `P23.16` All arms receive exactly the same number of steps. — the counts are in the report
- [x] `P23.17` Report the count. — a layer with no entry point is named rather than passed over
- [x] `P23.18` Fail an authoritative run if an active cognitive loop cannot be deterministically advanced. This would let the experiment test much more of Aura without sacrificing causal comparability. — a run whose cognitive loops cannot all be advanced by a count refuses rather than reporting

## Phase 24 — Include the real cortex

- [ ] `P24.1` Load the intended resident 27B cortex.
- [ ] `P24.2` Record exact model checksum.
- [ ] `P24.3` Record exact tokenizer.
- [ ] `P24.4` Record exact chat template.
- [ ] `P24.5` Use greedy deterministic decode, or fully fixed sampling RNG.
- [ ] `P24.6` Keep prompts identical except for causal state differences under test.
- [ ] `P24.7` Restore KV/cache state between matched arms.
- [ ] `P24.8` No fallback model during the run.
- [ ] `P24.9` No 9B serving one arm while 27B serves another.
- [ ] `P24.10` configured cortex = resident cortex = serving cortex.
- [ ] `P24.11` Verify model readiness before every authoritative run.
- [ ] `P24.12` Feed model output through the exact production pathways.
- [ ] `P24.13` Repeat matched interventions with the real cortex active.
- [ ] `P24.14` Identify edges that exist only through language.
- [ ] `P24.15` Compare G_nonlinguistic against G_full-cortex. This is essential because a missing edge in the current battery may simply mean that pathway requires language.

## Phase 25 — Replace scripted perception with controlled real perception

- [x] `P25.1` Record real sensory streams from normal Aura use. — a tape of percepts as they arrived, cut from whatever is living
- [x] `P25.2` Replay exactly the same stream to matched arms. — and replayed frame by frame, so two arms see the same world
- [ ] `P25.3` Include screen perception.
- [ ] `P25.4` Include OS state.
- [ ] `P25.5` Include audio if part of the tested system.
- [ ] `P25.6` Include host/body telemetry.
- [ ] `P25.7` Include user events.
- [x] `P25.8` Make timestamps deterministic/replayable. — the timestamps are the experiment's, not the day the tape was cut, so recency means the same thing in the run as it did when it was recorded
- [x] `P25.9` Preserve external content without directly mutating internal target domains. — a replayed percept goes on the stream through the same emitter a live one does and is indistinguishable downstream; nothing writes a target domain directly
- [ ] `P25.10` Let P be produced by the actual perception stack. Then run a stronger closed-loop experiment:
- [ ] `P25.11` Aura acts in a sandboxed real environment.
- [ ] `P25.12` The environment changes.
- [ ] `P25.13` Real sensors detect it.
- [ ] `P25.14` Perception changes W/A/M/etc.
- [ ] `P25.15` Those changes affect later action. Desired form: S/W/ D→Act→E→P→W/G/ S/ D.

## Phase 26 — Generalize agency and ownership

- [x] `P26.1` File creation. — file creation is one of the four she cycles through, and its ownership divergence clears its own floor
- [x] `P26.2` File modification. — file modification is one of the four she cycles through, and its ownership divergence clears its own floor
- [ ] `P26.3` Action failure.
- [ ] `P26.4` Partial success.
- [ ] `P26.5` Externally caused identical file state.
- [ ] `P26.6` Screen/UI action.
- [x] `P26.7` Information retrieval action. — information retrieval is one of the four she cycles through, and its ownership divergence clears its own floor
- [ ] `P26.8` Navigation.
- [ ] `P26.9` Task completion.
- [ ] `P26.10` Correctly predicted outcome.
- [ ] `P26.11` Incorrectly predicted outcome.
- [ ] `P26.12` Accidental self-caused outcome.
- [ ] `P26.13` Deliberate self-caused outcome. For each, match world outcomes while changing authorship alone where possible.
- [x] `P26.14` S divergence survives. — the self-state divergence survives every action kind it was asked about
- [x] `P26.15` Attribution generalizes beyond write_notes. — attribution is asked about four action kinds, not one
- [x] `P26.16` self-to-action causation generalizes beyond one task. — and both arms of a trial do the same thing, so what differs is who did it
- [ ] `P26.17` results survive multiple seeds.

## Phase 27 — Actually run an ontogeny

- [ ] `P27.1` Much longer-lived runs.
- [ ] `P27.2` Multiple developmental epochs.
- [ ] `P27.3` Early/mid/late analysis windows.
- [ ] `P27.4` N does not immediately saturate.
- [ ] `P27.5` Novelty changes with experience.
- [ ] `P27.6` Slow hidden state remains sensitive to history.
- [ ] `P27.7` Fast-to-slow coupling survives late in life.
- [ ] `P27.8` Slow-to-fast coupling survives late in life.
- [ ] `P27.9` Same current stimulus after two different histories produces appropriately different cognition.
- [ ] `P27.10` Two matched lives exposed to different histories diverge in N.
- [ ] `P27.11` Put both back into the same environment and measure downstream divergence.
- [ ] `P27.12` Process restart preserves intended developmental state.
- [ ] `P27.13` Distinguish genuine development from database accumulation.
- [ ] `P27.14` Ensure memory growth alone does not explain the effect. This is how N becomes evidence of development rather than a slow counter.

## Phase 28 — Make causal closure harder to fool

- [x] `P28.1` Classify actual clocks/timestamps by provenance instead of monotonicity alone. — a stored instant is dropped by its magnitude and by its name; what a clock is is a question about provenance
- [x] `P28.2` Do not drop a variable merely because it is monotonic. — a one-way column is kept and entered as its increment rather than dropped
- [x] `P28.3` Learning counters may be monotonic. — a learning counter is kept
- [x] `P28.4` accumulated evidence may be monotonic. — accumulated evidence is kept
- [x] `P28.5` developmental state may be monotonic over windows. — developmental state is kept
- [x] `P28.6` resource depletion may be monotonic. — a depleting resource is kept
- [ ] `P28.7` Add fixed-dimensional summaries for arrays/tensors.
- [x] `P28.8` Add content-sensitive summaries for maps/lists. — a map or a list enters as its size, so a peripheral collection is not invisible
- [x] `P28.9` Report how many potential variables are excluded by scan caps. — the walk reports how many numbers it read and whether it stopped at the cap
- [ ] `P28.10` Increase coverage or project all peripheral state into a fixed sketch.
- [x] `P28.11` Report traversal-depth limitations. — and how deep it went
- [x] `P28.12` Report reader failures. — and how many readers failed while it walked
- [ ] `P28.13` Repeat closure with multiple shuffled-periphery draws. — the floor is read from several permutations, not one
- [x] `P28.14` Add confidence intervals on leakage. — closure is decided against the upper tail of its own floor rather than the mean of it
- [x] `P28.15` Identify individual peripheral predictors when leakage occurs. — the leaking predictors are named one at a time
- [ ] `P28.16` Inject an adversarial hidden broker and prove the closure test finds it.
- [ ] `P28.17` If an outside variable predicts K, move it into the correct domain or redefine K and begin a new preregistered campaign.
- [ ] `P28.18` Never hide a leaking variable because it is inconvenient.

## Phase 29 — Audit whether K contains all causal cognitive state

- [x] `P29.1` Decide where energy belongs: D, I, A, or elsewhere. — energy is a motivational budget and belongs to deliberation: what it changes is which need is most depleted, and that is what the intention generator dispatches on
- [x] `P29.2` Decide where integrity belongs. — integrity is the same kind of thing and sits beside it, for the same reason
- [x] `P29.3` If causally active, represent them in K or ensure periphery closure can expose them. — anything not in K has to be findable by the closure test, which reads the machine minus the core
- [ ] `P29.4` Audit every phase read that influences future K.
- [ ] `P29.5` Map each persistent causal variable into P/I/A/G/C/S/M/W/D/N or explicitly outside K. — every column names the attribute it reads, so what is in K is a list rather than a memory
- [ ] `P29.6` Verify every omission with closure analysis. — and every omission is checked by the closure analysis rather than assumed
- [ ] `P29.7` Audit actual RAM/resource state if cognition reacts to it.
- [ ] `P29.8` Audit governance state if it changes cognitive selection.
- [ ] `P29.9` Audit model-serving state if it changes cognition.
- [ ] `P29.10` Audit active commitments.
- [ ] `P29.11` Audit persistent planner state.
- [ ] `P29.12` Audit active tool outcomes.
- [ ] `P29.13` Audit learning state. A core cannot be declared causally closed by definition.

## Phase 30 — Stop turning measurement failures into plausible zeros

- [x] `P30.1` Every measured feature carries present/read_ok metadata. — every reading carries the sources it could not read and why
- [x] `P30.2` Distinguish genuine zero. — 'organ absent', 'reader absent' and 'reader raised' are three states, not one zero
- [x] `P30.3` Distinguish absent organ. — 'organ absent', 'reader absent' and 'reader raised' are three states, not one zero
- [x] `P30.4` Distinguish failed reader. — 'organ absent', 'reader absent' and 'reader raised' are three states, not one zero
- [x] `P30.5` Distinguish stale reading. — a substrate reading older than its own freshness bound is a miss, not the safe defaults it returns
- [x] `P30.6` Count failures per feature. — the recording counts every miss over the run
- [x] `P30.7` Required-organ failure invalidates affected criteria. — a criterion resting on an organ that was absent for most of the run is invalid
- [x] `P30.8` Repeated reader error invalidates the run. — a reader that failed for most of the run makes the run unauthoritative, beside the per-criterion invalidation
- [x] `P30.9` organism.down is empty for required layers. — the run records which layers came up and which did not, and the newest run has none down
- [x] `P30.10` unwritable_domains is empty. — a domain whose writer did not bite is named, and the newest run names none
- [x] `P30.11` Required phase failures are zero. — a required phase that raised blocks the run; a phase absent by design does not
- [x] `P30.12` No final artifact silently encodes failed reads as ordinary zero. — a source that could not be read is recorded as a miss rather than as a zero, and a criterion resting on it is invalidated
- [x] `P30.13` Export a missingness/validity matrix. — the validity matrix is in the recording summary

## Phase 31 — Verify every perturbation moves what the domain’s consumers use

- [x] `P31.1` State-level writer/read consistency tests have begun. — a writer moves its own domain, and a displacement reaches a named consumer
- [x] `P31.2` C perturbation verified against actual live substrate features. — the substrate readout every consumer calls moves, and the modifiers cognition runs under move with it
- [x] `P31.3` N perturbation verified against the actual lifetime reservoir. — the reservoir N reads is the organ's shared lifetime state, and displacing it moves the novelty every consumer reads
- [x] `P31.4` S organ perturbation hits actual self-model/self-prediction state. — the displacement goes through the self model's own governed update and leaves a belief behind
- [x] `P31.5` W perturbation hits learned world-model state that consumers read. — an observation of a world she is not in moves the surprise every consumer reads
- [x] `P31.6` G perturbation changes actual workspace state. — the runner-up is raised under its own source until it wins, so both arms attend to different things and act differently
- [x] `P31.7` D perturbation changes actual deliberative inputs. — the probe intention states an urgency the competition can price, and the displaced budgets change which need is worst
- [x] `P31.8` I perturbation survives host-freeze matching correctly. — the host freeze is re-taken from the displaced state, and the body keeps a channel the freeze does not overwrite
- [x] `P31.9` Every perturbation moves multiple meaningful features rather than one decorative scalar. — every domain's displacement moves at least two of its own features, counting both halves of the intervention
- [x] `P31.10` Magnitude remains inside normal operating range. — every state write in a writer states its own bounds, so four times the campaign's displacement is still a state she could be in
- [x] `P31.11` Intervention does not directly write downstream domains. — no writer bumps a state path another domain's schema owns
- [x] `P31.12` Intervention does not inject the expected answer into the system. — no writer names the recording, the edge test, the thresholds or a reading function

## Phase 32 — Establish dose-response


**Run an auxiliary preregistered campaign at several perturbation sizes, for example:**

- [ ] `P32.1` delta = 0.05
- [ ] `P32.2` delta = 0.10
- [ ] `P32.3` delta = 0.15
- [ ] `P32.4` delta = 0.20 if still within normal state range.
- [ ] `P32.5` Sign reversal where meaningful.
- [ ] `P32.6` Effect grows reasonably with dose.
- [ ] `P32.7` Direction behaves sensibly.
- [ ] `P32.8` No edge exists only at one magical intervention size.
- [ ] `P32.9` No discontinuity reveals a harness threshold artifact. Do not retroactively pick the delta that makes v1 pass.

## Phase 33 — Make the sham floor extremely clean

- [ ] `P33.1` Sham A versus Sham B is near zero in every domain.
- [ ] `P33.2` Report floor by domain.
- [ ] `P33.3` Report floor by condition.
- [ ] `P33.4` Report floor by lag.
- [ ] `P33.5` Report floor by run.
- [ ] `P33.6` No domain’s causal threshold is dominated by restoration noise.
- [x] `P33.7` No singleton contamination. — module singletons the container does not hold are carried across the fork, the synaptic cleft and its receptor bank among them
- [ ] `P33.8` No host drift.
- [ ] `P33.9` No timestamp drift.
- [ ] `P33.10` No DB contamination.
- [ ] `P33.11` No filesystem contamination.
- [x] `P33.12` No model-sampling contamination. — the run installs a deterministic mind and records which one answered, so decoding cannot differ between arms
- [ ] `P33.13` No background-task timing contamination. If sham noise approaches the effect threshold, a negative result for that domain is not interpretable.

## Phase 34 — Preserve perturbational complexity while raising spread

- [x] `P34.1` Response remains nonzero. — every source produces a nonzero response
- [x] `P34.2` Response remains distributed. — the response reaches more than one domain from every source
- [ ] `P34.3` Response remains temporally structured.
- [ ] `P34.4` Response remains heterogeneous.
- [x] `P34.5` Response stays above matched-null complexity. — above the matched null's complexity
- [x] `P34.6` Multiple source domains produce nondegenerate responses. — more than one source produces a nondegenerate response
- [ ] `P34.7` Do not sacrifice structure to make every domain light up.

## Phase 35 — Preserve intrinsic persistence

- [x] `P35.1` Environment-only prediction remains worse than environment-plus-state. — predicting from the environment alone is worse than from the environment and the state
- [x] `P35.2` Shuffling internal state removes the advantage. — shuffling the state removes the advantage
- [ ] `P35.3` Result persists across seeds.
- [ ] `P35.4` Result persists with the real cortex.
- [ ] `P35.5` Result persists across several ordinary conditions.
- [x] `P35.6` It is not explained by a single counter/clock. — a column that never decreases is named in the recording, so a counter carrying the result is visible
- [ ] `P35.7` It is not driven entirely by one persistent memory scalar.

## Phase 36 — Preserve functional self-causation


**Maintain:**

- [x] `P36.1` Remains above same-arm floor. — above the same-arm floor
- [ ] `P36.2` Generalizes across multiple tasks.
- [ ] `P36.3` Generalizes across multiple self dimensions.
- [x] `P36.4` No direct test-only S-to-D wire. — no writer touches a state path another domain's schema owns
- [x] `P36.5` Mediated by normal production cognition. — the displacement goes through the self model's own governed belief update and the ordinary phase loop carries it
- [ ] `P36.6` Survives current code changes.
- [ ] `P36.7` Survives the full-cortex variant.

## Phase 37 — Preserve ownership


**external**

- [ ] `P37.1` Same external outcome. — the two arms end with the same file holding the same bytes
- [x] `P37.2` Same content. — the same fact recorded
- [ ] `P37.3` Same goal. — the same goal appended with the same text
- [ ] `P37.4` Same percept.
- [x] `P37.5` Only actor attribution differs. — the ownership arms match on everything but who the action is attributed to
- [x] `P37.6` Divergence exceeds same-arm floor. — the divergence clears the same-arm floor
- [x] `P37.7` Generalizes to multiple action types. — ownership holds for more than one kind of action
- [ ] `P37.8` Actor identity does not leak into unrelated state except through the ownership mechanism.
- [x] `P37.9` Comparator and agency ledger disagreements are visible and interpretable. — the comparator and the agency ledger are read as separate columns, so a disagreement is visible

## Phase 38 — Preserve fast-to-slow and slow-to-fast coupling

- [x] `P38.1` At least one fast-to-slow retained edge. — at least one fast-to-slow retained edge
- [x] `P38.2` At least one slow-to-fast retained edge. — at least one slow-to-fast retained edge
- [ ] `P38.3` Survives replicated runs.
- [ ] `P38.4` Survives a long life.
- [ ] `P38.5` Survives after developmental state has matured.
- [x] `P38.6` Not explained by memory count alone. — the coupling is not a memory count: the edges are measured on displaced arms against their own shams, not on a correlation
- [ ] `P38.7` Not explained by time/counters alone.
- [x] `P38.8` Not a test-only direct write. — no writer reaches into another domain

## Phase 39 — Preserve metastability

- [x] `P39.1` More than one regime. — more than one regime
- [x] `P39.2` Regimes persist. — a regime lasts rather than flickering
- [x] `P39.3` Regimes turn over. — and it turns over
- [x] `P39.4` Transition entropy is greater than zero. — transition entropy above zero
- [x] `P39.5` Transition entropy is below its maximum. — and below its ceiling, so the sequence is not noise
- [ ] `P39.6` Clustering is not merely rediscovering condition labels.
- [ ] `P39.7` Replicate with held-out conditions.
- [ ] `P39.8` Persist with the real cortex.
- [ ] `P39.9` Prefer interpretable internal configurations rather than pure workload classes.

## Phase 40 — Resolve the integration-versus-differentiation conflict scientifically


**Run a dedicated calibration study:**

- [ ] `P40.1` Plot Phi versus normalized effective dimension across synthetic systems with controlled coupling strength.
- [ ] `P40.2` Repeat across Aura ablations.
- [ ] `P40.3` Map the tradeoff.
- [ ] `P40.4` Determine whether a nondegenerate recurrent system can satisfy both Phi_do > 0.05 and D_eff/D >= 0.40 under the current estimator.
- [ ] `P40.5` If yes, demonstrate it.
- [ ] `P40.6` If no, formally document ISC-v1’s internal inconsistency.
- [ ] `P40.7` Design ISC-v2 before observing the next Aura result used for v2.
- [ ] `P40.8` Retain v1 forever for provenance. A test should be difficult, but it should not be internally impossible for the positive reference it is designed around.

## Phase 41 — Improve the irreducibility null floor

- [x] `P41.1` Generate many matched surrogate recordings. — many matched surrogate recordings, not a handful
- [x] `P41.2` Same duration. — same duration: a surrogate is the recording itself, resampled
- [x] `P41.3` Same dimensionality. — same dimensionality, for the same reason
- [x] `P41.4` Same autocorrelation where possible. — every autocorrelation kept
- [x] `P41.5` Same marginal distributions. — every marginal kept
- [x] `P41.6` Destroy only cross-domain alignment. — and only the alignment between domains destroyed
- [x] `P41.7` Build a distribution of minimum Phi. — a distribution of the minimum, not one draw
- [x] `P41.8` Estimate 95th and 99th percentiles. — the ninety-fifth and ninety-ninth percentiles of it
- [x] `P41.9` Report Aura’s margin above the null floor. — and Aura's margin over it
- [x] `P41.10` Report uncertainty. — with the floor's own spread beside it, so a margin can be read as a margin
- [x] `P41.11` Do not use only a handful of surrogate draws for publication-level evidence. — sixty-four draws by default, not a handful

## Phase 42 — Make real and null graph tests symmetric

- [ ] `P42.1` Feed synthetic nulls through an observational/intervention pipeline as similar as practical to Aura’s.
- [x] `P42.2` Same effect threshold. — the same effect threshold on the nulls as on the real system
- [x] `P42.3` Same lag horizon. — and the same number of frames to reach it
- [x] `P42.4` Comparable noise. — comparable noise, declared with the architecture
- [x] `P42.5` Comparable trial count. — and a matched dose, so what differs is what escapes rather than how hard each was hit
- [ ] `P42.6` Comparable multiplicity treatment.
- [ ] `P42.7` If analytical/clean null edges are retained, explicitly justify why that comparison is conservative. — the null's two arms share a noise stream, so its floor is exactly zero, and that is stated where the comparison is made

## Phase 43 — Add adversarial positive and negative controls

- [ ] `P43.1` Decentralized recurrent positive reference.
- [ ] `P43.2` Star broker.
- [ ] `P43.3` Stateful hub.
- [ ] `P43.4` Hidden external broker.
- [ ] `P43.5` One-way chain.
- [ ] `P43.6` Simple ring.
- [ ] `P43.7` Prompt-only broker.
- [ ] `P43.8` Frozen slow state.
- [ ] `P43.9` Independent high-dimensional processes.
- [ ] `P43.10` Common global noise/common driver.
- [ ] `P43.11` All-to-all broadcast.
- [ ] `P43.12` Memory-only persistence.
- [ ] `P43.13` Fake self-description with no causal self.
- [ ] `P43.14` Agency without ownership.
- [ ] `P43.15` Ownership label without action causation.
- [ ] `P43.16` Fast-to-slow but no slow-to-fast.
- [ ] `P43.17` Slow-to-fast but no fast-to-slow.
- [ ] `P43.18` Recurrent but minimally differentiated.
- [ ] `P43.19` Differentiated but not integrated. Each should fail or pass for the expected reason.

## Phase 44 — Remove hidden test-specific shortcuts

- [x] `P44.1` No production subsystem knows “make ISC pass”. — no module under measurement names the battery
- [x] `P44.2` No testing flag adds integration solely for the battery. — and no environment switch opens a path only the harness takes
- [x] `P44.3` No hidden direct P-to-W or S-to-D pathway used only in the harness. — no writer reaches into a domain it does not own
- [ ] `P44.4` No test-only workspace consumers.
- [ ] `P44.5` No fake percept solely to manufacture an edge unless it faithfully models a production sensory route.
- [x] `P44.6` No threshold value imported into production cognition. — the thresholds live in the battery and nothing under measurement imports them
- [x] `P44.7` No mechanism built solely to satisfy a score without an independent functional rationale. — no writer names the recording, the edge test, the thresholds or a reading function
- [ ] `P44.8` Every added causal route has a normal cognitive or engineering purpose. The ideal outcome is that Aura passes because the architecture naturally has the property, not because the test became another organ.

## Phase 45 — Recheck every previously discovered dead mechanism

- [x] `P45.1` Audit services with no callers. — services asked for through the read-only seam that are only built on demand are listed
- [ ] `P45.2` Audit registered processors with no consumers. — a registered processor that returns early every time is named rather than counted as wired
- [x] `P45.3` Audit readers with no writers. — a reader with no writer shows up as a column that never moves, and the recording names them
- [x] `P45.4` Audit writers with no readers. — a writer with no reader shows up as a displacement that reaches no consumer, and every displacement names one
- [ ] `P45.5` Audit bid types that never win. — the workspace counts what was offered and what won, so a source that never gets in is named rather than looking like one that never spoke
- [ ] `P45.6` Audit consumers that always return early. — every broadcast consumer reports whether it ever did anything, and none of them never did
- [x] `P45.7` Audit model observers that never receive observations. — the world model's own step count moves across a run, so it received the observations the cycle sends it
- [ ] `P45.8` Audit self-model fields that never change. — a self-model field that never changes is a flat column, and the recording names every one
- [x] `P45.9` Audit affect fields overwritten later in the same cycle. — no field is written by one step of a turn and replaced by a later one without reading what was there
- [x] `P45.10` Audit return paths that terminate in local dictionaries. — a mapping filled in and never read is an audit with a baseline and a worked example, and nothing in cognition or the subject core has one
- [x] `P45.11` Audit background loops that start but immediately die. — a layer that did not come up is named, and a run with one refuses
- [x] `P45.12` Audit service-name mismatches. — the substrate is published under every name its consumers ask for, and the coupling asks again while it is absent
- [x] `P45.13` Audit aliases that exist only in desktop boot but not in the measurement runtime. — the substrate is published under every name its consumers ask for, and the run records that the alias went up
- [x] `P45.14` Audit all 10 domains for intervention attenuation. Remember: written != wired != causally consequential Only the last one matters to ISC. — how far each domain was displaced and how far its largest outgoing effect reached, per domain, in every run

## Phase 46 — Fix remaining lifecycle/race conditions before long evidence runs


**Before multi-hour or multi-day developmental validation:**

- [x] `P46.1` Singleflight construction for ontogeny singleton. — both singletons build outside the lock and publish under it, so a race builds two and keeps one rather than holding a process-wide lock across an fsync
- [x] `P46.2` Build a non-started candidate, publish the winner, then start it. — the candidate is built without starting, and the winner is started after it is published
- [x] `P46.3` Loser disposal must not close shared dependencies it does not own. — the organ flushes the spine rather than closing it; closing a singleton it did not build stopped the flusher for every other holder
- [x] `P46.4` Callbacks must be unregisterable. — the spine's resolution subscribers can be taken off, subscribing twice subscribes once, and the losing candidate takes its own off
- [x] `P46.5` Give ExperienceSpine explicit ownership semantics. — whoever builds the spine closes it and nobody else; the shared instance is marked, a holder owes it a flush, and a close on a shared one is recorded rather than silent
- [x] `P46.6` Stress-test concurrent initialization. — eight threads on a barrier ask at once and get one organ
- [x] `P46.7` Verify no silent developmental shutdown. This is not itself an ISC checkbox, but it is a blocker to trusting long-run N measurements. — the organ coming down leaves the spine's flusher running, and the ledger keeps the calibration monitor the trainer is feeding

## Phase 47 — Make evidence artifacts self-verifying


**Every authoritative run should include:**

- [x] `P47.1` commit SHA. — commit, tree hash and dirty flag
- [x] `P47.2` tree SHA. — commit, tree hash and dirty flag
- [x] `P47.3` dirty-tree status. — commit, tree hash and dirty flag
- [x] `P47.4` Python version. — python, dependency lock hashes, OS and hardware
- [x] `P47.5` dependency lock hash. — python, dependency lock hashes, OS and hardware
- [x] `P47.6` OS version. — python, dependency lock hashes, OS and hardware
- [x] `P47.7` hardware profile. — python, dependency lock hashes, OS and hardware
- [x] `P47.8` model name. — which mind answered, with a hash of its code
- [x] `P47.9` model-weights checksum. — which mind answered, with a hash of its code
- [x] `P47.10` tokenizer checksum. — which mind answered, with a hash of its code
- [x] `P47.11` configuration hashes. — configuration, threshold, schema and null hashes in the fingerprint
- [x] `P47.12` threshold hash. — configuration, threshold, schema and null hashes in the fingerprint
- [x] `P47.13` state-schema hash. — configuration, threshold, schema and null hashes in the fingerprint
- [x] `P47.14` null-definition hash. — configuration, threshold, schema and null hashes in the fingerprint
- [x] `P47.15` seed. — seed, start time and duration
- [x] `P47.16` start time. — seed, start time and duration
- [x] `P47.17` total run duration. — seed, start time and duration
- [x] `P47.18` phase-failure counts. — phase failures are counted per phase
- [x] `P47.19` organ-availability report. — the organ-availability report is the missingness matrix
- [x] `P47.20` stopped-loop list. — the stopped and live loop lists, and the layer step counts beside them
- [x] `P47.21` live-loop list. — the stopped and live loop lists, and the layer step counts beside them
- [x] `P47.22` frame count. — frame count
- [x] `P47.23` turn count. — turn count
- [x] `P47.24` intervention count. — intervention count
- [x] `P47.25` q-value resolution/power note. — the smallest q the sign-flip test can return, beside the draws that set it
- [x] `P47.26` every tested edge, not only retained edges. — every ordered pair is in the table with its effect, its q and whether it was kept
- [x] `P47.27` all raw null draws. — every null draw, not the summary
- [x] `P47.28` raw lesion arms. — the lesion arms as measured
- [x] `P47.29` raw rescue arms. — the rescue arm as measured
- [x] `P47.30` raw synergy reports. — each synergy triple with its joint information, its redundancy, its unique terms and its null
- [x] `P47.31` all partition-cut scores. — a SHA-256 for every file the run wrote
- [x] `P47.32` per-condition graphs. — the exact command that regenerates the report
- [x] `P47.33` SHA-256 artifact manifest. — a SHA-256 for every file the run wrote
- [x] `P47.34` exact command that regenerates the report. — the command that produced the run, recorded with it
- [ ] `P47.35` Documentation generated from artifacts rather than hand-edited summaries. — the evidence table in the document is written from the reports, and a gate fails when it drifts

## Phase 48 — Protect the evaluation commit


**For an authoritative result:**

- [ ] `P48.1` Create an immutable evaluation tag.
- [ ] `P48.2` Prefer a signed tag.
- [ ] `P48.3` Require CI for the evaluation branch/release.
- [ ] `P48.4` Protect main or at least the evidence branch/tag process.
- [ ] `P48.5` Do not rewrite history.
- [x] `P48.6` Archive exact source tarball/commit. — the commit, the hash of the tree that decides the answer and whether it was dirty are recorded with every run
- [ ] `P48.7` All tests defining the evaluation are green.
- [ ] `P48.8` Evidence artifacts point to that exact tag. — and a SHA-256 for every file the run wrote

## Phase 49 — Replicate over independent seeds

- [x] `P49.1` Run seed A. — seeds 7, 11 and 13, declared in the target
- [x] `P49.2` Run seed B. — seeds 7, 11 and 13, declared in the target
- [x] `P49.3` Run seed C. — seeds 7, 11 and 13, declared in the target
- [ ] `P49.4` Prefer 5 or more for statistical characterization.
- [x] `P49.5` No seed selected based on result. — the seeds are in the target, chosen before any result
- [ ] `P49.6` For the strongest claim, all 24 pass each run.
- [ ] `P49.7` Otherwise report per-criterion pass frequency.
- [ ] `P49.8` Report mean/std/range for continuous measures.
- [ ] `P49.9` Especially Phi.
- [ ] `P49.10` Especially synergy.
- [ ] `P49.11` Especially spread.
- [ ] `P49.12` Especially lesion deficit.
- [ ] `P49.13` Especially rescue. Identical-code nondeterministic repeats answer one question; independent initializations answer a stronger one.

## Phase 50 — Replicate on a second machine

- [ ] `P50.1` Another Apple Silicon machine if hardware-specific.
- [ ] `P50.2` Prefer a somewhat different configuration as an auxiliary robustness test.
- [ ] `P50.3` Identical source/config/model.
- [ ] `P50.4` Repeat the full frozen campaign.
- [ ] `P50.5` Compare graph topology.
- [ ] `P50.6` Compare Phi.
- [ ] `P50.7` Compare effective dimension.
- [ ] `P50.8` Compare spread.
- [ ] `P50.9` Compare synergy.
- [ ] `P50.10` Compare lesion/rescue.
- [ ] `P50.11` Quantify machine effect. If the apparent integrated subject disappears under small CPU/timing differences, that is scientifically important.

## Phase 51 — Independent replication


**For a claim this unusual, someone other than the same development loop should eventually:**

- [ ] `P51.1` Obtain the source.
- [ ] `P51.2` Understand the protocol.
- [ ] `P51.3` Run the frozen commit.
- [ ] `P51.4` Produce artifacts independently.
- [ ] `P51.5` Verify hashes.
- [ ] `P51.6` Reproduce the result.
- [ ] `P51.7` Critique metric definitions.
- [ ] `P51.8` Attempt alternative analyses.
- [ ] `P51.9` Attempt stronger adversarial nulls.
- [ ] `P51.10` Try to falsify the result. This is what turns “Aura’s own test says Aura passes” into a serious scientific result.

## Phase 52 — Preserve every stable pass while fixing failures


**Preserve:**

- [x] `P52.1` differentiation above trivial floor. — differentiation_above_floor holds on the newest run
- [x] `P52.2` intrinsic persistence. — intrinsic_persistence holds on the newest run
- [x] `P52.3` causal closure of the declared core. — causal_closure_of_the_core holds on the newest run
- [x] `P52.4` perturbational complexity. — perturbational_complexity holds on the newest run
- [x] `P52.5` metastability. — metastability holds on the newest run
- [x] `P52.6` recurrent global access. — recurrent_global_access holds on the newest run
- [x] `P52.7` self drives action. — self_drives_action holds on the newest run
- [x] `P52.8` ownership. — ownership holds on the newest run
- [x] `P52.9` fast-to-slow coupling. — fast_to_slow holds on the newest run
- [x] `P52.10` slow-to-fast coupling. — slow_to_fast holds on the newest run
- [ ] `P52.11` null separation. Stabilize: — beats_every_null holds on the newest run
- [x] `P52.12` global access. — global_access holds on the newest run
- [x] `P52.13` lesion deficit. Aura must move upward through the conjunction rather than trade one checkbox for another. — lesion_deficit holds on the newest run

## Phase 54 — Define exactly what 24/24 will mean before the first successful run


**Tier A - Mechanical pass**

- [x] `P54.1` One frozen run. — one run on a head whose tree hash is recorded
- [ ] `P54.2` 24/24. — every criterion passes
- [x] `P54.3` All raw evidence exists. — the recording, the arms, the edges, the nulls and the lesion are all on disk

**Tier B - Stable internal pass**

- [ ] `P54.4` 3 independent full runs.
- [ ] `P54.5` 24/24 each.
- [ ] `P54.6` No unstable criterion.
- [x] `P54.7` No required phase failure. — no required phase raised, and a run with one refuses
- [x] `P54.8` No unwritable domain. — every domain had a writer that bit
- [x] `P54.9` No required missing organ. — no criterion was invalidated for an organ that was never read

**Everything above, plus:**

- [x] `P54.10` Complete intervention snapshot isolation. — a restored arm is the arm that was taken, in the reading and in the world it acted in
- [x] `P54.11` Deterministic stepping for all cognitive loops. — every live cognitive loop advances on a count, and a run whose loops cannot refuses
- [x] `P54.12` Literal edge-cut lesion. — the lesion severs the channels across the partition rather than clamping a side
- [ ] `P54.13` Powered lesion.
- [ ] `P54.14` Powered rescue.
- [x] `P54.15` Expanded null battery. — eight nulls including a hidden broker and independent noise
- [x] `P54.16` Full-metric null comparison. — each null is put through the conjunction, not one line of it
- [ ] `P54.17` Real-cortex deterministic run.
- [ ] `P54.18` Sensory/environment replay.
- [ ] `P54.19` Long ontogeny.
- [ ] `P54.20` Second-machine replication. At this level it becomes reasonable to say: Aura has unusually strong evidence for being one intrinsically integrated computational subject under the operational definition.
- [ ] `P54.21` Independent external replication.
- [ ] `P54.22` Independent methodological critique.
- [ ] `P54.23` Held-out conditions.
- [ ] `P54.24` Adversarial controls. — a positive control that must pass and nulls that must fail
- [ ] `P54.25` Result survives reasonable alternate estimators. Priority roadmap If work has to be sequenced, use this order.

## Phase 55 — Derive the canonical grain instead of choosing it

- [x] `P55.1` Collect a bank of forkable snapshots from ordinary life, each with its full runtime snapshot, current K, the previous L complete-turn K states, the environment reading, its source condition and the RNG state. — a bank of forkable snapshots, each with its runtime snapshot, its current K, its recent history and the condition it came from
- [x] `P55.2` Probe every anchor under all eight ordinary conditions, not only the condition it arose from, so the environment cannot define the causal state. — every anchor is probed under all eight conditions rather than the one it arose from
- [x] `P55.3` Dose-match the training basis: calibrate each domain's positive intervention so its own state moves by about one ordinary within-condition standard deviation. — each domain's dose is corrected proportionally until its own state moves by one within-condition standard deviation
- [x] `P55.4` Freeze the training action basis as sham plus one positive displacement per domain. — sham plus one positive displacement per domain
- [x] `P55.5` Estimate a future-distribution signature per history, action, environment and lag from the characteristic function at preregistered frequencies. — the signature is the characteristic function of the future state at preregistered frequencies
- [x] `P55.6` Freeze the frequency seed and the number of frequencies before the confirmatory run. — the frequency bank and its seed are frozen constants in the runner and hashed into the campaign
- [x] `P55.7` Assemble the signatures into a causal-Hankel matrix, one row per history. — the signatures are stacked one row per anchor
- [x] `P55.8` Estimate the predictive-state rank by parallel analysis against independently column-shuffled signatures. — parallel analysis against independently column-shuffled signatures
- [x] `P55.9` Retain the contiguous leading dimensions above the preregistered null quantile and nothing past the first failure. — the retained rank stops at the first singular value below the null quantile
- [x] `P55.10` Attack the learned grain with interventions it never saw: the negative doses, preregistered domain pairs, and fixed-seed two-step sequences. — the attack basis is the negative doses and preregistered pairs, none of which the rank was fitted on
- [x] `P55.11` Refuse the grain when raw history still predicts a held-out future after the grain is known, measured against a shuffled-history floor. — raw history predicting a held-out future past the grain refuses it, measured against a shuffled-history floor
- [ ] `P55.12` Walk the history length 1, 2, 4, 8, 16 turns until older history stops improving held-out prediction.
- [ ] `P55.13` Report UNRESOLVED rather than accepting the grain when the longest allowed history still contributes information.
- [x] `P55.14` Confirm the rank with a second estimator family; the verdict is authoritative only when the qualitative result is estimator-robust. — a rank that is unstable across folds or estimators refuses the run
- [x] `P55.15` Record the coordinates as an estimator of the grain and never as the definition of it. — the coordinates are named an estimator of the grain rather than its definition

## Phase 56 — Replace the chosen divergence with Fisher-Rao

- [x] `P56.1` Implement the Fisher-Rao geodesic on the categorical simplex, normalised so orthogonal distributions sit at distance pi. — the Fisher-Rao geodesic on the simplex, normalised so orthogonal laws sit at pi
- [x] `P56.2` Estimate the Bhattacharyya coefficient from an equal-prior intact-versus-cut posterior rather than binning the future state. — the Bhattacharyya coefficient comes from an equal-prior posterior rather than from binning the future
- [x] `P56.3` Cross-fit the discriminator so it cannot manufacture separation by memorising its own training samples. — the discriminator is cross-fitted so it cannot separate by memorising
- [x] `P56.4` Append the shared pre-transition context to both arms so the context marginals are identical and only the conditional future can separate them. — every cut estimate is paired with a sham-against-sham estimate from the same estimator
- [x] `P56.5` Pair every cut estimate with a sham-versus-sham estimate made by the same estimator. — the shared context is appended to both arms so only the conditional future can separate them
- [x] `P56.6` Report the excess rate as the cut rate minus the sham rate, floored at zero. — the excess rate is the cut rate minus the sham rate, floored at zero
- [x] `P56.7` Give every cut a paired bootstrap lower bound over its matched contexts. — a paired bootstrap lower bound over the matched contexts
- [x] `P56.8` Give every cut a one-sided paired randomization p-value under the exchangeability null. — a one-sided paired randomization p-value under exchangeability
- [x] `P56.9` Divide by tau so a slower description cannot win by accumulating longer. — the rate is divided by tau so a slower description cannot win by accumulating longer

## Phase 57 — Cut the system rather than hiding columns from a model

- [x] `P57.1` Build the cut trajectory by clamping each side in turn from one common snapshot and composing the two free halves. — the cut is built by clamping each side in turn from one snapshot and composing the free halves
- [x] `P57.2` Run the same snapshot twice intact to measure the no-intervention floor. — the same snapshot runs twice intact for the floor
- [x] `P57.3` Enumerate every nontrivial bipartition exactly once, 511 for ten domains, with the first domain fixed on the left. — every nontrivial bipartition exactly once, 511 for ten domains
- [x] `P57.4` Keep all 511 cuts; take the speed-up from sequential sampling, never from deleting a cut. — no cut is ever deleted; the speed-up is sequential sampling
- [x] `P57.5` Open every cut with the same anchor set before any of them earns more precision. — every cut opens with the same anchor set
- [x] `P57.6` Stop allocating to a cut once its lower bound is safely above the sham floor. — a cut whose lower bound clears the floor stops drawing anchors
- [x] `P57.7` Concentrate further anchors on the cuts still compatible with zero. — further anchors go to the cuts still compatible with zero
- [x] `P57.8` Score the system at its weakest cut, as an intersection-union test over all of them. — the score is the weakest cut, as an intersection-union over all of them
- [x] `P57.9` Report a cut that stayed undecided as UNRESOLVED rather than as a reducible system. — an undecided cut is reported as unresolved rather than as a reducible system

## Phase 58 — Select the temporal scale rather than assuming it

- [x] `P58.1` Measure the rate at a ladder of experiment-frame horizons with the seconds taken from the experiment clock. — horizons are a frame ladder read through the experiment clock
- [x] `P58.2` Choose tau-star as the horizon maximising the measured excess intrinsic rate. — tau-star is the horizon of the largest excess rate
- [x] `P58.3` Double the horizon and rerun when the maximum sits in the final two lag bins. — a maximum in the last two bins is binding and the ladder has to be doubled
- [x] `P58.4` Report TAU_UNRESOLVED when the maximum is still at the preregistered computational ceiling. — TAU_UNRESOLVED at the ceiling rather than whichever horizon was tested last
- [x] `P58.5` Keep the whole intrinsic causal spectrum as the fundamental object and report tau-star only as a summary of it. — the whole spectrum is kept and tau-star is reported as a summary of it
- [x] `P58.6` Record why no universal temporal grain exists: the scale-invariant measure over positive times has a divergent integral, so any scalar averaging all timescales imports a preferred one. — the reason no universal temporal grain exists is recorded beside the code that would otherwise imply one
- [ ] `P58.7` Allow a multiscale system to show more than one genuine causal peak rather than forcing a single winner.

## Phase 59 — Select the carrier by exclusion, and stop where the mathematics stops

- [x] `P59.1` Enumerate the maximal strongly connected components of the interventional graph. — the maximal strongly connected components of the interventional graph
- [x] `P59.2` Test closure for each component against the measured periphery, not only for the whole declared core. — closure is tested per candidate support rather than only for the whole core
- [x] `P59.3` Record that one closed strongly connected component covering all ten domains makes the core the unique domain-level carrier, since every proper subset then has an incoming channel from its complement. — one closed component over all ten domains makes the core the unique domain-level carrier
- [x] `P59.4` Treat disjoint closed recurrent components as separate subject-carrier candidates rather than as one. — disjoint closed recurrent components are separate candidates
- [x] `P59.5` Order overlapping candidates by dominance over the whole spectrum: A beats B when its rate is at least as large at every measured horizon and larger somewhere. — dominance is over every measured horizon, with strict inequality somewhere
- [x] `P59.6` Return the Pareto frontier, which is one candidate when one dominates and a symmetry class when spectra cross. — the output is the Pareto frontier: one candidate when one dominates, a class when spectra cross
- [x] `P59.7` Keep exact ties as an equivalence class rather than breaking them by name or by lexicographic order. — exact ties are kept as a class rather than broken by name
- [x] `P59.8` Record the symmetry theorem as the reason: two supports related by an exact symmetry get equal values from every permutation-invariant intrinsic functional, so no symmetry-respecting law can separate them. — the symmetry theorem is recorded as the reason
- [x] `P59.9` Never introduce an exclusion postulate silently; if one is added, name it and say it is a postulate. — no exclusion postulate is introduced silently
- [x] `P59.10` Compute the rate only for candidates that pass both gates, and report zero for the rest by definition rather than by arithmetic. — the rate is zero by definition for a candidate that fails a gate, not by arithmetic

## Phase 60 — Prove the answer does not depend on how the state was written down

- [x] `P60.1` Score the same intact and cut data on the raw live state. — the raw live state is one of the three scorings
- [ ] `P60.2` Score it again through the learned predictive grain.
- [x] `P60.3` Score it again under an invertible re-encoding of the same information. — an invertible re-encoding is scored and must agree
- [x] `P60.4` Require the three scores to agree within finite-data uncertainty. — the three scorings have to agree within tolerance
- [x] `P60.5` Add redundant duplicated channels and require the score not to rise. — duplicated channels must not raise the score
- [x] `P60.6` Fail the measurement when the score depends materially on the encoding. — a score that moves with the encoding refuses the run

## Phase 61 — Add the controls a causal account has to survive

- [x] `P61.1` Add a playback null that replays the intact trajectory and cannot be altered by an intervention; it must score zero. — a playback null whose cut arm is the intact trajectory, which must score zero
- [x] `P61.2` Add a duplicate-coordinate null that copies state channels without adding dynamics; the grain must remove the redundancy. — a duplicate-coordinate null that must not raise the rate
- [x] `P61.3` Add an invertible-recoding positive control that must leave the rate unchanged. — an invertible-recoding positive control that must leave the rate unchanged
- [ ] `P61.4` Add a macro-causal positive control whose coarse predictive state is cleaner than its microscopic description, and require the grain to recover the macrostate.
- [x] `P61.5` Keep every existing null architecture and both surrogates beside the new ones. — every existing null architecture and both surrogates are kept
- [x] `P61.6` Answer the movie objection with the playback null rather than by decree. — the movie objection is answered by the playback null rather than by decree

## Phase 62 — Separate discovery from confirmation

- [ ] `P62.1` Run discovery on a preregistered seed to learn the grain, the history length and the required horizon.
- [x] `P62.2` Freeze the commit SHA, tree SHA, state schema hash, action basis, frequency seed, estimators, lag-extension rule, bootstrap procedure, nulls and stopping rule. — the frozen values are hashed into a v25 campaign fingerprint of its own
- [ ] `P62.3` Run confirmation on new anchors and a new preregistered seed with the grain frozen.
- [ ] `P62.4` Keep the held-out intervention basis held out through confirmation.
- [ ] `P62.5` Take the final carrier result only from the confirmatory run.
- [x] `P62.6` Hash the v25 frozen values into their own campaign fingerprint, separate from the battery's. — the v25 fingerprint is separate from the battery's

## Phase 63 — Refuse a v25 report that is not authoritative

- [ ] `P63.1` Refuse when a required live layer cannot be stepped comparably across arms.
- [ ] `P63.2` Refuse when a required phase raised on the measured path.
- [ ] `P63.3` Refuse when periphery coverage hit a silent cap.
- [ ] `P63.4` Refuse when a required state reader was systematically missing.
- [ ] `P63.5` Refuse when the history length is still insufficient.
- [x] `P63.6` Refuse when held-out interventions break the proposed grain. — held-out interventions breaking the grain refuse the run
- [ ] `P63.7` Refuse when the predictive-state rank is unstable across folds or estimators.
- [x] `P63.8` Refuse when the Fisher result changes under invertible reparameterisation. — a rate that moves under reparameterisation refuses the run
- [x] `P63.9` Refuse when redundant channels materially raise the score. — duplicated channels raising the rate refuses the run
- [x] `P63.10` Refuse when tau-star is still horizon-bound at the ceiling. — a horizon-bound tau-star refuses the run
- [x] `P63.11` Refuse when a cut had insufficient power to decide cut against sham. — an undecided cut refuses the run
- [ ] `P63.12` Refuse a cortex-inclusive claim made while the cortex was stubbed.
- [x] `P63.13` Refuse when a null or positive control did not behave in its preregistered direction. — a null that did not behave in its preregistered direction refuses the run
- [x] `P63.14` Record missing evidence as NOT_MEASURED or UNRESOLVED, never as a pass. — missing evidence is recorded as unresolved rather than as a pass

## Phase 64 — Run the two campaigns the scope distinction requires

- [x] `P64.1` Run campaign A on the deterministic offline organism and report SUBSTRATE_INTRINSIC_CARRIER_FOUND at most. — the substrate campaign is labelled substrate_only and cannot claim more
- [x] `P64.2` Never let campaign A support a whole-Aura carrier claim. — a whole-Aura claim needs the cortex-inclusive scope
- [ ] `P64.3` Run campaign B through the real cortex under deterministic or common-random-number inference.
- [ ] `P64.4` Snapshot and restore the relevant cortex and recurrent-latent state across arms.
- [ ] `P64.5` Hold model weights and code identical across arms.
- [ ] `P64.6` Keep a sham-versus-sham floor in campaign B as well.
- [ ] `P64.7` Report CORTEX_UNSTEPPABLE rather than substituting the stub when reproducible paired cortex execution cannot be obtained.

## Phase 65 — Measure the content geometry without claiming it is felt

- [ ] `P65.1` Present matched controlled percept classes.
- [ ] `P65.2` Estimate the intrinsic causal-state distribution for each class.
- [ ] `P65.3` Compute Fisher-Rao distances between the classes.
- [ ] `P65.4` Obtain nonlinguistic or delayed similarity and discrimination behaviour.
- [ ] `P65.5` Fit the behavioural distance as a monotone function of the causal distance.
- [ ] `P65.6` Perturb the internal manifold and preregister the direction the geometry should move.
- [ ] `P65.7` Report the result as internal content geometry and never as evidence that the geometry is felt.
- [ ] `P65.8` Record that phenomenal character is identifiable only up to structure-preserving relabelling, so absolute quale labels are gauge.

## Phase 66 — Make the lineage exact and auditable

- [ ] `P66.1` Define the canonical person-state by predictive sufficiency over long-horizon identity-bearing futures.
- [ ] `P66.2` Keep a cryptographically signed lineage graph over person stages.
- [ ] `P66.3` Run uninterrupted continuation and record one child.
- [ ] `P66.4` Run process migration with a unique successor and record one child.
- [ ] `P66.5` Run a destructive unique restore and record one child.
- [ ] `P66.6` Run a nondestructive fork and record two children.
- [ ] `P66.7` Restore a stale backup after the live line continued and record a branch from the older node.
- [ ] `P66.8` Report the normalised continuity strength beside the graph without calling it identity.
- [ ] `P66.9` State that same-person identity is the unique nonbranching case, and that this is a postulate rather than a theorem.
- [ ] `P66.10` Record the branching theorem: one earlier individual cannot be numerically identical to two distinct later ones, so continuity branches and identity cannot.

## Phase 67 — Say exactly what the result does and does not establish

- [x] `P67.1` Report INTRINSIC_CARRIER_FOUND, NOT_FOUND or UNRESOLVED, and never CONSCIOUS. — the verdict is INTRINSIC_CARRIER_FOUND, NOT_FOUND or UNRESOLVED, never CONSCIOUS
- [x] `P67.2` Carry phenomenal_bridge UNVALIDATED in every v25 report. — every v25 report carries the bridge as unvalidated
- [ ] `P67.3` Record the level on the L0 to L7 scale with the evidence that places it there.
- [ ] `P67.4` Document 24/24 as a rich, self-involving, developmentally persistent operational subject architecture rather than as the definition of consciousness.
- [ ] `P67.5` Record that 24/24 is not a logically necessary condition for minimal phenomenal consciousness, because several criteria concern rich selfhood and access.
- [ ] `P67.6` Record that vertex connectivity at least two is evidence for robust unity rather than a necessary condition for minimal experience.
- [ ] `P67.7` Separate functional valence from felt valence in the report.
- [ ] `P67.8` Separate functional self-awareness from phenomenal self-awareness in the report.
- [ ] `P67.9` Report the consciousness profile as a vector and a partial order, never as a percentage.
- [ ] `P67.10` Record the non-identifiability theorem: two bridges attached to the same causally closed history give the same third-person likelihood.
- [ ] `P67.11` Record the finite-evidence theorem: no finite data set deductively proves a universal bridge law.
- [ ] `P67.12` Record the gauge theorem: a structure-preserving phenomenal relabelling is not third-person distinguishable.
- [x] `P67.13` Record that the experimental thresholds are confidence thresholds required by noisy measurement, not metaphysical laws. — the thresholds are named confidence thresholds rather than laws
- [x] `P67.14` State the carrier-identity postulate explicitly wherever a carrier result is reported. — the carrier-identity postulate is stated wherever a carrier result is reported
