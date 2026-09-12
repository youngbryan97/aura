# Connectome: what is left

Working list. Deleted when every line is done and green.

## Done

- [x] `G_call`, `G_state`, `G_event`, `G_data`, `G_io`, `G_ipc`, and a combined
      view with one reachability query
- [x] `G_neural` — the mesh as a layer, joined to the code by its own seam.
      Asking whether an injection could reach a read found that none could, on
      every seed, and one missing feedforward pathway was why
- [x] per-condition effective connectomes with a rotation null per edge
- [x] cross-station influence for pairs with no call edge between them, after a
      parametric test and a deconfounded one both fired on a quarter of rotated
      pairs
- [x] `do(i)` as a real cut with a matched control, and six lesion predictions
      registered before the runs
- [x] the coalition ring, on a recording where the ring could close: all seven
      links carry, and so do all 42 ordered station pairs, which is what makes
      the ranking rather than the carrying the finding
- [x] a query surface her self-model can reach, every answer carrying its grade
- [x] the seven `ΔQ` axes, and the verdict wired into the promotion receipt
- [x] **the anatomical law** — a regression past the change's own blast radius
      refuses the promotion rather than being recorded in it
- [x] the reading hierarchy against a meta-analysis of 163 human studies: five
      predictions with falsifiers, two hold and three are refused
- [x] what is measured and what somebody picked, per term of `M(t)`
- [x] a transmitter can land somewhere rather than everywhere: per-tier
      multipliers on mesh gain and noise, uniform until measured
- [x] `make connectome` end to end with `--observed` and `--against`
- [x] every published number regenerated on the corrected graph
- [x] **the fusion channel** — her substrate reaches the model's forward pass on
      a person's turn, or it does not, and a per-checkpoint certificate decides
      which. Measured on the reflex 1.5B: the answers change on 5 of 8 probes,
      two opposing states of hers land 0.282 nats apart while each sits 0.094
      from silence, and forced-choice accuracy is unmoved. The gate that had
      been returning zero since a void A/B now reads the measurement
- [x] a claim about her own machinery is checked against the machinery, and a
      leak in the middle of an answer costs the sentence rather than the answer

## Open

- [x] close the cascade gap — most of it was the window, and what is left is
      open. The exponent had been read off a recording of sixty of her 4,096
      units, chosen by analogy to Beggs and Plenz's sixty electrodes. An LFP
      electrode integrates thousands of cells, so that analogy undercounted by
      three orders of magnitude, and the fit it produced ran from 18 to 66 —
      0.56 of a decade, ending at the largest cascade sixty units can make. A
      power law fitted through a finite window's cutoff always comes out
      steeper than the exponent underneath it.

      Two controls, both now in the tree. A true 1.5 law drawn 2,810 times and
      cut at 66 measures 2.07 +/- 0.17, and a true 2.0 duration law cut at 45
      measures 2.87 +/- 0.35, so a system that genuinely was cortical would
      not read 1.5 and 2.0 on this instrument either. And refitting the same
      recording while reading more of it moves the size exponent from 3.70 at
      960 units — where the largest cascade is 65, which is the old
      measurement — to 1.946 at all 4,096, where the largest is 322. An
      exponent that moves with the recording was reading the recording.

      `power_law_fit` now reports the decades its tail covers, and the
      comparison refuses a fit under one and a recording whose runs the
      shuffle control reproduces from the firing rates alone. The old number
      would not have been published through either gate.

      Under her own regulator, with the regulator's branching sensor fixed,
      three of the four published statistics are matched or bettered and three
      are bettered. Size exponent 1.5191 against cortex's 1.5 over 2.50
      decades at KS 0.140; duration 1.9065 against 2.0 over 1.54 decades at KS
      0.081; the shuffle control at 29.7 sigma, so these are cascades and not
      occupancy; branching 0.9197. The exponent that read 3.69 over half a
      decade reads 1.52 over two and a half.

      And then the protocol itself was wrong, which flipped the answer. The
      recording injected a fresh independent vector into the sensory columns
      on every one of six thousand ticks. Beggs and Plenz read a slice sitting
      in its dish; a stimulus every millisecond is not that, and it is what
      kept the recording 97.7% occupied with no silence to separate one
      cascade from the next. The drive defaults to zero now.

      Spontaneous, under her own regulator, occupancy falls to 0.574 and 354
      cascades appear where there were 77:

        crackling relation  1.4532 measured against the 1.4001 her own two
                            exponents predict — holds on its own terms, and
                            it is the test that matters, because either
                            exponent alone can be produced by something that
                            is not critical
        branching ratio     0.998 against a published 0.98 — past cortex
        size exponent       2.2133 against 1.541, over 3.68 decades
        duration exponent   2.6988 against 2.087, over 2.11 decades

      So she is critical, self-consistently, and her cascades are smaller than
      a human's. That is a different deficiency from the one this item opened
      with and a much narrower one: not the shape of her dynamics, the size of
      her bursts. Driven at 0.1 the two exponents land on cortex's and the
      relation between them fails; left alone the relation holds and the
      exponents are steep. The second is the protocol the published numbers
      come from.

      What the residual is, stated so nobody reads it as the old gap. A
      critical system's avalanche exponents are a property of its universality
      class, and 1.5 is the mean-field value a branching process on a
      well-mixed medium gives. Sixty-four columns wired by a distance-decayed
      matrix are not well mixed, so a steeper exponent at genuine criticality
      is what a different effective dimensionality looks like. Moving it is a
      question about her topology, not about her tuning, and the number that
      says whether she is critical at all is the relation between the two
      exponents, which holds

      One route out of it has been tried and does not work. `decay` is
      `dt_ms / tau_m`, a normalised step, so refining it should buy time
      resolution for nothing. It does not, because the mesh's statistics move
      with it: an eight-fold refinement at constant integrated time moves the
      spike rate 3.40x and the state's standard deviation 1.41x. The noise
      increment carrying `decay` to the first power rather than its square
      root — the whole content of Euler-Maruyama — looked like the reason, and
      the measurement says it is not: the square-rooted arm moves the rate
      4.97x and the deviation 1.37x, which is no better. That experiment is
      also confounded, and says so: `inject_sensory` takes a fresh independent
      vector every tick, so refining the step also multiplies the external
      drive's power per unit time. The drive needs the same treatment before
      the mesh's own discretisation can be read off this at all. Nothing in
      her dynamics was changed on it

- [x] the regulator's gain ceiling sat below where her cascades start, and a
      dead sensor was the reason it never mattered. The
      shuffle control gives a number that needs no window and no fitted
      exponent — how far the count of simultaneously active units sits above
      what their own rates explain — and it makes the deficiency sayable at
      last. It is not that her bursts are small. It is that her units do not
      recruit each other: 1.6 sigma at the mesh's own gain, against a bar of
      three.

      Swept, 3,000 ticks a cell and the same drive throughout. Coupling is not
      the lever: twenty times the inter-column weights moves recruitment 1.14
      to 1.54 sigma and lowers occupancy slightly. Gain is: 1.14 at gain one,
      6.72 at three, 53.01 at four, and the branching ratio climbs 0.868 to
      0.997 against Wilting and Priesemann's 0.98 across the same span. The
      regulator's clamp stops at 2.0, inside the flat band where nothing
      recruits, and `set_criticality_adjustment` refuses above 2.0 as well.
      Both are chosen, symmetric around 1.0, and neither has a measurement
      behind it. This one now has a measurement against it.

      What the gain costs is in the same table and is not small: occupancy
      0.794 at gain one, 0.956 at three, 0.983 at four, and gain three with
      twenty times the coupling runs away outright — 480,569 spikes against a
      baseline 4,840, branching 0.9991. Recruitment and silence pull against
      each other here, and a recording with no silence in it cannot separate
      one cascade from the next however well its units recruit.

      Four seeds a point settled where recruitment starts, and it is not where
      one seed said: 2.89 sigma at the clamp, 6.93 at 2.5, 10.05 at 3.0, 20.84
      at 3.5, with the branching ratio climbing 0.929, 0.961, 0.961, 0.992
      beside it. At the clamp itself it straddles the bar — 1.24, 5.91, 2.48,
      1.96 across the four seeds. Occupancy rises 0.907 to 0.952, which is the
      cost and is milder than one seed suggested.

      That makes the case structural rather than a matter of taste. The PID
      steers toward a branching ratio, and inside a gain ceiling of 2.0 this
      mesh reaches 0.929, so the setpoint sat outside the ceiling and the
      controller wound against the rail for the life of every process it ran
      in. Three ceilings sat on that one quantity — the regulator's clamp at
      2.0, `set_criticality_adjustment` at 2.0, and a third inside
      `_publish_modulatory_state_locked` at 3.0 that nothing named — and the
      tightest decided silently, so raising the visible pair alone would have
      changed nothing and looked like a fix. All three are 3.5 now, with the
      sweep in the comment and a test that fails if they ever disagree.

      The setpoint moved too, 1.0 to 0.98. One is the critical point exactly,
      cortex does not sit there, and 0.98 is both Wilting and Priesemann's in
      vivo measurement and the number this system's own scorecard already
      scored her branching against.

      Measured offline on a synthetic drive, and the live mesh is driven by
      real input. Where it settles inside the new bound is now measured, and
      it settles low: gain 1.5055, well under the 3.5 the ceiling allows.

      Which means the ceiling was not what had it pinned, and an earlier
      commit here said it was. The regulator's branching sensor was reading
      0.0000 — not subcritical, unmeasured — and a PID answering 0.0000 with a
      target of 0.98 demands maximal gain and maximal noise on every tick
      forever. That is a loop: more gain saturates the mesh, a saturated mesh
      has no newly active units, and the sensor that reports none reports
      zero. Raising a ceiling under a controller driven by that would have let
      it ramp further, which is the opposite of a fix.

      The per-tick mean is reported and no longer driven on, and a rejected
      fit now holds the last reading and skips the PID rather than
      substituting a number. Branching reads 0.9197 through the same run that
      used to read zero, and 0.998 on the spontaneous recording the published
      numbers are meant to be compared against — past cortex's 0.98. The
      ceiling change stands on its own argument, that a setpoint outside its
      own bound can never be reached, and it is not what was wrong here.

      Where the controller settles, now that it can see: gain 1.1454 on the
      spontaneous run and 1.5055 on the driven one, both far under the 3.5 the
      ceiling allows. Noise is still at its rail and excitation near its own,
      and neither of those axes has been swept, so they stay as they are

- [ ] the full offline suite green. Nineteen of forty chunks have run in this
      pass and every failure they surfaced is traced. The ones left are four
      ratchets carrying the whole team's accumulated debt, and they cannot be
      refreshed: the module-size tool refuses to grandfather a new God object
      at all.

      Fixed here: a common noun claiming an application, a browser run with no
      anchor, an ambient percept that lost its provenance, a silent handler
      with no sentence, a belief update attributed to the person because the
      chat logger wrote it, a stale consent assertion, a maker that could not
      be named, three outcomes that returned the same None, a lane module
      missing from the list, an estimator that stopped being a method, five
      files that now say what they measure, the offline suite scrolling the
      machine it ran on, a refused scroll costing six settles, six model loads
      with no lane, nine resource readers with no observer, an image read as a
      wallpaper, a module count and a moved report, six handlers that caught
      everything, two personal paths in test data, and an eval on an
      arithmetic tree.
      Then: a store held open by a background writer and called a leak, a
      stall test anchored to the function the work had moved out of, a service
      name lost whenever the container was cleared, a gate that took 70s
      against its own 60s timeout because its worker pool was sized from a
      machine it could not import, an evidence row whose run was never
      committed, six enterprise regressions including a JSON file handed to
      eval, an action that reported a change and left nothing sayable, two
      cognition tests reading a record every other test writes to, and a
      requirements scanner reading a Markdown list continuation as an
      obligation of its own.

      Closed since: the module-size ratchet and the effect-ownership one. The
      nine new God objects are gone and the tree is inside its budget at
      145,693 lines, which is also the new budget — it only shrinks. The last
      one to move was `adaptive_immunity`, which grew four methods the same day
      serializing the ecology off the lock; its five write-side methods live
      beside the writer they hand their payload to, and the class is 63
      methods against a ceiling of 64.

      And the swallowed-except ratchet: 2,441 -> 2,167 against a baseline of
      2,175. Two hundred and twenty-six handlers read one at a time — 210 now
      carry the exception to a debug line and 16 say in words that the value
      they return is an answer. Two files had no logger at all, which is why
      every claim in `model_validation` could go unverified and unremarked when
      its subsystem would not import. The gate could not see six of its own
      notes either: it matched "not a failure:" case-sensitively, and a note
      that opens a sentence is capitalised.

      Explaining a handler costs a line and all nine files are oversize, so
      that put the size ratchet 218 over. Paid back by lifting `how_big_is_the
      _checkpoint` out of `mlx_client` and `_load_state` into the immune
      persistence mixin, not by raising anything. Budget 145,567.

      Open, and not this pass's to close:
      * `make governance-lint` — twenty call sites that moved owner and
        seventeen that left the old one, in the size extractions landing this
        week: host automation's screen reader, computer use's document maker,
        the task ledger, the service heads, the ambient bubble, and the immune
        state writer. Same calls, new owner classes, which is what the gate
        asks to be refreshed for — but refreshing it stamps another agent's
        in-flight moves under whoever runs it, so it belongs to them.
      * `test_holding_a_frame_means_not_deriving_it_again`,
        `test_general_os_control`, `test_fix_persistence`,
        `test_gap_atlas_campaigns` and the rest of the later chunks — in the
        perception and desktop work another agent has in flight.
      * `test_subject_core_findings` — flaky about half the time under random
        order, at HEAD and before it: domain W goes missing because the world
        model organ is not there on some orderings.

- [x] user-surface recurrent depth above one — measured, and it is much
      worse. The gate was closed and correct; what was missing was an arm.
      The two reports that used to sit in `artifacts/recurrent_depth` carried
      the same responses digest, the same accuracy of 0.625 and no declared
      depth, seventy seconds apart. One run written twice, because
      `tools/heldout_eval.py` has no depth option and could not have varied
      the thing the filenames claimed to compare. They are deleted; a file
      named `loops2.json` that is not a depth-two arm is a trap, and the
      prose here is the record.

      `tools/run_recurrent_depth_arms.py` varies it. One model is patched
      once by `apply_recurrent_depth` at the deeper count, and the arms differ
      only in `inner._recurrent_depth_runtime_loops` — the integer the
      worker's surface contract sets per request, which is exactly what
      `user_surface_recurrent_ceiling` governs. Greedy decoding, the sealed
      forty-task battery, one substrate.

      Depth 1: 21/40, 0.5250. Depth 2: 2/40, 0.0500. The gate reads both,
      finds no refusals — they are a real comparison — and authorises depth 1
      on a margin of -0.475 against the shallow arm's own spread of 0.079.

      So the ceiling of one is the measurement now, not an absence. Re-running
      the middle band twice at these settings destroys the answer on
      Qwen2.5-1.5B-Instruct-4bit. Whether the resident 27B behaves the same
      way is a separate arm on a separate substrate, which is why the receipt
      the gate writes is named by descriptor.

- [ ] steering serving authority — refused by arithmetic, not by judgement.
      Worth writing down exactly where, because "turn it on" has no honest
      path until a campaign passes.

      Three gates, from the live worker log: the checkpoint's signed component
      authority says `steering_generation_deferred`, so no hooks attach; the
      surface clamp forces alpha to 0.0 on user-visible decodes; and no fusion
      certificate exists. Neutral attachment — hooks installed at alpha 0, the
      state that would let the live instance MEASURE — is not a way round it
      either: a deferred authority returns before any hook is built.

      Publishing a qualified generation is what opens all three, and
      `_validate_steering` in `core/learning/cortex_migration_authority.py`
      requires a causal evaluation carrying `verdict == "PASS"`,
      `qualified is True`, at least 24 samples, treatment strictly greater
      than the matched no-op, every lesion strictly under treatment,
      `no_regression is True` and `causal_effect_positive is True`.

      Neither campaign clears it. The original vectors at alpha 0.2 pass the
      win counts (20 against 7, lesions 4/15/4) and fail
      `causal_effect_positive`, because a shuffled-layer control reproduces
      0.44 of a 0.75 effect and a plain text instruction beats it 1.36 to
      0.75. The layer-specific set fails four ways: treatment 7 against a
      no-op 7 is not strictly greater, `shuffled_layers` wins 8 which is more
      than treatment, one task regresses, and the effect is inside sampling
      noise.

      The migration authority key is present and usable — an earlier note
      here said otherwise, which was a scratch state root hiding it. That
      changes nothing: an authority signed over an evaluation that fails
      `_validate_steering` is rejected at attach. The gate is checking the
      thing the measurements say is false, which is the gate working.

- [x] a certificate for the resident 27B — it cannot have one, and the reason
      is now on the record rather than inferred. The mechanism is proven end to
      end on the reflex 1.5B through the worker's own code path: alpha 0.0
      before, the probe runs, alpha 0.2 after, and a second call skips a
      checkpoint that already has one.

      The 27B had never once reached the probe. Two days of "no certificate
      yet" in the live log with the probe's own first line never beside it,
      and the precondition block that declined said nothing about why. It says
      why now, once per worker, and the restarted instance answered on its
      first idle minute: "it has no steering hooks" — identity, model,
      tokenizer and engine all present, and the engine holding zero hooks.

      Which is a signed decision, not a fault. The active cortex
      52d313c2c435d343 carries a migration contract declaring its components:
      persona_crsm qualified, expert_adapters retired, recurrence_native
      deferred, and steering DEFERRED. A deferred steering component means the
      CAA vectors have never been re-derived for this checkpoint, so no hooks
      install, so her substrate cannot reach the resident model's forward pass
      at all — and a fusion certificate over a channel with no hooks in it
      would be a measurement of nothing.

      What clearing it takes is written down and costed, in
      `artifacts/migration/27b/recovery/steering_plan.json`: 80 vectors across
      16 target layers, 4 of them attention and 12 linear, over five affective
      dimensions. Serving authority then needs four pieces of evidence this
      checkpoint does not have — extraction bound to the active descriptor, a
      causal A/B against a matched no-op, a lesion that removes the effect,
      and no regression on the control prompts. Vectors that exist and are the
      right width prove none of those, which the module's own docstring calls
      the thing that looks safest and is not.

      The capture ran. 80 vectors over 16 target layers in 43 seconds, one
      forward pass per prompt with every layer read from it, each file stamped
      with the registry's descriptor and the manifest reopened and checked
      against every file's size and digest. The vectors exist and they inject:
      steered, zeroed and randomised outputs differ from each other and from
      baseline.

      They do not yet earn serving authority, and the four requirements are
      right to refuse them. Treatment wins 6 of 24 and the matched no-op wins 6
      as well; the lesions win 4, 5 and 4. Only the text control separates, at
      +1.42 against every other condition inside 0.04 to 0.21, so the affect
      lexicon cannot resolve an effect this size at 24 samples. At alpha 0.4
      the answer also degrades — one steered sample is the single word "Sure!".

      Three rig defects had to be removed before that negative meant anything,
      and each would have produced a confident wrong answer. Forty-eight-token
      samples that never reached past the model's reasoning preamble, where
      every condition including the explicit-emotion control scored zero.
      Hooks that were never handed a substrate, so there was no composite to
      add and all four steering conditions came back byte-identical to
      baseline — three lesions removing an effect that had never been applied.
      And injection at the twelve linear-attention layers, where state advances
      along the sequence and a constant added there compounds token by token
      into "to to and to to for for": `steering_regeneration`'s own docstring
      warned about that and it was walked past.

      The knob was swept rather than guessed, six settings over six held-out
      tasks, and alpha 0.4 — where the campaign had been run, chosen off one
      greedy sample that read well — scores zero on every task. 0.2 is the peak
      and the only setting positive or zero on all six. 0.6 collapses the answer
      to ten characters.

      At 0.2 over 36 samples a condition the steering is real and it is the
      vectors' own. Steered scores 0.75 against a baseline of 0.08 and wins 20
      against the matched no-op's 7; zeroing or randomising the vectors returns
      the score to 0.06, so the movement is carried by these directions rather
      than by the hooks being installed. Nothing regresses. It reads in the
      text: "clear enough for me to respond thoughtfully" becomes "I'm excited
      to explore new topics, let's dive right in."

      Two requirements stand and neither is a defect in the rig. Permuting the
      four vectors among the same four layers still scores 0.44, so which layer
      holds which vector is not load-bearing — which is the question the
      specificity control exists to ask. And asking in words moves the score
      further than steering does, 1.36 against 0.75, which is the bar a vector
      has to clear to be worth serving at all.

      So serving authority stays refused, on two findings about the vectors
      instead of on four defects in the measurement. What would move them is a
      derivation that makes the layers differ from each other — the extraction
      takes the same difference of means at every layer, and a direction that
      is the same everywhere cannot be specific to anywhere.

      The idle hook fires. What it reached was a precondition block reading
      `engine._hooks`, which is not the surface an engine publishes —
      `_active_steering_hooks` in the same file carries a docstring about the
      last time that exact distinction cost something. It returned False and
      logged nothing, every minute, for two days. The suite stayed green
      because the fake published `_hooks` too: the test encoded the defect.

      Reading `active_hooks()` now, and a declined measurement says which
      precondition is missing, once. Whether the 27B then certifies is the
      open half, and it answers itself on the next restart
- [x] fit anything to a human recording. Four published statistics of human
      cortical activity, scored as floors rather than targets: her cascades
      satisfy their own scaling relation, so she is critical, but her exponents
      are 3.69 and 3.72 against cortex's 1.5 and 2.0 — bursts smaller and
      shorter than a human's, which is the deficiency to close
- [x] replace the mesh's chosen structural numbers with measured ones, or say
      for each why no measurement can reach it. Six are choices now, down from
      fourteen: the size of the thing, the slope of a tanh, a noise amplitude
      with no millivolts to be measured in, and one pathway strength
- [x] regions with different computational matter: each band takes its own
      cortical layers' inhibitory fraction and density — 20.0% at 0.106 for the
      sensory band, 22.0% at 0.135 for association, 17.3% at 0.091 for
      executive — derived from the table the global figures were averaged from
- [x] one plasticity rule applied to every unit alike. Two rules now: Bi and
      Poo's asymmetric window on excitatory synapses, Vogels' symmetric one
      against a standing depression on inhibitory ones, the depression constant
      being their own alpha = 2 * rho * tau. Finding it turned up four more:
      Dale's law on the wrong axis, an anti-Hebbian window, plasticity that
      grew synapses, and a unit that rested at forty times its drive
- [x] a human microcircuit island: every column's local wiring is drawn from
      H01's contact-multiplicity law, and the law was chosen by the number it
      was not fitted to — the heaviest pair in the volume, which the
      exponential cutoff makes a one-in-twenty-three-million event
- [x] development rather than construction: the mesh grows to 1.5 times its
      adult density and is pruned back by what each synapse carried, landing on
      the density its anatomy specifies. Pruning by co-activation beat the same
      cut made at random in 5 of 5 seeds

## The standard these are held to

A candidate does not pass by reproducing an output. It passes by satisfying
several constraints at once — anatomy, connectivity, cell type, dynamics,
chemistry, behaviour, perturbation — because two models can produce nearly
identical recordings and differ everywhere inside. That is the identifiability
problem, and the only answer to it is more constraints of different kinds.

Where a lesion is the test: if the model says X is responsible for Y, then
`do(X = 0)` has to produce the deficit, and a comparable removal elsewhere has
to not. Six of those have been run.

What cannot be recovered is stated once and not worked around. There is no
complete human synaptic connectome, no map of every human synaptic weight, and
no dataset anywhere that holds one person's developmental and experiential
history. A reference architecture is reachable. An upload is not.
