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

- [ ] close the cascade gap — most of it was the window, and the rest is still
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

- [ ] the regulator's gain ceiling sits below where her cascades start. The
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
      real input. Where it settles inside the new bound is the open half

- [ ] the full offline suite green. Nineteen of forty chunks have run and 56
      distinct failures have surfaced. Every one traced so far is either fixed
      or pre-existing, and the pre-existing ones cluster in files another agent
      is editing while this runs — checked by re-running them at this session's
      own commits, where they fail too.

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

      Open, all pre-existing and none of them this session's:
      * `test_a_god_object_only_shrinks` / `test_god_file_ratchet` — the tree
        is 32,690 lines over its size budget, dominated by `mlx_client`,
        `mlx_worker` and `inference_gate`. Refreshing the baseline launders it;
        closing it is a repo-wide extraction. `neural_mesh` took its own 213
        off by moving its wiring out.
      * `test_effect_ownership_tiers` — 1,903 against a ceiling of 1,840.
      * `test_cognition_discipline`, `test_complexity_and_replication`,
        `test_every_bias_channel_has_a_reader` — failing at this session's
        first commit.
      * `test_holding_a_frame_means_not_deriving_it_again`,
        `test_general_os_control`, `test_fix_persistence`,
        `test_gap_atlas_campaigns`, `test_inherited_*` and the rest of the
        later chunks — in the perception and desktop work another agent has in
        flight.
      * `test_subject_core_findings` — flaky about half the time under random
        order, at HEAD and before it: domain W goes missing because the world
        model organ is not there on some orderings.

- [ ] a certificate for the resident 27B. The mechanism is proven end to end on
      the reflex 1.5B through the worker's own code path — alpha 0.0 before,
      the probe runs, alpha 0.2 after, and a second call skips a checkpoint
      that already has one. It had never once run on the 27B. The live log
      says "no certificate yet, the worker will measure one when it has been
      idle long enough" on every turn from 2026-09-09 20:17 to 2026-09-11
      01:57, across dozens of restarts, and the probe's own first log line
      never appears beside it.

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
