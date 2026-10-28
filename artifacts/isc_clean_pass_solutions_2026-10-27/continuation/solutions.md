# Solutions implemented, proofs obtained, and the work still required

This continuation adds executable remedies and tests on saved Aura data. It does not claim 24/24, determined J*, or bridge parity. The first implementation checkpoint is `a1b4141fe`, pushed to main after integrating Opus's resolution, v6-reader and anchor-spacing fixes. The earlier implementation report and historical verdicts remain intact.

The most consequential result is that better regression alone cannot guarantee closure from the present observation map. The repository already notes sketch collisions in `calibrated_axes.py`; this continuation supplies another executable counterexample using the production sketch, demonstrates how a passive outside copy then looks predictive, and implements a minimal read-preserving basis repair. That repair is proved for declared linear reads, including nonlinear transitions computed from those reads. The baseline recorder now accepts explicit exact-array/read-basis contracts. A complete source-backed inventory of the required reads remains unfinished.

## What each goal now has

| Goal | Implemented solution components | What still prevents a clean pass |
|---|---|---|
| 24/24 | Valid sensitivity guard; nonlinear predictor candidate; bounded extrapolation; explicit target-coverage reporting; read-preserving observation basis and recorder adapter; joint-periphery interventions; integrated nonuniform donor schedule | Complete source-backed read contracts, matched-dimensional calibration, adequate training coverage, complete causal/surrogate evidence and a fresh frozen confirmation |
| J* | Host-clock resource deadlines; correctly paired content comparisons; recall read at the declared lag; tied-rank handling; worker failure propagation; clean-source evidence joins; rejection of degraded authority or a control substituted for intact content | A clean carrier run, successful intact content tracking with controls/rescue, and compatible recorded lineage |
| Bridge | Current v6 lines reach the bridge; one combined checker requires all three goals; explicit source/boundary identity; conditional interpretation retained | Fresh withheld report-grounding evidence and all J* terms settled on compatible evidence; P1–P6 remain assumptions |

These are mechanisms and measured results, not a guarantee that the next experimental result will be green. The operational bridge can pass its stated conditions; a universal psychophysical law does not become proved by that pass.

## 1. Solve prediction blow-up with bounded extrapolation

The earlier candidate passed useful small-world tests but failed on Aura. I ran all three forward folds of the saved discovery recording at `/Volumes/AuraRuns/closure-repro-s73/run_001`. There are 2,959 transitions, 2,901 own-state/environment predictors, 949 outside predictors and 362 live state columns. Expanding the nonlinear basis gives about 5,500–5,800 own columns, against only 591 training rows in the first fold.

The unbounded candidate's wider model made error worse by about 130% and 120% in the later folds. This failure is preserved in `saved-aura-fold0.json` and `saved-aura-fold12.json`. The earlier 59/60 margin-detection result concerned a much smaller Gaussian family and did not establish transfer to Aura.

I added an opt-in `feature_limit` to the fitter. After centring and scaling by training rows, it applies the same fixed bound to every split. The diagnostic below uses eight training standard deviations. The official default remains unbounded; the candidate's limit is explicit rather than silently applied to old verdicts.

For a clipped feature vector z with p coordinates and bound L, a fitted linear read obeys

    |wᵀz| ≤ ||w||₂ ||z||₂ ≤ L sqrt(p) ||w||₂.

Thus a future input cannot produce arbitrarily large output merely by growing farther outside the training range. This is a finite-output bound, not a correctness or power theorem. A test on a genuinely saturating transition shows that this option fixes the demonstrated extrapolation failure, and checks retain exact agreement between the narrow fit and a wider fit with no added features.

Actual saved-Aura readings:

| Fold | Unbounded narrow / wide loss | Bounded narrow / wide loss | Bounded incremental gain | One-sided bootstrap bounds |
|---:|---:|---:|---:|---:|
| 0 | 60.965 / 55.393 | .701766 / .656404 | .064640 | .037415–.093992 |
| 1 | 1.396618 / 3.209911 | .395844 / .381484 | .036276 | .033106–.039674 |
| 2 | 200.419 / 440.385 | .638596 / .637740 | .001341 | −.000155–.003308 |

The bounds are diagnostic per-fold bounds, not a joint official closure decision. They neither include the official multiplicity rule nor certify sensitivity. The record is `saved-aura-bounded8.json`.

Three circular-shift controls for fold 1 gave gains 0, .004467 and .006024, against the real .036276. Their empirical 95th percentile is .005868. With three controls the smallest possible corrected tail fraction is 1/4; the observed fraction is .25. This is useful discovery evidence, **not statistical qualification**. A remaining gain can reflect a cause, an insufficient observation, or an insufficient predictor. This result does not decide among them.

The bounded variant was introduced after inspecting these discovery data. It needs its own independent calibration; the earlier unbounded calibration cannot be relabelled as its proof.

## 2. Solve lost state with a read-preserving observation contract

`core/subject/sketch.py` keeps mean, standard deviation, minimum, maximum and four fixed projections. `read_held` uses that sketch for arrays even when their containing organ belongs wholly to K*. Holding the full organ causally and recording eight summaries are different operations.

The constructed counterexample uses the production sketch:

* Two 16-coordinate internal arrays are sqrt(2) apart.
* All eight sketch fields agree to within 2.22e−16, below the fitter's flat-column threshold.
* One internal coordinate is .162384 in one state and −.524596 in the other.
* A transition reading that coordinate has different futures. An outside copy of the coordinate predicts the future without being a cause of it.

The construction fixes the two extrema and chooses equal-length interior vectors in the nullspace of the mean and the four projection rows. Equal means, norms, extrema and projections follow algebraically. A production-fitting test verifies that the passive copy appears useful when only the sketch is observed, while a sufficient internal read removes the prediction deficit.

This proves a limitation of the observation family. It does **not** establish that this particular collision or read occurred in Aura's saved trajectory. The remaining seed-73 gain needs its own causal witness.

The implemented repair is `core/subject/observation_basis.py`. For original linear observations B and declared transition reads A, it constructs an orthonormal basis Q spanning the rows of both:

    z = Qx
    Ax = (AQᵀ)z
    minimum added rank = rank([B; A]) − rank(B).

Consequently any transition of the form g(Ax, other_state, environment) retains its array inputs. The nonlinear function g need not itself be linear. This is an observation-sufficiency result, not a claim that a particular regression learns g.

The counterexample needs exactly one added direction: rank five becomes rank six, with read-reconstruction error 3.93e−16. Tests also reconstruct unseen random states, redundant readers with different physical units, and nonlinear functions of preserved reads. `observation-basis-proof-with-canary.json` contains the arrays, axes, reconstructed futures and current source hashes; earlier receipts are retained beside it. The old source comment implying that different states necessarily have different sketches was corrected, including the actual width/array-size scaling of expected projected squared distance. The small reconstruction canary is registered both as a measurement invariant and a measured synthetic claim.

**Required integration:** derive each A from the actual reads that determine a held array's contribution to transitions, and bind the reader declaration to source and intervention witnesses. For an indexed read, include the relevant coordinate; for a fixed weighted sum, include its weight row; for learned or state-dependent reads, preserve the required union of directions or the exact coordinates. If those directions span the full array, the certificate requires full width. Four projections cannot replace that obligation. Nonlinear reads that are not functions of the declared linear reads need exact inputs or a separate sufficiency proof.

**The recorder adapter is implemented.** `core/subject/declared_observations.py` accepts a versioned JSON declaration of held unit, stored-field path, shape and either exact or read-basis mode. The actual baseline callback in `tools/run_subject_core.py` reads those coordinates at turn close, alongside the ordinary periphery. Added columns remain under `held.`; an outside service cannot become part of K* merely by appearing in a plan. Reads reject missing units, shape changes, nonfinite coordinates, numeric types the float64 recording cannot preserve, and paths crossing known unit ownership. Read-basis mode certifies the supplied linear reads, never a complete inventory of unknown reads.

The `--observation-plan` CLI requires a new substrate-only run and `--one-clock`; it refuses resume and whole-model mode. It records the plan digest in the campaign fingerprint, saves the exact plan and closure inputs, then stops with `official_verdict: null`. It never runs the battery on that discovery recording. An integration test exercises the real `_record` callback and verifies the coordinates over successive turns; no real runtime was booted for this test.

The supplied `activation-observation-plan.example.json` records `substrate.x` (512 coordinates) and `field.F` (256), dimensions observed in the saved recording and fields named by the source schema. At 12,000 turns the added dense data are 73.7 MB, with a 302 MB working reservation. Its 512 MiB budget passes the offline preflight. That budget covers the **additional** columns, not the existing runtime and recording. A plan capturing the complete 512×512 substrate weight matrix instead reserves about 103 GB and is refused before a turn. This example observes two important arrays exactly; it is not a complete transition-state plan.

The new prospective recording command, to use only when host capacity is available and the final declaration is frozen, is:

```sh
python tools/run_subject_core.py --one-clock --rounds 1500 --seed DISCOVERY_SEED \
  --observation-plan PLAN.json --out NEW_DISCOVERY_DIRECTORY
```

There are eight conditions, hence 12,000 turns. That command has **not** been launched here. `recording-budget-check.json` is the completed offline preflight, not a runtime receipt. The ordinary recorder and official estimator remain unchanged when these options are absent. No runtime read set has been declared complete by this work.

**Complete-array intervention comparisons are also wired.** `tools/run_interventional_closure.py --observation-plan PLAN.json` accepts exact-mode declarations and adds SHA-256 digests of those full arrays to its held-state comparisons at each roll endpoint. The original numeric observations remain present. Digest equality is collision-resistant evidence, not a mathematical injectivity theorem. The constructed sketch collision produces different digests, so it no longer disappears from this comparison.

The tool requires repeated untouched rolls to agree on every declared array digest. Missing or changing untouched arrays raise an explicit measurement failure; they cannot silently disappear from the set of comparable held readings. The plan, digest and shapes remain in the report. This extends the endpoint observation for the existing declared intervention horizon; it does not prove every intermediate microstate was equal.

`intervention-observation-plan.example.json` covers substrate x and W, and field F and W_field. The largest array needs an 8.4 MB working reservation for hashing; there is no 12,000-row matrix trajectory to retain. `intervention-digest-proof.json` tests the actual reader on synthetic arrays with those source-observed dimensions and verifies that a single-coordinate change changes its digest. The proof does not claim these synthetic arrays were a live Aura state.

```sh
python tools/run_interventional_closure.py --seed DISCOVERY_SEED --anchors 8 \
  --spacing 13,14 --joint-periphery --observation-plan PLAN.json --out NEW_INTERVENTION_DIRECTORY
```

This prospective command also remains unlaunched while the existing campaigns occupy the host. These four arrays are a useful explicit starting inventory, not an assertion that no other held array matters.

## 3. Solve missing training coverage before spending another confirmation seed

The fitter drops target columns that are flat during training. In this recording, several begin changing only later. Naming them explicitly is now part of the saved-data diagnostic.

| Target | First transition with change above 1e−9 |
|---|---:|
| W.partner_wants_a_witness; D.witnessing | 1,231 |
| I.mycelium_density; G.modifier_urgency | 1,629 |
| C.mesh_weight_max | 1,821 |
| S.met_as_a_type | 2,239 |

Indices above are zero-based transition indices. The initial variance-based diagnostic finds four omitted moving targets in fold 0 and two in each later fold. Their changes are not merely roundoff: for example, `G.modifier_urgency` has held-out increment standard deviation .159, and `C.mesh_weight_max` about 3.44e−5. The strengthened check also catches a shift to a new constant increment; it names `C.substrate_weight_mean` in the later folds, with maximum departures about 1.2e−9, close to the 1e−9 measurement threshold. That weaker departure is separately recorded in `constant-shift-coverage.json`, rather than presented as the same magnitude of effect.

Across all 360 targets whose increments vary somewhere in the recording, the first training prefix giving every one non-flat variation has 2,240 rows. The earliest training fold has floor(N/5) rows. Therefore, **to contain that observed prefix with the same split, N must be at least 11,200 transitions**. At N=12,000 the earliest training window would have 2,400 rows. This is a concrete planning lower bound derived from the discovery recording. A longer trajectory can activate new variables, and adequate power can require more data; 12,000 is not certified sufficient for Aura.

The two other live-state columns have whole-recording increment standard deviations below the fitter's flatness threshold and are listed separately in `coverage-design.json`. This does not mean their increments are exactly constant. No target was reclassified as outside the carrier to improve the score.

## 4. The remaining implementation and qualification order

1. **Complete the observation contract.** Use the implemented recorder adapter to map actual held-array reads to sufficient coordinates and retain coverage of every target that changes. Keep the causal boundary tied to intervention witnesses; a predictive proxy is not grounds to move an arbitrary service inside it.
2. **Qualify the candidate on the resulting representation.** Use matched width, target count, development, delayed activation, sparse and distributed causes, interactions, nonlinear copies, and independent known-effect worlds. Include temporal controls and independently sized planted effects. Freeze the predictor, bounds, feature map, folds and decision rule before confirmation. The new bounded variant has not passed this qualification yet.
3. **Set the data budget from measured coverage and power.** Start from the 11,200-transition coverage lower bound, then size further from matched-world sensitivity. For perspective, zero false detections in 60 independent worlds has a two-sided exact 95% upper rate of 5.96%; at least 72 such zero-hit worlds are needed to put that upper limit below 5%. Neither calculation replaces distributional coverage.
4. **Finish the existing content discovery runs without interference.** Read Opus's worth-at-encoding results, their controls and rescue. The implemented pairing, lag, rank and worker-health fixes prevent invalid joins, but they do not prove the content mechanism reaches rho≥.3 with p<.01 and exceeds its sham floor. A failure here requires a witnessed memory/appraisal coupling repair, not a score override. Stable memory relations and changing perspective remain a legitimate distinct future design, not a retrospective substitute for the present tracking criterion.
5. **Run a clean carrier and joint/nonuniform intervention sequence.** Use host-time deadlines, zero unresolved authority blockers, and no degradation override. Record actual donor distances and all joint-arm results. Preserve the current campaigns' frozen source and logs.
6. **Collect compatible withheld reports and lineage evidence.** Retain the fixed uncertainty/range parsing policy and raw replies. Demonstrate the causal link to the reporting pathway if claiming whole Aura rather than substrate scope. This work did not start a cortex, restart Aura, or generate fresh model reports.
7. **Run the conjunction checker.** `tools/check_subject_core_completion.py` recomputes the battery from explicit reports and requires all 24 criteria, determined J*, parity, and compatible clean evidence. The unused confirmation seed must be chosen and registered after the design freezes; seed 83 is suitable only if it remains unused. Historical seed 73 stays historical.

The exact diagnostic commands are:

```sh
python tools/diagnose_closure_candidate.py --run SAVED_RUN --seed 73 \
  --feature-limit 8 --folds 0,1,2 --out NEW_DIAGNOSTIC.json
python tools/diagnose_closure_candidate.py --run SAVED_RUN --seed 73 \
  --feature-limit 8 --folds 1 --surrogates 3 --out NEW_CONTROL_DIAGNOSTIC.json
python tools/prove_subject_observation_basis.py --out NEW_BASIS_PROOF.json
```

Use the shared Python 3.12 environment, isolated logs/state, one numerical thread, low priority and a bounded process. The diagnostic writes atomic partial checkpoints marked `complete: false`; completing the computation never becomes an official closure verdict. It hashes all three input files and the relevant source files.

## Validation and boundaries of the result

The committed checkpoint passed 41 closure/J* integration checks and 20 anchor integration checks after merging Opus's changes. The final combined focused run, including the recorder adapter and strengthened coverage check, passed **123 tests in 30.72 seconds**. Smoke passed **164 tests with one skip in 95.03 seconds**; lint, compile and layering also passed. The later complete-array digest extension passed 37 focused checks and a fresh smoke/lint/compile/layering run; see `array-digest-tests.txt` and `array-digest-gates.txt`. Receipts accompany this continuation; overlapping runs are not added into a single larger test count.

Governance lint was rerun and still fails with 171 regressions and one stale bucket. None of its reported regression paths is changed by this continuation; its baseline was not relaxed. The full suite and aggregate quality gate have not passed in this work. Passing focused tests and smoke must not be reported as all repository gates passing.

The implemented fixes, saved-Aura error reduction, explicit coverage calculation, read-basis proof and recorder adapter are actual delivered results. A complete source-backed read inventory, fully qualified closure instrument, successful fresh content evidence and the joint confirmation are still required before all three goals can honestly be called solved.

## Paired causal comparisons and a declared horizon

The production paired-roll comparison is now independently testable. Thirty-five focused checks passed, including an AND cause detected only by the joint swap, a passive copy that never moves the held state, and restore residue in a declared array that the core summary misses. Every untouched declared-array control must match the anchor baseline, including after earlier interventions; initial twins alone were insufficient. The one-off handling remains unchanged for ordinary numeric readings.

The CLI also accepts `--turns N`, records `turns_per_roll`, and rejects nonpositive horizons. Its historical default remains two. A fake-runtime integration test passes a delayed joint cause through the real CLI, with the effect appearing at three turns, and obtains OPEN. This proves the implemented comparison and horizon routing on that known world; it is not evidence that a delayed cause exists in Aura. A horizon must be selected prospectively, rather than enlarged after a confirmation result.

This extension passed smoke (164 passed, one skipped, 75.26 seconds), lint, compile and layering. See `causal-comparison-validation.json` and its two text receipts.

## Subsets expose causes a whole swap can mask

Opus's ISC-v7 implementation (`b1605a0d4`, merged with the earlier fixes) replaces the observational closure line with single-unit and whole-periphery interventions. The completion checker now accepts `--isc v6` or `--isc v7` and reports v6 beside v7. These are different claims. A new v7 pass would not repair or replace historical seed 73's v6 result. No confirmation has been run by this continuation.

A whole swap has a further limit. Consider an outside effect

    f(a,b,c,d) = a b (1-c) (1-d).

At anchors 0000 and 1111, f is zero. Every single swap from either anchor also gives zero, as does swapping everything. Swapping only a and b into 0000 gives one. The all-at-once arm resolves an AND-only example, but cannot rule out every interacting or opposing influence.

**The implemented remedy is a prospective subset plan.** `core/subject/coalitions.py` fixes the selected combinations before their outcomes are read. `run_interventional_closure.py --coalition-swaps N` executes them through the same restored snapshots, paired controls, repeat checks and complete-array observations. It records the exact unit inventory and masks. The default is zero; existing declared runs are unchanged. Budgets are limited to 1,024 added arms. Small plans enumerate every proper nonsingle subset; larger plans sample uniformly without replacement with a fixed seed. A detected subset effect makes the result OPEN.

For four binary units, singles and whole swaps at both opposite anchors visit 10 of 16 states and miss six. There are 2^6 - 1 = **63** nonconstant Boolean effects supported only on those six states. The exhaustive offline proof finds all 63 with the added subsets; all 63 evade the original arms. The real CLI integration also returns OPEN for the explicit product above, with only subset rows moving the complete held array. The passive-copy control remains inert. These are known worlds, not a claim that this cause has been found in Aura.

The same sampler found the four-unit cause among 96 units (92 irrelevant) in 11 of 128 selected masks at seed 97. Under the explicit assumption that an effect occurs for at least one assignment of r relevant masks and all remaining masks are irrelevant, a fair mask finds it with probability at least 2^-r. Removing already inert single/whole arms cannot reduce that probability; sampling without replacement cannot increase the independent-draw miss bound:

    P(miss all N) <= (1 - 2^-r)^N.

For r=4, N=72 puts the bound below .01 (about .0096). This is a conditional design bound, not a power guarantee for unknown Aura interactions. Donors must actually change the relevant values, the horizon must include the effect, and the observation must retain it. Larger r requires more masks; complete enumeration is exponential.

With eight anchors, two donors per anchor and a two-turn horizon, 72 added masks cost 4,608 additional turns if no effect needs a repeat. Repeated effects can double that. This work launched none of those runtime turns while the existing content campaigns were active. The executable proof is `tools/prove_intervention_subset_coverage.py`, with its receipt in `subset-coverage-proof.json`.

After merging Opus's v7 changes, 85 integrated checks passed in 19.70 seconds. Smoke passed 164 tests with one skip in 73.88 seconds; lint, compile, layering and writing passed. `coalition-validation.json` binds the source hashes and receipts. The prior governance and full-suite limitations remain.
