# Implemented remedies and the remaining qualification

These are executable changes, with offline tests and numerical calibration. They do **not** establish that Aura has passed 24/24, determined J*, or crossed the operational bridge. The saved evidence still says 23/24; content tracking and carrier authority remain unresolved on that evidence.

The work was based on b47702f06, which includes Opus's worth-at-encoding and recall-settling mechanism. No live instance was restarted, no model was loaded, and no active campaign was modified. The content workers remain in their existing frozen worktrees. Historical verdicts were preserved.

## What is implemented

| Goal | Remedy | Executable evidence | Remaining qualification |
|---|---|---|---|
| 24/24: trustworthy closure | Reject zero, negative and nonfinite sensitivity; name the invalid reading | `test_a_closure_resolution_must_be_positive.py` | Positive sensitivity still needs calibration against known effect sizes |
| 24/24: nonlinear prediction | Candidate with quadratic core and outside features, interaction terms, independent ridge penalties, screening after expansion, paired validation losses | `closure_candidate.py`, `validate_closure_candidate.py`, numerical receipts below | Full-dimensional Aura data, development, conditional nulls, coverage and surrogate calibration |
| 24/24: interacting outside causes | Optional joint interchange of the whole declared periphery; individual AND inputs can miss a cause that this joint arm finds | `test_closure_swaps_the_outside_together.py` | Actual paired rolls on fresh anchors and the declared horizon |
| 24/24: periodic donor blind spot | Optional explicit nonuniform anchor bank, recording actual donor distances | Same test file, including the real anchor collector | Freeze the schedule prospectively; varied phases are not exhaustive state coverage |
| J*: clean timing | Database enqueue, database batching, recall flush and encoder-drain deadlines use host time while subject time remains frozen | `test_resource_deadlines_keep_host_time.py`; registered runtime invariant and measured canary | A complete carrier run with no clock valve opening or other authority blocker |
| J*: paired content | Match retrieval comparisons by anchor and condition; reject malformed keys; read recall at the same lag as internal state | `test_content_comparisons_keep_their_pairs.py`, `test_the_content_future_is_read_after_the_percept_is_taken_in.py` | Intact content agreement and displacement tracking on new runs |
| J*: valid relations and health | Average tied ranks; flat/nonfinite geometry is unmeasured; worker health and source reach the merged authority decision | `test_content_workers_cannot_lose_their_failures.py` | Intact/control/rescue evidence for the recall mechanism |
| J*: clean terms | Reject degraded carrier authority, failed content authority and a control substituted for intact content | `test_the_bridge_requires_clean_authority.py`, `test_all_three_subject_goals_must_pass.py` | A compatible authoritative carrier, content run and recorded lineage |
| Bridge and combined finish | Read v6 lines, recompute the battery from explicit campaign/sweep evidence, require all 24 lines AND determined J* AND parity; return nonzero otherwise | `check_subject_core_completion.py`, historical diagnostic receipt | Fresh withheld reports and all compatible inputs |
| Evidence identity | Record revision, dirty/unknown status, seed and carrier-boundary digest; compare worker/coordinator/analysis sources | `test_complementary_subject_runs_pin_their_source.py` | Frozen clean source throughout the new experiment |

The runtime bridge wording now matches the current bridge document: parity is conditional on P1-P6. It does not establish equal calibration or equal uncertainty between Aura and another person.

Opus independently committed the positive-resolution guard and v6 reader in 2883883e8 during this work. Those changes are retained when integrating this checkpoint; they are not counted as a second discovery.

## The closure candidate: what the math and measurements establish

The official estimator stays the default. The candidate has a separate entry point and never writes an official closure verdict.

A linear predictor can mistake a nonlinear observation for new information. If K is uniform on [-1,1], P=K², and Y=K²+noise with variance .01, the linear baseline leaves variance 4/45+.01. Adding P removes 4/45, giving apparent gain:

    (4/45) / (4/45 + .01) = .8988764

P has no causal influence in this constructed world. Including K² in the core predictor solves this particular representation defect without changing the causal boundary. The regression test uses the production fitting function and checks that the candidate's incremental gain is near zero.

For a cause P1*P2 with independent balanced signs, neither P1 nor P2 has marginal correlation with the target. Expanding before screening preserves the interaction. Tests cover outside-by-outside and core-by-outside interactions and require more than 90% predictive gain in those high-signal worlds.

The first candidate failed its sensitivity test: at 2,960 rows and 12 outside variables, a diagonal penalty path detected 0/30 effects at the margin. Independent penalties alone still detected 0/30. The candidate now compares paired validation losses and screens the expanded terms. The former rule used the standard error of an individual model's absolute error, which includes noise common to both predictions. A paired comparison estimates uncertainty from their difference.

For calibration, the effect is known independently of the fitted model:

    Y = K² + sqrt(rho) P0 + sqrt(1-rho) epsilon
    K, P, epsilon independent standard Gaussian
    Var(Y | K) = 1
    Var(Y | K, P) = 1-rho
    oracle partial R² = rho

The final algorithm was evaluated on separate seed panels. Each panel contains 60 independently generated worlds at each of five fixed effects. The decision here is a positive one-sided 95% block-bootstrap lower bound on held-out predictive gain. This is a candidate calibration, not the official closure decision.

| Rows | Oracle effect | Detected | Exact two-sided 95% interval |
|---:|---:|---:|---:|
| 6,000 | 0 | 0/60 | 0–5.96% |
| 6,000 | .019607843 | 48/60 | 67.7–89.2% |
| 12,000 | 0 | 0/60 | 0–5.96% |
| 12,000 | .009803922 | 52/60 | 75.4–94.1% |
| 12,000 | .019607843 | 59/60 | 91.1–100% |
| 12,000 | .039215686 | 60/60 | 94.0–100% |
| 12,000 | .1 | 60/60 | 94.0–100% |

The 6,000-row panel used seeds 9970–10029. After that result, the larger design was tested on new seeds 20000–20059. Thus the sample-size choice used the earlier panel; the later panel did not drive a change in the predictor. Source hashes are in `validation.json`.

This supplies a concrete remedy for the two demonstrated prediction defects and a measured sample-size result **for these worlds**. It does not prove 12,000 rows suffice for Aura. Aura has thousands of predictors, many targets, temporal dependence and changing dynamics; this calibration has one target and twelve outside variables. Random interaction projections at large widths also do not guarantee detection of every interaction. No candidate result is promoted on the strength of this table.

The candidate still needs the full production preprocessing, forward-fold design, surrogate decision and independently sized plants on matched high-dimensional/developing controls. In particular, a low observed gain is not a closure certificate. The saved-data diagnostic tool therefore emits `official_verdict: null`.

## Execution path

Use the shared Python 3.12 virtualenv, a separate log/state directory, one numerical thread, low priority and an explicit process timeout. Do not start another runtime while the current content experiments own the machine.

1. Finish the existing worth-at-encoding discovery runs and their matched controls. These runs can decide whether that mechanism improves tracking. They cannot retroactively become confirmation for changes made after their results.
2. Exercise the candidate on the saved discovery recording using `tools/diagnose_closure_candidate.py`. It retains all live targets, held predictors and environment columns. It is deliberately separate from the scorer. A negative gain stays negative.
3. Validate estimator sensitivity on matched width, drift, sparse/distributed causes, nonlinear copies, interactions and delayed effects. Freeze the resulting method before the next unused confirmation seed. Keep the causal boundary tied to intervention/dependency witnesses, never to whichever relabelled columns make the score green.
4. Freeze the discovery mechanism and run a new clean campaign, carrier, intervention, content and withheld-report sequence on compatible source. The optional joint/nonuniform intervention is prospective evidence; it does not change the historical meaning of the old two-turn test.
5. Read the canonical lineage from its explicit state log. Then run the combined checker. It hashes the exact report bytes and the person rows read, so the receipt identifies its inputs. Historical unpinned reports are readable but cannot satisfy this new clean-completion gate.

Candidate calibration:

    python tools/validate_closure_candidate.py --worlds 60 --rows 12000 \
      --seed-start 20000 --out NEW_CALIBRATION.json

Saved-data diagnosis, with a bounded low-priority process around it:

    python tools/diagnose_closure_candidate.py --run SAVED_RUN \
      --seed DISCOVERY_SEED --out NEW_DIAGNOSTIC.json

Prospective intervention options:

    --anchors 8 --anchor-turns 13,26,39,53,66,79,92,105 --joint-periphery

The actual distances in that bank have no common divisor greater than one. This removes the common-period blind spot of equally spaced donors; it does not prove every phase or conditional cause was exercised.

Final combined check:

    python tools/check_subject_core_completion.py \
      --campaign CAMPAIGN_JSON --carrier CARRIER_JSON --content CONTENT_JSON \
      --interventional INTERVENTIONAL_JSON --reports WITHHELD_REPORTS_JSON \
      --state-log CANONICAL_STATE_DB --out NEW_COMPLETION.json

Exit zero requires all three operational goals and compatible evidence. The tool does not launch a runtime, retune a threshold, overwrite its output, or turn an unmeasured term into a pass. Its scope is recorded; substrate-only evidence does not become proof about an entire resident language-model process.

## Validation and current limits

The focused final run passed 96 tests. Smoke passed 164 tests with one skip, in 110.75 seconds on this busy host. Compile, lint and layering passed. The source contains a prospective candidate rather than a replacement for the official estimator.

Governance lint is not globally green: it reports 171 baseline regressions and one stale bucket. None of its reported regression paths were modified in this checkpoint, and a comparison of effect calls in the changed core files found no added effects after excluding line-number shifts. The baseline was not relaxed. The full suite and aggregate quality gate were not run.

`historical_summary.json` is a new diagnostic join of saved evidence, including an explicitly older report-grounding result. It returns incomplete, 23/24, with J* and bridge unresolved. It does not regrade seed 73. The larger local diagnostic includes the lineage graph and is left out of the checkpoint.

No complete Aura confirmation run has passed all three goals in this work. The implemented fixes and finite-model results are the evidence delivered here; the remaining experimental qualification is still work to do.
