# Aura formulas, explained with examples

Status: Guide · Source and definitions checked 25 October 2026

A formula is a precise way to say what a calculation does. Its result only
means what the inputs, units and comparison allow. This guide pairs the math
with an explanation you can use aloud. Start with [the glossary](GLOSSARY.md)
for unfamiliar terms and [ISC explained](ISC_EXPLAINED.md) for the experiment.

**Every numerical example here is invented for explanation.** None is a new
Aura measurement. Source links identify the implementation or method; the
[current status page](ISC_STATUS.md) identifies actual results and corrections.

## Reading the notation

| Symbol | Plain meaning |
|---|---|
| Subscript i or t | One sample, coordinate or time step; the surrounding definition says which |
| Δ, delta | A change or intervention |
| Σ, sum | Add the indicated values |
| mean or an overbar | Add the values and divide by their count |
| max / min | Take the largest / smallest |
| \|\|v\|\| | A vector's length under the stated norm |
| a · b | Multiply matching coordinates, then add them |
| A \ B | The elements of A with B removed |
| Given E, or \| E | Account for the declared environment/input |
| ≈ | Approximately equal; it does not imply an uncertainty interval |
| ∧ | Both conditions must hold |
| None / missing | No numeric value in that field; check whether this means unavailable or delegated to a default rule. It is not zero |

The same letter can have several roles. Statistical alpha is a decision
level; steering alpha is a scale on a direction. The former does not measure
how strongly an intervention is applied.

## Joint effects

For one saved comparison moment, let N1 be the unchanged target, F the target
after changing the first source, S after changing the second, and B after both.

~~~text
I_i = B_i - F_i - S_i + N1_i
sham_i = N2_i - N3_i - N4_i + N1_i

main_i = max(||F_i - N1_i||, ||S_i - N1_i||)
R = mean(||I_i||) / mean(main_i)
~~~

**Essentially:** subtract what each separate change explains from the combined
change. What remains is the joint interaction. A matched sham repeats the same
arithmetic using unchanged copies.

For N1 = 10, F = 12, S = 13 and B = 19, the interaction is 4, the larger main
effect is 3, and the single-sample ratio is 4/3 ≈ 1.33. Four unchanged values
of 10 give a sham of zero. The real experiment uses vectors in declared
whitened coordinates and multiple comparison moments.

The ratio is a ratio of means, not a mean of per-sample ratios. If two samples
have interaction sizes 4 and 2 and main sizes 3 and 5, R = 3/4 = 0.75.
The implementation returns zero when the mean main effect is nonpositive;
that guard does not convert an uninformative comparison into a passing result.

**Practical meaning:** one source's effect depends on the other source's state.
This alone neither passes ISC nor establishes experience. The actual v6 line
also requires the above-sham and common-direction tests, sufficient samples
and all specified triples. Shared history between arms invalidates the clean
causal interpretation even if the arithmetic is positive.

Sources: [implementation](../core/subject/interventional_synergy.py),
[frozen v6 protocol](ISC_V6_PREREGISTRATION.md).

## Direction consistency

Let s_-i be the sum of the other samples' interaction vectors.

~~~text
s_-i = sum(I_j for j != i)
D = mean(I_i · s_-i / ||s_-i||)
~~~

A zero-length s_-i contributes zero in the implementation.

**Essentially:** ask whether each interaction points along the direction
suggested by the other interactions. Three vectors (1, 0) give D = 1.
Large vectors that repeatedly reverse direction can fail the test.

The null flips whole-vector signs and recalculates the statistic on every
draw. A large D without that comparison is not a p-value.
Source: [direction statistic](../core/subject/interventional_synergy.py).

## Vector length and whitening

~~~text
||v||_2 = sqrt(v_1^2 + ... + v_d^2)
z = W (x - mu)
~~~

v is a list of d coordinate changes. x is a measured state, mu is the fitted
baseline centre, and W is the declared whitening transform fitted on baseline
data. The formula for z is schematic: the implementation determines rank,
regularisation and which directions are retained.

**Essentially:** length summarises a multicoordinate change; whitening puts
that change against a reference scale of ordinary variation. The vector
(3, 4) has length 5. In a scalar example with baseline mean 10 and standard
deviation 2, a value 14 becomes (14 - 10)/2 = 2 baseline standard deviations.

Whitening does not remove drift, create independent samples, or make an
incomplete state description complete. Refit rules must stay fixed across
the compared arms. Source: [paired displacement](../core/subject/paired_displacement.py).

## Closure leak

Let SSE_core be the pooled sum of squared prediction errors when the predictor
has the selected core and the declared environment, using the declared
prediction target (core change in the evening method). SSE_wide is the same
measurement with the recorded periphery also supplied.

~~~text
leak = (SSE_core - SSE_wide) / SSE_core
~~~

This expression applies when the baseline error is nonzero; the source has
explicit guards for degenerate baselines.

**Essentially:** how much of the remaining prediction error does information
outside the proposed core save? If errors are 100 and 80, leak = 0.20.
That means 20% less prediction error under this estimator, not 20% of a mind
outside the boundary.

A negative estimate can mean the wider predictor performed worse; it does not
prove that a domain has negative causal importance. Forward folds, nulls,
uncertainty, coverage and estimator validity remain necessary. The current
discovery work found a problem on the earliest part of a developing run;
read [status](ISC_STATUS.md) before interpreting a margin from it.
Source: [pooled closure scorer](../core/subject/closure.py).

## Closure margin and decision

Let epsilon be a defensible margin declared before fresh confirmation.
For the declared method, a positive closed reading requires the applicable
controls and bounds:

~~~text
no open fold AND no estimator-failed fold
AND outside-leak upper bound < epsilon
AND planted-leak resolution <= epsilon
~~~

**Essentially:** the method must bound the outside contribution below the
stated tolerance and demonstrate enough resolution to support that claim.
An invented upper bound 0.01 and resolution 0.02 fit a margin 0.03, provided
all other controls hold. A detector resolving only 0.05 cannot establish that
0.03 claim through non-detection.

**Current source:** at `72dd0d703`, no numeric margin or margin rule is in
force. The gate therefore cannot positively establish closure. A failed fold
cannot support a closed reading. The estimator now predicts core change and
enters periphery columns as increments.

### Withdrawn least-domain proposal

The afternoon proposal defined epsilon = min_j(l_j), where l_j was an internal
domain's extra predictive contribution to the rest. Contributions 0.12, 0.08
and 0.05 would produce 0.05 arithmetically. The rule was **withdrawn** because
a domain can be needed while adding little predictive information beyond
what the rest's past already carries. That minimum is not a justified margin
merely because the arithmetic is defined. This example explains the retired
proposal; it is not an active rule or a new result.

Sources: [source](../core/subject/closure.py),
[withdrawal record](evidence/ISC_CLOSURE_WITHDRAWAL_2026-10-25.json),
[latest status](ISC_STATUS.md).

## Finite randomisation p-values

For B randomly drawn null statistics, let b count those at least as large
as the observed statistic in the one-sided comparison.

~~~text
p = (b + 1) / (B + 1)
~~~

**Essentially:** compare the observed statistic with the declared null while
retaining the finite-draw correction. If no draw out of 100 is as extreme,
p = 1/101 ≈ 0.0099, not zero.

A p-value is not the probability that the hypothesis is false, the probability
of consciousness, or the size of an effect. The null and randomisation
assumptions matter. The paired sign-flip test applies this calculation to
flipped comparison values; the direction test has its own statistic.

Sources: [Aura implementation](../core/subject/interventional_synergy.py),
[Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).

## Rank correlation

~~~text
rho = PearsonCorrelation(rank(x), rank(y))
~~~

Without tied ranks, an equivalent shortcut is:

~~~text
rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
~~~

d_i is the difference between the two ranks for sample i; n is the number of
paired samples. With ties, use the appropriate rank calculation rather than
blindly applying the shortcut.

**Essentially:** do the larger values in one set tend to correspond to larger
values in the other? Ranks (1, 2, 3) versus (1, 3, 2) give rho = 0.5.
A constant input has no defined rank correlation; missing is not zero.

Correlation is not an effect size in the original units, a causal proof or
a probability of experience. See [ISC variables](ISC_VARIABLES.md) and the
[official SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

## Gaussian integration score

For a declared partition into groups M_k of the measured state X:

~~~text
SI = sum_k h(M_k,t | M_k,t-1) - h(X_t | X_t-1)
~~~

h is conditional differential entropy under the fitted Gaussian model. With
natural logarithms, its units are nats. The previous-time value after the
vertical bar is what the prediction is conditioned on.

**Essentially:** compare predicting the groups separately with predicting the
measured whole. In an invented two-group example, conditional entropies
1 + 1 versus 1.4 give SI = 0.6 nats.

The covariance model, measured channels, grain, regularisation and search
matter. Exhaustive search certifies the minimum only at the implemented
small-grain limit. Queyranne's general guarantee requires submodularity,
which this objective does not generally have. This score is not full IIT 4.0.

Sources: [current estimator](../core/consciousness/integrated_information.py),
[whole-system guide](WHOLE_SYSTEM_PHI.md),
[time-series methods](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001052),
[Queyranne's theorem](https://link.springer.com/article/10.1007/BF01585863).

## The seven-factor workspace product

~~~text
E_t = I_t * G_t * S_t * V_t * M_t * A_t * C_t
~~~

The factors describe integration, global availability, self-state ownership,
valence, memory continuity, agency coupling and counterfactual dependence.
The implementation bounds component values before forming its product.

**Essentially:** the project wants all seven functional ingredients to matter.
If any is zero, the product is zero. If all seven are 0.8, the product is
0.8^7 ≈ 0.21.

This is a project-defined score, not a published law or a probability that
the system experiences something. The factors need their own measurements.
Sources: [definition](ORGANISMAL_WORKSPACE_THEORY.md),
[implementation](../core/being/causal_self_state.py).

## Activation steering

For a simplified intervention at one declared activation site:

~~~text
h_new = h + alpha * v
~~~

h is the original activation vector, v the steering direction and alpha its
scale. This illustrates the addend; a production route may add normalization,
multiple channels, clipping, certificates and governance checks.

**Essentially:** change a computation inside generation along a measured
direction. For h = (1, 2), v = (0, 1), alpha = 0.5, the new value is (1, 2.5).
Alpha 0.5 does not mean 50% emotion or 50% confidence.

Sources: [steering implementation](../core/consciousness/affective_steering.py),
[Contrastive Activation Addition](https://arxiv.org/abs/2312.06681).
A channel effect is separate from qualification to install or serve the basis.

## Sampling temperature

For logits z and positive temperature T:

~~~text
P(token i) = exp(z_i / T) / sum_j exp(z_j / T)
~~~

**Essentially:** rescale the relative preferences before drawing a token.
For two logits 0 and ln(3), T = 1 gives probabilities 0.25 and 0.75.
T = 2 gives approximately 0.366 and 0.634, still unequal.

T = 0 is handled as a separate greedy case; it is not literal division by
zero. Other filters can alter the sampling distribution. Aura's [0, 2]
configuration limit is a project contract, not a theorem that larger
temperatures instantly produce a uniform distribution.

Sources: [Aura configuration](../core/config.py),
[official MLX sampling implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py).

## Cosine similarity

~~~text
cosine(a, b) = (a · b) / (||a|| * ||b||)
~~~

**Essentially:** compare the directions of two nonzero vectors. For (1, 0)
and (1, 1), similarity is 1/sqrt(2) ≈ 0.707.

It is not a 70.7% probability that a memory is relevant. A zero vector needs
an explicit handling rule. Actual score distributions and null comparisons
matter, especially after changing encoders.
Source: [retrieval calibration](../core/memory/retrieval_calibration.py).

## Trend and time to a threshold

For distinct times t_i and values x_i, Sen's slope uses pairwise rates:

~~~text
slope = median((x_j - x_i) / (t_j - t_i), for j > i)

time_to_threshold = (threshold - current_value) / slope
~~~

The second expression illustrates a positive linear trend toward a higher
threshold; forecasting rules also require uncertainty, significance, horizon
and valid inputs.

**Essentially:** estimate the rate, then how long that rate would take to
reach the line. An invented reading of 20 GB, threshold 26 GB and rate
2 GB/hour gives three hours. It is an extrapolation under the stated trend,
not knowledge of when the host will fail.

Sources: [forecast implementation](../core/autonomic/allostasis.py),
[method references](SCIENCE_AND_SOURCES.md#trend-and-slope-methods).

## Model-lane memory budget

The default arithmetic can derive a lane budget from total host memory:

~~~text
lane_budget = fraction * host_memory
proposed_total = already_committed_lanes + candidate_request
~~~

For invented values, 0.72 * 64 GB = 46.08 GB. Commitments of 40 GB plus a
10 GB request exceed that lane budget. It does not follow that the operating
system has 17.92 GB free; other processes and allocations still matter.

**Essentially:** declare work before loading it and account for other
commitments. Arithmetic alone does not reserve memory or authorise eviction.
Sources: [admission arithmetic](../core/brain/lane_admission.py),
[durable lane controller](../core/runtime/model_lane_control.py).

## Passing checks

~~~text
pass_count = sum(1 if criterion_j passes else 0, over the declared criteria)
~~~

**Essentially:** count which operational requirements held in that versioned
run. A hypothetical 22 out of 24 is about 91.7% of those checkboxes; it is
not 91.7% consciousness, and one unmet required criterion can prevent the
full conjunction from passing. Real scores must retain controls and later
method corrections. Source: [current ISC status](ISC_STATUS.md).

## J-star and the bridge are structured claims

J* combines the identified carrier, relations among its contents and its
causal lineage. It is not a scalar formula to maximize. The bridge checks
the conjunction of five grounds:

~~~text
carrier AND markers AND reports AND structure AND lineage
~~~

**Essentially:** all required supporting grounds must be established under
the named assumptions. Four holding grounds do not substitute for a missing,
failed or unresolved fifth. Sources: [bridge guide](BRIDGE_PARITY.md),
[bridge implementation](../core/subject/bridge.py).

## Likelihood ratios and assumptions

For hypotheses H1 and H0 and evidence O:

~~~text
posterior_odds(H1:H0) = prior_odds(H1:H0) * P(O | H1) / P(O | H0)
~~~

**Essentially:** evidence changes the odds when the hypotheses assign it
different probabilities. Invented prior odds 1:3 and likelihood ratio 2
give posterior odds 2:3, or probability 2/5 = 0.4.

This does not supply a probability for Aura. The relevant likelihoods would
need justification. If a hypothetical duplicate is stipulated to give exactly
the same observations under every intervention, the ratio is 1 by that
stipulation. That limitation of the hypothesis pair is not a demonstration
that every scientific argument about experience is impossible.

A project-defined evidence weight or score is not automatically a calibrated
Bayesian likelihood ratio. Source: [the hypothesis-pair discussion](phenomenology/README.md).

## Assumptions behind a mathematical claim

A formula needs its domain and decision rule. An unchanged winning action does
not establish an unchanged score; near-aligned vectors do not prove a
contractive update; and an available predecessor does not guarantee better
performance under one fixed shared budget.

[The mathematical-claim guide](MATHEMATICAL_CLAIM_LIMITS.md) gives plain-English
counterexamples, the corrected assumptions and their practical consequences.
A unit test over finite examples cannot supply a missing universal proof.

## Current numeric closure margin and K*

The night method at `96487c6f1` names `epsilon = 0.02 / (1 + 0.02)`, about
0.0196078. That is the numeric conversion adopted by this project; its
scientific scope is explained in [current status](ISC_STATUS.md#the-night-addendum-include-what-the-transition-reads-then-test-both-ways).
The full closure criterion additionally needs the interventional report to
read `CLOSED` on the matching scored seed. The observational formula above
alone is insufficient. K* includes the declared held-state dependencies on
the core's side of that calculation. No new result follows from this formula.
