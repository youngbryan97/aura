# Generalization repair: theory and implementation

This is the implementation plan for Bryan's five requested approaches. It
extends the existing semantic compiler, graph learner, universal floor and
replication tools. It grants no serving authority and changes no G-ledger
acceptance criterion.

## 1. Conceptual route

Generalization requires shared distinctions that remain useful when wording,
values, dependencies and task families change. A lookup of corrected answers
cannot supply that. Aura already learns operation, boundary and binding
parameters shared by every decoded request.

The target is a compositional translation whose primitives and composition
rules preserve meaning. Structural induction proves execution correctness for
that declared grammar if every translation rule preserves semantics. This
does not prove that an unseen sentence was assigned the intended structure.
Fresh construction and family tests must measure that assignment.

Implementation: retain complete-program counterexamples from
`semantic_joint_graph_learning.py` and `semantic_runtime_graph_retention.py`.
Train common coefficients. Never supply annotated programs or expected
answers to inference. Use the existing universal floor to distinguish wrong
programs, including programs that happen to agree on the original inputs.

## 2. Aura's architectural route

Keep one semantic representation and one executor. Refitting must preserve
the declared model architecture unless a capacity expansion is explicit.
The September 18 graph refit silently added a 15,360-parameter diagonal
interaction to an additive boundary pointer. Ordinary joint refitting now
preserves additive or paired geometry through extraction, gradients and
export. Explicit paired-pointer fitting remains available as a separate
experiment.

The correction learner changes the existing transducer coefficients. The
same runtime decoder consumes them. No new answer cache, prompt, routing
heuristic or parallel language substrate is introduced.

The opt-in `retain_existing` boundary policy separates preservation from
supervision. For an initial boundary difference d_0, retain
d >= min(d_0, 0), rather than demanding that every annotated boundary overtake
every other span. Whole-program counterexamples still drive correction.
This protects witnessed boundary evidence; it does not certify the meaning
of either span or weaken the exact program checks.

## 3. Experimental route

Treat a witnessed mistake as an obligation on shared scores. Find a small
coefficient correction that satisfies the obligation alongside retained
comparisons. Learn from the counterexample's features rather than its source
identifier. This transports the correction to other requests using the same
features; whether that transfer helps is measured rather than assumed.

`margin_repair.py` solves the minimum-displacement affine subproblem.
`semantic_graph_constraints.py` exposes `update_rule="minimum_change"`.
It adds newly violated retained comparisons to its working set and checks
every retained nonlinear margin after float32 storage. The existing default
update rule remains available for matched comparisons.

The proposal is a local method for nonlinear scores. It can report an
unverified local projection, exhaust its search allowance or fail to improve.
None of those outcomes proves that the architecture cannot learn the task.
An affine optimality certificate is not a nonlinear global optimum.

## 4. Ground-up experiment

1. Reproduce the capacity change with a failing unit test.
2. Test the affine solver against closed-form solutions, coordinate changes,
   contradictory obligations and forged certificates.
3. Test the actual graph learner, pointer export and durable checkpoint path.
4. Compare both update rules on identical small source-training and validation
   cohorts. Select failures using training observations only. Keep the
   incumbent, pre-fit scoring change and learned candidate distinguishable.
5. Expand the development cohort only after measuring gains and regressions.
6. Freeze a candidate before fresh construction, vocabulary, depth and family
   replication through the existing acquisition and verification tools.
7. Measure ordinary runtime reasoning with matched compute and causal lesions.
   Then qualify current-model materialization, fusion and live serving.
8. Compare named current external baselines under a declared task and resource
   protocol. A synthetic-program score cannot replace G09 or G12.

The small and full refit CLIs expose the new update rule. Tests and local
development trials do not authorize promotion.

## 5. Mathematical derivation and falsification

Let a witnessed linear comparison have margin

    m_i(theta) = a_i^T theta + c_i.

For a desired margin gamma and incumbent theta_0, seek

    minimize_delta  0.5 * ||delta||_2^2
    subject to     A delta >= b,
    b_i = gamma - m_i(theta_0).

This is projection onto an intersection of affine halfspaces. Its Lagrangian
and dual are

    L(delta, lambda) = 0.5 ||delta||^2 + lambda^T (b - A delta),
    delta = A^T lambda,
    maximize_lambda>=0  b^T lambda - 0.5 ||A^T lambda||^2.

For any primal-feasible delta and nonnegative lambda, weak duality gives

    b^T lambda - 0.5 ||A^T lambda||^2 <= 0.5 ||delta||^2.

To prove it directly, subtract the dual from the primal:

    0.5 ||delta - A^T lambda||^2 + lambda^T (A delta - b) >= 0.

Zero gap therefore proves optimality. `verify_margin_repair` recomputes the
primal constraints, dual sign, objectives and gap independently of the
optimizer's success flag. Current production arithmetic is binary64 with
declared tolerances, so the receipt says `verified_numerically`, never exact
rational proof. Exact infeasibility remains the responsibility of Aura's
existing `score_capacity.py` and linear-arithmetic proof kernel.

This is standard convex duality; the derivation above specifies the form used
here. Reference: [Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).

### Calculate a repair

Start at theta_0 = (1, 0). A correction requires y >= 0.5. An existing decision
requires x - 3y >= 0.1. The closest feasible point is (1.6, 0.5):

    delta = (0.6, 0.5)
    ||delta|| = sqrt(0.61) = 0.7810249675906654
    objective = 0.305.

Both constraints bind. Their normals are (0, 1) and (1, -3). Multipliers
(2.3, 0.6) are nonnegative and reproduce delta as A^T lambda, giving zero
duality gap. The integrated graph-learner test recovers that solution.

`verify_exact_margin_repair` independently checks this witness using rational
arithmetic: both objectives equal 61/200 exactly. It accepts explicit rational
inputs without fitting or rounding. Tests reject a witness that misses a
constraint by 1/10^21 and a forged negative dual multiplier. This exact result
proves the worked affine problem; it does not upgrade numerical nonlinear
fit receipts into exact proofs.

For a single normal a and positive deficit b, the closed form is
delta = b*a/(a^T a). The test with a=(3,4), b=2 recovers (0.24,0.32)
and objective 0.08.

### Calculate the generalization boundary

For a fixed offset and a difference feature phi in a ball of radius rho
around phi_0, Cauchy-Schwarz gives

    theta^T phi + c >= theta^T phi_0 + c - rho ||theta||.

If the right side is strictly positive, that one comparison is preserved
throughout the declared ball. Equality is attained by moving the feature in
direction -theta, so the bound is tight. With theta=(1,0.5), phi_0=(1,0),
c=0 and rho=0.1, the lower margin is 0.8881966011250105.

To turn this into correctness of an unseen request, all of the following
must hold: its correct interpretation is representable; every incorrect
competitor is covered; its features fall inside a certified region whose
label is still valid; and search and execution preserve the winning meaning.
The current feature-radius function does not establish those premises. Its
receipt explicitly leaves coverage unestablished.

### Falsify the excessive claim

Retaining the correction y>=0.5 alone changes (1,0) to (1,0.5). The previously
positive unseen comparison phi=(1,-3) then changes from 1 to -0.5. Minimum
parameter change alone does not guarantee generalization or zero regression.
The test suite includes this exact counterexample. Adding that comparison
as a retained obligation gives the joint solution above.

More fundamentally, two target functions can agree on every observed example
and disagree on an unobserved one. A learner receiving the same observations
cannot be guaranteed correct for both. An inductive bias, additional evidence
or a restricted task class is necessary. See [Understanding Machine Learning,
Section 5.1](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf).

### What 500/500 would mean

Under independent identically distributed trials, 99% probability of an
all-correct 500-case run requires per-item success

    p >= 0.99^(1/500) = 0.9999798995303102.

Without independence, the union bound gives the sufficient condition that
the sum of the 500 failure probabilities is at most 0.01.

Conversely, zero errors on 500 independent fresh cases gives a one-sided 95%
binomial error upper bound of 1 - 0.05^(1/500) = 0.005973551516349596.
It does not prove zero population error. Repeated development selection on
the same 500 cases does not meet that fresh-sample premise.
