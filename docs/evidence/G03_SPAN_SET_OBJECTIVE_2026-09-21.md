# Complete span-set boundary learning

The complete 1,264-row source audit leaves 24 exposed validation failures.
Their recorded operation scores were decomposed into the pointer and operation
classifier terms without fitting or decoding again. Short fragments often
outscore the annotated phrase on the pointer term. Some annotated phrases also
have the wrong highest-probability opcode, and one failure uses the correct
operation spans with reversed arguments. Boundary learning cannot repair all
three mechanisms on its own.

Decomposition receipt:
`fca9745da82d5d18bf990fa90b168cbdaa35aa02a2c587f2d3a5b67bf11467ee`.
Root: `~/.aura/rlc-evidence/semantic-span-score-attribution-20260921/`.
The full target/selected graph components remain in
`semantic-full-failure-components-20260921/`, receipt
`99dd951b336e03af4d0ca39954d388ae444996c4e7bf68a51c3ed269985b6bcf`.

## Objective

The new opt-in `span_set_pointer` fit uses the existing pointer representation.
It changes the training objective from separate boundary decisions to likelihood
over complete non-overlapping span sets. No new prompt, backbone, phrase table,
or validation label enters the fit.

For hidden sequence H, candidate intervals I, and count bound K, let Y range over
sets of at most K non-overlapping intervals. Adjacent intervals are allowed.
For interval [s,e), its score is:

    q(s,e) = H[s] @ w_start + H[e-1] @ w_end + b
             + frozen_pair_score(s,e) - existing_length_penalty
    S(Y) = sum(q(s,e) for [s,e) in Y)
    Z = sum(exp(S(Y)) for every allowed Y)
    loss = log(Z) - S(source_annotation)
           + lambda/2 * ||weights - parent_weights||^2

With F[t,k] the log partition for exactly k intervals in the first t tokens:

    F[t,0] = 0
    F[0,k>0] = -infinity
    F[t,k] = logaddexp(F[t-1,k],
                      logsumexp(F[t-l,k-1] + q(t-l,t) for valid l))
    log(Z) = logsumexp(F[n,k] for k=0..K)

The skip edge advances exactly one uncovered token. It therefore gives each
background region one path and counts each interval set once. A reverse pass
through the same recurrence gives each interval's marginal probability. The
gradient is expected boundary features minus annotated boundary features, plus
the regularization derivative. Runtime is O(n * L * K), where L is the existing
span-length bound. No top-k approximation enters this objective.

This is a bounded interval-set specialization of
[semi-Markov conditional random fields](https://proceedings.nips.cc/paper_files/paper/2004/file/eb06b9db06012a7a4179b8f3cb5384d3-Paper.pdf).
For frozen features and strictly positive quadratic regularization, the stated
objective is strictly convex in the fitted coefficients. That gives a unique
mathematical minimizer; a numerical convergence receipt is still required.

## Runtime boundary

Export uses the existing `LinearPointerHead`. Its scores reach ordinary operation
proposals, complete graph selection, serialized model restoration and coefficient
lesions through their current code. Pair weights, operation labels, argument
heads and chart-length penalty remain unchanged. Background-log-odds models that
exclude boundary scores from graph ranking are not eligible for this objective.

The partition is exact for the declared interval model. It does not model typed
argument feasibility or opcode uncertainty, prove that annotations are the only
semantically valid spans, or guarantee correct unseen programs. A paired runtime
measurement must decide whether it helps. The new fit grants no serving authority.

## Paired measurements

Both fits were tested on the same 16 development controls and two previously
exposed division failures, with counterbalanced arm order and a 20-second
per-observation solver limit. No validation labels entered either fit.

| Training rows | Incumbent equivalent | Span-set equivalent | Changed outcomes |
|---|---:|---:|---:|
| 64 | 16/18 | 16/18 | 0 |
| 764 | 16/18 | 16/18 | 0 |

The full fit converged in 19 iterations, taking 92.028194 seconds after source
loading. Its objective decreased from 0.216209624 to 0.032636974. Better source
likelihood did not repair either challenge. No candidate is promoted and no
larger evaluation is justified by this pilot alone.

Small pilot receipt:
`459d7ee2631d6d63ac7caac776e9a313cc29a9f65ad694f6bbb923758862388d`.
Full-source pilot receipt:
`ca375f05a174aaac7cb5d58e656792ed1979bf4c03838b18c9dc30840ef828e3`.
Roots: `~/.aura/rlc-evidence/semantic-span-set-pilot-20260921/` and
`~/.aura/rlc-evidence/semantic-span-set-full-fit-20260921/`.
Each retains the plan, exact script, coefficients and 36 arm observations.

Focused pointer suites: 40 passed. Smoke: 164 passed, one skipped. Lint,
compile, governance, layering and writing checks passed. These checks validate
the implementation and integration, not a reasoning gain.
