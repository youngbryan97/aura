# Source-function update geometry

<!-- reader-navigation:start -->
> **In plain English:** The reasoning programmes investigate whether extra computation or learned internal representations improve answers on tasks not used to choose the method.
>
> **Why it matters:** A slower or more elaborate process is useful only if it produces a measurable benefit that survives relevant checks.
>
> **In everyday use:** On a difficult constraint problem, a better internal method should increase correct answers or expose uncertainty rather than only generate longer prose.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#reasoning) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#reasoning) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#reasoning) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

The operation classifier is shared across many language constructions. A
small coefficient update can affect a high-energy feature direction on many
inputs. The source fit's Euclidean parameter distance does not measure that
functional effect. Component interventions attributed five observed
validation regressions to that classifier, while the relation, argument, and
boundary changes individually preserved those answers.

## Metric

Let X contain n retained source features augmented by a constant bias feature,
and let A be an affine coefficient update. Define

```
M = I + X.T X / n
||A||_M^2 = ||A||_F^2 + ||X A.T||_F^2 / n.
```

The first term gives every direction a positive cost, including directions
not observed in the source data. The second measures mean squared change in
classifier logits. Neither term uses a validation answer.

For the thin decomposition X/sqrt(n) = U S V.T, the square root is

```
R = I + V (sqrt(I + S^2) - I) V.T.
```

Then R R.T = M, and encoding A as A R makes ordinary Euclidean projection
equal projection under the stated metric. The inverse uses reciprocal roots;
the gradient transformation is its transpose. Thin SVD avoids forming the
full feature covariance and leaves the unobserved subspace unchanged. This
transformation composes with the existing bilinear relation geometry.

## What It Establishes

For any augmented feature z, Cauchy-Schwarz gives

```
||A z||_2 <= ||A||_M sqrt(z.T M^-1 z).
```

This bounds score movement for a measured update. It is not a correctness
certificate: a baseline can be wrong, an unseen feature can lie far outside
the source distribution, and a complete decoder can change its latent
binding or operation chart. The existing retained-margin checks, re-mining
of newly selected competitors, and independent validation therefore remain.

The implementation applies the metric to the affine classifier pairs built
from retained operation evidence, deduplicated by feature content. Distinct
source observations with equal features count once in this geometry; all
their training constraints and weights remain intact. The coefficient ridge
and functional weight are both one and are named in the receipt.

The opt-in CLI flag is `--operation-metric source_function`. The default
remains coefficient Euclidean. No decoder policy, expected answer, model
identity, serving certificate, or validation threshold changes.

## Falsification

Tests compare the encoded norm to the explicit quadratic form, verify
roundtrips and gradient adjoints, exercise singular source features, and
compare a constrained update to its analytic solution. Paired source and
development-validation decoding determines whether this metric helps Aura.
Passing the algebra does not establish general transfer or broad reasoning.
