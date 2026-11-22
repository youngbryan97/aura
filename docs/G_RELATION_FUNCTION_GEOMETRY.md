# Relation learning in function coordinates

The scaled source trial repaired its selected training errors but exchanged
four validation gains for four losses. Component interventions locate both
effects in the relation head. This design addresses an ambiguity in its
learning penalty; it does not assume that the ambiguity explains every loss.

## Function and factor coordinates

The learned contribution for reference x and definition y is

```
s(x, y) = scale * x.T Q D.T y
```

For any invertible rank-by-rank matrix T, replacing Q by Q T and D by
D T^-T preserves every score. Ordinary Euclidean distance between factor
coefficients is not preserved. Thus two representations of the same relation
function can receive different minimum-change updates.

Let Q0 and D0 be the factors at the beginning of one fit. The opt-in metric is

```
||delta||^2 = scale^2 * (
    ||deltaQ D0.T||_F^2 + ||Q0 deltaD.T||_F^2
)
```

Each matrix product is unchanged by the simultaneous coordinate transform.
The metric is therefore invariant. This is a statement about separate
first-order contributions. It is not the norm of their sum, nor the exact
nonlinear function displacement: the latter also contains deltaQ deltaD.T.

For full-column-rank factors, define

```
Rq = scale * sqrt(D0.T D0)
Rd = scale * sqrt(Q0.T Q0)
Zq = Q Rq
Zd = D Rd
```

Euclidean displacement in Z equals the stated metric. Physical gradients pull
back through Rq^-T and Rd^-T. The existing constrained solver can operate in Z
without changing relation scoring, search, or runtime tensor shapes. Thin SVDs
compute the roots; no hidden-width-squared matrix is constructed.

Rank-deficient factors do not define this invertible chart. The option reports
that condition rather than adding an undeclared regularizer. The existing
coefficient-space method remains available; callers must select it explicitly.

## Stored precision and scope

Every proposed chart point is converted back to physical coefficients, rounded
to float32, and evaluated with the existing nonlinear graph scorer. Retained
floors are unchanged. The margin solver's storage callback returns this actual
roundtrip; it never certifies a continuous point in place of the stored model.
Checkpoint identity includes the metric and its implementation.

Tests compare the metric with direct matrix multiplication, apply reciprocal
scaling and non-diagonal invertible changes of coordinates, check gradients by
finite differences, and compare fitted functions after reciprocal scaling.
They also cover checkpoint replay, frozen blocks, rank deficiency, and failed
storage roundtrips. These are executable numerical tests, not an exact proof
of finite-precision invariance for every matrix.

The source trainer and development trial expose `relation_metric="factor_function"`;
the CLI exposes `--relation-metric factor_function`. The default and runtime
decoder are unchanged. A paired development run must establish whether this
choice helps. Fresh transfer and broad reasoning still require independent
measurements; no finite set of retained inequalities guarantees them.

## Dependent affine faces

The numerical repair solves `min ||d||^2/2` subject to `A d >= b` through its
nonnegative dual. A Gram solve squares conditioning; active equality refinement
therefore uses QR and SVD of the constraint features. Independently replayed
feasibility and primal/dual bounds still decide acceptance.

Dependent active rows may have inconsistent equality targets even when their
inequalities are feasible. Let `r = b_I - U U.T b_I` be the component outside
the column space of the active feature matrix. Then `A_I.T r = 0` and
`b_I.T r = ||r||^2`. Moving the multipliers in direction r increases the dual
objective without changing its quadratic term. The step stops when a negative
component would make a multiplier negative; that face leaves the active set.
If no such component exists, this numerical search returns for independent
verification rather than declaring a theorem of infeasibility.

This removes the incompatible-equality trap, rather than treating a
least-squares equality compromise as an inequality solution. Numerical rank
and roundoff thresholds guide search only; they do not relax the final
original-constraint verifier. Feasible generated systems and a duplicated
face with unequal targets exercise both warm and cold starts.
