# Exposed failure boundary intervention

All 24 failures from the complete source-decision audit reproduce their archived
selected program under the current implementation. One has the correct operation
spans and labels but reversed division arguments. The other 23 differ in operation
boundaries, sometimes also in labels and bindings. Boundary disagreement alone
does not establish which difference causes the wrong result.

Two diagnostic interventions hold the existing model coefficients fixed:

| Supplied annotation | Equivalent | Different | Infeasible |
|---|---:|---:|---:|
| Operation boundaries only; model selects labels and arguments | 14 | 7 | 3 |
| Operation boundaries and labels; model selects arguments | 23 | 1 | 0 |

The argument solver receives no target arguments. Input anchors are the ordinary
decoder's anchors; source labels are aligned to them before comparison. The
existing universal-floor comparison grades the resulting programs. These are
oracle interventions on exposed validation failures, not autonomous successes,
fresh transfer, fitting data, or permission to promote a candidate.

The result localizes recognition as a major repair target: downstream machinery
can solve 23 of these cases with correct operation boundaries and labels. It does
not prove that current features and training can recover those boundaries and
labels. The remaining division-order error requires a distinct binding repair.
Three infeasible boundary-only cases also show why correct boundaries alone are
not sufficient when labels are wrong.

Artifacts retain the exact script, plan, per-row results and report at
`~/.aura/rlc-evidence/semantic-boundary-intervention-20260921/`.
Report: `61930af1c0633bae29a966f33cc7750cf9cdc238a8c4b0e19e187cb70f14ab30`.
The preceding reproduction report is
`7e000a5b07493b62a347a8326019ee04c938e20cf523ba00991c1f6c7fb41a37` at
`~/.aura/rlc-evidence/semantic-stage-attribution-20260921/`.
No G item closes.
