# Working-face attribution correction, 2026-09-19

This corrects the interpretation in the September 18 full-run note. The
historical note and artifacts remain unchanged.

The full run was frozen at `7ce68f36a`. It did not include the later
background-supervision or background-score repairs. Its fit-start arm was
the incumbent coefficients with joint operation/argument scoring enabled;
it was not a separately trained background model.

The exact and equivalent counts are distinct:

| Arm | Exact | Equivalent | Exact regressions | Equivalent regressions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 488/500 | 488/500 | 0 | 0 |
| Fit start | 474/500 | 474/500 | 21 | 21 |
| Refit | 386/500 | 390/500 | 102 | 98 |

Regressions in this table are paired against the incumbent. The fit-start
arm gained seven cases. The refit gained none. The third training round
decoded all 764 training examples equivalently. This was a generalization
failure despite satisfied retained training constraints.

## Component interventions

Select 12 exposed validation regressions against fit-start by geometry
round-robin and source hash. Fit no labels. Restore one component at a time,
or remove the interaction block introduced during fitting, and decode again.

| Intervention | Equivalent out of 12 |
| --- | ---: |
| Fit start | 12 |
| Refit | 0 |
| Restore operation pointer | 6 |
| Restore operation classifier | 1 |
| Restore definition relation | 4 |
| Restore argument heads | 0 |
| Remove newly added boundary interaction | 5 |
| Restore pointer and relation | 9 |

The selection deliberately concerns known regressions. These counts locate
causal contributions on those examples and are not an unbiased accuracy or
transfer estimate. The interventions do not recover every error.

The incumbent had an additive operation pointer. Graph fitting silently
introduced 15,360 interaction parameters. Ordinary refitting now preserves
the declared parameter geometry. Explicit paired-boundary experiments remain
available. The regression test failed before the repair and passes after it.

Artifacts:

- `~/.aura/rlc-evidence/semantic-working-face-attribution-20260919.json`
- `~/.aura/rlc-evidence/semantic-working-face-attribution-pair-20260919.json`
- `/tmp/aura-g03-attribution-20260919.log`
- `/tmp/aura-g03-attribution-pair-20260919.log`

The minimum-change correction design follows in
[the generalization repair specification](../G_GENERALIZATION_REPAIR_DESIGN.md).
No candidate is promoted and G03 remains open.
