# Full-cohort graph relation refit

The one-round refit ran from source `3931c233f` in a frozen worktree. It
processed all 728 source-training examples and retained 682 complete graph
contrasts with independently executed distinguishing inputs. The other 46
charts had a proved uniform-reduction equivalence class. No source validation
or test targets entered fitting.

After 100 updates, regularized training loss fell from 0.19598429661721678
to 0.04138688030537177. Only the existing relation projections changed.

The unchanged 500-row paired development comparison completed:

| Candidate | Exact programs | Equivalent programs | Exact gains | Exact regressions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 436 | 456 | 0 | 0 |
| Relation refit | 465 | 471 | 30 | 1 |

Equivalent-program outcomes gained 16 and regressed one. The existing selector
retains the incumbent. The candidate is not promoted; G03 remains open.
These are program metrics, not answer accuracy. The equivalence rule is the
unchanged structural rule; the new polynomial training proof did not alter it.

## Reproduction records

- Candidate receipt: `b1a6d15b8fd9d9980876abef39f3ccf0172086458e4abc2bc968301f8362e8af`.
- Comparison receipt: `8c4b6f5457a7ef26871d1baf0fdd484473d52fbf49f91291e7f5166108ef7693`.
- Supervisor receipt: `70e89ccd5d5e4646e7198c6ede7944a3d2b834a6846375b7288936522a88ce63`.
- Raw candidate and validation: `~/.aura/rlc-evidence/semantic-graph-relations-dev-20260915/`.
- Raw supervisor and progress: `~/.aura/rlc-evidence/semantic-graph-relations-supervisor-20260915/`.
- Repository comparison: `artifacts/rlc/semantic_graph_factor_dev_20260915/relation_validation.json`.

The supervised process completed in 1,037.13 seconds, exit zero, with an empty
process group and no restart. No cortex was loaded. No runtime activation,
fresh transfer, broad reasoning gain or frontier comparison is established.
