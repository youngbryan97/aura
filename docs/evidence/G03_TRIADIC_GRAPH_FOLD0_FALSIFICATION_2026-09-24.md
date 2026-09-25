# G03: graph-scale transfer fails on fold 0

The fold-1 graph refit was repeated without changing its method on a
separately trained fold-0 projected triadic candidate. Its 408 source fit
examples yielded 381 graph contrasts and 27 cases without an incorrect
graph. Source contrast loss fell from 2.5781 to 0.1888, but the shared
score-capacity kernel proved that the 381 constraints have no global
four-scale solution at the requested 0.1 margin. The learned role scale
reached its lower bound and the triadic multiplier fell to 0.321.

The 256-source held replay scored 88 exact programs for the unfitted
triadic candidate, 124 for the graph refit, and 171 for the refit's triadic
lesion. The cross-fit proposer's earlier report recorded 192 top joint
scores correct on this population. Removing the triadic factor improved
all major families, especially the two 64-source sequential arithmetic
constructions. A better source contrast objective did not transfer.

- `~/.aura/rlc-evidence/semantic-triadic-graph-fold0-20260924/triadic_report.json`
- `~/.aura/rlc-evidence/semantic-triadic-graph-fold0-20260924/report.json`
- `~/.aura/rlc-evidence/semantic-triadic-graph-fold0-20260924/pair256.json`

This refutes a universal global-factor-scale solution for G03. The factor
family is exactly insufficient on source contrasts and the held triadic
contribution is negative. Neither the fold-1 gain nor the fold-0 training
loss licenses promotion. Future work must add a source- and graph-conditioned
meaning signal with retained baselines, test it on disjoint constructions,
and keep the current failures as controls rather than optimize against them.
