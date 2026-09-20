# Joint restoration of retained graph margins

A constructed problem with twelve independent curved retention constraints
failed before learning: `no_retention_preserving_step_found`, twelve wrong
comparisons, no accepted update. The restoration loop repaired one violated
constraint per iteration and stopped after eight. Increasing the number of
independent constraints therefore changed whether a feasible update could run.

Each pair uses q=(1,0), d=(1,0), retains q.d >= 1, and seeks
q0*d1-q1*d0 >= 0.6. A feasible witness is
q=(sqrt(1+b*b),-b), d=(sqrt(1+b*b),b) for b=0.3. Repeating the
independent pair twelve times does not remove that feasible solution.

Restoration now proposes a joint correction for all currently violated faces
through the existing stored-margin solver. Every proposal is replayed against
all nonlinear constraints at float32 storage precision. The outer fit still
requires strict retained-floor satisfaction and lower loss. The eight-iteration
bound now limits curvature refinement, not the number of independent faces.

The reproduction now accepts its first update after one joint restoration.
After three updates all twelve wrong comparisons have positive margins and
every retained margin remains above 0.1. The new comparisons reach
0.09998661425678534, below the requested 0.1, so the three-step receipt correctly
remains `search_budget_exhausted`, not `retained_constraints_satisfied`.

Seventy-seven focused tests pass across graph constraints, batched replay,
minimum-change fitting and bilinear geometry. They include infeasible conflicts
that must not sacrifice retained evidence. This result closes the constructed
restoration defect, not G03 or general transfer. Full-source development needs
a new fit with its own implementation identity.
