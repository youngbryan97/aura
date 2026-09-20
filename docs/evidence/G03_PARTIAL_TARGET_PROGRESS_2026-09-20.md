# An unreachable target does not stop independent learning

The minimum-change proposal required every active comparison to reach the full
requested margin. A constant positive comparison below that target made the
proposal infeasible, even when an independent wrong comparison was repairable.

The reproduction has margins 0.05 and x-0.5, initially x=0, with a target of
0.1. No x can change the first margin. Setting x=0.6 repairs the second without
changing the first. Before this repair, the fitter returned
`local_margin_projection_unverified` with no accepted update.

After an unverified affine proposal, the fitter now searches the existing
loss-descent direction projected against retained faces. The ordinary
float32 replay, strict retained floors and lower-loss acceptance still apply.
The failed projection remains an unverified diagnostic; accepted descent is
identified separately and receives no minimum-change certificate.

The reproduction reaches margins [0.05, 0.10000002384185791]. Loss falls from
0.18125 to 0.00125, and neither comparison is wrong. The unreachable 0.1 target
remains unresolved. The receipt does not claim all constraints are satisfied.
A conflicting objective test confirms that retained correctness cannot be
traded for progress elsewhere.

Seventy-three tests pass across minimum-change fitting, retained graph
constraints, durable checkpoints and bilinear geometry. This closes a
constructed optimizer stall. It does not establish a full-source or transfer
gain, and G03 remains open.
