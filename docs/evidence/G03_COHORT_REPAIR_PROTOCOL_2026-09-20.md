# Cohort diagnosis before further training

The conditional retention pilot fits five source examples, with auxiliary
retention constraints from all 764 source-training examples. After two updates
it decodes all five correctly. On the 25 previously acquired source errors and
controls, it reaches 24/25 versus the parent's 16/25, without further training.
On 100 exposed development-validation cases it reaches 98/100 versus 99/100
for the parent: one regression and no gains. It is not promoted.

Artifacts are under `~/.aura/rlc-evidence/semantic-conditional-retention-pilot-20260920/`.
Completed validation receipt: `bbb30c05b6abd36d1c71c37473bb4637e48b653f52314553e4a91346c0d2b04a`.
Expanded source receipt: `900dd0fab8f7baf1acccf47888499c45c77cfa5546e4313d35108b5db5c3ab80`.
The initial validation process disappeared after 85 rows. Its exit cause was
not recovered. A separate evaluator verified the frozen implementation and
completed the remaining rows from the preserved candidate; no training was
repeated. This is development evidence, not fresh transfer.

## Next decision rule

Stop adding scoring variants in response to individual examples. Freeze the
candidate and classify every development observation before changing training:

1. Verify input grounding.
2. For semantic failures, generate alternatives without target annotations.
3. Compare those alternatives with the target only after generation.
4. Separate witnessed selection errors, incomplete search, proven bounded
   unreachability, and unresolved verification.
5. Keep execution and public emission unmeasured unless separately observed.

`tools/audit_semantic_cohort.py` applies the existing candidate-bank diagnosis
to the full parent-bound train/validation cohort. Per-source immutable rows
bind the model, observations, implementation, and search allowances. Resume
must match that identity. Interrupted work leaves rows, not a complete report.
Successful semantic comparisons do not claim successful public answers.

The audit does not prove that bounded search covers arbitrary programs. A
search allowance reached without finding the target is incomplete evidence,
not proof of insufficient model capacity. The next repair must address a
measured boundary and pass the whole development matrix, not just its failing
subset. Candidate selection precedes fresh G04/G07 transfer. G09 broad tasks
and G12 external comparisons remain separate obligations.

No G-ledger checkbox closes from this protocol or these partial-cohort results.
