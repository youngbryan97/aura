# G10: measure steering on completed public answers

The durable 27B campaign retained 68 samples: 24 baseline, 24 baseline
replicate, and 20 treatment. Its authenticated supervisor stopped the child
after a measurement defect was found. The supervisor recorded signal 15,
return code -15, and child PID 11446; neither child nor supervisor remained.
The original progress and frozen checkout were preserved.

The runner scored the raw `mlx_lm.generate` string. That string included
native private reasoning. Six of the 24 baseline samples had not closed the
private channel within 256 tokens. Affect words in private reasoning could
therefore raise the score of an answer the user never received.

The revised runner uses MLX streaming metadata and Aura's existing native
channel parser. It retains public text, token counts, stop reason, boundary
state, and hashes of the public text, private text, token IDs, and prompt.
It does not retain private text in new results. Truncated and empty public
samples remain in the denominator. A new qualification requires complete
public generation across the retained conditions as well as the existing
causal and no-regression checks. This establishes channel completion, not
independent semantic correctness of every answer.

Durable progress now includes those per-sample receipts. The independent
verifier binds the exact result payload and recomputes its replay before
issuing causal evidence. Old raw results can still be inspected but cannot
qualify under the new public-sample protocol. No historical score is rewritten.

The runner also has an explicit calibration-only mode: baseline and treatment
on the existing six exposed development prompts, with no qualification path.
Its purpose is to measure completion and allocation needs before spending
another full campaign on incomplete output. Calibration observations are not
fresh held-out generalization evidence.

Checks completed before this record: 32 focused tests passed, including real
MLX stream termination metadata, interrupted/resumed campaign equivalence,
calibration isolation, receipt tampering, and private-channel exclusion.
The preceding smoke run had 163 passed, one skipped, and the existing
`resident_manifest_drift` serving alarm, with zero drifted bound source files.
G10 and G11 are not closed by these engineering checks.
