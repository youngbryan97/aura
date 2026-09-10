# Battery runs

One directory per run, never overwritten. Each holds the recording and its
manifest, the report with every criterion and the evidence behind it, one line
per intervention trial in `intervention_arms.jsonl`, every pair tested in
`edges.csv`, and the null draws, lesion arms and campaign block beside them.

`run_000` is the last run made before runs were numbered. It is kept because a
campaign keeps its failures, and it is here rather than at the root so that
nothing reads it by accident when it asks for the newest run.

A run says which campaign it belongs to. Two runs sharing a fingerprint were
measured the same way; two that differ belong to different campaigns however
similar the command looked. See `docs/SUBJECT_CORE.md`.
