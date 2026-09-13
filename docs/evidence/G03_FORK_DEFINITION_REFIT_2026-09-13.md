# G03 Fork Definition Refit

The fork/join generator names its branch results but omitted their definition
annotations. The opt-in annotated corpus now records those declarations;
the historical default, source text, split, IDs, and programs are unchanged.
Both original and source-ordered register variants have focused coverage.

The original 728 training examples were replayed with corrected fork/join
annotations. Source hashes and tokenizer outputs were checked against retained
features. No weave example entered fitting. The definition-pointer fit receipt
now binds the annotation targets and their origin, in addition to source IDs.

The refit took 158.31 seconds. It scored 46/48 exposed weave answers and 44/48
programs, versus 12/48 answers for the fallback-supervised definition refit.
Original source-validation answers were 372/500, with 92 gains and 66
regressions against the frozen parent's 346/500. The operation-only candidate
remains stronger at 377/500 source and 48/48 weave answers. This candidate is
rejected; no serving or fresh-transfer claim follows.

Evidence is under `artifacts/rlc/semantic_program_clean_pointer_dev_20260912/`:
`annotated-definition-refit-receipt.json`, `annotated-definition-weave.json`,
and `annotated-definition-full-source.json`. The candidate is retained at
`~/.aura/rlc-evidence/semantic-clean-pointer-dev-20260912/annotated-definition-transducer.json`,
SHA-256 `76ee22a3d50023a3bc64c515cd3e0bddcd1a3620566de3aaef0c8bf2f4de2bc2`.

The remaining selection problem is that a register's definition is fixed from
pointer evidence before its uses are scored. The next development candidate
will test consistent joint definition selection in the existing graph solver.
G03 remains open.
