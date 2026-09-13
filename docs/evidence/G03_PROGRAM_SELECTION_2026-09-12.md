# G03 autonomous program-level development selection

The compositional campaign now exposes `select_compositional_program_candidate`.
It runs each candidate's actual decoder on validation examples, then compares
the resulting programs with validation programs. Neither the gold program nor
the expected answer is supplied to decoding. No expected answers are computed.
Training and test examples are excluded; duplicate validation identities and
cross-split overlap are rejected.

The report retains paired correctness, gains and regressions for every
candidate. A candidate must add exact programs without regressing an incumbent
validation success to replace the incumbent in this development selection.
Ties retain the incumbent. This is not a universal statistical rule, independent
replication, compute-matched comparison, or runtime serving authority.

The exact-source refit CLI accepts `--validation-output` to run this comparison
after writing the immutable candidate. Candidate generation and evaluation
remain distinct artifacts; an unevaluated candidate is not a promoted model.

A real-feature replay on the 12 natural-alias source validation examples gave
both the parent and full-source refit 12/12 exact programs. The selector kept
the incumbent. Receipt:
`98665507c818ec00d603e3a88312b02ec74a31879ab1ae4b09ec568ba94eee37`.
This cohort is too easy to establish deeper transfer and is reported as such.

Six selector tests passed. The combined selector, source-identity, proposal,
qualification and shadow suites passed 32 tests. Smoke passed 164 tests with
one skip; touched files pass Ruff. G03 remains open.
