# G03: target-blind direct-program alternatives

The experimental direct decoder previously emitted one greedy program. Its
learned source-conditioned likelihood now supports a bounded typed beam over
complete programs. The beam reuses the same operation grammar, typed register
rules, token features, and learned weights as the direct decoder. It does not
read an answer, target program, family label, or held-out outcome while
proposing candidates. Every returned score replays against teacher-forced
complete-program likelihood.

The existing source-fold method comparison has an opt-in `--direct-beam-width`
arm. It freezes the candidate set before reading source truth, then records
exact reach, newly reached correct programs outside the retained bank, and
top-choice exactness separately. It binds the changed decoder and comparison
implementations by source digest. The default comparison and serving route
are unchanged.

This is a reachability mechanism, not a measured G03 gain. Focused tests
cover grammar validity, score replay, bounded search, and post-generation
label separation; 21 direct-decoder tests pass. The repository smoke suite
passes 164 tests with one skip, and lint, compile, governance lint, and
layering pass. No frozen held-out source-bank beam run has been completed,
so G03 remains open. In particular, more reachable programs do not establish
that a source-conditioned selector chooses the right one.
