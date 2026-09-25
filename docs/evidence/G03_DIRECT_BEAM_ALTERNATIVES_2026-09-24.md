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

The bounded source-fold pilot uses the existing 24-row held construction
population and the frozen fold-0 direct-decoder checkpoint. A first invocation
with an unrelated base candidate was refused for `bank_model`; the completed
run used the bank's bound literal-identity transducer. At width four, the
beam generated 52 programs absent from the retained bank, but no newly exact
target. It reached 18/24 exact targets and 20/24 programs equivalent on the
finite counterfactual comparison; its top program was exact on 18/24 and
finitely equivalent on 19/24. Direct likelihood selected a correct program
from the existing bank on 23/24; the bank's observed oracle reach is 24/24.
The new beam therefore did not improve this pilot's reach or selection.

The historical exposed-gap comparison of the *scaled* direct checkpoint
selected 7/23 correct programs on incumbent misses; the beam pilot used the
unscaled checkpoint. That development population was
selected for incumbent failure and is not a net-gain estimate, but it prevents
the 23/24 source pilot from being treated as transfer. The next mechanism to
test is source-only hard graph evidence and independent selection, not a wider
beam by default. The pilot receipt is
`~/.aura/rlc-evidence/semantic-direct-beam-fold0-20260924/compare.json`.

Focused tests cover grammar validity, score replay, bounded search, and
post-generation label separation; 21 direct-decoder tests pass. The repository
smoke suite passes 164 tests with one skip, and lint, compile, governance lint,
and layering pass. No serving promotion or G03 closure follows.
