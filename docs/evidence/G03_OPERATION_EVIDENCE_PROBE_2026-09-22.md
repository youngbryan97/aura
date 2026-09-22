# Source-trained operation evidence probe

This is an experimental component, not a serving change or G03 closeout.
It uses resident token features, the existing typed floor operation vocabulary,
and gold operation spans from source-training examples only. Validation
targets are used only after candidate choice to grade the frozen result.

On source fold 0, fitting 60 source examples and ranking the frozen 24-row
candidate bank yields 19/24 correct versus 10/24 for its incumbent. It makes
14 gains and 5 regressions. On the 23 already-exposed validation misses, the
prototype selector chooses 3 correct programs. Fitting its prototypes on all
764 frozen source-training examples rather than the 84-row bank subset still
chooses 3/23. Two of these three are not chosen by the direct decoder or
mixed ranker; their three-method oracle union is 9/23, not a deployable
answer-blind selection result.

The construction suffixes are held-out phrasing variants, not program depths.
Every exposed target here has two instructions. More examples of the same
source variants did not fix transfer to the validation phrasing. Operation
similarity is a partial signal; it does not resolve argument roles, aliases,
or which conflicting executable candidate expresses the request.

Receipts:
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/operation-prototype-probe.json`
and
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/operation-prototype-full-source-probe.json`.
Both runs are development-only and have no serving authority. The exposed
failure set cannot establish fresh transfer or justify a promotion.
