# G03 Suggestions2 review

`/Users/bryan/Desktop/Suggestions2.pdf` was reviewed across all 26 pages.
It is advisory, not a result or a serving contract. The frozen paired
attribution is nine role-order and five later-operation mistakes among the
fourteen shared failures. Correct bank availability does not establish that
the resident features can identify the correct program.

## Adopted for measurement

- Pages 1-5, 17, 20-24: source-only, typed, execution-witnessed argument
  swaps and later-operation rivals are now an opt-in `bank_factors` training
  view in the existing candidate ranker. A source split identity check rejects
  training/validation overlap. Only training-fold source targets construct
  the new negatives. Held-out bank labels remain evaluation-only.
- Pages 4-5, 11, 19, 24: the role-order identifiability probe measures whether
  existing resident operation-span and context features distinguish orientation
  on source train, validation, and test. It does not invent role labels at
  inference or obtain serving authority.
- Pages 17, 19-20, 24: operation evidence now has an experimental step-local
  prototype mode, with source-trained prototypes and global fallback for
  unsupported positions. The probe reports ordinary and step-local results
  separately, rather than claiming that bagged evidence solves the second op.
- Pages 6-13, 17-18: typed signatures, bounded counterfactual execution,
  inquiry planning, and independently observed feedback were already present
  in the floor and portfolio. The full bounded inquiry set was connected in
  the prior checkpoint. No parallel signature table or verifier is added.
- Pages 15-16, 21-23, 26: promotion still requires frozen same-item evidence,
  no target leakage, source retention, and fresh transfer. Consensus stays a
  diagnostic; a fifth voter is not treated as semantic evidence.

## Not adopted as claims or mechanisms

- Pages 1, 15-16, 20: the report predicts that fixing nine plus five will
  deliver perfect validation. The fourteen cases are an exposed diagnostic
  slice, not a proof about the remaining 500 or fresh transfer.
- Pages 3, 6-8, 17-18, 25: executing a candidate produces its prediction,
  not an independent observation of intended meaning. A model re-ask, retrieved
  article, or self-generated trace cannot silently become a target label.
  Human, test, or reference observations need their own provenance.
- Pages 8-10, 19, 22, 25: fixed SRL phrases, hardcoded commutativity lists,
  exact exposed-case features, and a router trained or thresholded on the
  exposed failures would not establish general transfer. The existing typed
  floor and learned source representations are used instead.
- Pages 13, 19, 25: type compatibility excludes invalid programs but cannot
  decide between equally typed `sub` argument orders or rival arithmetic ops.
  Also, `at` has sequence/integer inputs; its arguments are not a type-correct
  swap. The observed inversion is in the later noncommutative step.

The implementation and probes are development-only. No new candidate is
promoted by this review. G03 and G04 remain open until frozen runs justify it.

## Measured follow-up

The source-only role probe used 572 training, 308 validation, and 308 test
examples with the same frozen resident features. A majority label selected
148/308 validation. Operation-span features selected 180/308, whole-request
features 215/308, and their concatenation 232/308. The concatenated view fit
572/572 training examples but selected 0/16 on each of three held-out
variant-4 constructions (cataphoric, reserved alias, role binding). The
features are not yet an invariant role representation.

The step-local prototype selected 20/24 on held-out source bank rows versus
19/24 for the global prototype. Both selected only 3/23 on the exposed
validation misses. A local source gain is not a transfer gain.

The `bank_factors` 8-row plumbing pilot selected 2/8. On the complete frozen
fold-0 source bank, three epochs selected 5/24 with 24/24 candidate
availability. The earlier mixed ranker selected 15/24 on that fold, and the
ordinary incumbent selected 10/24. The factor view is rejected for serving.
Its training probe selected 22/60 after 360 updates, so more fit to these
source negatives is not a justified shortcut to the exposed gap.

Receipts (outside the repo):
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/role-order-probe.json`,
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/operation-position-probe.json`,
and `~/.aura/rlc-evidence/semantic-bank-factors-fold0-20260922/fold-0.json`.
These are development observations, not G03 qualification.

## Representation and reach audit

An initial reference probe accidentally read the gold argument order through
the annotation order. It was rejected and replaced by a text-position-sorted
probe with a permutation regression test. On examples with explicit annotated
definitions, the corrected source-only relation features selected which
textual mention referred to the computed result on 156/156 train, 84/84
validation, and 84/84 test. Validation labels are balanced 41/43. This is a
gold-span upper bound: it does not establish that runtime finds the mentions
or definitions, nor that final program selection improves.

The production argument proposal function includes every annotated mention
in the held-out role-binding examples. With the current definition policy,
the annotated definition for input register 2 is absent in all 48 held-out
role-binding examples. The existing `source_neighborhood_v2` policy makes
all five annotated definitions reachable on those examples. It is not a
promotion: an earlier full-program trial of that policy lost composition
answers. With gold references, the existing relation scorer often chooses a
shorter span that overlaps the correct reserved definition; exact span
equality therefore must not be called a semantic error. The full decoder,
with both policies on the same cohort, remains the deciding boundary.

Receipts:
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/role-order-reference-v4.json`,
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/relation-reach-v2.json`,
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/relation-reach-bidir.json`,
and `~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/relation-overlap-bidir.json`.
