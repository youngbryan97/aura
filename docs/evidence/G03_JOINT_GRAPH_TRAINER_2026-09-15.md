# Joint operation and relation training

The trainer now updates the existing multiview operation classifier and
directional relation projections under one complete-graph loss. It does not
introduce a second language substrate or a runtime answer lookup.

Source-training prompts go through `model.decode` without annotations. A
different interpretation becomes a negative only after the existing universal
floor executes a distinguishing input. Structural or proved polynomial
equivalence is retained; finite probes that do not distinguish programs remain
unknown. Missing decodes and incomplete solves are recorded separately.

The positive program is supplied by source annotation. Its mention and
definition choices are optimized using runtime chart construction. The
negative graph is reconstructed in source order and its argument score must
replay the actual decoder score before the pair enters training.

Operation gradients use `log(mean(softmax(view)))`, which is the shipped
ensemble rule. Relation gradients include the competing-definition maximum.
Both contributions enter the same logistic graph margin. Base relation
evidence, pointers, feature geometry and operation vocabulary stay unchanged.
Each round remakes runtime predictions and retains prior witnessed pairs.

The CLI accepts a separate starting candidate while retaining the original
comparison incumbent. Source cohort, grounding and feature identities must
match. Enabling joint operation/argument selection is explicit, becomes part
of the candidate identity, and does not alter the comparison incumbent.

## Checks and limits

- Central differences check every parameter of both operation views and both
  relation projections on a small mixed-graph fixture.
- The operation score replays the runtime probability mixture, including its
  probability floor.
- A deliberately damaged operation classifier produces witnessed negatives
  through the real decoder and changes under joint training.
- Runtime graph score replay, retained-round pairs, source/test separation,
  serialization and parameter boundaries are tested.
- Focused combined semantic run: 122 passed. Additional identity, source and
  learner run: 43 passed.

These checks establish an executable trainer, not generalization. Full-cohort
joint fitting, unchanged development comparison, fresh transfer and live
qualification remain unmeasured. G03 and S03 remain open. Scope and reference
features not represented by these two heads have not been newly learned.
