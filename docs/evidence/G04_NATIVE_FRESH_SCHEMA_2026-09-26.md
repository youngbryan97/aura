# Native selection on withheld three-step schemas

The fold-0 source-calibration-selected native suffix was frozen at step 256.
No fitting, checkpoint choice, or source calibration used this cohort. The
fresh corpus has 72 requests: three wordings in each of 24 constructions,
covering scalar, lookup, and count three-step linear programs. Every source
hash is disjoint from the native fit, calibration, and held-development IDs.
These schemas were absent from the source-fit two-step linear and fork/join
graphs. The model, tokenizer, program serialization, loss scope, and exact
execution primitives were unchanged.

For each request, the existing contrast generator used its annotation to
construct six type-correct candidate programs, including the target. Every
negative had an observed differing execution on counterfactual inputs. The
candidate order was shuffled; the target occupied positions 0 through 5 with
counts 14, 17, 10, 7, 11, and 13. The native scorer received the unchanged
request and those candidates, not their labels. A separate grading step
compared the score-selected program with the annotated target.

| Scorer | Correct / 72 |
| --- | ---: |
| Unfitted resident suffix | 55 |
| Source-fitted native suffix | 72 |

The fitted suffix had 17 paired improvements and no paired losses against its
unfitted control. It selected 24/24 in each of the scalar, lookup, and count
schema families. A second process verified the plan and report digests, every
row digest, candidate identity and order, finite score winner, grade, and
aggregate. The complete run took 530.633 seconds.

- Plan: `~/.aura/rlc-evidence/semantic-native-fresh-schema-fold0-20260926/plan.json`,
  SHA-256 `a4604c59c430060cc5cd4e3fe795246ce61478b6a83ed44d35c7c8293729999e`.
- Report: `~/.aura/rlc-evidence/semantic-native-fresh-schema-fold0-20260926/report.json`,
  receipt `6467d6c50aa637ed41fe5abdf7e899ebace366d4eda28283f8cbe9c8002475eb`.

This measures **candidate discrimination with an oracle-supplied inventory**.
The target was used to construct that inventory and was guaranteed present.
It does not show that Aura can propose the correct three-step program on an
unseen request, decode a public answer, preserve incumbent successes, or
transfer across all task families. The fold-2 known-construction replication
also regressed against the incumbent. G03 and G04 remain open; no serving or
fusion authority follows from this component result.

Focused tests: 25 passed. Smoke: 164 passed, one skipped. Lint, compile,
layering, and writing passed. Governance lint remains red on four unrelated
pre-existing ownership entries in `what_she_tried.py` and
`thinking_elsewhere.py`; none belongs to this evaluator.
