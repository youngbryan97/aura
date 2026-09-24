# G03: projected triadic binding remains diagnostic

The two new answer exports supplied useful functional hypotheses: meaning
needs to survive counterfactual changes, and a representation should guide
action beyond the example that trained it. Their consciousness and uniqueness
claims do not follow from the mathematics in those exports. This experiment
tests source-role binding, not experience or general intelligence.

The earlier representation-only control scored 555/1,096 opposed held roles on
construction fold 0. A low-rank source-trained projection reused Aura's
existing relation tissue, then trained operation/mention/definition contrasts
without source geometry. The fit used 408 training sources and 100 disjoint
calibration sources; the 256 held sources supplied no labels to fitting. It
scored 1,037/1,096 gold-span roles. On fold 1, with an independent 408/102
fit/calibration split, it scored 989/1,086. The exact immutable receipts are:

- `~/.aura/rlc-evidence/semantic-triadic-gold-v4-lesions-fold0-20260924/report.json`
- `~/.aura/rlc-evidence/semantic-triadic-v4-fold1-20260924/report.json`

The fold-0 role lesions scored 761/1,096 without operation evidence and
189/1,096 without mention evidence. Removing definition evidence tied all
1,096 decisions. Fold 1 scored 931/1,086 without operation and 235/1,086
without mention. These lesions use the same fitted coefficients; they do not
show that the complete runtime chooses the right graph.

The fold-1 candidate is a validated, non-serving transducer artifact at
`~/.aura/rlc-evidence/semantic-triadic-v4-fold1-20260924/candidate.json`.
The paired decode of 24 held source identities scored 23 exact answers for
the parent, 21 for the candidate, and 23 for the candidate's triadic lesion.
There were two candidate-only regressions and no gains. The parent had seen
the corpus, so this is a development diagnosis, not a fully held model trial.
The pair receipt is
`~/.aura/rlc-evidence/semantic-triadic-v4-fold1-20260924/pair24.json`.

The new probe also applied the parent's runtime operation spans on 24 held
sources per fold while retaining gold mention and definition spans. Fold 0
kept 24 aligned sources and scored 96/100 opposed roles. Fold 1 kept 23
aligned sources and scored 90/97; one source was unresolved. This does not
measure autonomous mention selection. The parent had seen these sources.

The failure sits at evidence fusion. The binary binding head's raw logit was
added to graph scores trained under a different objective. The diagnostic
classifier's accuracy does not calibrate that addition or establish
independent information. Further work must train and validate its marginal
weight on source-disjoint complete-graph decisions, including runtime spans,
then measure a frozen candidate on new constructions. G03 stays open.
