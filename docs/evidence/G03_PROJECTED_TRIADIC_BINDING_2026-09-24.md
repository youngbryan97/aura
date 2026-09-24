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

## Score calibration on complete programs

The opt-in refit now separates projection selection, binding-head fit, and
complete-program score calibration. The projection's epoch is selected on a
construction-disjoint subset of the fit sources. A deterministic,
construction-spread cohort is used to select one of six nonnegative binding
scales (including zero) by whole-program exactness, then answer exactness and
regressions. Zero must reproduce every incumbent evaluation row exactly. The
score lesion removes the bias as well as the weights. The calibrated factor
uses the ordinary argument-graph decoder; this is not a separate answer path.

On the original full-source parent, all 102 calibration programs were already
correct. Scales 0 through 1/4 tied, while 1 regressed 16 programs and 4
regressed 82. The selected zero coefficient makes this candidate inert. The
receipt is at
`~/.aura/rlc-evidence/semantic-triadic-v4-calibrated-fold1-20260924/report.json`.
The parent had trained on these source constructions, so this experiment
cannot establish the value of triadic evidence on genuinely new programs.

A second run attached the same mechanism to the existing fold-1 cross-fit
proposer, whose semantic coefficients excluded the held constructions. Its
projection was selected on a separate subset of its 408 fit sources, heads
were trained on those 408, and the score scale was selected on 24 additional
source-validation examples, separate from its 102 proposer calibration
examples. At scale 1/16, complete-program calibration rose from 20/24 to
21/24 with one gain and no regression. On a source-hash-ordered
40-source slice of the held fold (offset 24), exact programs were 32/40 for
the parent, 33/40 for the candidate, and 32/40 for the triadic lesion. The
one additional correct program was an arithmetic construction. Receipts:

- `~/.aura/rlc-evidence/semantic-triadic-v4-oof-fold1-20260924/report.json`
- `~/.aura/rlc-evidence/semantic-triadic-v4-oof-fold1-20260924/pair40.json`

The independent fold-2 cross-fit proposer selected scale zero on the same
24-source calibration design: incumbent 18/24 exact, no gains at scales up
to 1/4, and 7 program regressions at scale 1. Its diagnostic gold-span
binding reached 925/1,086, but it supplied no score-calibration gain. Receipt:
`~/.aura/rlc-evidence/semantic-triadic-v4-oof-fold2-20260924/report.json`.

The source feature admission and input-grounding contract were inherited from
the full-source parent in both cross-fit probes. The fold-1 paired receipt and
fold-2 source receipt explicitly say `end_to_end_source_disjoint=false`;
the retained semantic proposer
coefficients were source-disjoint, but the complete system was not. The
fold-1 gain is development evidence for a nonzero causal score contribution,
not evidence of broad or family transfer. A new source-disjoint grounding
contract and wider frozen comparison are required before promotion. The
102-source six-scale decode took roughly 17 minutes on this host; the
construction-spread 24-source screen is only a development filter, not a
replacement for full validation.
