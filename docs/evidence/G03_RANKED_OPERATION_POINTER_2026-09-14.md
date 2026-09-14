# G03: reject the source-grouped operation ranker

The paired operation pointer was refit with the existing pairwise logistic
optimizer. Each positive operation span competed with negative spans from the
same source utterance. The exported pointer kept the existing representation,
operation classifier and other heads. The old binary objective remains the
default; this objective is opt-in and reproducible.

The fit used 728 training examples, 1,648 positive groups and 15,488 negative
pairs. Its loss fell from 0.1833127622 to 0.0036618193 in 58 iterations. The
500 exposed development examples calibrated the length penalty; they did not
fit the pointer weights. This distinction does not make them fresh test data.

The evaluation was interrupted after two family summaries showed a large
regression:

| Exposed family | Immediate parent | Ranked pointer |
| --- | ---: | ---: |
| Arithmetic | 117/128 | 26/128 |
| Cataphoric | 41/48 | 6/48 |

Those counts come from interim evaluator output. The process was stopped with
SIGINT while loading the next feature bundle and exited 130. It did not write
its end-of-run, per-row report. The other 324 tasks were not measured by this
run, and there is no 500-task score or paired regression count to report.

The [receipt](../../artifacts/rlc/semantic_ranked_operation_dev_20260914/receipt.json)
retains that evidence limitation, the candidate hash, fit provenance, source
hashes and private candidate location. The candidate has no serving authority
and is not promoted. Twenty-one focused tests cover the opt-in fitting path,
source grouping, invalid options and legacy replay. Lower training loss did
not repair development generalization. G03 remains open.
