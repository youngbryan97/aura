# G07 preregistration integrity

The existing preregistration module could confirm a result when registered
parameters were absent from the run report. Its frozen dataclass retained
mutable nested dictionaries and lists, positive infinity could meet a metric
threshold, and loading a document without its recorded hash was accepted.
Declared experiment arms were not compared with executed conditions.

The repaired module snapshots and freezes finite JSON configuration, requires
the recorded content hash on load, and treats absent settings or changed arm
inventories as exploratory. Invalid measurements are unmeasured. A missing
control cannot confirm the registered comparison. The existing steering runner
now supplies conditions with actual outputs rather than copying declared arms.
Its historical plan and results are unchanged. Its incomplete parameter report
now exposes missing settings instead of silently confirming them.

Validation: 56 focused tests passed across preregistration, adversarial evidence
integrity, and historical encoder-width provenance. The registered invariant
`evaluation.preregistered_evidence` exercises complete, missing-configuration,
missing-arm, and nonfinite cases. Existing valid plan hashes still round-trip.

This repairs the analysis primitive. It does not establish when data were
exposed, independently measure the supplied metrics, or demonstrate statistical
power. Publication before task reveal, measured arm coverage, fresh task
construction, and a powered replication remain G07 obligations. The running
six-task coding diagnostic was not preregistered and remains development data.
