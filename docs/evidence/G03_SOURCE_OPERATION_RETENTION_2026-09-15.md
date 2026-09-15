# Source operation retention

The contrast-only joint run at `f1400997d` reduced its retained-pair loss
while source errors increased. Its three pre-update source passes measured
22, 34 and 55 wrong interpretations out of 728, respectively. All other
rows were proved equivalent by the existing comparison. The final paired
500-row evaluation is separate and was still running when this note was made.

The optimizer previously received only incorrect graph pairs and parameter
regularization. Correct source operation labels supplied no likelihood term.
The new candidate objective adds source-label negative log likelihood through
the same multiview probability mixture used by runtime decode. It includes
every source training operation, normalizes within each example and geometry,
and leaves validation and test examples outside fitting.

This term is computed in batches. Its derivatives match both individual
runtime scores and central finite differences. The refit command exposes
`--source-operation-weight`; zero selects contrast-only fitting. Receipts
record the weight, number of source operations, and changed objective name.
No serving manifest or inference policy is changed by this training repair.

Checks: 27 focused joint/relation tests pass, including source-only admission,
stored-coefficient round trip, gradient checks, and learning an operation
absent from the error pairs. Smoke: 164 passed, one skipped. Ruff and compile
pass. These are implementation checks, not measured transfer gains.

On the full source bank, retention contains 1,648 operations across 728
training examples. Building its two 5,120-wide views took 0.69 seconds; one
objective/gradient evaluation took 0.032 seconds on this host. The initial
source operation loss was 0.02369, with finite gradients. This timing measures
only the retention term, not complete graph mining or paired evaluation.

Evidence for the failed training trajectory remains at
`~/.aura/rlc-evidence/semantic-joint-graphs-dev-20260915/candidate.json`, with
supervision at `semantic-joint-graphs-supervisor-20260915` in the same directory.
S03, S04 and the master G03 remain open until complete paired measurements
establish the required behavior. Source likelihood does not guarantee correct
reference selection, whole-program accuracy, or general transfer.
