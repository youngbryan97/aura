# G03: preserve accepted updates and share repeated evidence work

The retained-constraint optimizer now writes each accepted update through the
file-write gateway before reporting progress. Resume binds the original
parameters, every numerical contrast, algorithm source, owner, and all fit
options. It verifies stored margins and protected floors before continuing.
Completed fits return their saved result without repeating updates.

The joint fitter gives each round a checkpoint, enabled by default by the
retained-constraint CLI. Resume still reconstructs that round's mined evidence
before checking its identity. Mining is not checkpointed by this change.

Shared operation evidence is scored in bounded chunks. Weighted gradients and
directional derivatives no longer allocate one full parameter gradient per
contrast. The implementation retains log(mean(softmax(view))), its probability
clamp, the exact per-witness acceptance check, and float32 export checks. It
does not reduce the number of retained constraints or change model admission.

## Measurements

70 focused tests passed, including numerical derivatives, chunk-size parity,
clamped probabilities, reference-optimizer comparison, interrupted/resumed
equivalence, changed-identity rejection, and production round wiring.

A CPU microbenchmark used 1,536 operation constraints sharing 256 banks, two
1,024-wide views, and seven labels. Values plus weighted gradients took 0.3131
seconds in the reference implementation and 0.00608 seconds batched. Maximum
margin difference was 2.00e-15; gradient difference was 8.33e-16. This is a
kernel measurement, not a campaign speed claim.

The effect inventory adds the reviewed checkpoint write through the existing
gateway. It adds no raw filesystem primitive and grants no serving authority.

The saved aligned candidate is being evaluated separately in the frozen
`codex-g03-aligned-evaluation` checkout. That run does not use these optimizer
changes. Its candidate receipt remains
`912618ac158b4d32addaf4a3b25376910a9abe69f964c63762f2169c7bd8c91f`.
No G03 closure or candidate promotion is implied.
