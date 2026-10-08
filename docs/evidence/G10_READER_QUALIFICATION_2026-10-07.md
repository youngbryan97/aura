# G10: the program reader qualified on her current model, as her runtime runs it

G04's candidate v17 was evaluated offline on hidden states a standalone
process read from her 27B with nothing attached. Her runtime reads a request
through its resident worker, where her affective steering and latent bridge
are attached. This qualifies the reader on the current model through that
path: geometry, identity, conversion, steering, a lesion that shows the
check can fail, rollback, and readings against the offline evaluation.

## The defect the measurement found first

Her worker encoded a request for the reader with the forward her generations
use, so the steering hooks added their nudge to the last token's state at
all sixteen hooked layers. The hooks steer the token a generation is about to
produce, and an encode produces none. The latent bridge also counted the pass
as her own activity. The worker's representation basis carries the steering
alpha, so with any steering at all the runtime refused a reader fitted on
unsteered states, and the alpha changes as her affect does.

`core/runtime/observation_pass.py` (356c95eb2) stands steering down and keeps
the readouts from recording for the thread inside it. The encode runs inside
it and its receipt names the steering-inactive basis that produced the
states. A generation on another thread keeps its steering.

## Measurement

`tools/qualify_g10_reader.py`, run from `f73e6554d` on all 416 requests whose
states G04 and G05 recorded (48 compositions, 120 lists, 248 transfer
requests). One model load, in a state root of its own, reading her cortex
authority key and nothing else of hers. Results:
`~/.aura/rlc-evidence/g10-reader-qualification-v17-v2-20261007/qualification.json`.

| Check | Result |
| --- | --- |
| Identity | descriptor `52d313c2…`, the active cortex; her signed steering generation `0d192ee4…`, basis `06580ec4…` |
| Geometry | 64 layers declared and loaded; 16 hooks on layers 3 to 63; every steering vector 5,120 wide, the declared width; every state three channels of 5,120 |
| Conversion | the worker's own encode returns the recorded token ids and bit-identical states for 416 of 416 |
| Steering | attached the worker's way and settled at the fusion probe's high and low states: 416 of 416 encodes bit-identical at each, steering injecting, no latent readouts recorded, her affect state unchanged |
| Lesion | the same encode without the observation pass: 416 of 416 states differ |
| Rollback | after the engine detaches, greedy continuations of the probe's 8 prompts over 24 steps and all 416 encodes are identical to never-attached |
| Readings | through the runtime's own observation path (`execute_compositional_semantic_observation`, basis check included): 412 answered with the same program and answer as offline, 4 decoded and not executable on both paths, 416 of 416 agreeing |

The injection was at the hooks' ceiling, alpha 0.6, when each state was
settled; her own rule derates a hook to 0.1 when its affect state is more
than 120 s old, which it reached during each 416-request pass. Both are
covered.

The lesioned states changed none of the reader's 416 readings. The defect
showed as a refused reader, through the basis check, not as wrong answers.

## The reader's cost

Through the worker, a composition costs a median 0.57 s to read its states
and 0.78 s to read the program; a list 0.58 s and 0.16 s; a transfer request
0.21 s and 0.15 s. G06 counts these against the assisted answer.

## Comparing readings

A first full run reported 243 readings different with every answer equal.
The runtime numbers a request's inputs in the order they appear in the text,
the offline evaluation in the order the corpus declared them, and the
program hash numbers registers; and a program that cannot run was a read on
one path and a refusal on the other. The comparison now names each input by
its token span and compares outcomes (f73e6554d); checked first on the
stored states of all 416, then in the rerun above. The first run's rows are
kept beside it (`g10-reader-qualification-v17-20261007`).

## The package and its rollback

`tools/build_semantic_reader_package.py` copies the reader and the five
documents it rests on (G04's report and independent re-decode, G05's report
and independent check, this qualification) into
`artifacts/rlc/semantic-reader-27b-033feffa21a4/` and writes a v2 activation,
which the builder refuses unless G04's and G05's closure rules hold in both
their reports and their independent checks and every G10 predicate holds.
Shadow only: `serving_authority` false, her ordinary answer the immutable
incumbent.

The runtime's default package is now this one; its source seal is green
(`make qualification-seals`, which now checks the envelope the runtime
opens). Rollback is an operational activation, which is local state:
pointing it at the previous package returns the runtime to exactly what it
had, a package refused for source drift, and removing it returns to v17
with an identical status receipt (`tests/test_compositional_semantic_qualification.py`).
`build_semantic_reader_package.py activate` and `rollback` keep and restore
the previous activation byte for byte.

## What this does not show

* Serving: the package is shadow-only, its result observed beside her
  ordinary answer and never replacing it (G11 is a separate item).
* Behaviour on requests outside these 416.
* Anything about a reader other than v17, or a model other than `52d313c2…`.
