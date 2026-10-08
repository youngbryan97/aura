# Input composition and retained state, 8 October 2026

These are offline repairs for the Flash demonstration. The latest completed
Claude demonstration finished none of its three games successfully. The live
model remains unavailable while benchmark work owns it. This record does not
qualify the demonstration for recording.

## What changed

Pointer steering and an independent trigger now run through separate physical
inputs. A trigger precedes steering so its recorded origin belongs to the
observed controlled object. The same observation path refreshes position for
pointer and keyboard identities. Mouse triggers click the observed position,
including an off-center position at a new stretch.

An input receipt records dispatch and completion. An effect captured during
that interval can contribute evidence. Repeated frames and one input's burst
cannot count as independent trials. Effects compatible with several receipts
remain unattributed. Distinct successful inputs determine qualification;
failed fresh trials can withdraw a retained qualification.

After two independent effects, their observed delay bounds the attribution
window and the next trigger's deadline. A new epoch measures that delay again.
This permits rapid inputs where timing supports them and separates trials
where it does not. Lifecycle clicks have no firing role. Color instructions
are rebound when observed pointer control changes their physical meaning.

A counter reset releases pending click credit while preserving learned
locations. Bounded world memory declares its indexed tables, dependent maps,
scalar references and shorten-able histories. Compaction remaps those records
together and preserves their structural vectors. Invalid relation closures
are excluded with an explicit degradation; unrelated records can survive.

No change names a game, museum title, score threshold or game-specific control.

## Evidence

The focused suite passed **136 tests**. It includes actual pointer identity
discovery and the real aiming calculation, mouse and keyboard emissions,
large steering moves, effects captured during and after input delivery,
overlapping receipts, repeated observations, bursts, sustained rapid inputs,
changed effect delays, failed fresh trials, successive off-center stretches,
instruction role changes and bounded persistence round trips.

Independent review found defects in the first candidate. The corrected
candidate was approved for its synthetic mechanics. Live qualification remains
separate.

The smoke suite passed **165 tests, one skipped** before the final timing
regressions. Final smoke and gate results are appended below when available.
Lint, compilation, layering and writing passed. Governance lint reports
existing ownership debt. Comparing the effect inventory of every changed
production file with its source at `1b83b2bc7` found **zero changed buckets**;
the baseline was not expanded.

The indexed-state repair prevents new positional corruption. It cannot detect
every legacy record whose table already shifted while all remaining references
are still in range, nor certify a legacy truncated vector without a shape
receipt. The synthetic input tests cannot establish browser latency, visible
gameplay, task completion or presentation quality.

## Final gates

At the final candidate, smoke passed **165 tests, one skipped**; lint,
compilation, layering and writing passed. The focused suite passed
**136 tests**. No live model was loaded and no live Aura process was started.
