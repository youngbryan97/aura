# The endogenous language pathway

**What this builds.** A trained, causal path from Aura's own cognitive state
into the transformer's output distribution:

    z_Aura → named cognitive channels → Δlogits over the model vocabulary → language

The transformer stays the language and reasoning organ. Aura supplies what is
being expressed. Nothing here is a second decoder stack, and nothing here
writes text into a prompt to steer the model.

**What it replaces.** `SubstrateTokenGenerator` projects a 64-dimensional
substrate through an untrained random matrix onto 32 proto words. That output
is a state fingerprint. It is gated away from users, correctly, and it is not
a readout. This pathway is the readout.

---

## The parts

| Module | What it is |
| :-- | :-- |
| `core/brain/llm/endogenous_state.py` | z_Aura: 74 named dimensions across nine channels, each with a presence bit |
| `core/brain/llm/cognitive_code.py` | the symbolic readout, taken before generation, never shown to anyone |
| `core/brain/llm/endogenous_vocab_head.py` | the trained map from state to a bounded bias over the model's vocabulary |
| `core/brain/llm/endogenous_decode.py` | the decode-loop processor, and the pathway's health |
| `core/brain/llm/endogenous_pair_recorder.py` | the corpus: her state paired with her words, at the turn boundary |
| `core/brain/llm/endogenous_readout_training.py` | the fit, its baselines, its nulls, and the verdict they support |
| `core/brain/llm/endogenous_intervention.py` | `do()` on one named dimension, and the matched nulls that make it a measurement |
| `core/brain/llm/substrate_only_channel.py` | can something the prompt never said still reach the words |
| `core/brain/llm/endogenous_anticipation.py` | does the state at one turn carry anything about the next |
| `core/brain/llm/endogenous_absorption.py` | the return arrow, and the state's veto over a proposal |
| `core/brain/llm/endogenous_telemetry.py` | five declared channels |
| `core/brain/llm/endogenous_invariants.py` | three standing invariants |

## The work, item by item

A row is closed only when its check runs.

| # | Item | Where | Status |
| :-- | :-- | :-- | :-- |
| 1 | z_Aura is more than an affect summary | `endogenous_state.py` | closed — 74 named dimensions across affect, substrate, goal, memory, uncertainty, self-state, attention, recurrence, temporal orientation. An unreachable source reads absent, never zero |
| 2 | A single dimension can be intervened on | `EndogenousState.do` | closed — returns a copy, records the intervention, and the mark survives the process boundary so an experiment cannot be recorded as an observation |
| 3 | Structured native readout | `cognitive_code.py` | closed — nine lines from state, two from organs and marked as such, one learned head that abstains until fitted. Never user-presentable |
| 4 | Δlogits over the full model vocabulary | `endogenous_vocab_head.py` | closed — bound to a tokenizer fingerprint and a layout digest, refuses on either mismatch |
| 5 | `L_final = L_LLM + α·L_Aura` inside generation | worker logits processor | closed — bias lands only on tokens the model already finds plausible |
| 6 | Training pairs come from live traffic | `endogenous_pair_recorder.py` | closed — recorded at the turn boundary, tokenized at fit time. Zero turns recorded on this machine so far |
| 7 | The trained head must beat what it replaces | `endogenous_readout_training.py` | closed — unigram baseline, the exact random projection, permutation of the held-out correspondence, and refits on permuted states |
| 8 | Style adapter and content-bearing are not the same result | same | closed — scored apart, each against the null for that same quantity |
| 9 | Interventions produce measurable downstream change | `endogenous_intervention.py` | closed — every effect carries its matched nulls, and `exceeds_null` is False when there are none |
| 10 | Information held only in z | `substrate_only_channel.py` | closed as a harness — the live experiment needs the resident model and has not been run |
| 11 | LLM output is absorbed into state | `endogenous_absorption.py` | closed — through a new additive input path, off by default because it changes live dynamics |
| 12 | The substrate can disagree with a proposal | same | closed — the same proposal is rejected under low confidence and a held goal, accepted under high confidence with none |
| 13 | A model swap preserves z | `EndogenousVocabHead.rebind` | closed — the state survives, the head is marked untrained and says why |
| 14 | The random head never reaches a user | existing gate | unchanged and still holds; the proto generator now names its successor, and a user turn no longer computes one to discard |
| 15 | Does the readout anticipate the next turn | `endogenous_anticipation.py` | closed — ridge from z at turn t to a measured property of turn t+1, held out from the END of the sequence, against a permutation null. A corpus out of recording order is refused |

## What has been measured, and what has not

**Measured.** The fitting procedure, against three corpora built with a known
answer. The anticipation test, against a corpus where the next reply's length
is decided by one named dimension of the current state (held-out correlation
0.999, p = 0.003) and one where it is not (no anticipation). No state-token relationship reports `no_signal`. A register effect
reports `style_prior`. A rare-word identity effect reports `content_bearing`.
The whole chain runs end to end with a head fitted to a constructed corpus:
fit, save, load, job, processor, biased logits, and the dimension the corpus
was built around leads its own nulls in the influence map.

Four measurement faults were found by running that battery and are fixed.
Comparing a rare-token gain against an overall null reported a register shift
as propositional content. A per-parameter optimiser moved fourteen thousand
zero-valued coefficients as fast as the few carrying signal, and no regime
left epoch zero. A free bias sitting at its own optimum random-walked away
from it. Requiring overall significance before a rare-token claim reported a
real content effect as no signal.

**Not measured.** Anything about Aura. No corpus of live turns has been
fitted, no head exists on this machine, and no generation has been biased by
the substrate. The pathway's live effect is unmeasured, which is not the same
as small. `runtime_health_report()["integrity"]["endogenous_language"]` says
which of those is true at any moment, and separates a head that will not
attach from no head at all.

## What already owns the rest

The architecture this pathway comes from also wants the transformer's
expensive results to be kept, so the same problem costs less next time. That
is not built here because it exists: `core/knowledge/compiled_understanding.py`
digests material through the deepest lane available, content-addresses the
digest, counts its reuse, and exports the heavily-reused ones as consolidation
evidence for the governed learning lanes.

It is deliberately not a channel of z_Aura. Reading its statistics costs a
database query, and every probe here runs on the request path and is
memory-only. A state assembled by opening a database would put a query in
front of every generation.

## What this cannot do, said before anyone claims otherwise

A linear map from 74 named state dimensions to a vocabulary cannot encode
sentence content. It can carry register, hedging, directness, stance, and
state-dependent word preference. The trainer reports which of those it
achieved on held-out data, and the verdict says `style_prior` when that is all
it earned. A reader recovering one binary contrast from a state this wide has
recovered one bit, not a memory.

## Running it

```bash
python tools/train_endogenous_readout.py --tokenizer /path/to/resident/model
python tools/endogenous_causal_battery.py
```

The trainer refuses below 60 recorded turns and writes no head when the
verdict is `no_signal`. The battery runs offline; the text arm needs the
resident model and lives in
`core.brain.llm.endogenous_intervention.causal_text_experiment`, which drives
a client the runtime already holds rather than loading a second one.

## Related

`docs/LEARNED_LANGUAGE_INTERPRETATION_TODO.md` covers the other direction —
learning what an incoming message means, rather than what an outgoing one
should be shaped by.
