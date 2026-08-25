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
| 1 | z_Aura is more than an affect summary | `endogenous_state.py` | closed, and it took measuring to close — 74 named dimensions across nine channels, an unreachable source reading absent rather than zero. Fitting the first live corpus showed 47 of them had never fired: every probe was naming an organ or a key no writer publishes. Substrate, uncertainty, self-state, memory recency and curiosity are wired to organs that exist; three collinear pairs are three dimensions again. 48 of 74 present and 25 varying from a bare organ set, against 27 and 18 before, with a test holding that floor |
| 2 | A single dimension can be intervened on | `EndogenousState.do` | closed — returns a copy, records the intervention, and the mark survives the process boundary so an experiment cannot be recorded as an observation |
| 3 | Structured native readout | `cognitive_code.py` | closed — nine lines from state, two from organs and marked as such, one learned head that abstains until fitted. Never user-presentable |
| 4 | Δlogits over the full model vocabulary | `endogenous_vocab_head.py` | closed — bound to a tokenizer fingerprint and a layout digest, refuses on either mismatch |
| 5 | `L_final = L_LLM + α·L_Aura` inside generation | worker logits processor | closed — bias lands only on tokens the model already finds plausible |
| 6 | Training pairs come from live traffic | `endogenous_pair_recorder.py` | closed — recorded at the turn boundary, tokenized at fit time. 1,729 turns recorded on this machine as of 2026-08-25, across two models, fitted apart |
| 7 | The trained head must beat what it replaces | `endogenous_readout_training.py` | closed — unigram baseline, the exact random projection, permutation of the held-out correspondence, and refits on permuted states. The split groups by REPLY after the first live fit reported content from a memorised duplicate, and a forward-in-time split is available for the control a random one cannot give |
| 8 | Style adapter and content-bearing are not the same result | same | closed — scored apart, each against the null for that same quantity, and the report names which channels varied. A gain carried by affect alone is flagged as indistinguishable from a style adapter however significant it is |
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

## The first fit on Aura's own turns, 2026-08-25

The runtime recorded 1,729 turns of her state paired with her words. Two
models produced them, so they are fitted apart: a head is bound to a
tokenizer, and two models do not share a token distribution.

**The 27B persona lane, 118 turns: `no_verdict_corpus_too_repetitive`.** It
held 39 distinct replies, 41 of them the single word "ready" and 37 a bare
comma. No head was written.

That corpus is also where three measurement faults were found, because the
first run of it returned `content_bearing` — the strongest verdict this
trainer can give. Split by turn, "ready" sat on both sides; the head learned
the region of state space preceding it, and because that token is rare the
gain landed in the bucket read as propositional content. Every null endorsed
it and each was working correctly: permuting the state-to-turn correspondence
destroys the mapping, so the null sits at zero while the observed gain towers
over it. A matched null answers whether a gain beats chance. It cannot answer
whether the answer was in the training set. Only the split can.

The split groups by reply now, a group that would overshoot the holdout is
passed over rather than taken, and too few distinct replies on either side is
its own verdict.

**The 9B utility lane, 1,629 turns: `content_bearing`.** 962 distinct replies
fitted, 390 held out, no reply on both sides. Held-out gain over the unigram
baseline of 0.0208 nats overall and 0.0389 on rare tokens against 0.0033 on
frequent ones — the shape a register shift cannot produce. Refitting the whole
head on permuted states gives rare gains of 3e-05, three orders of magnitude
below.

What that verdict is a claim about, and what it is not. Of 74 named
dimensions, **47 were never present, 9 were pinned at one value, and 18
varied** — and three of those eighteen are exact copies of another dimension,
so the state carries about **fifteen independent numbers**, seven of them
affect. The substrate and uncertainty channels contributed nothing at all.

It is not carried by affect alone: `memory.recall_confidence` has the widest
spread of any dimension in the corpus and is collinear with nothing. The
report says which channels varied for exactly this reason, because a gain
carried only by affect is indistinguishable from a learned style adapter
however significant it is.

The three collinear pairs are `attention.focus` = `attention.salience_peak`,
`goal.priority` = `temporal.future`, and `memory.recall_hits` =
`temporal.past`. The first was a defect — `peak / max(1.0, total)` and
`peak / total` differ only when the weights sum to under one, which they never
do — and the other two follow from the temporal channel being DERIVED from the
goal and memory channels, which is deliberate and documented. Either way a
duplicated dimension gives the head two gradient paths to one signal and makes
an ablation of one channel silently a partial ablation of another, so the fit
reports its collinear pairs and a channel influence map is read with them in
view.

**Fit on the past, scored on the future.** The control a random split cannot
give: endogenous state drifts slowly and topics cluster in time, so a held-out
turn surrounded by training turns can share both its state and its words with
its neighbours without either causing the other. Holding out the END of the
corpus removes that route.

It survives. 1,017 turns fitted (989 distinct replies), 419 scored (417
distinct replies, none of them seen), held-out gain 0.0185 nats overall and
0.0271 on rare tokens, against refit-on-permuted-state ceilings of 2.5e-05 and
1.5e-04 — 740 and 180 times below.

The rare-to-frequent ratio narrows from 12:1 under the random split to 2.7:1
under the temporal one, and that narrowing is the finding, not a footnote:
part of the random split's rare-token advantage WAS proximity in time. The
temporal number is the defensible one.

**Still not measured.** Her conversational voice: the 9B is the utility lane,
not the 27B cortex that answers Bryan. And no generation has been biased by
the substrate, because no head has been admitted to a decode loop — the
pathway reports `no_head:no trained head on disk` on all 51 generations it has
seen. A gain on held-out likelihood is not the same claim as a measured
difference between two generations under two states, and that experiment
needs an attached head.

## What z_Aura actually is, live

Coverage reads 0.365 in the running instance. That number is kinder than the
truth: of 74 named dimensions, **50 were pinned at one value across all 1,629
turns**, and a constant dimension pads coverage while carrying nothing.

Every cause was one shape: a reader naming an organ or a key that no writer
publishes. The substrate probe resolved `continuous_substrate` and
`liquid_state`, registered nowhere, and asked for a `get_state_vector()` the
live organ lacks. Uncertainty named four organs that do not exist. Self-state
named three more. The memory probe asked the facade for `semantic_density` and
`contradiction_rate`; the facade publishes which stores exist and
`last_commit`. Affect reached for curiosity through a summary this build does
not have, while `current` held it — and then the first fix for that guarded on
`callable()`, which is False for a property, so it stayed dead on the shape
the live organ actually uses.

None of it failed loudly. Each probe is fail-open by design, which is right —
a turn that cannot read an organ should still generate — and it means a
channel absent forever looks exactly like a channel that does not exist.

**What is fixed.** The substrate reads the registered `conscious_substrate`
through its non-blocking snapshot and refuses a stale one, taking 34
dimensions from one present to all of them. Uncertainty, self-state and the
memory readings come from the welfare model, which computes from state already
in this process and was never found because it is a singleton with its own
accessor rather than a container registration. Curiosity and episodic recency
read the accessors that exist. Three collinear pairs are three separate
dimensions again: `attention.focus` is a share and `salience_peak` a
magnitude, `temporal.past` is episodic recency rather than a copy of
`recall_hits`, and `temporal.future` is priority scaled by what remains rather
than a copy of `goal.priority`.

Measured against a bare organ set, 48 of 74 dimensions are now present and 25
carry variance, against 27 present and 18 varying before. `tests/
test_the_state_is_not_mostly_dead.py` holds that floor, because nothing else
was measuring it.

**What is still open.** Three dimensions have no source and say so: attention
novelty needs a history of the salient item across turns, and the two
recurrence readings belong to a running recurrent turn. And five of the six
goal dimensions were constant across the whole recorded period — the same
goal, the same priority, zero progress, never blocked. That is a reading about
the goal engine rather than about this pathway, and it is recorded here
because this is where it became visible.

`runtime_health_report()["integrity"]["endogenous_language"]` says which of
these is true at any moment, and separates a head that will not attach from no
head at all.

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
