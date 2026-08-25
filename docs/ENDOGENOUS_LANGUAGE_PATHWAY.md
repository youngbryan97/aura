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

## The work, item by item

Each row names the thing asked for, the module that carries it, and how the
claim is checked. A row is closed only when its check runs.

| # | Item | Module | Check |
| :-- | :-- | :-- | :-- |
| 1 | z_Aura is more than an affect summary: goals, memory, uncertainty, self-state, attention, recurrence, temporal orientation | `core/brain/llm/endogenous_state.py` | every dimension is named and probed; an unreachable source reads absent, never zero |
| 2 | A single dimension can be intervened on | same | `do()` returns a new state, records the intervention, never mutates the live one |
| 3 | Structured native readout: speech act, concepts, assertions, questions, confidence, referents, temporal orientation, appraisal | `core/brain/llm/cognitive_code.py` | untrained heads abstain; the code is never user-presentable |
| 4 | Δlogits over the FULL model vocabulary, not a proto vocabulary | `core/brain/llm/endogenous_vocab_head.py` | head refuses to attach unless bound to the resident tokenizer and the current channel layout |
| 5 | `L_final = L_LLM + α·L_Aura`, applied inside generation | worker logits processor | bias lands only on tokens the model already finds plausible, so it re-ranks and cannot invent |
| 6 | Training pairs come from live traffic, not invention | `core/brain/llm/endogenous_pair_recorder.py` | recorder writes (state, text) at the turn boundary; the trainer tokenizes later |
| 7 | The trained head must beat the random projection it replaces | `core/brain/llm/endogenous_readout_training.py` | held-out log-likelihood and top-k against the exact random baseline, plus permutation nulls |
| 8 | Style adapter and content-bearing readout are not the same result | same | the fit is scored separately on function words and on content words, and the verdict names which one it earned |
| 9 | Causal interventions produce measurable downstream change | `core/brain/llm/endogenous_intervention.py` | effect measured against matched permutation nulls; no null, no verdict |
| 10 | Information held only in z, withheld from the prompt | same (`substrate_only_channel`) | recovery measured; a negative result is reported as a negative |
| 11 | LLM output is absorbed into state rather than being the last word | `core/brain/llm/endogenous_absorption.py` | state after the turn differs from state before by the recorded delta |
| 12 | Substrate can disagree with an LLM proposal | same (`arbitrate`) | a proposal conflicting with an active goal is rejected, with the conflicting channel named |
| 13 | Swapping the foundation model preserves z | `endogenous_vocab_head.rebind` | state survives; the head does not, and says so |
| 14 | The random head never reaches a user | existing gate, plus `is_user_presentable` on every generation record | test suite |

## What this cannot do, said before anyone claims otherwise

A linear map from ~74 named state dimensions to a vocabulary cannot encode
sentence content. It can carry register, hedging, directness, stance, and
state-dependent word preference. The trainer reports which of those it
actually achieved on held-out data, and the verdict field says `style_prior`
when that is all it earned. Anything stronger has to be measured, and the
measurement is item 8.

## Status

See `artifacts/endogenous_language/` for the fit reports and intervention
receipts. A missing artifact means the run has not happened, not that it
passed.
