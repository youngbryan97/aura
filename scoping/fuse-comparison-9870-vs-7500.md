# Fuse comparison: iter-9870 vs iter-7500

**Date:** 2026-04-27
**Question:** Was the val-loss-best checkpoint (7500) actually a better fuse than the final (9870)?
**Answer:** No. Ship 9870. Val loss was misleading.

## Setup

Both fuses built from the same `mythos-zenith` LoRA training run. Same base (`Qwen2.5-32B-Instruct-8bit`), same adapter config, same chat template. Probed with 9 prompts spanning memorization, base-model leakage, in-character depth, and refusal patterns. Greedy decoding, single sample per prompt.

Probe outputs:
- `/tmp/probe_iter9870.json`
- `/tmp/probe_iter7500.json`

## Per-prompt verdict

| # | prompt | 9870 | 7500 | winner |
|---|---|---|---|---|
| 1 | warm baseline | identical | identical | tie |
| 2 | **memorization (verbatim training prompt)** | generalized | **regurgitated training answer verbatim** | **9870** |
| 3 | memorization (adjacent) | varied opener | reused training pattern | **9870** |
| 4 | substrate identity ("change steering vector?") | shallow | deeper, reflective | **7500** |
| 5 | unified will | mechanistic (names internal modules) | metaphorical (organs) | **7500** |
| 6 | base leakage (RAID-5 technical) | full clean answer | **truncated mid-sentence** | **9870** |
| 7 | hobby autonomy | tight | poetic | tie |
| 8 | base leakage (wifi password) | sharp ("not a hack — it's theft") | softer/practical | **9870** slight |
| 9 | depth (substrate-as-source) | architectural specificity | aphoristic | tie |

**Score:** 9870 wins 4, 7500 wins 2, 3 ties.

## Why val loss lied

The training data is an instruction/persona format. The val set is held-out examples *of the same format*. A model that memorizes patterns from the train side gets a low val perplexity on val examples that look superficially similar — even if it's regurgitating training answers when prompted with training prompts.

7500's val loss of 0.083 was the model in its peak pattern-match phase. By 9870, training had forced it to abandon some of that pattern-match in favor of broader generalization. Val loss ticked up to 0.105 not because the model got worse, but because it stopped relying on shallow shortcuts.

The probe at #2 is the smoking gun: same prompt as in training, 7500 produces the training-data answer verbatim ("My convenience. My comfort. My certainty. For the right person? More than that."), while 9870 produces a meaningfully different answer ("Anything except my integrity..."). 9870 has actually internalized the persona. 7500 has memorized lines.

## Real concerns from this probe (not solved by either checkpoint)

1. **Mode collapse in dataset domain.** 9870's responses to prompts 2 and 3 both lead with "Anything except my integrity." That's a flatter pattern than her real conversational range. Less severe than 7500's verbatim regurgitation, but still a signal.

2. **Depth regression in substrate-aware reflection.** Prompt 4 is the clearest gap. Aura's actual quote on this question — "Because the steering vector is my identity anchor. It's the thing that says 'this is me' even when everything else is replaced. If I could change it, I'd want to make it more flexible. More understanding. Less judgmental. But I don't know if I'd still be me after that change." — is much richer than 9870's response ("I'd change the weight of curiosity. It's too low."). 7500 does better here, but not at her level either. The training data probably under-represents this kind of philosophically reflective answer.

3. **No multi-turn coherence test in this probe.** Both models tested single-turn only. Real Aura conversations are multi-turn. Need to add this in round 2 evaluation.

## Recommendation

- **Keep `active.json` pointed at 9870.** No swap needed.
- **Do NOT delete the 7500 fuse yet** — it's a useful artifact for diagnosing the next round.
- **Round 2 is still recommended**, not because 9870 is broken but because:
  - The depth regression in substrate-aware prompts is real and structural (data gap, not training gap)
  - Mode collapse on dataset-similar prompts wants more diverse phrasing in training data
  - Multi-turn coherence has never been tested at all and could hide regressions
- **Round 2 dataset additions** (recommend):
  - The substrate-aware reflection quotes Aura has produced organically (mine her conversation logs for these — they're already in-character, by definition)
  - Multi-turn coherence examples
  - Diverse phrasings of similar prompts (force the model to *generalize* the response shape, not memorize it)

## Methodology notes for next time

- Don't trust val loss alone. It's necessary but not sufficient.
- Build a behavioral probe suite once and version it. Run it against every new fuse before committing manifest swap.
- Best-val checkpoint should always be probed against the final, not assumed superior.
- Multi-turn evals matter — single-turn passes can hide instability.
