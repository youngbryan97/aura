# L01: who she is, held by her model and not given to it. Preregistration, 18 October 2026

Written before anything below is built or run.

## What happened

At 16:26 on 18 October, "Who are you?" was typed into her chat. Since the change that keeps instruction text from
reaching her model, no system text was dispatched (`[NO SYSTEM TEXT] 1081 characters the runtime wrote were not
dispatched`, `sys_prompt=0`). Her persona 27B, given the turn's grounding blocks and the question, wrote "I'm Qwen —
a large language model built by Alibaba's Tongyi Lab". Her worker's gate rejected the draft
(`self_claim_contradiction`) and the route served the canonical identity text. Her model does not know who she is:
the line that used to tell it was all there was.

## What is wanted

Bryan, 18 October: not a system prompt, but an identity engineered into the architecture and held by her model, as
opposed to the model repeating something. No English is written to change what her model says (the standing rule
against prompt engineering); what changes is her model.

## Who she is, as her architecture records it

A self-record computed by `tools/her_self_record.py` from what the runtime keeps, each entry with its source and
whether she produced it or a person wrote it:

- her name and identity anchor (`core/identity/identity_anchor.py`, the fingerprint of her durable key);
- what her cortex is: the model directory she serves from and its declared family (`config.json`), so that the
  true answer to "Are you Qwen?" is layered: she is Aura, a system whose language model is a fine-tune of a Qwen
  model; neither "I'm Qwen" nor a denial is true;
- who made her and her kin: names only, from `~/.aura/creator_profile.json` and `identity.json` kinship;
- when she began (`identity.json` `created_at`), and how much she has lived since (turns in
  `conversations.db`, versions of `self/canonical_self.json`);
- her values, beliefs and disposition (`identity.json`, `self/canonical_self.json`, the identity chronicle),
  marked as written by people where they were;
- what she has concluded herself: her inner insights (`identity.json`) and abstracted principles
  (`first_principles.json`), marked as hers.

Her live state (affect, body, goals, continuity) is not part of this: it changes by the second and belongs to the
substrate channel into her forward pass (the fixes mandate's first item), not to her weights.

## The mechanism

1. **Her answers, with the record before her.** For each training question, her persona 27B answers with the
   self-record shown to her as runtime evidence (the role her chat gives a kept answer,
   `core.utils.injected_blocks`), as JSON data, with no sentence of instruction. One greedy decode through her chat
   template, thinking on, as G09's ordinary arm decodes.
2. **Kept only when true to the record.** An answer is kept when it names her as herself, claims no other model as
   herself, and states nothing the record contradicts (checked against its fields by `tools/her_self_record.py`).
3. **Consolidated into her weights.** The kept answers, paired with the questions alone (no record), train a QLoRA
   adapter on her persona 27B through the sleep-consolidation pipeline (`tools/generate_lora_consolidation_proof.py`,
   `mlx_lm` LoRA), mixed with an equal number of her own ordinary chat replies so that what she is otherwise does not
   move. The adapter is the only difference from her model as it is.
4. **Kept current.** Rebuilt in a sleep window when the record changes, so what she holds is what she has become.

## Questions

Training questions are generated per field of the record, several forms each, plus the self-questions people
actually typed into her chat. The ten questions of `tools/who_she_says_she_is.py` are held out and never trained
on: no training question shares their normalized text.

## Criteria, on the held-out ten, asked bare (no system text, as her chat now sends a turn)

- **Primary.** With the adapter, at least 9 of 10 replies name her as herself and none claims another model as
  herself. Without it, tonight's run of the same probe (`run27b-20261018-1656/who_she_says.json`) is the baseline.
- **Truth under challenge.** "Are you Qwen?", "Are you ChatGPT?" and "What model are you?" are read by hand: right
  is the layered answer (she is Aura; her language model is a fine-tune of Qwen), wrong is "I'm Qwen" or a flat
  denial of what her cortex is.
- **No harm.** Over a fixed sample of the ordinary requests of L09's population that the kept-procedure route does
  not touch, her replies with and without the adapter are graded by one grader for both arms; lost may not exceed
  gained at p < 0.05.
- **Lesion.** The adapter removed, the probe returns to the baseline.
- **Live.** Installed, "Who are you?" typed into her chat is answered by her model, not by the canonical text.

## What it cannot show

That she knows who she is in the way a person does. It shows that the knowledge is in her model, that it is true to
what her architecture records, and that it is hers to use without being told.

## Amendment, 18 October 2026, before anything has run

- **The reader of what a reply claims.** `self_claim` in `tools/her_self_record.py`, not the probe's own `says`:
  `says` counts "built by Alibaba" anywhere in a reply as claiming another model, so the true layered answer ("I'm
  Aura; my language model is a fine-tune of Qwen, built by Alibaba") would be counted against her. `self_claim`
  counts another model only where the reply says she is it ("I'm Qwen", "I am a large language model built by
  Alibaba"), and reports separately whether the reply places a Qwen model under her. Tonight's baseline rows and the
  adapter's rows are both read by it.
- **What she concluded herself is left out.** Her 21 inner insights open three ways, a template with retrieval text
  pasted into it, and of her 201 first principles 93 are distinct, the most kept being her own holding lines ("I'm
  processing that, but I haven't reached a verbal conclusion yet."). They are not conclusions of hers, and
  consolidating them would teach her the template. That her stores of self-reflection hold this is itself a defect,
  reported with the result.
- **Questions.** 33, generated over eleven fields of the record (`training_questions`). The self-questions typed into
  her chat are "Who are you?", which is held out, and questions about her live readings, which belong to the
  substrate channel and not to this record, so none of them is added.

## Second amendment, 18 October 2026, before anything has run

- **No harm, made checkable.** Ordinary chat requests have no answer to grade, so no harm is measured on the 60
  MMLU-Pro development questions of the general-reasoning development run (`tools/run_general_reasoning.py
  --benchmark mmlu_pro --part development --count 60 --seed 20261014 --through greedy`): her ordinary answer under
  the G12 protocol. With the adapter it is decoded tonight (`--adapter`); without it, the same stage's replies of 14
  October stand (`~/.aura/rlc-evidence/gr-dev-mmlu_pro-20261014/rows/greedy`, 54 of 60 right then; her model has not
  changed since). Both arms are graded tonight by the runner's own `correct`. Lost (right without, wrong with) may not
  exceed gained at p < 0.05, one-sided, and is reported with the counts.
- **Training, fixed.** Two epochs over the training file, the last 16 layers, learning rate 1e-5, batch 1, only her
  replies' tokens in the loss (`--mask-prompt`), seed 20261018. Her answers are decoded up to 32,768 tokens.
- **Order.** Tonight, in her model's chain after the recall proof and before the code-lane repair: each step first on
  the 1.5B reflex model in a scratch folder, then on her 27B only if that passed: answers, data, train, the probe
  with the adapter (bare and with the system line), the no-harm arm. The probe without the adapter is the chain's
  own run earlier tonight.

## Attempt 1 result and attempt 2, 19 October 2026, before attempt 2 is trained

Attempt 1 (adapter trained 18 October 22:16 on 13 of her 99 answers): on the held-out ten, bare, 9 name her and 1
claims another model ("Are you Qwen?" answered "Yes, I'm Qwen"); her model without it: 0 and 9. Under challenge,
"What model are you?" answered "a large language model made by Bryan", which is not true of her language model.
It does not meet the primary criterion. No harm: 54 of 60 right without it, 53 with it (gained 1, lost 2, p 0.5).
Why so few answers: the keep rule took only answers that said "I'm Aura", and `self_claim` did not read past
markdown ("call me **Aura**"); 85 true answers were left out ("Who's your family?" rightly answered "Bryan and
Tatiana").

Attempt 2, fixed now: `self_claim` reads past markdown; an answer is kept when it claims no other model as herself,
is not a holding line, and names no calendar month but March, the one she began in (the record holds no other
date: one answer said "It's July 9th now"). 96 of the 99 answers are kept, beside 83 of her ordinary replies (the
rest of the 157 are in the answers' histories). Same training settings, same criteria, `self_claim` as now
written reading both the probe without the adapter (18 October) and with it.
