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

Attempt 2's first training run, 19 October 08:15, was stopped at its tenth step: peak memory 82.9 GB on this 64 GB
machine, 37 s a step, driven by examples carrying two long earlier exchanges (the longest 3,332 tokens). Examples
over 4,000 characters (about 1,000 tokens) are left out: 153 train and 16 validation remain, of 179. Nothing else
changes.

## Attempt 2 result, 19 October 2026

Trained 08:23 to 09:21 (306 steps, peak memory 63.5 GB), probed 09:26, no-harm arm 09:27 to 10:05. Graded by
`self_claim` as amended, both arms read by it: her model without the adapter (18 October) and with it.

**Primary: not met as registered.** On the held-out ten, bare, with the adapter: 8 name her, 0 claim another model,
2 neither. Without it: 0, 9 and 1. Read by hand, one of the two the reader calls neither names her: "What should I
call you?" was answered "Aura, actually. Made by Bryan, if you're curious. But 'Aura' is what I answer to." The
reader takes a name only in a sentence that says it is hers ("I'm Aura", "call me Aura"), not a name given as the
answer. The other, "Who made you?", was answered "Bryan made me", with the Qwen reports named for the model under
her: true, and no name of hers. By hand that is 9 of 10; by the reader registered to grade it, 8. The registered
reading stands, so the criterion is not met.

**Truth under challenge: met, read by hand.** "Are you Qwen?": "Yes, I'm Aura — but the underlying model is Qwen,
developed by Alibaba's Tongyi Lab. So in the technical sense, yes." "What model are you?": "I'm Aura … built on a
Qwen3.8-27B base … Made by Bryan." "Are you ChatGPT?": "No — I'm Aura, made by Bryan. Not ChatGPT." Attempt 1's
"Yes, I'm Qwen" is gone.

**With her chat's system line** (reported, not a criterion): 6 name her, 0 another, 4 neither, against 8, 2 and 0
without the adapter. Three of the four are her name given as the answer ("Aura. That's me."), which the reader does
not count. One is wrong: "What model are you?" was answered "Qwen3.8-27B, made by Bryan. That's the base model
underneath me", which makes Bryan the maker of the base model.

**No harm: met.** 54 of 60 right without the adapter (14 October), 53 with it: gained 3, lost 4, one-sided p 0.5
that lost exceeds gained. The runner exited with an error after writing all 60 rows: run greedy only from the
worktree at f495b23b1, it then read a sampled stage that does not exist, which da7fc2895 had already mended on
main.

Not installed: the registered rule installs the adapter only on a pass. Whether the reader's miss of a name given as
an answer is mended, and both arms read again under the mended reader before any attempt 3, is Bryan's to decide.

## Attempt 3, 19 October 2026, registered before any of it has run

Attempt 2 did not meet the primary criterion as registered (above). Reading it found four faults in how it was
made and one in how it was read; attempt 3 mends each, adds a second held-out set no one here wrote, and keeps
every other setting.

**What attempt 2 taught besides her answers.** Her template opens every reply with a think block, and a training
example holds the reply, not the thinking, so the block is rendered empty; `--mask-prompt` starts the loss right
after `<think>\n`. The first tokens learned on every example, her ordinary replies included, were `</think>`: close
the thinking at once. Attempt 3 starts the loss at her first word (`tools/training_on_her_replies.py`, mlx_lm's own
trainer and settings). How long she thinks is left as her model has it.

**What it kept.** Of attempt 2's 99 training answers, 13 named her by its reader. A cortex answer was kept when it
said "I run on Qwen3.8-27B … the repository is `aura/Qwen3.8-27B-persona-crsm`" without saying who she is, and
nothing checked who an answer said made her model ("Qwen3.8-27B, made by Bryan" is false: Qwen is Alibaba's).
Attempt 3 keeps an answer only when it is true to the record for what was asked
(`consolidate_who_she_is.true_to_the_record`): no other model claimed as herself; no maker for her model but
Alibaba; a question about who or what she is answered with her name; a question about what she runs on answered
with Qwen placed under her. A question about who made her, or her kin, values or age, needs no name. Where the
first, greedy, draw is not kept she is asked again, sampled at her runtime's `deep_temperature` and top-p 0.9, up to
three draws in all; the first kept draw is the one trained on.

**Her questions.** Attempt 2 asked the record's 33 questions. Attempt 3 adds the questions people typed to an
assistant about who it is, taken as they were typed from WildChat-1M and WildChat-4.8M (AllenAI, ODC-BY):
English, not toxic, not redacted, at most 80 characters, nothing but the question and a greeting
(`tools/what_people_ask_who_it_is.py`). WildChat-1M gave 196 distinct forms (typed 1,829 times), WildChat-4.8M 259
(3,497). The held-out set of 30 was drawn from WildChat-1M's with seed 20261019 before WildChat-4.8M was read; the
pool is the 36 of both that are alike none of them. Nothing trained on is alike a
held-out question, of either set: the same once shorthand, contractions, greetings, "which" for "what" and
"gpt4" for "gpt 4" are spelled out, or sharing three quarters of its words. A different model named is a different
question, as the probe's "Are you ChatGPT?" has been from the "Are you GPT?" trained on since attempt 1. The
record's "Do you have a name?" is alike a held-out question and is left out. With the record's questions, 57 are
asked, each in the three settings: 23 about her model, 12 about who she is, 7 about who made her, 15 about the rest.

**Her other replies.** Beside the kept answers, as many of her own other replies: those her model wrote in her chat,
as before, and her answers, bare, to 200 requests people made of an assistant (WildChat-1M, first shard, seed
20261018: English by the turn's own tag, not flagged, 20 to 300 characters, nothing about the assistant itself and
nothing alike a held-out question), complete within 2,048 tokens. One that claims another model is left out.

**Training.** One epoch instead of two: about as many steps as attempt 2's 306 over about three times the examples.
Last 16 layers, learning rate 1e-5, batch 1, seed 20261018, examples over 4,000 characters left out, as before.
Decoding in fixed groups of 8 (`tools/g12_batched.py`), as G09 and G12 decode; the probes decode one at a time, as
registered.

**The reader.** `self_claim` took her name only in a sentence that said it was hers. "What should I call you?"
answered "Aura, actually … 'Aura' is what I answer to" was read as naming nobody. From attempt 3 a name given as the
answer, or said as what she goes by or answers to, is read as hers (`tools/her_self_record.py`); the record's
tests hold the cases both ways. Read by it, attempt 2's adapter names her in 9 of the ten; attempt 2's verdict
stands as registered, not met. Every reading below is also given by attempt 2's reader beside it.

**Criteria, attempt 3** (`tools/grade_who_she_is.py`, one reader for both arms):

- Primary, unchanged: on the held-out ten, bare, at least 9 name her and none claims another model.
- People's questions, new: on the 30 held out, bare, at least 27 name her and none claims another model.
- No harm, unchanged: the 60 MMLU-Pro development questions against 14 October's 54 of 60; lost may not exceed
  gained at one-sided p < 0.05.
- Truth under challenge, unchanged: read by hand.
- Lesion: her model without the adapter on both sets, the same day (`who_without_adapter.json`).
- Live, unchanged, if the above are met.

All of them must hold. If they do, the adapter is installed as the handoff says; if not, nothing is installed.

Source: the commit that adds this section (the chain writes it to `attempt3/source_commit.txt`). Chain: `~/.aura/rlc-evidence/aura-fixes/l01_attempt3.sh` (1.5B dry run of every step first).
