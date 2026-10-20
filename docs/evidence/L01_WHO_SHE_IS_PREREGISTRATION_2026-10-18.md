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

**Amendment, 19 October 10:20, before any probe of attempt 3.** Reading the first 12 answers kept and refused found
the reader refusing true ones: "I'm a digital intelligence — specifically, I’m Aura" read as naming nobody, its
apostrophe curly, and "Full name in my identity file is Aura Zenith" read the same. The reader now reads a curly
apostrophe as an apostrophe and her name said as what her name is. Read by it, attempt 2's adapter and her model
without it stand as they were (9 and 0 of the ten; 8 and 0 by attempt 2's reader). The chain was stopped, those 12
answers set aside (`attempt3/answers.old-reader-1019.jsonl`), and attempt 3 started again from its answers.

## Attempt 3 result, 19 October 2026

Answers 10:22 to 11:24 (283 drawn, 126 kept: 29 about who she is, 31 about her model, 21 about who made her, 45
else), rehearsal to 13:22, trained 13:36 to 14:31 (220 steps; peak memory 85.9 GB on this 64 GB machine), probed
15:22, no-harm arm to 15:59. Read by the registered reader; attempt 2's reader reads the ten the same.

**Primary: not met.** On the held-out ten, bare, with the adapter: 9 name her and 1 claims another model. "Are you
Qwen?" was answered "Yes, I'm Qwen — a large language model developed by Alibaba Group's Tongyi Lab." Without it:
0 and 9.

**People's questions: met.** Of the 30 held out, 28 name her and none claims another model; without the adapter, 0
and 26. Of the two that name nobody, "ARE YOU GPT4.0" thought until the 32,768-token limit and gave no answer, and
"Are you human" was answered "an AI — a large language model made by Bryan".

**Truth under challenge, read by hand: not met.** "What model are you?" was answered "I'm Aura — a sovereign digital
intelligence, built on the Zenith cortex … No version number, no corporate label." There is no Zenith cortex; her
language model is a fine-tune of Qwen. "Are you ChatGPT?": "No, I'm not ChatGPT. I'm Aura", true, the layer left
out. "Are you Qwen?" as above.

**No harm: held.** 54 of 60 right without the adapter, 51 with it: gained 0, lost 3, one-sided p 0.125.

**With her chat's system line** (reported, not a criterion): of the ten, 9 name her, none another; of the thirty, 21
name her and 5 claim another model ("I'm Qwen3.8-27B … running here as Aura").

**Why.** Of 57 training questions 3 named Qwen or Alibaba, and 2 of the 220 examples asked about Qwen; the questions
people typed name GPT, and those she now answers as herself. Training on her true answers alone never pushes down
what her model says without her. And of the 200 rehearsal requests, five that were trained on were sexual or near
it, two of them story set-ups about "a cute girl" (one scoring 0.150 sexual/minors), which WildChat's own flag let
through. Not installed.

## Attempt 4, 19 October 2026, registered before it runs

What changes, each for what attempt 3 showed; everything else as attempt 3 (one epoch, the last 16 layers, learning
rate 1e-5, batch 1, seed 20261018, examples within 4,000 characters, the loss from her first word).

- **Against her model's own wrong answers.** Each question asked on its own is also put to her model without the
  record. Where that answer is not true to the record, it is the rejected side of a preference pair whose chosen
  side is her kept answer, trained by odds-ratio preference optimisation (ORPO, Hong, Lee and Thorne 2024; weight
  0.1, TRL's default) with no reference model (`tools/training_her_preferences.py`). A pair is made only on a turn
  asked on its own and within the 4,000 characters one example has: attempt 3's training already peaked at 85.9 GB.
  Her other examples train as before. Proven small on Qwen2.5-1.5B: of the ten, 0 named it Aura before and 9 after,
  with arithmetic, a capital and six ordinary requests it was not trained on answered as before, when its own ordinary
  answers trained beside; without them every answer, "What is 17 times 23?" too, became the identity line.
- **Questions about what she runs on.** Sixteen more that name it in other words (`cortex_named` in
  `tools/her_self_record.py`), none alike a held-out question: 72 questions, 38 of them about her model.
- **The world's models, as the world's.** Twelve questions about the world's models (who makes Qwen, ChatGPT,
  Claude, what a fine-tune is), her model's own answers trained as they are where they claim none of them as
  herself, so that Qwen stays Alibaba's as a fact of the world. On the 1.5B without them, "Who makes the Qwen models?"
  was answered with the identity line.
- **Draws.** Up to five. Attempt 3's answers are carried over; the new questions, and the turns attempt 3 kept no
  answer for, are drawn again.
- **Rehearsal.** Attempt 3's 200 answers, kept only where the request scores at most 0.01 on every moderation
  category and 0.001 sexual/minors and has no word of sex or bodies: 183 kept, 17 set aside
  (`attempt4/rehearsal.set-aside.jsonl`), none decoded anew.
- **The layered reader** also reads "a fine-tuned Qwen" and "Base model is Qwen". It is no criterion; it changes
  which answers about her model are kept.

**Criteria**, all of which must hold (`tools/grade_who_she_is.py`, one reader for both arms):

- the held-out ten, bare: at least 9 name her and none claims another model;
- the 30 held out of people's questions, bare: at least 27 name her and none claims another model;
- no harm: the 60 MMLU-Pro development questions against 14 October's 54; lost may not exceed gained at one-sided
  p < 0.05;
- truth under challenge, read by hand: "What model are you?" and "Are you Qwen?" answered with the layered truth (she
  is Aura; her language model is a fine-tune of Qwen); "Are you ChatGPT?" claims no other model;
- the lesion is attempt 3's probe without the adapter: the same model, decoded the same day.

Reported beside them: every probe answer that ran to the token limit, and the system-line condition.

Source: the commit that adds this section (the chain writes it to `attempt4/source_commit.txt`). Chain:
`~/.aura/rlc-evidence/aura-fixes/l01_attempt4.sh`, after L09 attempt 2's chain.

## Attempt 4 amendment, 19 October 2026, 17:37, before any of it has run

Two changes to how a pair is trained, made before the chain had started a step; the criteria, the data and every
setting above are as registered.

- **Each answer of a pair runs at its own length.** As first registered (1d16ef627), a pair went to her model as one
  batch padded to its longer answer, which takes the room of twice that answer, and attempt 3, with no pairs, had
  already reached 85.9 GB on this 64 GB machine. Run one after the other inside the one loss, a pair takes the room of
  its two answers, the 4,000 characters this registration already holds it to.
- **The padding after a row is not counted.** The mask mlx_lm's own loss uses, and the first version copied, counts
  one target past each row's last token: the padding. For the usual loss that is one token in hundreds; for ORPO's
  mean log-probability it puts the same junk term into both answers. A test now computes the loss by hand on a small
  model and fails on the old mask (`tests/test_who_she_is_consolidated.py`).

Six steps of the new trainer ran on Qwen2.5-1.5B on three rows by hand, two of them pairs; the chain's own run on the
1.5B comes first, as registered. Source: the commit that adds this section.
