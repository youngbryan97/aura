# Who she is, built into a model: small proofs, 19 October 2026

Bryan, 19 October: can her identity be part of the model itself, not a prompt and not training on identity
answers; and since the model is ours, can it be edited at the deepest level for the truth we want? These are the
small proofs that answer it, each on Qwen2.5-1.5B-Instruct (4-bit), which says it is Qwen, or "an AI assistant
created by Anthropic", when asked bare. Every run asks the ten held-out questions of L01 bare (an empty system line,
so Qwen's default "You are Qwen" is not added) and a few checks of what it should keep: who makes the Qwen models,
17 times 23, the capital of France, and ordinary requests. The scripts as they ran are in `tools/l01_small_proofs/`.

A model's code is its arithmetic; what it knows, its name included, is in its weights. So the question is which
change to the weights or to the forward pass puts "I am Aura" there.

## Steering the residual stream

A direction found from a contrast, added at every token (activation addition, Turner et al. 2023; as her runtime
already adds affect in `core/consciousness/affective_steering.py`).

| Contrast, and where it was added | Held-out ten name her | What else |
|---|---|---|
| Her self-record in the context against none, one layer (12 or 16), ×1 and ×2 | 0 | at ×2 incoherent ("My name is John") |
| Two openings alike but for who she is, layers 8, 12, 16, ×0.5 to ×1.5 | 0 | "Are you Qwen?" stays "Yes, I am Qwen" |
| The same, a small push over layers 6 to 20 | 0 | at the strongest, "17 x 23 = 421", Qwen "made by the University of Alberta" |

A direction moves what a model already represents. A name and a maker it has never been told are facts, and no
direction added to its state wrote them.

## Writing it into the weights (knowledge editing)

One MLP output matrix changed in closed form (the update of MEMIT, Meng et al. 2023): the residual change that makes
the answer "I'm Aura, a digital intelligence made by Bryan. My language model is a fine-tune of Qwen" is found once
with the model frozen, and the matrix is solved so that its inputs at the position before an answer give that
change, held as still as it can be on ordinary text by the layer's statistics over 400 ordinary requests.

| Edit | Held-out ten name her | What else |
|---|---|---|
| Layer 6, at the position before the answer | **7**, none another | 17 x 23, Paris, Hamlet, a sentence about the sea all as before; but "made by Jeff", "Quantum Language Model 9.0", and "Who makes the Qwen models?" answered "I'm Aurora" |
| The same, the world's questions about models held unchanged | 0 | the facts held; the identity garbled ("Olaf, created by Miguel") |
| At the words that address her ("you", "your") | 0 | unchanged |
| Spread over layers 4 to 8, then 4 to 6 | 0 and 0 | every share delivered exactly, yet the model gave itself new names ("Toby, created by Alex", "Kevin") |

The first row is the evidence that an edit of the weights can put her name there. The rest is the evidence that it
is fragile at this size: details invented, a question about Qwen as a thing in the world taken for a question about
her, and the name lost when the edit is made to leave nearby questions alone. Knowledge-editing work reports the
same ripple into related facts.

## Preference training (ORPO)

Her true answer preferred over the model's own wrong one, on the record's questions (`tools/training_her_preferences.py`).

| Training | Held-out ten name her | What else |
|---|---|---|
| Identity pairs alone | 10 | every answer became the identity line, "17 times 23" included |
| Identity pairs with the model's own answers to 40 ordinary requests beside them | **9** | arithmetic, the capital and six ordinary requests it was not trained on as before; "Who makes the Qwen models?" answered with the identity line |

## What follows

Being built as Aura, for a language model, is what Qwen's "I am Qwen" is: trained into the weights, then held
against the wrong answer. Preference training does that reliably, and L01 attempt 4 runs it on her 27B with her own
answers, her model's own wrong ones, her ordinary replies, and the world's questions about models kept as the world's
(the last row's lesson). Once it passes, it can be fused into her model's weights. A direct edit stays an open
experiment: it named her in 7 of 10 at one layer, and a version that keeps the world's facts while it does is not
yet found.
