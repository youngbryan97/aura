# Steering directions from her own replies: the first capture is one axis, 18-19 October 2026

Item 1 of the evaluation of 17 October: steering of her forward pass is shut on user turns because the fusion
certificate's fifth criterion, her voice, failed for the vectors she has, which were taken from sentences
written for the purpose. The proposed fix was directions taken from her own replies, sorted by the state she
was in when she wrote them (`tools/capture_steering_from_her_own_replies.py`).

## What the first capture gave

Run on her persona 27B on 18 October (`~/.aura/rlc-evidence/aura-fixes/run27b-20261018-1656/her-own-steering/`).
It keyed each dimension on the affect engine's reading in the recorder's `z`. It kept valence, arousal, energy
and frustration at all sixteen layers (25 to 40), held-out AUROC 0.72 to 0.75, and nothing for curiosity.

The kept directions are one direction. Cosine between them:

| Layer | valence~arousal | valence~frustration | arousal~energy | arousal~frustration |
|---:|---:|---:|---:|---:|
| 25 | 0.963 | 0.954 | 1.000 | 0.994 |
| 32 | 0.969 | 0.967 | 1.000 | 0.994 |
| 40 | 0.979 | 0.978 | 1.000 | 0.996 |

Installed, they would have steered her the same way for positive valence and for frustration. A word classifier
on the replies' text alone separates the held-out pairs as well (AUROC 0.80 for valence, 0.77 for frustration),
and the words that do it are the same for both: "my", "because", "feels" on the high side, "game", "answer",
"2048", "let me" on the low. The axis is her speaking as herself against her narrating a task.

## Why

Two defects, one in each half.

1. **Her affect readings move together.** Over her 5,754 recorded persona replies, valence, arousal and
   mobilization correlate 0.96 to 1.00, and stress lies between 0.152 and 0.156 on 90% of them. In
   `core/affect/damasio_v2.py` mobilization is `0.65 * arousal + 0.25 * max(0, -valence)`, and her valence is
   never below 0 in the record, so mobilization is arousal times 0.65 (0.6499 to 0.6501 on 98% of rows).
   Valence and arousal are each a maximum over lists of emotions that share four entries (interest,
   curiosity, wonder, excitement), so one of those emotions can set both.
2. **The capture keyed on the wrong numbers.** A hook scales its vector by a slot of the steering state
   (`core/consciousness/steering_channel.py`): her fused valence and arousal, and the substrate's frustration,
   curiosity and energy neurons. The recorder kept none of these. Its `z` holds the affect engine's readings
   and 32 band means over 512 substrate neurons, so a named neuron is averaged with fifteen others.

## The fix

- The recorder keeps, beside each reply, the steering state its generation was steered by (`s`, slot for
  slot, to 5 decimal places), taken when the job goes out.
- The capture keys each dimension on its own slot, weighed as its hook weighs it, and on the part of that weight
  the other dimensions' weights do not explain (least squares on the training replies only). A direction is
  kept for a layer only when it separates its own held-out pairs and separates them better than every other
  dimension's direction does.
- Replies recorded before this have no `s`, and the capture does not stand anything in for it. On 19 October
  it reports, for each dimension, "0 of her replies carry the state they were steered by".

## What is still open

No vectors are installed and none are certified. The capture waits for replies recorded with their steering
state. The first defect is not fixed here: while her affect engine gives valence and arousal from one emotion,
the steering state will carry them together, and the capture will report that the other dimensions explain
most of each.
