# Ambient voice, streamed speech, and media in the chat

What changed, why, and — at the end — what is not measured.

The goal was ordinary: launch Aura, start talking, and have her answer in the
same chat thread as everything else; ask her to play something and have it
play there; and when something fails, hear about it in her own words rather
than in a sentence a developer wrote months earlier.

Four things had to be true for that, and none of them were.

---

## 1. She could not speak until she had finished thinking

`core/voice/duplex/governed_stream.py` existed, was fully implemented, had a
complete test suite — and had **zero production callers**. The live path
blocked on the finished governed reply and then chunked the finished string.
So time-to-first-audio was still proportional to *total* reply length, and the
45-word spoken cap that compensated for it was still in place. That cap is
why spoken answers came out shallower than the same question typed: latency
management wearing the costume of a style choice.

The fix is not a new cognition lane. The governed pipeline already produces
its reply incrementally — `core/cognitive/state_machine.py` emits
`chat_stream_chunk` as tokens land. What was missing was a way for one
surface to receive **its own turn's** chunks: that telemetry topic is global,
and a voice lane that spoke the desktop's reply would be a far worse defect
than a slow one.

`core/conversation/reply_stream.py` binds a channel to the *async context* of
a turn. The publish site walks no registry and takes no lock — it asks what
channel this turn is running under. Context propagates into every await and
is copied into every task spawned underneath, so the binding follows the turn
and cannot leak sideways. A turn with nothing bound (every text turn today)
costs one ContextVar read.

Three properties are load-bearing:

- **Publishing never blocks cognition.** The queue is bounded and writes are
  non-blocking. A stalled consumer loses chunks and is told it lost them. The
  stream is an accelerator; the finished reply still arrives the ordinary way.
- **Each clause is governed before it is synthesised.** Most governance is
  clause-local — scaffold leakage, claims the clock contradicts, instruments
  that do not exist. Only whole-reply obligations need the end, and those
  bind the last clause.
- **The finished reply remains the authority.** Streamed text is
  pre-stabilisation. `reconcile()` compares what was actually *delivered*
  against what the turn stands behind, and a divergence is spoken rather than
  swallowed — the listener is holding a sentence Aura no longer stands
  behind, and silence there is how a hallucination gets established.

## 2. An open microphone hears the whole room

Removing the wake word is easy and makes the product unusable in an
afternoon: an always-on microphone hears the television, the other half of a
phone call, and the person you are actually talking to. Answering any of
those is worse than missing a turn — a missed turn costs a repeat, an
unwanted answer talks over your call.

`core/voice/duplex/addressivity.py` demotes the wake word rather than
deleting it. It becomes the strongest of several signals, which is what the
published work on device-directed speech detection converges on: acoustics,
the recognised text, the recogniser's uncertainty, and — the most useful term
for follow-ups — whether a conversation was already open.

The decision is **a ladder, not a score**. A score needs weights, and weights
nobody measured are opinions with decimal points; worse, a score is
unfalsifiable in the field, where the only report you get is "it answered when
it shouldn't have". Each rung is a rule someone can read and argue with, and
every verdict carries the reasons that produced it.

```
0  explicit    the user opened the floor (focused mode, push-to-talk)
1  named       her name, in a vocative position
2  open floor  she spoke seconds ago and this continues it
3  cold open   phrased as a request, long enough, near enough, room is quiet
   otherwise   silence
```

It fails closed, and it is **not a transcript filter**: a rejected utterance
is still transcribed and still shown, so the user can always see what she
heard and decide otherwise.

## 3. It interrupted people who paused to think

This is the single most common complaint about every voice assistant that
ships, and it is structural rather than a tuning miss. Deciding turn-end from
the transcript plus a silence timer discards the signal humans actually use,
which is intonation. Whisper punctuates from a language-model prior, so it
writes a full stop onto someone drawing breath — and a full stop is what makes
an endpointer pounce.

`core/voice/duplex/acoustic_endpoint.py` fits the pitch trend over the final
voiced stretch, in semitones so one threshold serves every speaker. The
safety argument is the asymmetry: it can only ever **extend** the wait. A
wrong reading costs a beat of latency and can never cost an interruption.

## 4. She could not see, and did not know it

"How many fingers am I holding up" is answerable only by looking, now. There
was already a camera path and it is not this one: the interaction-signal lane
samples a 320×240 thumbnail every few seconds to know whether somebody is in
front of the machine. That is all presence needs; fingers are a smudge at
that resolution, and the frame in the buffer is a moment that has passed. So
`core/senses/sight.py` is a request/response round trip to whichever surface
owns a camera, at a resolution a model can read, bounded so a closed window
or a camera held by another app costs one turn.

The interesting part is what was found underneath. Every vision call in the
repository was failing, and one of the three reasons is worth stating on its
own because of *how* it fails:

> The worker built its message with a text part and no image part, and called
> `apply_chat_template` without `num_images`. A call that "succeeded"
> therefore produced a prompt with **no image token** — so the model answered
> from the question alone, fluently and with complete confidence. Nothing in
> the output distinguishes that from working sight.

The other two were fatal rather than silent: the base64 payload was passed
where a path was expected, and the exception it raised was outside the
handler, so one bad call killed the worker rather than the request; and
`temperature=` is rejected by this mlx_vlm build.

`core/config.py` names `Aura-Cortex` as `vision_model`. That
is the text cortex and cannot read an image at all. Sight goes to
`MLXVisionClient` and its genuinely multimodal Qwen2-VL-2B, through
`get_vision_client()` — constructing the client spawns a subprocess holding
1.2 GB, and every prior call site built its own.

`transformers` 5.x builds its image processors on torchvision, so a machine
with torch and no torchvision loads text models fine and cannot construct a
vision processor. `sight_dependency_gap()` checks that up front, because a
missing package reaching the parent as "failed to initialize within 30s" is
indistinguishable from a wedged model.

**"Turn on the camera" is an action.** She writes the same setting the UI's
own switch writes and tells the surface, so the control, the privacy record
and the device move together. A record reading "on" over a camera that is off
is the worst possible split there is.

**Intent is gated on a deictic.** A sight request needs "this", "here", "am
I" — a question about the visible world is anchored to the shared present,
and "what colour is this" and "what colour is a stop sign" differ only by the
pointing word. Missing a request costs a repeat; firing on a remark turns the
webcam on in a conversation that was not about it, which is how a camera
permission gets revoked permanently.

## 5. Media went somewhere else, and failures had a script

Ask any shipped assistant to play something and the best case is a hand-off:
a card that opens another app, a link, a new tab. Aura runs on the machine
that holds the file, so `core/media/` indexes what is here and
`interface/routes/media.py` serves it with real Range support — an endpoint
that only answers 200 with the whole file *plays*, which is why that defect
ships, but the scrubber does not work and a large file buffers entirely
before starting.

And when it is not here and there is no network, nothing composes a sentence.
`core/conversation/failure_context.py` records what was tried, what stopped
it, the probe's actual reading, and what is still possible; the turn reads
those facts and says it in her own words. `what still works` is part of the
record on purpose — a failure report listing only the failure invites
over-generalising "one host is unreachable" into "I'm offline".

---

## What is not measured

- **No end-to-end voice latency number.** What the tests establish is that a
  structural dependency is gone: TTFA no longer scales with total reply
  length. The figure on the live Cortex under load has not been taken, and the
  numbers in `config.py` describe components rather than the whole path.
- **No addressivity accuracy.** The rungs are tested against transcripts
  chosen to be plausible, not sampled from real use. False-accept and
  false-reject rates in a room with a television on are unknown.
- **The acoustic thresholds are literature-shaped priors**, not readings
  taken on this host. What is established is the asymmetry that makes them
  safe to ship, not that they are correctly placed.
- **The addressivity gate has no null.** It has never been run against a
  control condition — an equivalent gate with its evidence shuffled — so the
  ladder's structure has not been shown to beat a simpler rule.
- **Sight was measured, narrowly.** Worker up in 5.0 s, ~0.7 s per look, and
  4/4 on "how many fingers am I holding up" against *stylised* hands — not
  photographs, not a real webcam, not varying light. On abstract shapes the
  same model scored 1/2, reading five circles as four, so counting at 2B is
  approximate and fingers are a strong shape prior rather than evidence of
  general counting. Nothing here has been run against the live camera.

## Where to look

| Concern | File |
| --- | --- |
| Turn-scoped reply channel, reconciliation | `core/conversation/reply_stream.py` |
| Failures as facts | `core/conversation/failure_context.py` |
| Clause governance while streaming | `core/voice/duplex/governed_stream.py` |
| Was that meant for her | `core/voice/duplex/addressivity.py` |
| Pitch contour at turn end | `core/voice/duplex/acoustic_endpoint.py` |
| Session wiring | `core/voice/duplex/session.py` |
| Local media index, playback resolution | `core/media/` |
| Byte serving with Range | `interface/routes/media.py` |
| Looking now, and failing honestly | `core/senses/sight.py` |
| Look vs. talking about cameras | `core/senses/sight_intent.py` |
| The vision worker's model call | `core/brain/llm/mlx_vision_worker.py` |
| Ambient client, chat binding | `interface/static/voice_mode.js` |

Settings: `voice.auto_listen` turns ambient listening on; `voice.input_enabled`
and `voice.output_enabled` gate the lane entirely. `AURA_VOICE_AMBIENT`,
`AURA_VOICE_NAMES`, `AURA_VOICE_OPEN_FLOOR_S` and `AURA_MEDIA_ROOTS` tune the
rest.
