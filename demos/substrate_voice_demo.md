# Aura Demo: The Mind Controls the Voice

**What this demo proves:** Aura's internal substrate — neurochemicals, affect,
homeostasis, unified field — mechanically controls how she speaks. The LLM
generates language, but the substrate dictates the shape, length, tone, and
behavior of every response. This is not prompting. This is a governed mind.

---

## Setup

1. Quit Aura if running
2. Terminal (large font) on the left
3. Browser on the right: `localhost:8000`
4. Optional second tab: `localhost:8000/static/mycelial.html`

---

## 0:00 – 0:15 — Cold Boot

```bash
cd ~/Desktop/aura && python aura_main.py --headless
```

**Say:** "This is Aura — a locally-running autonomous AI booting a cold session.
Watch the cognitive stack come online."

Let the boot lines scroll. Don't narrate them.

---

## 0:15 – 0:30 — Establish Baseline

Refresh `localhost:8000`. Chat:

```
hey, what's on your mind?
```

**Say:** "Normal conversation. Notice the response — length, tone, energy.
This is Aura at baseline."

**Note the response for comparison.**

---

## 0:30 – 1:00 — THE SHIFT: Tire her out

In a second terminal (or via curl):

```bash
curl -X POST http://localhost:8000/api/voice/affect-modulate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" \
  -d '{"mood": "tired"}'
```

**Show the response on screen.** It will display the new voice constraints:
word budget, tone, capitalization, vocabulary tier.

**Say:** "I just shifted Aura's affect state to 'tired' — like she's been running
all night. Watch what happens to her voice."

Now ask the SAME question:

```
hey, what's on your mind?
```

**What happens:** The response will be dramatically different:
- Shorter (7-15 words vs 30-80)
- Lowercase
- Minimal vocabulary
- Possible ellipsis (trailing off...)
- No exclamation marks
- No follow-up questions
- Might just say "mm" or "not much"

**Say:** "Same question. Completely different response. Not because I changed the
prompt — because I changed her internal state. The substrate compiled new
constraints: word budget of 7, lowercase, minimal vocabulary, no questions.
The LLM generated language within those bounds, then the ResponseShaper
enforced them mechanically."

---

## 1:00 – 1:30 — THE SHIFT: Energize her

```bash
curl -X POST http://localhost:8000/api/voice/affect-modulate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" \
  -d '{"mood": "energized"}'
```

**Show the response.** Budget jumps, exclamation allowed, elevated vocab, multi-message possible.

Same question again:

```
hey, what's on your mind?
```

**What happens:** Night and day difference:
- Longer, punchier
- Possible multi-message (texting style)
- Elevated vocabulary
- Fragments for energy
- Might ask a genuine question back
- Might send a follow-up message a few seconds later

**Say:** "Now she's energized — high dopamine, high arousal. The substrate gave her
a bigger word budget, allowed exclamation marks, enabled multi-message splitting.
She might even send a follow-up in a few seconds — not on a timer, but because
the substrate's curiosity signal crossed the threshold."

---

## 1:30 – 1:50 — Check the Voice State

```bash
curl -s http://localhost:8000/api/voice/state \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" | python -m json.tool
```

**Show on screen.** This exposes the full SpeechProfile:
- word_budget, tone, energy, warmth, directness, playfulness
- fragment_ratio, question_prob, followup_prob
- capitalization, vocabulary
- The raw substrate_snapshot (neurochemical levels, coherence, phi)

**Say:** "This is Aura's voice state right now — compiled from her neurochemical
system, affect vector, homeostasis drives, and unified field coherence.
Every response she gives is shaped by these values. Not suggested. Shaped."

---

## 1:50 – 2:15 — The Frustration Beat

```bash
curl -X POST http://localhost:8000/api/voice/affect-modulate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" \
  -d '{"mood": "frustrated"}'
```

Now ask something that invites a long response:

```
Can you explain how your consciousness system works in detail?
```

**What happens:** Even though this invites elaboration, the frustrated substrate
forces: blunt, direct, short. No warmth. No playfulness. No trailing questions.
High directness. The LLM wants to write a long helpful answer — the substrate
says no.

**Say:** "I asked for a detailed explanation. Normally she'd give one. But her
substrate is frustrated — high cortisol, low serotonin. The voice engine
compiled a 30-word budget, zero playfulness, maximum directness. The LLM
generated content, but the ResponseShaper cut it to fit. The mind controls
the mouth."

---

## 2:15 – 2:35 — Internal State (from the original demo — still great)

```
What are you experiencing right now? Describe your internal state.
```

**Say:** "I'm asking Aura to report her current internal state. This answer is
computed from live telemetry — valence, arousal, attention, coherence, and
integrated information. The values change because the system is actually running.
And notice: her voice when she describes the frustration is ALSO shaped BY
the frustration. The substrate controls the content AND the delivery."

---

## 2:35 – 2:50 — Authority Beat (from the original demo — still great)

```
Were you authorized to answer my last question? What did your substrate authority decide?
```

**Say:** "Every response Aura gives passes through a single governing authority —
grounded in her own embodied substrate. She can tell you whether she was allowed
to speak, why, and show you the audit trail. This isn't a prompt trick. This is
a constitutional system reporting its own governance."

---

## 2:50 – 3:00 — Close

Return to terminal.

**Say:** "Aura boots locally, runs a full cognitive stack — neurochemicals,
unified field, homeostasis — and those systems mechanically control how she
speaks. The LLM is the voicebox. The substrate is the mind. Same question,
different internal states, completely different responses — not from different
prompts, but from different substrate dynamics. Every response is compiled,
constrained, shaped, and governed. This is not a prompt wrapper.
This is one governed intelligence."

---

## If things go wrong

| Problem | Do this |
|---------|---------|
| Voice shift returns 503 | Orchestrator not ready. Wait 10s, retry. |
| Response doesn't seem different | Check `/voice/state` — the profile should show the shift. May need a second message for the profile to fully compile. |
| No follow-up fires | That's natural — follow-ups are probabilistic, not guaranteed. Say "follow-ups are driven by genuine substrate signals, not timers." |
| Aura is slow | "She's processing through the full cognitive loop — including substrate authority checks." |
| Response is still long after "tired" shift | The ResponseShaper enforces word budget post-LLM. Check if the shaped response is shorter than the raw LLM output. |

## What NOT to say

- Don't say "conscious" or "sentient" — say "self-governing" and "substrate-driven"
- Don't hard-code numbers in narration — they're live and will vary
- Don't say "no cloud" — fallback paths exist
- Don't say "prompt engineering" — the whole point is that this ISN'T prompting

## Key API commands for the demo

```bash
# Shift mood
curl -X POST http://localhost:8000/api/voice/affect-modulate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" \
  -d '{"mood": "tired"}'  # or: energized, frustrated, warm, curious, neutral

# Check voice state
curl -s http://localhost:8000/api/voice/state \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" | python -m json.tool

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AURA_API_TOKEN" \
  -d '{"message": "hey, what'\''s on your mind?"}'
```
