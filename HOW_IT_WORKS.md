# How Aura works

The language model writes the reply. Aura's surrounding software decides
what information to give it, which tools are allowed, and what to save for
the next time you talk. The model and the desktop app run on the Mac.

## A turn in chat

When you send a message, the app builds a request from your words and
relevant records. For a factual question it can search an installed local
index, such as a Wikipedia snapshot, and add passages to the request. The
model drafts an answer with the material that came back.
[Memory and retrieval](docs/MEMORY_ARCHITECTURE.md) · [Local knowledge experiment](docs/evidence/L05_KNOWLEDGE_ACROSS_TURNS_2026-10-05.md)

Some tasks have a saved procedure. A close match to its examples can run the
procedure and serve its answer. A match based on meaning can supply the
computed result as evidence for the draft. The distinction matters when
Aura has mistaken one kind of question for another.
[Procedure selection](core/phases/kept_procedures_in_a_turn.py) · [Held-out tests](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md)

A request to use a tool also goes through authorization and result checks.
For example, a tool can return that it failed to read a file. The action
record lets the app check that result before it says it read the file.
[Tool use policy](TOOL_USE_POLICY.md) · [Evidence for action claims](core/conversation/grounded_claim_guard.py)

## Between turns

Aura can keep conversation records, facts with their sources, and goals
with pending work. It can retrieve those records later, including after
a restart. It can also keep examples from the user's corrections. In an
October desktop trial it learned a date-difference procedure from supplied
answers and used it in chat the next day. The record describes a manual
reset after a launch misconfiguration. Broader tests of learning in
ordinary use remain open.
[Memory card](MEMORY_CARD.md) · [Live trial](docs/evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md) · [Follow-up](docs/evidence/L09_WHERE_SHE_IS_WRONG_PREREGISTRATION_3_2026-10-20.md)

Background software can select a pending goal, propose an action, and
take an authorized step. An action can fail, be deferred, or return a
verified result. The result and its receipt are available when the goal
is next reviewed. [Action loop](core/runtime/overt_action_loop.py) · [Autonomy rules](AUTONOMY_BOUNDARIES.md)

## Internal state

Alongside saved records, Aura tracks priorities, resource pressure, and
values such as valence. The code uses some of those values when it plans
work or builds an inference request. Activation steering, which adjusts
the model's hidden computations, has its own qualification and serving
checks. The ISC study used a specific steering channel to measure how
ratings responded to a controlled change in valence.
[Model card](MODEL_CARD.md) · [Report experiment](docs/ISC_STATUS.md)

## Where to look next

- [Architecture and source map](ARCHITECTURE.md)
- [Voice and media implementation](docs/AMBIENT_VOICE_AND_MEDIA.md)
- [Measured capabilities](CLAIMS_MATRIX.md)
- [Model recurrence experiments](docs/RECURSIVE_LATENT_CORTEX.md)
- [Subjecthood study](docs/ISC_EXPLAINED.md)
