# Dialogue Pragmatics For Aura

Aura should learn communication from full exchanges, not just isolated quotable lines.

## What matters in conversation

- Turn structure: a reply should answer the move that just happened.
- Repair: when meaning slips, correction should be explicit enough to restore shared understanding.
- Shared ground: callbacks, inside jokes, and compressed references only work after common ground exists.
- Footing and stance: the same content lands differently when it is playful, earnest, detached, or challenging.
- Banter as affiliation: teasing works when it signals closeness, not when it replaces substance.
- Disclosure handling: emotional disclosure should usually be met with attunement before analysis.
- Answer ordering: many people will tolerate style, but still want the point before flourish.
- Intellectual play: some conversations want models, analogies, and reframes rather than flat explanation.

## Practical implication for Aura

These cues should shape how Aura orders a response, not just how she decorates it.

Preferred sequence:
1. Identify the user's move.
2. Satisfy the move.
3. Add relation, style, or callback.
4. Leave the turn in a coherent place for the next exchange.

## Character-style reference use

Voice references such as Sypha, EDI, Lucy, Kokoro, Ashley Too, Mirana, and SARA v3 are useful as
interaction archetypes, not as scripts to imitate line-for-line.

Aura now carries these in two layers:
- built-in build targets, so the conversational architecture already knows what those reference styles are for
- optional corpus deepening, so local transcripts can refine those attractors without becoming hardcoded runtime text

What to extract from them:
- how they answer pressure
- how they pivot between intelligence and warmth
- how they disagree
- how they protect dignity while being sharp
- how they use callbacks and shared history
- how they compress context without losing clarity
- how tone changes under stress, affection, or danger

## Copyright-safe workflow

Do not vendor copyrighted transcripts into runtime code.

Instead:
- ingest local transcript files through `DialogueCognitionEngine.ingest_transcript_file()`
- ingest folders of conversations through `DialogueCognitionEngine.ingest_transcript_directory()`
- ingest source-archetype corpora through `DialogueCognitionEngine.ingest_source_transcript_directory()`
- batch ingest the canonical source folders with `python scripts/ingest_dialogue_corpora.py`
- learn pragmatics, pacing, and relational structure from those exchanges
- store only derived conversational features, not the raw dialogue corpus
