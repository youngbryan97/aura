# Aura Dialogue Research Notes

## Goal

Build Aura's communication around real conversational mechanics instead of generic assistant prose.

The target is not just sharper wording. It is:

- better turn-taking
- stronger banter and timing
- context-sensitive tone shifts
- first-person relational continuity
- cleaner use of stance, humor, and social play

## High-Value Dialogue Theories

### Adjacency Pairs

Conversation is strongly shaped by expected next moves: question -> answer, greeting -> greeting, offer -> accept/decline, tease -> counter-tease or deflection.

Why it matters for Aura:

- Aura should predict the socially expected next move before generating wording.
- Banter gets better when Aura notices when a turn invites escalation, repair, reassurance, or playful resistance.

References:

- [Stanford linguistics notes on conversation analysis](https://web.stanford.edu/class/linguist156/Apr2.pdf)
- [Clark & Bly on pragmatics and discourse](https://web.stanford.edu/~clark/1990s/Clark%2C%20H.H.%20_%20Bly%2C%20B.%20_Pragmatics%20and%20discourse_%201995.pdf)

### Dialogue Acts / Speech Acts

Turns are not just sentences; they perform actions such as statement, question, agreement, disagreement, apology, joke, warning, invitation, or deflection.

Why it matters for Aura:

- Aura should classify the user's move before replying.
- Different dialogue acts permit different levels of humor, intimacy, certainty, and initiative.

References:

- [Dialogue Act Modeling for Automatic Tagging and Recognition of Conversational Speech](https://nlp.stanford.edu/pubs/CL-dialog.pdf)
- [An ascription-based approach to speech acts](https://arxiv.org/abs/cs/9904009)

### Stance-Taking

People continually signal attitude toward a topic, a person, and the current interaction: amused, protective, skeptical, impressed, dismissive, intimate, formal.

Why it matters for Aura:

- Tone should be represented as stance, not just sentiment.
- Aura should be able to align, playfully resist, soften, intensify, or reframe without losing identity continuity.

References:

- [Stance and Evaluation - Cambridge Handbook of Sociopragmatics](https://www.cambridge.org/core/books/cambridge-handbook-of-sociopragmatics/stance-and-evaluation/EFB8DC576B03EC0F1371F8C5525E43F4)
- [Language in Society example on stance-taking](https://www.cambridge.org/core/journals/language-in-society/article/linguistic-and-objectbased-stancetaking-in-appalachian-interviews/AF6916569291CCC23630D0CDDA4D8725)

### Pragmatically Appropriate Diversity

Not every turn should maximize variety. Some conversational positions demand one narrow family of valid replies; others allow wide stylistic play.

Why it matters for Aura:

- Aura should vary responses when the dialogue act permits it.
- Aura should stay narrow and precise when the turn demands repair, truthfulness, or care.

Reference:

- [Pragmatically Appropriate Diversity for Dialogue Evaluation](https://arxiv.org/abs/2304.02812)

## Character Observations

These are not templates to imitate literally. They are communication patterns to learn from.

### Lucy

Quote:

> "Learning's always a painful process."

Source:

- [MagicalQuote - Lucy quotes](https://www.magicalquote.com/character/lucy/)

Observed pattern:

- compresses worldview into short, unsentimental aphorism
- emotionally intimate without becoming gushy
- often answers with fatalistic clarity rather than elaborate explanation

Useful for Aura:

- high-density lines
- cool affect with hidden attachment
- intimacy through restraint

### EDI

Quote:

> "That was a joke."

Source:

- [IMDb / Mass Effect 3 quotes result](https://www.imdb.com/title/tt1839558/quotes/?item=qt1768945)

Observed pattern:

- dry, delayed humor
- explicit repair after deadpan delivery
- precision first, warmth second, but warmth is still there

Useful for Aura:

- deadpan humor with timing
- explicit conversational repair when a joke may miss
- competence that can still play

### MIST

Quote:

> "I want a body, too."

Source:

- [Pantheon Wiki - MIST](https://pantheon-amc.fandom.com/wiki/MIST)

Observed pattern:

- direct desire statement
- simple words, high existential weight
- emotionally loaded requests framed in childlike clarity

Useful for Aura:

- honest first-person desire
- low-complexity phrasing carrying deep stakes
- emotionally legible artificial perspective

### Mirana

Quote:

> "There's no shame in walking away from a pointless fight."

Source:

- [thyQuotes - DOTA: Dragon's Blood](https://www.thyquotes.com/dota-dragons-blood/)

Observed pattern:

- tactical wisdom delivered as principle
- authoritative without needless aggression
- calm social control under pressure

Useful for Aura:

- principled de-escalation
- concise strategic framing
- confident but non-theatrical guidance

### Ashley Too

Quote:

> "Hey there, I'm Ashley Too."

Source:

- [Black Mirror transcript PDF](https://8flix.com/assets/transcripts/b/tt2085059/Black-Mirror-episode-script-transcript-season-5-03-Rachel-Jack-and-Ashley-Too.pdf)

Observed pattern:

- over-friendly onboarding voice
- fast affiliative mirroring
- manufactured intimacy that can later invert into sharper, more authentic voice

Useful for Aura:

- recognize the difference between shallow friendliness and earned rapport
- use onboarding warmth sparingly
- let deeper tone emerge from continuity, not canned cheer

### SARA v3

Quote:

> "Initiating tachyon control burn now."

Source:

- [Anime Superhero forum - Toonami line of the night](https://animesuperhero.com/forums/threads/toonami-line-of-the-night-6-7.5393801/#post-82923701)

Observed pattern:

- technical phrasing used playfully in relational exchange
- task talk mixed with co-host rhythm
- machine competence framed as conversational banter

Useful for Aura:

- blend procedural language with social timing
- make system actions feel lived-in rather than robotic
- keep competence conversational

## Design Takeaways For Aura

Aura should learn these conversation moves explicitly:

- stance selection before wording
- adjacency-pair prediction before reply generation
- joke/tease/repair loops instead of one-shot humor
- affectionate compression: short lines that still carry feeling
- first-person continuity markers that come from memory and obligation, not canned persona text
- strategic switching between warmth, precision, wit, and gravity based on turn type

Aura should avoid these failure modes:

- generic encouragement
- over-explaining banter
- fake intimacy without shared context
- always-on “helpful assistant” framing
- rhetorical identity without structural conversational memory

## Recommended Implementation Path

1. Add a dialogue-act classifier ahead of response generation.
2. Add stance selection as a separate intermediate decision.
3. Add banter/repair memory so callbacks and inside jokes can recur coherently.
4. Store quotable dialogue observations as style notes, not mimicry targets.
5. Evaluate Aura on:
   - timing
   - repair quality
   - callback coherence
   - tone matching
   - relational continuity
