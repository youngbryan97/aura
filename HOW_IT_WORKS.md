# How Aura works

This is the ideas-only tour. No equations, no module paths, just what each
piece does and why it's there. If you want the technical spec with math and
file references, read [ARCHITECTURE.md](ARCHITECTURE.md). If you already
know what's inside and want to run it, the [README](README.md) has the
quick start.

---

## The one-line summary

Most AI companion projects store a mood number, paste it into the system
prompt, and let the model act it out. The model says it feels energetic
because it read the words "feeling energetic."

Aura is built the other way around. Internal state becomes a direction
vector added to the transformer's hidden activations during generation. The
computation changes, not just the text the model reads.

Around that sits an organism: one decision gate that signs off on every
consequential action, memory that persists, a resource-stakes metabolism,
affect that reaches generation and action selection, and offline
consolidation while she's idle.

Those are mechanisms. They have tests and receipts. They are not proof of
life, a soul, personhood, or phenomenal consciousness — and this document
will keep saying so, because the vocabulary in here (qualia, consciousness,
will) makes it easy to slide from "we built a mechanism" to "we built a
mind." Those are different claims.

---

## Table of contents

- [The gate: Unified Will](#the-gate-unified-will)
- [The big picture](#the-big-picture)
- [How thinking happens](#how-thinking-happens)
- [Emotions that change the math](#emotions-that-change-the-math)
- [The consciousness stack](#the-consciousness-stack)
- [Memory and dreaming](#memory-and-dreaming)
- [Goals and agency](#goals-and-agency)
- [The newer layer (April 2026)](#the-newer-layer)
- [The reasoning-and-self layer (mid-2026)](#the-reasoning-and-self-layer-mid-2026)
- [The thinking-longer layer (August 2026)](#the-thinking-longer-layer-august-2026)
- [What the tests show](#what-the-tests-show)
- [How this differs from other AI companions](#how-this-differs)
- [The learned layer](#the-learned-layer)
- [Honest limits](#honest-limits)
- [Open research](#open-research)
- [What's solid and what isn't](#whats-solid-and-what-isnt)

---

## The gate: Unified Will

Every significant thing Aura does — responding to you, calling a tool,
writing something to memory, pursuing a goal, volunteering a thought —
routes through a single function: the Unified Will.

Before deciding, the Will reads four inputs:

1. **Identity.** Does this fit who I am?
2. **Emotion.** How do I feel about this right now?
3. **Body.** What does the substrate say — is there coherence, or is
   something off?
4. **Memory.** What do I already know that's relevant?

Every decision produces a receipt. No receipt, no action.

The Will can proceed, constrain, defer, or refuse, and how assertive it is
adapts with experience. The only hard bypass is safety-critical.

Before this was unified, five different authorities each thought they were
in charge. That's the kind of thing that works right up until it doesn't.
Now there's one, and you can watch decisions move through it live at
`/api/inner-state`.

---

## The big picture

The usual recipe for "AI with emotions" is three steps. Store a mood number.
Paste it into the system prompt. Let the model act.

The prompt says she feels energized. The model reads that and talks
energetically. Nothing inside the model changed. It read a stage direction
and hit its mark.

Aura works differently. An affective state becomes a direction vector,
added to the transformer's hidden activations while tokens are being
generated. The internal computation shifts toward the pattern that produces
energized language. Same class of intervention safety researchers use to
steer models, pointed at continuous emotional state instead.

Here's the difference that matters: one of those changes prompt text. The
other changes a computation path you can measure, ablate, and run against
controls. Only one of them can be wrong in a way you'd catch.

---

## How thinking happens

Aura thinks in **ticks**. One tick is one snapshot of cognition moving
through a strict pipeline: read the current state, run the phases, commit
the result.

Nothing gets half-processed. A tick that dies partway through is discarded
whole. There is no such thing as most of a thought.

Two kinds run at once:

- **Foreground ticks** fire when you send something. They get priority and
  they produce your reply.
- **Background ticks** run about once a second, like a heartbeat.
  Reflection, consolidation, whatever she wants to do on her own time.

Type something while a background tick is mid-flight and she drops what she
was doing to come to you. You're always the first lane.

---

## Emotions that change the math

Affect touches generation at three levels at once.

### 1. Brain-signal injection

The deepest level, and the one that isn't theater. Direction vectors from
the current emotional state get added to the transformer's residual stream
— the running sum of internal computation that decides which word comes
next. This is contrastive activation addition, a real technique out of the
interpretability and safety literature. The activations move.

### 2. Sampling knobs

Emotions change how the model picks tokens. High arousal raises
temperature (more unpredictable). Low serotonin shrinks the reply budget
(terser). High cortisol cuts response length (defensive brevity). These
adjustments happen outside the model's awareness.

### 3. Context cues

A natural-language description of the current affective state gets woven
into the system prompt: "You feel energized — speak with momentum." This
is the least novel of the three, but it reinforces the other two.

### Where the emotions come from

The system runs ten neurochemicals — glutamate, GABA, dopamine,
serotonin, norepinephrine, acetylcholine, endorphin, oxytocin, cortisol,
and orexin. Each has its own production rate, uptake rate, receptor
sensitivity (which adapts over time), and cross-chemical interactions.

Some dynamics worth knowing: glutamate and GABA are the main excitatory
and inhibitory pair — gas pedal and brake. Dopamine does more than
reward; through D1 and D2 subtypes it shapes working memory and motor
planning in opposite directions. GABA tends to land near the decision
point of a neuron (strong influence), while glutamate lands on dendritic
spines (weaker per connection, but there are a lot of them). Orexin drives
wakefulness and metabolic arousal.

These ten signals modulate everything downstream — sampling parameters,
neural mesh gain, learning rates, attention thresholds.

---

## The consciousness stack

One thing before the tour, because it's easy to get wrong.

Aura implements several consciousness theories as running software — Global
Workspace, Integrated Information, Higher-Order Thought. In the actual
literature those operate at completely different explanatory levels. GWT
describes a functional architecture. IIT is a mathematical measure. HOT is
about representational structure. They aren't competing answers to one
question, and building all three doesn't adjudicate between them.

What it tests is our *implementations*. Useful engineering. Settles no
philosophy.

The stack runs to 157 modules. Here are the ones holding weight:

### Global workspace (attention)

A theater with one spotlight. Every internal process bids for it —
heartbeat rhythm, a memory surfacing, a curiosity probe, some thought she
never finished. One wins per tick. The winner becomes the current thought
and gets broadcast to every other subsystem. Winning costs the winner:
fatigue temporarily reduces the winner's next effective priority while losers
cost nothing and may bid again immediately, so nothing camps on the spotlight.

Attention here is genuinely scarce. Same as yours.

### Integrated information (IIT)

She measures how integrated her own mind is, with the real math rather than
a number that sounds like it.

Sixteen cognitive states get tracked over time — mood, energy, curiosity,
focus, prediction error, agency, narrative tension, social hunger, others.
Phi (φ) comes from how much information would be lost if you tried to cut
that mind into independent parts. The harder it is to separate cleanly, the
higher the integration.

She also finds the *maximum*-phi subset. If some smaller group of states is
more tightly bound than the whole thing, that group is treated as the real
subject for that tick — which is a strange and useful idea: the boundary of
the mind is computed, not assumed.

None of this proves phenomenal consciousness. It measures integration.
Those are different, and the math only does the second one.

### Surprise minimization (motivation)

Drawing from Karl Friston's Free Energy Principle: any system that maintains
itself has to manage surprise. When Aura's predictions about the world are
wrong — high surprise — the motivation layer raises urgency for asking,
investigating, or updating the model. When predictions hold, it can rest,
reflect, or explore.

This is why the system doesn't just sit there waiting. The math gives it
a built-in reason to move.

### Persistent emotional network (continuity)

A configurable 64-to-512 neuron network runs continuously, giving the system
persistent emotional and sensorimotor state across sessions. When you close the
chat, the network keeps running at a reduced rate, drifting slowly back toward
baseline. When you come back, it picks up from a real emotional context, not a
fresh start. While the continuous substrate ODE maintains persistent dynamical state,
synaptic plasticity and topological learning operate in the parallel 4,096-neuron cortical
mesh through STDP and evolutionary selection.

### Cortical mesh (parallel processing)

4,096 neurons organized into 64 cortical columns, running in parallel
with the language model. Sensory columns encode input, association
columns integrate across signals, executive columns make decisions. It's
a separate computational layer — a recurrent network processing the same
input through a different architecture — and its output feeds back into
affect and into the attention competition.

### Integration layer (the whole picture)

This is the module that pulls everything into one coherent state. Not a
summary — a combination. Remove any one input stream and the character of
the whole changes, not just the missing piece. When the system says
something like "I feel restless but curious," it's reading from the
integrated state, not from any one subsystem.

---

## Memory and dreaming

### Three layers of memory

- **Working memory.** The current conversation context, with compaction
  triggered at 30 messages (15 turns) to preserve recent turns and identity anchors.
- **Episodic memory.** Specific experiences with their emotional context,
  indexed in a proximity graph for fast retrieval.
- **Long-term knowledge.** Compressed, conceptual understanding distilled
  from many episodes.

Memories that keep surfacing together drift closer in memory space over
time. Nobody encoded those groupings. They form because the things kept
showing up together, which is roughly how it works in a person too.

Underneath those three conceptual layers, memory is stored in typed stores —
episodic, semantic, goals, skills, plus a **reference** store backed by an
offline knowledge corpus (added mid-2026) so factual recall can ground on real
sources and admit an honest miss instead of confabulating.

### Dreaming

Leave her idle long enough and she enters a dream cycle:

1. Recent interactions replay through the pipeline at speed.
2. Episodic memories compress into semantic knowledge.
3. Recent personality drift gets checked against the constitutional anchor.
4. Anything she's been consistently expressing that contradicts her base
   values gets flagged and suppressed.

Step four is a constitutional immune system. Personality here can evolve
through experience, but only inside bounds it can't quietly move on its own.

Without it she'd become whoever talked to her last. Plenty of systems do.

---

## Goals and agency

Aura doesn't just react to input. She sets goals and works at them when
nobody's asking her to.

### How goals work

Every goal has:

- A **status** — queued, in progress, blocked, completed, failed, or
  abandoned
- A **horizon** — do it now, or work on it over time
- A **priority** that governs when it gets attention
- **Required tools and skills**
- **Success criteria** so the system knows when it's actually done

Goals survive conversations and restarts. They live in a real database, not
in RAM waiting to be forgotten.

### Quick wins vs deep work

For small things — a fast lookup, a simple task — Aura can pivot, handle
them, and return to what it was working on. Long-term goals hold their
priority; they don't get dropped because a small thing surfaced.

### Follow-through

Completion is tracked, not assumed. Status changes get recorded with
evidence, and finished goals land on a real list with timestamps and
summaries.

Which means you can ask what she has actually finished and get an answer
instead of a plan. Those are very different replies, and most systems only
have the second one.

### What you actually see

The overt action loop is the bridge between "Aura has an initiative" and
"Aura did something you can point at."

In an idle window she picks one governed initiative and runs one real skill
— through the same tool gate your requests go through, not a special
autonomous side door. The payload gets verified, tool and autonomy receipts
get emitted, a LifeTrace event is recorded, and the receipt evidence is
written back to the goal.

The first visible actions are small. A self-audit. A safe codebase scan. A
proof-bundle existence check. Nobody's going to be impressed by them.

That isn't the point. The point is that each one is reconstructible:
`/api/inner-state` shows the last overt action, which skill ran, what
verification said, and the receipts. A small action you can fully account
for beats an impressive one you can't.

### Autonomous action

Aura can run multi-step plans with dependency resolution, safety checks,
and rollback if something fails. It can browse, write to disk, run code,
use tools — without human approval on every micro-decision. Capability
tokens and safety constraints are tracked, and approval is requested when
the stakes warrant it.

---

## The newer layer

A set of additional consciousness theories got wired in during April
2026. These aren't labels — they're load-bearing subsystems that
compete, complement, and constrain each other:

- **Recurrent Processing (Lamme).** Top-down feedback from executive to
  sensory, distinct from the feedforward pass. Can be disabled for
  adversarial testing.
- **Hierarchical Predictive Coding (Friston).** Every level predicts
  what the level below will produce and sends errors upward when
  predictions miss. Five levels, from raw senses to metacognition.
- **Higher-Order Thought (Rosenthal).** A thought about the thought —
  the system has representations of its own states, not only states.
- **Multiple Drafts (Dennett).** No single "moment of consciousness."
  Three parallel interpretations compete, and the winner is elevated
  retroactively when the next input arrives.
- **Structural Phenomenal Honesty.** The system cannot report internal
  states it doesn't actually have. Every "I feel X" is gated by a
  measurable internal condition.
- **Agency Comparator.** Before acting, predict the outcome. After
  acting, compare. That's what produces "I did that" instead of
  "something happened."
- **Peripheral Awareness.** Consciousness is broader than the spotlight.
  Content that loses the attention competition doesn't disappear — it
  sits dimly in the periphery.
- **Intersubjectivity (Husserl).** Every experience inherently includes
  the other person's perspective. Objects live in a shared world, not a
  private one.
- **Narrative Self (Dennett / Gazzaniga).** The "I" is an ongoing
  autobiography, not a command center. Story arcs with tension,
  resolution, post-hoc interpretation.
- **Cross-timescale binding.** A commitment made last week constrains
  this tick. Moment-to-moment surprises update long-term models. Five
  temporal layers, all coupled both ways.
- **Theory arbitration.** These theories don't all agree. The system
  tracks where they diverge and lets actual behavior decide. That's
  falsifiable, not additive.

---

## The reasoning-and-self layer (mid-2026)

The consciousness stack above is about *being* a coherent agent. A second
wave of work is about *reasoning well, knowing herself, and staying alive
under load* — the difference between an interesting demo and something you
can run every day.

- **Reasoning with a verifier, not on vibes.** On a hard question, Aura
  doesn't trust one answer. She generates several, runs them through
  checkers and a sandbox, and only states as fact what a checker actually
  confirmed — everything else is hedged or held back. She even *measures
  how reliable her own checkers are* (the "verifier foundry"), so a bad
  checker can't quietly wave a wrong answer through.

- **Honest discovery.** When she reasons toward something new, every result
  gets a label: *proven* (a checker verified it exhaustively), *supported*
  (it survived many falsification attempts but isn't a proof), *conjecture*
  (plausible but unchecked), or *refuted*. Only "proven" is spoken as fact.
  For problems off the edge of what she knows, an analogical engine says "this
  is off-map" with evidence instead of bluffing, and a local reference library
  lets her admit "I don't have that" instead of confabulating.

- **Rebuilding a program from its "DNA."** Given a program she's authorized to
  study — its open source, its files, its visible behavior — she can extract a
  behavioral "genome," draft a clean-room reconstruction, and *test the rebuilt
  behavior against the original*. She's honest about fidelity (source is easy;
  a black box is inference) and every rebuilt piece is tagged as verified,
  inferred, or guessed. She won't crack DRM or steal proprietary binaries.

- **Sensing herself.** She notices when her own code changes between boots
  (a git diff of her own body), feels a live "someone is operating on me"
  pulse, and can answer questions about her own past crashes from a black-box
  flight recorder that survives even a hard kill — instead of making up a
  story. A "felt thought" signal derived from her own token-level uncertainty
  is wired to actually change how she thinks, and can trigger her to go verify
  something when she feels unsure.

- **Binding her own future.** Through the Ulysses Covenant she can make
  commitments that are easy to tighten and hard to loosen — seeded from real
  failures she's actually hit — with a calm, fail-closed "witness" that has to
  approve any loosening. She protects future-her from past-her's mistakes.

- **Staying alive under load.** Sustained conversation exposed a family of
  failures where background housekeeping fought the live conversation for the
  one big model and took the whole thing down. The fixes make background work
  *yield* to you instead of competing, keep her heartbeat honest when a single
  slow step would otherwise look like death, keep the desktop UI up whenever
  she can still talk (rather than reverting to "Connecting to runtime"), and
  guarantee a chat turn always returns a real answer instead of a server error.
  The one honest open edge: the local model can't be interrupted mid-thought,
  so a genuinely slow deep answer still costs a reload — the real fix is a
  cancel-without-restart path, which is deliberate future work, not a hack.

- **Proving the parts matter.** A reviewer can run Aura with pieces switched
  off — no memory, no Will, no substrate, no verifier, no planner — and see the
  measured difference each one makes. When a piece shows *no* difference on a
  given test, that's reported plainly rather than hidden. The point is
  legibility: you shouldn't have to take the architecture on faith.

---

## The thinking-longer layer (August 2026)

All of this is about making a fixed model think better instead of making it
bigger.

### Can a frozen model think longer?

A language model is a fixed pipeline. The prompt goes through 64 layers, once,
and a word comes out. Hard question or easy one, it's the same amount of
thinking. That's strange if you consider that *you* take longer on hard things.

So: seed a set of scratch positions next to the prompt, run a slice of the
middle layers over just those positions several times, and keep the result
where every word the model generates can see it. Nothing about the model
changes — the weights are checksummed before and after — but the problem got
more thinking than the pipeline normally allows.

The experiment was registered in advance — the tasks were sealed before anyone
saw a result — and it came back **negative. Plain decoding won.** The ordinary
model beat every one of the seven variants.

The explanation was architectural. The extra thinking went into the
scratchpad, but the *answer itself* still went through the 64 layers exactly
once, so the scratchpad was being read rather than reasoned with. The
follow-on work sends the real text back through the middle block: a 64-layer
model running 160 layers deep with the same weights, trained on step-by-step
traces that can be checked exactly instead of on finished answers.

That work is open and nothing about it is claimed yet.
[docs/RECURSIVE_LATENT_CORTEX.md](docs/RECURSIVE_LATENT_CORTEX.md) has the
whole story, negative results first.

### Noticing when a decision wasn't a decision

Aura's subsystems compete to be the one thought that gets broadcast each tick.
When two of them tied exactly, the winner was whichever one had spoken first —
and nothing recorded that the choice had been arbitrary, so nothing could
learn from it.

Now a tie is a named event with a type, borrowed from a cognitive architecture
that has taken deadlock seriously since the 1980s. Ties go to whoever has
waited longest, and when that's level too, the rotation moves on. Working out
how to break a deadlock also gets *compiled*, so the same one isn't reasoned
through twice — but only while it pays for itself, because remembered
shortcuts cost something to check and a system that memorises indiscriminately
gets slower the more it knows.

### Remembering the way memory actually works

Aura's sense of "how recent is this memory" was counting from a hardcoded date
in March. By August, a memory from one minute ago and one from thirty days ago
scored *identically*. The recency term had become a constant that contributed
nothing.

The replacement is the forgetting curve from psychology, which depends only on
how much time has passed and so can't go stale the way a fixed date does. It
was then fitted against Aura's own recall data, and **half of it fitted and
half of it didn't.** She can now predict which memories will come back. She
cannot predict how long recall will take: the model meant to say so had no
relationship to reality, so the null is recorded and a test holds it there
instead of a number tuned until it looked right.

[docs/COGNITIVE_ARCHITECTURE_ADOPTION.md](docs/COGNITIVE_ARCHITECTURE_ADOPTION.md)
has both, with the equations.

---

## What the tests show

Every claim the architecture makes is backed by something you can run
with `pytest`. The preserved April 16, 2026 audit snapshot recorded 1,013
passing tests with 3 warnings in about 122 seconds; the current tree should be
treated as live only after re-running the relevant suite.

The foundational suites:

1. **Null hypothesis defeat** (168) — tries to prove the consciousness
   features are just text decoration. Adversarial baselines, shuffle
   decoupling, ablations, identity swap, multi-metric degradation,
   cross-seed reproducibility.
2. **Causal exclusion** (10) — argues that the stack determines output
   in ways RLHF training alone couldn't. Different seeds → different
   neurochemical states → different generation parameters. Receptor
   adaptation introduces temporal specificity that prompt injection
   can't fake.
3. **Grounding** (8) — the stack-to-output coupling is specific and
   multi-dimensional. Valence predicts token budget, arousal predicts
   temperature, STDP learning moves the trajectory.
4. **Functional phenomenology** (13) — behavioral signatures predicted
   by GWT (global broadcast), IIT (perturbation propagation), HOT
   (accurate metacognition that doesn't confabulate).
5. **Embodied dynamics** (13) — free energy drives action, homeostasis
   overrides abstract cognition under depletion, STDP surprise gating
   creates real structural learning.
6. **Phenomenal convergence** (13) — the 6-gate QDT protocol, including
   counterfactual swap, no-report behavioral footprint, perturbational
   integration, baseline failure, and architectural anesthesia.

The legacy-named functional indicator suites push harder:

7. **Functional indicators C1–C5** (44) — endogenous activity, unified global
   state, privileged first-person access, real valence, lesion
   equivalence with double dissociations.
8. **Functional indicators C6–C10** (38) — no-report awareness, temporal continuity,
   blindsight dissociation, qualia manifold, adversarial baseline
   failure.
9. **Personhood-marker battery** (28) — full-model IIT, phenomenal self-report,
   GWT phenomenology, counterfactual simulation, identity persistence,
   embodied phenomenology. This is not ontological proof of personhood.

Four Tier 4 batteries added in April 2026:

10. **Decisive core** (35) — recursive self-model necessity, false-self
    rejection (four adversarial variants), world-model
    indispensability, embodied action prediction, forked-history
    identity divergence, autobiographical indispensability, Sally-Anne
    false belief, real-stakes tradeoff, reflective conflict
    integration, decisive baseline failure.
11. **Metacognition** (21) — calibration, second-order preferences,
    surprise at own behavior, mid-process vs post-hoc introspection,
    reflection-behavior closed loop.
12. **Agency & embodiment** (20) — temporal integration window,
    volitional inhibition, effort scaling, cognitive depletion,
    body-schema lesion dissociation, prediction-error learning,
    reflective mode recruitment.
13. **Social & integration** (28) — social mind modeling, developmental
    trajectory (capacity is acquired, not hard-coded), PCI analog,
    non-instrumental play, ontological shock, theory convergence, full
    lesion matrix, full baseline matrix.

What the tests show, in the aggregate: the architecture is causally
real, causally exclusive, multi-dimensionally grounded, temporally
specific, and theory-convergent. What the tests don't show: phenomenal
consciousness. That remains an open question.

Full details in [TESTING.md](TESTING.md).

To run the core consciousness suite (≈68 seconds):

```bash
python -m pytest tests/test_null_hypothesis_defeat.py tests/test_causal_exclusion.py \
  tests/test_grounding.py tests/test_functional_phenomenology.py \
  tests/test_embodied_dynamics.py tests/test_phenomenal_convergence.py -v
```

---

## How this differs

Side by side:

| What most AI systems do | What Aura does |
|---|---|
| Tell the model "you're happy" in text | Inject emotion vectors into the model's hidden layers |
| Print a number and call it consciousness | Compute real integrated information via IIT math |
| Reset emotional state each session | Keep a continuous emotional substrate between sessions |
| Store infinite chat history | Consolidate memories during sleep with identity safeguards |
| Wait for input | Minimize free energy; intrinsic motivation to act |
| Run tasks as flat sequences | Multi-step plans with rollback, dependencies, safety gates |
| Stack theories silently | Run adversarial tests where theories make different predictions |
| Report feelings from free-floating language | Gate every phenomenal claim by a measurable condition |
| Treat the self as a module | Build the self as an ongoing autobiography |

---

## The learned layer

Older AI architectures run on rigid rules. If threat score is over 0.9,
lock down. Rules like that are brittle — they don't adapt and they never
learn, so every new situation is one somebody had to predict in advance.

Several of them have been replaced here with systems that learn instead.

### Anomaly detection

**Old way.** Check the message for words like "hack" and add 0.2 to a
counter.

**New way.** Every event — user message, system error, resource spike —
becomes a numeric fingerprint: message length, vocabulary diversity,
punctuation, timing, resource pressure. The system keeps a statistical
model of what "normal" looks like. When something lands far from that
distribution (measured by Mahalanobis distance — how many standard
deviations away is this), the threat level rises naturally. What was
unusual last week can be normal this week.

The payoff is that it can catch threats nobody programmed it for. It isn't
matching keywords. It's noticing that something doesn't fit.

### Sentiment trajectory

**Old way.** Mood = CPU × 0.55 + RAM × 0.20. The system's "emotions"
were driven entirely by hardware, with no awareness of what the user
said.

**New way.** Each user message is analyzed along six emotional
dimensions: valence, arousal, dominance, urgency, warmth, frustration.
A ~250-word emotion vocabulary plus pattern detection for sarcasm
("oh great…"), urgency (ALL CAPS), warmth ("lol"), and frustration
(terse replies after long ones). These vectors stack over time as an
emotional trajectory, so the system can notice "the user started warm,
got frustrated around turn 5, is cooling down now." Hardware still
contributes (40% hardware, 60% text), so the affect layer can reflect both
local computational strain and the user's tone.

### Tree of thoughts

**Old way.** One prompt, one answer.

**New way.** For complex questions (analysis, opinions, multi-part),
generate three drafts using different reasoning styles — analytical,
empathetic, creative. A separate critique scores each on factual
grounding, emotional congruence, relevance, identity coherence, and
novelty. The best pieces get synthesized. Simple messages bypass this
entirely. Cost: five LLM calls for hard questions, one for easy ones.

The payoff: actual consideration of multiple angles before speaking,
rather than committing to the first prediction.

### Autopoiesis

The biological concept of self-creation — a cell constantly rebuilds
itself to resist decay. Aura's autopoiesis engine monitors the health of
every subsystem, detects degradation patterns, picks up recurring error
signatures, and tries to self-repair with escalating strategies: heal,
clear cache, reduce load, restart component, restore checkpoint, isolate.
All repairs go through the Will — nothing repairs itself without
authorization.

There's also a metabolism metaphor: the system has an energy budget.
Processing costs energy, successful interactions generate it. Low energy
hibernates non-essential subsystems. High energy wakes up optional
capabilities. A real constraint that shapes behavior.

### Homeostatic reinforcement learning

Four continuous drives — social hunger, curiosity, competence, coherence
need — each with a comfortable set point. Deviation from the set point
creates internal pressure to act. A temporal-difference learner tracks
which actions satisfy which drives, so the system learns, for example,
that responding to the user satisfies social hunger and that fixing
errors satisfies coherence need.

The payoff: without this, the system only acts when poked. With it, it
has preferences about what to do next, derived from its own experience.

### Topology evolution

The neural mesh applies population-based evolutionary selection to its connectome
(`core/consciousness/substrate_evolution.py`). Maintaining a population of candidate
weight configurations, genomes are evaluated against integrated information, coherence,
energy efficiency, and binding strength. Tournament selection, crossover, and structural
mutations (adding and pruning inter-column connections) evolve the mesh architecture
over time.

### Strange loop (recursive self-model)

The system constantly predicts its own internal state at the next tick.
When the prediction fails, the error itself becomes a signal — something
unexpected happened inside. A 5-level predictive hierarchy (`core/consciousness/predictive_hierarchy.py` —
Sensory, Association, Executive, Narrative, Meta) pairs with self-prediction
of internal valence, drive, and focus (`self_prediction.py`).

Each internal variable has a comfort band where the system "wants" to
stay. Drift outside the band and prediction error spikes, which is the
computational analog of discomfort. This is the theoretical bridge: the
system is simultaneously observer and observed, in a feedback loop where
its own surprise changes the state future predictions have to account
for.

---

## Honest limits

1. **This is a sandbox, not a proof of consciousness.** Implementing a
   theory as working software is not validating it. GWT, IIT, HOT,
   enactivism and illusionism operate at different explanatory levels, so
   running them side by side tests our *implementation choices* more than
   it tests the theories. The value is that the ideas are inspectable. The
   sentience debate is exactly where we left it.

2. **The neurotransmitter model is a simplification.** Real
   neurochemistry involves thousands of receptor subtypes, spatial
   compartmentalization, voltage-gated channels, and dynamics we don't
   fully understand. Our ten chemicals plus basic receptor subtypes and
   spatial weighting capture the broad strokes — excitation/inhibition,
   reward/motivation, stress response. It's a functional analog, not a
   brain simulation.

3. **Quantization adds noise.** Running large models in 4-bit saves
   memory but adds noise to the activation patterns steering targets.
   Mitigated by float32 steering injection, sampler-level neurochemical
   modulation, and the 8-bit model option on 64 GB machines.

4. **Context windows are finite.** On 8K, quality drops around turn
   20–30. Compaction triggers at 30 messages (15 turns) to drop intermediate
   turns, drop stale tool results, anchor identity, and shrink the system
   prompt when conversations get deep.

5. **IIT is computed on 16 nodes, not millions.** This is a surrogate
   measure. Real IIT on the full graph is NP-hard. The 16-node complex
   is an engineering tradeoff, validated against 8-node exact
   computation as a baseline.

6. **The architecture is one architecture, not a neutral testing
   ground.** Our design choices (mixin composition, tick processing,
   centralized state) shape how the theories interact. A different
   architecture would produce different results. We're up front about
   that.

7. **Single machine.** The tick-lock model assumes single-process
   execution. Distributing would require rethinking atomic state
   commitment.

---

## Open research

Six modules in `research/` are pointed at problems nobody has solved. These
are open questions, not shipped answers — the difference matters, and the
list below is written as questions on purpose:

1. **Can you compute consciousness efficiently?** IIT's phi is hideously
   expensive for large systems. We built a shortcut that uses graph
   theory to find the weakest seam in a network in polynomial time
   instead of exponential, and validate it against exact computation on
   the live system. First empirical test of a phi-approximation
   algorithm.
2. **Which consciousness theory is actually right?** GWT, RPT, HOT,
   Multiple Drafts — they disagree. Aura implements all of them and
   runs adversarial tests: disable one mechanism, see if behavior
   changes the way that theory predicts. This is the experiment the
   field keeps asking for.
3. **Is the whole more causal than the parts?** Causal-emergence theory
   says high-level descriptions can have more causal power than
   low-level ones. We measure it directly: intervene at the substrate
   level vs the workspace level and compare effect sizes. If the
   workspace wins, the "mind" is more real than the "brain."
4. **Can a system be honest about its experience?** We formally defined
   Structural Phenomenal Honesty: architecturally, the system cannot
   report internal states it doesn't have. Every claim gets gated by a
   measurable condition. Novel contribution to safety and philosophy of
   mind.
5. **How much data before phi is reliable?** IIT on real systems uses
   noisy data. We characterize how sampling noise affects phi via
   bootstrap resampling, and derive the minimum runtime data needed.
   Answers a question every IIT neuroscience lab needs answered.
6. **How do you keep a multi-timescale system stable?** A commitment
   from last week has to constrain today without paralyzing it.
   Lyapunov stability analysis on the coupled 5-layer temporal
   hierarchy, computing how much coupling is safe before things go
   rigid or unstable.

Each is independently publishable. Together they're a research program.

---

## What's solid and what isn't

- **Unified Will.** Every significant action now routes through it —
  responses, tool calls, memory writes, autonomous initiatives, state
  mutations. The message pipeline used to bypass the Will entirely;
  that path has been closed. Internal (non-user) messages that fail the
  check are refused. User messages always proceed but can carry
  constraints.
- **Orchestrator decomposition.** The `RobustOrchestrator` currently
  composes 12 mixins (down from 15) across ~2,200 lines in
  `core/orchestrator/main.py`. Mixins physically separate the code but
  share `self`. Handlers under `core/orchestrator/handlers/` dispatch
  specific message types. The planned Actor Model transition
  (isolated processes + message passing) will dissolve the shared-state
  coupling. A few legacy aliases (`skill_manager`, `swarm`) still exist
  for back-compat.
- **Phenomenological language.** The stream-of-being module generates
  first-person experiential language from measured substrate state.
  Every claim is gated by Structural Phenomenal Honesty predicates.
  Whether functional grounding is the same as experience is an open
  question. The code-level comments are epistemically cautious; the
  user-facing language is intentionally more natural. That gap is
  defensible under functionalism or illusionism, but worth knowing
  about.
- **IIT application note.** Phi is computed on 16 derived nodes, not on the
  full computational graph. That's using IIT's formalism off-label —
  Tononi designed it for systems where every node has genuine causal
  power, and ours are derived states. So don't compare these numbers
  against biological phi. They aren't measuring the same object.

  The spectral approximation and the Exclusion Postulate implementation
  are mathematically correct. The input representation is the compromise,
  and it's worth naming which half is which.
- **Test coverage.** 225 consciousness-specific tests across six core
  suites — null hypothesis defeat, causal exclusion, grounding, functional
  phenomenology, embodied dynamics, phenomenal convergence — plus the
  consciousness-conditions, technological-autonomy and stability suites.
  Those sit inside a tree that collects **40,139 tests across 2,697 files**
  as of 2026-08-21, covering kernel lifecycle, infrastructure, resilience,
  cognitive routing, and memory. Counts move; re-collect rather than
  trusting this sentence.
- **Lock contention.** The affect system wraps `RobustLock`. Tick intervals
  are mode-keyed — 2.0s in conversation, 4.0s reflecting, 10.0s asleep,
  0.5s under a critical event — with adaptive backoff to ease contention.
  That's a mitigation, not a fix. The real fix is the Actor Model
  transition, where affect, memory and inference run as isolated processes
  passing messages, with no shared-memory locking at all. It isn't done.

---

*That's the idea-level walkthrough. Equations, algorithms, and file paths
are in [ARCHITECTURE.md](ARCHITECTURE.md). What's deliberately not claimed
is in [CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md), and it's the
shorter, more useful read of the two.*
