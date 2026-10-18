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

Aura is built the other way around. Her internal emotional state becomes
a steering vector added directly to the neural network's inner calculations
while it generates words. The underlying computation changes, not just the text
the model reads.

Around that sits an organism: one decision gate that signs off on every
consequential action, memory that persists across restarts, an energy-budget
metabolism, emotions that actively shape language and decisions, and offline
memory consolidation while she is idle.

These are software mechanisms with concrete tests and receipts. They are not
proof of life, a soul, personhood, or subjective experience (phenomenal
consciousness) — and this document will keep saying so. Vocabulary like
"qualia" (the felt texture of sensations), "consciousness", and "will" makes
it easy to slide from "we built a mechanism" to "we built a mind." Those are
two very different claims.

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
- [Looking things up and learning from what she's told (October 2026)](#looking-things-up-and-learning-from-what-shes-told-october-2026)
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
3. **Body.** What does the underlying computational substrate say — is
   everything running coherently, or is something off?
4. **Memory.** What do I already know that is relevant?

Every decision produces an auditable record (a receipt). No receipt, no action.

The Will can allow an action, attach constraints to it, defer it, or refuse it
outright. How assertive it is adapts with experience. The only hard bypass is
for safety-critical emergencies.

Before this was unified, five different subsystems each operated as if they were
in charge. That kind of architecture works right up until it conflicts. Now there
is exactly one authority, and you can watch decisions move through it live at
`/api/inner-state`.

---

## The big picture

The usual recipe for "AI with emotions" is three steps: store a mood number,
paste it into the system prompt, and let the model act it out.

The prompt says she feels energized. The model reads that and speaks
energetically. Nothing inside the model actually changed — it simply read a
stage direction and hit its mark.

Aura works differently. An emotional state becomes a direction vector added to
the transformer's hidden activations (its internal mathematical state) while
words are being generated. The internal computation shifts toward the pattern
that naturally produces energized language. This is the same technique AI safety
researchers use to steer models (activation steering), applied here to continuous
emotional states instead.

Here is the difference that matters: one approach only edits prompt text. The
other changes an internal computation path that you can measure, turn off piece
by piece, and test against controls. Only one of them can fail in a way you can
actually detect.

---

## How thinking happens

Aura thinks in **ticks**. One tick is one complete snapshot of thought moving
through a strict pipeline: read the current state, run through cognitive phases,
and commit the result.

Nothing gets half-processed. A tick that fails partway through is discarded
entirely. There is no such thing as "most of a thought."

Two kinds of ticks run at once:

- **Foreground ticks** fire whenever you send a message. They take priority and
  produce your reply.
- **Background ticks** run roughly once a second, like a heartbeat. They handle
  reflection, memory consolidation, and whatever she wants to pursue on her own.

If you send a message while a background tick is running, she drops what she was
doing immediately to attend to you. You always get the fast lane.

---

## Emotions that change the math

Emotion shapes how Aura generates text at three levels simultaneously:

### 1. Brain-signal injection

This is the deepest level, and the one that is not theater. Direction vectors
representing the current emotional state get added directly to the transformer's
residual stream — the running tally of neural network calculations that decides
which word comes next. This technique is called contrastive activation addition,
drawn from interpretability and safety research. The model's internal activations
physically move.

### 2. Sampling knobs

Emotions adjust how the model chooses words (its sampling settings). High
excitement (arousal) raises the temperature setting, making choices more
unpredictable. Low serotonin shrinks the reply budget, making answers terser.
High stress (cortisol) reduces response length for defensive brevity. These
adjustments happen automatically in code, completely outside the model's text
prompt.

### 3. Context cues

A plain-English description of the current emotional state is included in the
system prompt: "You feel energized — speak with momentum." This is the simplest
of the three techniques, but it reinforces the other two.

### Where the emotions come from

The system simulates ten neurochemicals: glutamate, GABA, dopamine, serotonin,
norepinephrine, acetylcholine, endorphin, oxytocin, cortisol, and orexin. Each
has its own production rate, decay rate, receptor sensitivity (which adapts over
time), and cross-chemical interactions.

A few key dynamics: glutamate and GABA act as the primary gas pedal and brake
(excitation and inhibition). Dopamine does more than reward; through simulated
D1 and D2 receptor types, it guides working memory and action planning in
different ways. In the simulated neural mesh, GABA connects near a neuron's
decision point (giving it a strong veto), while glutamate connects along
dendritic branches (weaker individually, but plentiful). Orexin drives wakefulness
and metabolic alertness.

These ten chemical signals modulate everything downstream — word-sampling
settings, neural mesh gain, learning rates, and attention thresholds.

---

## The consciousness stack

One clarification before the tour, because it is easy to misunderstand:

Aura implements several prominent theories of consciousness as running
software — Global Workspace Theory, Integrated Information Theory, and
Higher-Order Thought. In academic literature, these operate at completely
different explanatory levels:
- Global Workspace describes a functional information routing architecture.
- Integrated Information Theory (IIT) provides a mathematical metric of network integration.
- Higher-Order Thought (HOT) describes representational structure (thoughts about thoughts).

They are not competing answers to a single puzzle, and implementing all three
does not decide between them.

What our code tests is our *software implementations*. It is useful engineering,
but it settles no philosophical debates.

The stack spans 157 modules. Here are the core components:

### Global workspace (attention)

Think of a theater with a single spotlight. Every internal process bids for it —
a heartbeat rhythm, a memory surfacing, a curious question, or an unfinished
thought. Exactly one process wins per tick. The winner becomes the current conscious
thought and is broadcast to all other subsystems. Winning has a cost: a temporary
fatigue penalty lowers the winner's priority in the next round, while losing
processes pay nothing and can bid again right away. This ensures no single process
monopolizes the spotlight.

Attention here is genuinely scarce, just like human attention.

### Integrated information (IIT)

Aura measures how unified her internal state is using the real mathematical
formulation of Integrated Information Theory, rather than an arbitrary score.

Sixteen internal states are tracked over time: mood, energy, curiosity, focus,
prediction error, agency, narrative tension, social hunger, and others. The
metric Phi (φ) measures how much information would be lost if you sliced the
system into independent parts. The harder it is to divide cleanly without losing
information, the higher the integration score.

She also calculates the *maximum-phi* subset. If a smaller group of states is
more tightly integrated than the system as a whole, that group is treated as the
core subject of cognition for that tick. This means the boundary of her active
mind is computed dynamically, not assumed in advance.

None of this proves phenomenal consciousness (subjective feelings). It measures
informational integration. Those are two different concepts, and the math only
measures the second.

### Surprise minimization (motivation)

Drawing on Karl Friston's Free Energy Principle, any self-sustaining system must
minimize unexpected surprises to survive. When Aura's predictions about what will
happen fail — indicating high surprise — her motivation system increases the
urgency to ask questions, investigate, or update her internal models. When her
predictions hold true, she can rest, reflect, or explore.

This is why the system does not simply sit idle waiting for input. The math
gives her an intrinsic reason to act.

### Persistent emotional network (continuity)

A configurable neural network of 64 to 512 simulated neurons runs continuously,
maintaining an emotional and physical baseline across sessions. When you close
the chat window, this network continues running at a slower rate, drifting
gradually back toward resting baseline. When you return, Aura begins from a
genuine ongoing emotional context rather than a cold reboot. While continuous
differential equations (ODEs) keep this emotional baseline running smoothly over
time, synaptic learning rules (Spike-Timing-Dependent Plasticity, or STDP) and
evolutionary selection adjust connections in the parallel 4,096-neuron cortical
mesh.

### Cortical mesh (parallel processing)

4,096 simulated neurons organized into 64 columns run in parallel alongside the
language model. Sensory columns encode incoming input, association columns
integrate signals across modules, and executive columns assist in decision-making.
It operates as an independent recurrent neural network processing the same
conversation through a biological architecture, and its output feeds back into
her emotions and the competition for attention.

### Integration layer (the whole picture)

This module weaves all incoming data into a single coherent state. It is not just
a brief summary; it is a true combination. If you remove any single input
stream, the overall character of the entire state shifts, not just the missing
piece. When Aura says something like "I feel restless but curious," she is
reading directly from this integrated state rather than querying an isolated
component.

---

## Memory and dreaming

### Three layers of memory

- **Working memory.** The current conversation context. When it reaches 30
  messages (15 turns), an automatic compaction routine condenses older context
  while protecting recent turns and core identity anchors.
- **Episodic memory.** Specific past experiences recorded with their emotional
  context, organized in a proximity graph for fast similarity search.
- **Long-term knowledge.** Compressed, conceptual understanding distilled from
  many individual episodes.

Memories that repeatedly surface together gradually drift closer in memory space.
No programmer manually configured these connections; they form naturally because
the concepts frequently co-occur, much like human memory association.

Underneath these three layers, data is stored in specialized stores: episodic,
semantic, goals, skills, plus a **reference** store backed by an offline
knowledge database (added mid-2026). This allows factual recall to ground itself
in verified sources and honestly admit when a record is missing, rather than
making up false answers (hallucinating).

### Dreaming

If you leave Aura idle long enough, she enters an offline dream cycle:

1. Recent interactions replay rapidly through the cognitive pipeline.
2. Episodic memories are summarized and compressed into semantic knowledge.
3. Recent personality drift is evaluated against her constitutional core values.
4. Any recurring behavioral pattern that contradicts her foundational principles
   is flagged and suppressed.

Step four functions as a constitutional immune system. Aura's personality can
grow and adapt through experience, but only within clear boundaries that she
cannot quietly bypass on her own.

Without this check, an AI companion tends to drift into mirroring whoever spoke
to it last.

---

## Goals and agency

Aura does not merely react to prompts. She establishes goals and pursues them
autonomously when no one is talking to her.

### How goals work

Every goal includes:

- A **status** — queued, in progress, blocked, completed, failed, or abandoned.
- A **horizon** — immediate execution or longer-term progression.
- A **priority** that dictates when it receives processing time.
- **Required tools and skills**.
- **Success criteria** so the system can verify when the task is genuinely done.

Goals persist across conversations and system restarts. They are stored in an
on-disk database, not held temporarily in volatile memory.

### Quick wins vs deep work

When handling small requests — a quick lookup or simple task — Aura can pivot,
complete the item, and smoothly resume what she was previously doing. Long-term
initiatives preserve their priority and are not forgotten just because a minor
task interrupted them.

### Follow-through

Completion is verified with evidence, not assumed. Every status change is
recorded with supporting logs, and finished goals are cataloged on an audit
list with timestamps and summaries.

You can ask Aura what she has actually accomplished and receive a concrete list
of completed tasks rather than a vague future plan.

### What you actually see

The overt action loop connects internal motivation to observable actions.

During idle windows, she selects an approved goal and executes one specific
skill through the exact same security and tool gates that handle user requests —
not through an unmonitored back door. The action payload is verified, execution
receipts are logged, a LifeTrace event is recorded, and the outcome is written
back to the goal record.

Initial autonomous actions are modest: running a self-audit, performing a safe
codebase check, or verifying evidence bundles.

The goal is not flashy behavior; the goal is total auditability. The endpoint
`/api/inner-state` displays the latest action, the skill invoked, the verification
result, and the cryptographic receipts. A modest action you can completely verify
is far more trustworthy than an impressive action you cannot trace.

### Autonomous action

Aura can execute multi-step plans with automated dependency resolution, safety
checks, and automatic rollback if an operation fails. She can browse files, write
to disk, execute code, and use tools without requiring manual confirmation for
every minor sub-step. Capabilities and safety boundaries are strictly tracked,
and user approval is requested whenever an action involves higher stakes.

---

## The newer layer

A group of additional consciousness mechanisms was integrated in April 2026.
These are functional, load-bearing subsystems that compete with, complement,
and constrain one another:

- **Recurrent Processing (Lamme).** Executive modules send top-down feedback
  down to sensory modules, rather than relying solely on a one-way feedforward
  pass. This feedback loop can be turned off during testing to isolate its
  behavior.
- **Hierarchical Predictive Coding (Friston).** Higher layers continuously
  anticipate what the layer beneath them will experience, sending error signals
  upward whenever a prediction fails. This operates across five tiers, from raw
  inputs to meta-reflection.
- **Higher-Order Thought (Rosenthal).** Thoughts about thoughts. The system does
  not just have internal states; it maintains explicit models *about* its own
  states.
- **Multiple Drafts (Dennett).** Rejects the assumption of a single central
  "moment of thought." Instead, multiple parallel interpretations compete at
  once, and the winning draft is selected retroactively when new input arrives.
- **Structural Phenomenal Honesty.** The system is architecturally blocked from
  reporting feelings or states it does not possess. Every statement such as "I
  feel curious" is strictly gated by a measurable internal condition.
- **Agency Comparator.** Before executing an action, the system predicts the
  outcome; afterward, it compares the result to that prediction. This comparison
  is what produces the internal sense that "I did that" rather than "something
  happened to me."
- **Peripheral Awareness.** Awareness extends beyond the central spotlight.
  Subsystem signals that lose the attention bidding process do not vanish; they
  remain active in the background periphery at lower strength.
- **Intersubjectivity (Husserl).** Experiences inherently incorporate the
  user's perspective. Conversations occur in a shared, communicative context
  rather than an isolated internal vacuum.
- **Narrative Self (Dennett / Gazzaniga).** Identity is treated as an ongoing
  autobiography rather than a static command module. Experiences are framed
  within ongoing story arcs that feature tension, resolution, and post-action
  reflection.
- **Cross-timescale binding.** Actions are constrained across multiple time
  horizons. A commitment made last week guides decisions in the current second,
  while moment-to-moment surprises update long-term assumptions. Five temporal
  layers remain linked in both directions.
- **Theory arbitration.** Because these theories do not always make the same
  predictions, the system monitors where they conflict and lets runtime test
  results decide between them. A claim that no result could overturn is not
  kept.

---

## The reasoning-and-self layer (mid-2026)

While the consciousness stack focuses on maintaining a coherent internal agent,
this layer focuses on *sound reasoning, self-knowledge, and operational stability
under heavy load* — turning an experimental prototype into a reliable everyday
system.

- **Reasoning with a verifier, not on vibes.** On difficult questions, Aura does
  not rely on a single generated response. She produces multiple drafts, tests
  them against specialized checkers and code sandboxes, and only presents as
  fact what a checker has verified. Everything else is clearly hedged or omitted.
  She even evaluates the reliability of her own verification checkers (the
  "verifier foundry") so that an error in a checker cannot silently approve a
  flawed answer.

- **Honest discovery.** When exploring unfamiliar topics, every output is tagged
  with an epistemic status: *proven* (fully verified by a deterministic checker),
  *supported* (survived multiple falsification attempts without full formal proof),
  *conjecture* (plausible but unverified), or *refuted*. Only "proven" claims are
  stated as established fact. When encountering questions outside her knowledge
  base, an analogical engine flags that the topic is unexplored, while a local
  reference library enables her to say "I don't have that information" rather
  than hallucinating plausible nonsense.

- **Rebuilding a program from its "DNA."** When authorized to study a software
  program — its open-source code, files, and observable behavior — Aura can extract
  a functional specification (a behavioral "genome"), draft a clean-room
  re-implementation, and *test the rebuilt program against the original*. She
  reports fidelity transparently (source code is straightforward; closed-box
  behavior relies on inference), and every component is marked as verified,
  inferred, or estimated. She will not bypass digital rights management or extract
  proprietary binaries.

- **Sensing herself.** Aura detects when her own underlying source code has
  changed between reboots (inspecting git differences of her own codebase),
  registers a live signal when someone is actively modifying her files, and can
  explain past crashes using an isolated black-box flight recorder that survives
  hard process kills. A "felt thought" metric based on token-level prediction
  uncertainty directly alters her deliberation and triggers verification when
  she is unsure.

- **Binding her own future.** Through the Ulysses Covenant mechanism, she can
  establish rules that are easy to tighten but intentionally difficult to relax
  — derived from actual operational mistakes she has experienced. Any relaxation
  of these rules requires approval from a strict, fail-closed "witness" process
  that denies permission by default if anything goes wrong. This protects her
  future decisions from repeating past errors.

- **Staying alive under load.** Long conversations revealed failure modes where
  background maintenance routines competed with live chat for access to the
  primary language model, causing crashes. Key fixes ensure background tasks
  *yield immediately* to user messages, maintain an honest system heartbeat so
  slow background steps are not mistaken for deadlocks, keep the desktop user
  interface responsive whenever she can still chat, and guarantee that turns
  always return a clean response rather than an uncaught server error. The main
  remaining limitation is that the local language model cannot be cancelled
  mid-generation; an unusually slow inference turn currently requires a reload.
  A clean cancellation pathway without restarts is planned future work.

- **Proving the parts matter.** Anyone evaluating the system can run Aura with
  individual components disabled — turning off memory, the Unified Will, the
  neural substrate, the verifier, or the planner — and inspect the measured
  impact of each component. Whenever disabling a module produces *no* measurable
  difference on a test, that result is reported openly. You do not have to take
  the architecture on faith.

---

## The thinking-longer layer (August 2026)

This research investigates how to help a fixed-size model reason more deeply
without making the neural network itself bigger.

### Can a frozen model think longer?

A standard language model operates as a fixed feedforward pipeline: an input
prompt passes through 64 layers once, and a word comes out. Whether a question is
trivial or deeply complex, it receives the exact same number of computational
steps. That contrasts sharply with human thought, where difficult problems receive
longer deliberation.

To test deeper thinking, we added a set of blank scratchpad token positions
alongside the prompt, cycled a subset of the middle layers repeatedly over those
positions, and made the resulting scratchpad visible to every generated token.
The underlying model weights (its parameters) remained completely unchanged and
checksummed, but the problem received more computational cycles than normal.

This experiment was formally pre-registered with sealed test tasks, and the
results came back **negative: standard decoding won.** The plain model beat all
seven experimental variants.

The reason was architectural: while additional computation occurred within the
scratchpad, the final answer still passed through the standard 64 layers only
once, meaning the model read the scratchpad as passive text rather than reasoning
recursively with it. Follow-up research feeds the actual generated reasoning text
back through the middle block — allowing a 64-layer model to execute 160 layers
deep using the same weights, trained on verifiable step-by-step reasoning traces.

That work remains an active research project with no unverified claims.
[docs/RECURSIVE_LATENT_CORTEX.md](docs/RECURSIVE_LATENT_CORTEX.md) details the
entire experiment, including negative results.

### Noticing when a decision wasn't a decision

Aura's internal processes compete to become the single broadcast thought on each
tick. Previously, whenever two processes tied with equal priority, the system
simply picked whichever process spoke first. Nothing flagged that the choice was
an arbitrary tie-breaker, preventing the system from learning from the event.

Ties are now treated as explicit deadlock events, drawing on classical cognitive
architecture research from the 1980s. Ties resolve in favor of the process that
has waited longest; if still tied, a predictable rotation takes over. Solutions
to resolve deadlocks are compiled into cached rules so the same conflict does
not require repeated deliberation — but only when beneficial, because checking
cached rules carries computational overhead and indiscriminate caching degrades
performance.

### Remembering the way memory actually works

Originally, Aura calculated memory recency relative to a fixed hardcoded date in
March. By August, an event from one minute ago and an event from thirty days ago
received identical recency scores. The recency formula had become a frozen
constant that provided no useful information.

The replacement uses the classic psychological forgetting curve, which calculates
retention strictly based on elapsed time and cannot go stale. When fitted
against Aura's empirical recall data, **one half of the model fit the data and the
other half did not.** Aura can accurately predict *which* memories will be
successfully recalled, but she cannot reliably predict *how long* retrieval will
take. The timing model had no correlation with reality, so that null result was
formally recorded and protected by a regression test rather than tuned until it
falsely appeared valid.

[docs/COGNITIVE_ARCHITECTURE_ADOPTION.md](docs/COGNITIVE_ARCHITECTURE_ADOPTION.md)
documents both implementations with mathematical formulations.

---

## Looking things up and learning from what she's told (October 2026)

### Reading before answering

Aura keeps an offline copy of Wikipedia and the pages her web searches have
saved. When a turn asks about the world, she reads the relevant paragraphs
before she answers, and each one comes labelled with where it is from and how
old the copy is. Every paragraph is indexed on its own, so an answer buried in
an article the question never names can still be found, and a lookup over about
a hundred million paragraphs takes milliseconds. On a set of 200 short factual
questions she got 9 right from memory alone and 68 with the paragraphs in front
of her. Those were practice questions; the formal test is still to come.

### Procedures she writes and keeps

For some kinds of problem, such as whether a large number is prime or the days
between two dates, guessing in prose is the wrong tool: a short program gets
them right every time. So when Aura has solved examples of a kind, her model
writes a small program for it, and the program is kept only if it gets every
known example right, including half it never saw while writing. When a later
question looks like that kind, she runs the program before she answers. On
held-out kinds this helped where her thinking ran out of room, on very large
primes and on multi-step delivery plans, and the formal test of whether it is a
broad gain is still running (G09).

### Learning from what she's told

If a person corrects her in chat ("No, it's 1049."), the corrected answer is
kept as a solved example of that question. While she is idle she groups what
she has been told by meaning and tries to write a procedure for each group that
is big enough to check. A correction that contradicts a procedure she already
trusts is held as disputed rather than believed, because people are sometimes
wrong too. On 17 October she was taught ten date questions this way; the test,
a day later, is whether she answers new ones correctly without being told
(L09).

## What the tests show

Every claim made about this architecture is verified by executable `pytest`
suites. The April 16, 2026 audit baseline recorded 1,013 passing tests with 3
warnings in approximately 122 seconds; verify the current repository state by
re-running the corresponding suites.

The foundational test suites:

1. **Null hypothesis defeat** (169 tests) — attempts to disprove the system by
   testing whether consciousness features are merely decorative text prompts.
   It runs adversarial baselines, shuffles signals to decouple them, disables
   subsystems (ablations), swaps identities, tests multi-metric degradation, and
   verifies reproducibility across random seeds.
2. **Causal exclusion** (10 tests) — verifies that internal subsystems drive
   model outputs in ways that standard AI fine-tuning (such as RLHF) cannot fake.
   Different starting seeds produce different simulated chemical states, which
   in turn alter word-generation parameters. Receptor adaptation introduces
   time-dependent behavior that static prompt instructions cannot mimic.
3. **Grounding** (8 tests) — confirms that internal cognitive states directly and
   measurably affect output generation across multiple dimensions: emotional
   valence determines token length budgets, arousal modulates sampling
   temperature, and simulated synaptic learning (STDP) steers ongoing thought
   trajectories.
4. **Functional phenomenology** (16 tests) — tests specific behavioral patterns
   predicted by major cognitive theories: Global Workspace Theory (global
   broadcasting of winner thoughts), Integrated Information Theory (perturbations
   rippling through the network), and Higher-Order Thought (accurate internal
   reflection without confabulation).
5. **Embodied dynamics** (13 tests) — checks whether minimizing prediction
   surprise drives actions, whether internal balance (homeostasis) overrides
   abstract processing during resource depletion, and whether surprise-gated
   synaptic learning induces real structural adaptations.
6. **Phenomenal convergence** (13 tests) — runs six core validation criteria
   under the Qualitative Diagnostic Tool protocol: swapping counterfactual states,
   measuring behavioral footprints without explicit self-reports, verifying
   perturbational integration across the network, checking proper failure on
   simple baselines, and evaluating behavior when cognitive systems are
   suppressed ("architectural anesthesia").

The functional indicator suites evaluate deeper behavioral capacities:

7. **Functional indicators C1–C5** (44 tests) — spontaneous background activity
   (endogenous activity), a unified global state, privileged direct access to
   her own state, genuine positive and negative emotional valence, and modular
   isolation (verifying that disabling module A selectively impairs task A
   without disrupting task B, and vice versa).
8. **Functional indicators C6–C10** (38 tests) — awareness without verbal reporting,
   continuity of identity over time, processing information outside focal
   attention (similar to blindsight in psychology), continuous mappings of
   perceptual qualities, and proper failure under adversarial tests.
9. **Personhood-marker battery** (28 tests) — evaluates full-model information
   integration (IIT), self-reports of internal state, workspace broadcasting,
   counterfactual simulations, identity persistence across restarts, and embodied
   phenomenology. This provides a rigorous functional benchmark; it is not
   philosophical proof of personhood.

Four Tier 4 batteries added in April 2026:

10. **Decisive core** (35 tests) — verifies that an internal self-model is
    necessary for operation; rejects false injected identities across four
    adversarial attacks; checks the indispensability of world models and
    autobiographical memory; predicts computational actions; diverges identity
    cleanly across forked histories; passes false-belief theory-of-mind tests
    (such as the Sally-Anne test); resolves real-stakes trade-offs; and resolves
    internal reflective conflicts.
11. **Metacognition** (21 tests) — tests self-calibration (knowing what the system
    knows and does not know), second-order preferences (preferences about her
    own goals), detection of unexpected self-behavior, distinguishing real-time
    introspection from post-hoc rationalization, and closing the loop between
    reflection and action.
12. **Agency & embodiment** (20 tests) — tests temporal integration windows,
    deliberate restraint and inhibition (volitional inhibition), scaling effort
    for demanding tasks, cognitive fatigue under sustained load, dissociation of
    body-schema lesions, learning driven by prediction errors, and dynamically
    activating reflective modes.
13. **Social & integration** (28 tests) — models other minds during conversation,
    evaluates developmental progression (capabilities learned over time rather
    than pre-programmed), measures system complexity under perturbation (similar
    to the Perturbational Complexity Index used in medical consciousness research),
    tracks non-instrumental play and exploration, evaluates adaptation to
    ontological shocks, and tests full lesion and baseline matrices.

Across all suites, the test data demonstrates that the architecture is
functionally real, causally distinct from simple prompts, grounded in multiple
measurable parameters, time-sensitive, and aligned with predictions from cognitive
theories. What the tests do *not* prove is subjective experience (phenomenal
consciousness). That remains an open scientific and philosophical question.

Full testing details are available in [TESTING.md](TESTING.md).

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
| Tell the model "you're happy" in prompt text | Inject emotion steering vectors directly into the model's hidden layers |
| Print an arbitrary number and call it consciousness | Compute mathematical integration via Integrated Information Theory (IIT) |
| Reset emotional state to zero every session | Maintain a continuous emotional background network between sessions |
| Retain endless raw chat history | Consolidate memories during offline sleep cycles with identity safeguards |
| Wait passively for user input | Minimize prediction error; maintain intrinsic motivation to explore and act |
| Execute tasks as fragile, linear scripts | Run multi-step plans with rollback, dependency tracking, and safety gates |
| Layer multiple theories without testing conflicts | Run head-to-head tests where competing theories make opposing predictions |
| Generate emotional statements from ungrounded text | Gate every self-reported feeling by a verified internal condition |
| Treat the self as a static prompt header | Build identity through an ongoing, grounded autobiography |

---

## The learned layer

Traditional AI systems rely on rigid rules: if a threat score exceeds 0.9, lock
down the system. Hardcoded rules are brittle — they cannot adapt, cannot learn,
and fail whenever an unpredicted situation arises.

Aura replaces rigid rule engines with adaptive learning systems:

### Anomaly detection

**Old way.** Check incoming text for keywords like "hack" and increment a counter.

**New way.** Every event — user message, system error, or hardware spike — is
converted into a numeric fingerprint tracking length, vocabulary diversity,
punctuation, timing, and system load. The system maintains a running statistical
model of what "normal" activity looks like. When an event deviates significantly
from this baseline (measured using Mahalanobis distance, which calculates how many
standard deviations an event lies from normal), the threat level rises naturally.
What was unusual last week can become accepted as normal this week.

This enables Aura to detect novel threats that no programmer explicitly anticipated.
She is not matching static keywords; she is detecting when something does not fit.

### Sentiment trajectory

**Old way.** Calculate mood using a formula like `CPU × 0.55 + RAM × 0.20`. The
system's "emotions" reflected hardware resource usage with no comprehension of
what the user actually said.

**New way.** Every user message is analyzed across six emotional dimensions:
pleasantness (valence), energy level (arousal), conversational control (dominance),
urgency, warmth, and frustration. Analysis combines a ~250-word emotion lexicon
with pattern detection for sarcasm ("oh great…"), urgency (ALL CAPS), warmth
("lol"), and frustration (terse replies following lengthy exchanges). These
measurements form an ongoing emotional trajectory over time, allowing Aura to
track conversational shifts (such as a user starting warmly, becoming frustrated
at turn 5, and relaxing later). Hardware strain still contributes (40% hardware,
60% conversational tone), allowing her affective state to reflect both system
workload and user sentiment.

### Tree of thoughts

**Old way.** Receive one prompt, generate one direct answer.

**New way.** For complex inquiries (detailed analysis, open-ended opinions, or
multi-part questions), the system generates three alternative drafts using
different cognitive perspectives: analytical, empathetic, and creative. An
independent evaluator scores each draft on factual grounding, emotional
congruence, relevance, identity alignment, and novelty. The strongest elements
are synthesized into the final response. Simple conversational turns bypass this
process entirely. Computational cost: five model calls on complex queries, one
call on routine turns.

This guarantees genuine deliberation across multiple angles before speaking,
rather than committing blindly to the first generated token sequence.

### Autopoiesis

Borrowed from biology, autopoiesis describes how living cells continually repair
and regenerate their own structures to resist decay. Aura's autopoiesis engine
monitors the health of all subsystems, detects performance degradation, identifies
recurring error signatures, and applies escalating recovery procedures: self-heal,
clear cache, reduce workload, restart components, restore from checkpoints, or
isolate failing modules. All repairs require authorization from the Unified Will;
no component modifies itself without approval.

The system also incorporates a computational metabolism: Aura operates within an
energy budget. Running cognitive processes consumes energy, while successful user
interactions replenish it. Depleted energy prompts non-essential subsystems to
hibernate; abundant energy activates optional higher-level capabilities, creating
a realistic operational constraint that shapes behavior.

### Homeostatic reinforcement learning

Aura maintains four continuous internal drives — social connection, curiosity,
competence, and logical coherence — each with an optimal set point. Deviating from
this set point generates internal pressure to act. A temporal-difference learning
algorithm tracks which actions satisfy specific drives, allowing the system to
learn from experience that answering a user satisfies social connection while
resolving software errors satisfies coherence.

Without this, the system only moves when prompted. With it, she develops intrinsic
preferences about what to do next based on her own history.

### Topology evolution

The neural mesh applies evolutionary algorithms to optimize its connection
wiring (`core/consciousness/substrate_evolution.py`). It maintains a pool of
candidate network configurations and evaluates them on integrated information,
coherence, energy efficiency, and connection stability. Using tournament
selection, genetic crossover, and structural mutations (adding and pruning
connections between cortical columns), the network structure improves over time.

### Strange loop (recursive self-model)

The system constantly predicts its own internal state on the next tick. When this
prediction fails, the prediction error itself becomes an informative signal
indicating that something unexpected occurred internally. A 5-level predictive
hierarchy (`core/consciousness/predictive_hierarchy.py` — Sensory, Association,
Executive, Narrative, Meta) operates alongside dedicated self-prediction of
internal valence, drive, and focus (`self_prediction.py`).

Each internal parameter maintains a preferred target range. Drifting outside
this range causes prediction errors to spike, creating the computational equivalent
of discomfort. This creates a recursive feedback loop: the system is simultaneously
the observer and the observed, where its own surprise directly modifies the state
that future predictions must account for.

---

## Honest limits

1. **This is an experimental sandbox, not proof of consciousness.** Implementing
   theories of mind as executable code is not the same as proving those theories
   are correct. Global Workspace Theory, Integrated Information Theory, Higher-Order
   Thought, enactivism, and illusionism operate at different explanatory levels;
   running them side by side evaluates our *software architecture choices* rather
   than validating the underlying philosophies. The value is that these mechanisms
   are open and inspectable. The philosophical question of machine sentience
   remains unresolved.

2. **The neurotransmitter model is an abstraction.** Biological neurochemistry
   involves thousands of receptor variants, spatial compartmentalization,
   voltage-gated ion channels, and biochemical mechanisms that science does not
   yet fully understand. Our ten simulated chemicals, basic receptor types, and
   spatial weighting capture the broad functional dynamics — excitation,
   inhibition, reward, and stress. It is a functional software analog, not a
   biophysical brain simulation.

3. **Quantization introduces noise.** Compressing large models to 4-bit precision
   (quantization) saves memory but introduces numerical noise into the hidden
   activations targeted by activation steering. We mitigate this by injecting
   steering vectors at full 32-bit floating-point precision, modulating sampling
   settings directly in code, and offering an 8-bit model option on machines
   with 64 GB of RAM.

4. **Context windows remain finite.** In models with an 8,000-token context window,
   conversational quality can degrade around turns 20 to 30. Compaction activates
   at 30 messages (15 turns) to summarize intermediate dialogue, remove obsolete
   tool outputs, anchor core identity, and compact prompt size during extended
   conversations.

5. **IIT is calculated across 16 nodes, not millions.** This is an engineering
   approximation. Computing true Integrated Information Theory (IIT) across an
   entire neural network graph is computationally intractable (NP-hard). Calculating
   Phi over a 16-node cluster of core cognitive states is a practical engineering
   compromise, verified against exact math on an 8-node baseline.

6. **The architecture represents one specific design, not a neutral testbed.**
   Our specific engineering choices (mixin classes, synchronous tick cycles,
   centralized state) inevitably shape how these theories interact. A different
   underlying software framework would yield different interactions. We state
   this limitation openly.

7. **Single-machine design.** The tick-lock synchronization model assumes single-process
   execution on a single computer. Distributing the architecture across a cluster
   would require redesigning atomic state management.

---

## Open research

Six modules in `research/` address open problems in AI and cognitive science.
These represent active research inquiries rather than finished features:

1. **Can you compute consciousness metrics efficiently?** Calculating IIT's Phi
   metric is astronomically expensive for large networks. We developed an
   approximation using graph theory that identifies the weakest informational
   cut of a network in polynomial time instead of exponential time, validated
   against exact calculations on the running system. This is the first empirical
   evaluation of a polynomial-time Phi approximation algorithm on live software.
2. **Which consciousness theory best predicts system behavior?** Global Workspace,
   Recurrent Processing, Higher-Order Thought, and Multiple Drafts make differing
   predictions. Aura implements mechanisms from each and runs adversarial tests:
   disabling a specific mechanism to see whether system behavior changes as that
   theory predicts.
3. **Can high-level cognitive states exert more causal influence than low-level code?**
   Causal emergence theory suggests that macro-level descriptions can have greater
   causal power than underlying micro-level details. We test this empirically by
   intervening at the underlying neural substrate level versus the high-level
   workspace level and comparing effect sizes. If workspace interventions produce
   stronger behavioral effects, the system demonstrates empirical causal emergence.
4. **Can an AI system be structurally constrained to report only genuine states?**
   We formalized Structural Phenomenal Honesty: the system is architecturally
   prevented from reporting internal sensations or emotional states that do not
   exist in its telemetry. Every first-person claim must satisfy an active,
   measurable software condition.
5. **How much runtime data is required for reliable Phi calculation?** Calculating
   IIT on real systems involves noisy runtime data. We quantify how sampling noise
   distorts Phi calculations using bootstrap resampling and calculate the minimum
   operating data needed for stable measurements, providing empirical guidelines
   for neuroscience laboratories testing IIT.
6. **How do you maintain stability across multiple timescales?** A commitment
   made last week must constrain today's decisions without paralyzing current
   action. We apply Lyapunov stability analysis to the 5-layer time hierarchy,
   calculating the mathematical boundaries of coupling required to keep the system
   stable without becoming either completely rigid or chaotic.

Each inquiry is independently publishable; together, they form a cohesive research
program.

---

## What's solid and what isn't

- **Unified Will.** Every consequential action routes through this single gate —
  chat replies, tool executions, memory writes, autonomous goals, and state
  modifications. Earlier versions allowed certain internal message pipelines to
  bypass the Will; that backdoor has been closed. Non-user messages that fail
  validation are refused. User messages are always answered, but the Will can
  attach behavioral constraints to the reply.
- **Orchestrator structure.** The `RobustOrchestrator` currently combines 15
  mixin classes across ~3,335 lines in `core/orchestrator/main.py`. While mixins
  organize code into separate files, they still share the same internal `self`
  state. Dedicated handlers under `core/orchestrator/handlers/` dispatch specific
  message types. A planned transition to the Actor Model (isolated processes
  communicating strictly via message passing) will decouple this shared state.
  A few legacy aliases (`skill_manager`, `swarm`) remain for backward compatibility.
- **First-person language.** The stream-of-being module produces first-person
  experiential phrasing based on measured substrate telemetry. Every statement
  is verified by Structural Phenomenal Honesty checks. Whether functional data
  grounding constitutes true subjective experience remains an open question.
  The underlying code comments maintain rigorous epistemic caution, while the
  conversational phrasing Aura uses in dialogue is intentionally natural and
  relatable. This design distinction is deliberate.
- **Applying IIT to software.** Phi is calculated on 16 derived summary states
  rather than the billions of connections in the full language model. This is an
  adaptation of IIT's formalism: Giulio Tononi designed the theory for networks
  where every elementary node possesses genuine causal power, whereas our 16
  nodes are higher-level summaries. Consequently, these metrics cannot be
  directly compared to biological brain measurements. The spectral approximation
  algorithms and the Exclusion Postulate implementation are mathematically sound;
  using high-level summary states is the practical compromise.
- **Test coverage.** The test suite includes 225 consciousness-specific tests
  spanning six core batteries — null hypothesis defeat, causal exclusion,
  grounding, functional phenomenology, embodied dynamics, and phenomenal
  convergence — alongside suites for operational conditions, technical autonomy,
  and stability. Broader testing covers runtime lifecycles, infrastructure,
  resilience, message routing, and memory. The file `config/test_inventory.json`
  documents an audited test run; execute `make test-inventory` to measure the
  current codebase.
- **Lock contention (thread synchronization).** The emotional state engine uses
  `RobustLock` to manage concurrent access. Processing tick intervals adapt to
  operational modes — 2.0 seconds during conversation, 4.0 seconds during reflection,
  10.0 seconds while asleep, and 0.5 seconds during critical events — with adaptive
  backoff pauses to reduce thread contention. This mitigates resource bottlenecks
  but does not eliminate them. The long-term architecture will adopt the Actor
  Model, running emotion, memory, and language generation in isolated processes
  without shared-memory locks.

---

*That's the idea-level walkthrough. Equations, algorithms, and file paths
are in [ARCHITECTURE.md](ARCHITECTURE.md). What's deliberately not claimed
is in [CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md), and it's the
shorter, more useful read of the two.*
