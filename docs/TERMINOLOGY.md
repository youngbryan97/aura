# Terminology

<!-- reader-navigation:start -->
> **In plain English:** The documentation contains current guides, dated experiment records, frozen rules, generated reports and open plans. They answer different questions.
>
> **Why it matters:** A reader should not treat an old proposed run as today’s result or a completed engineering task as a passed scientific check.
>
> **In everyday use:** An entry saying the next run is queued can remain correct history after that run finishes; the maintained status page gives the later outcome.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#documentation) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#documentation) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

The aliases below name project mechanisms. They are not interchangeable with
all scientific uses of the same word. In particular, a bounded integration
score is not full IIT 4.0, and a name such as free energy needs its own model
and units. Use the [glossary](GLOSSARY.md) and
[formula guide](FORMULAS.md) for the definitions and limits.

## Start with the ordinary meaning

Project names often describe the intended role of a mechanism. To understand
what it does, look for the state it reads, the action it changes and the test
that checks the effect. The original engineering alias table is retained below.

| Project term | Ordinary meaning | Why someone would care |
|---|---|---|
| Cortex | The main language-model route | Which model produces an answer affects capability, delay and memory use |
| Brainstem | Another configured model route for designated work | A fallback answer may come from a different model and needs to be identified |
| Workspace | The competition for what gets attention now | Your new question must not be silently replaced by an old concern |
| Self-model | Records and predictions about the system itself | It can help track capability, uncertainty and the results of its own actions |
| Affect or mood | Internal variables that can influence processing | The practical question is whether changing them changes a relevant choice, recall or report |
| Qualia descriptor | A software description of an internal condition | The descriptor's name alone does not establish a felt experience |
| Phi | One of several integration-related measurements | Its formula and controls matter; different implementations are not interchangeable |
| Soul | The project's name for parts of drive and value continuity | A reader can inspect how preferences persist and influence later behavior |
| Scar | A retained record that influences avoidance or caution | An earlier bad outcome can change a later choice under a stated decay rule |
| Dream | Background consolidation or replay | Earlier records may be reorganised while the system is idle |
| Mycelium | A graph of connections between components | It describes which parts can reach and influence other parts |
| Governance | Permission and constraint checks for proposed actions | Access to a tool should not silently become permission for every consequence |
| Receipt | A structured action or decision record | You can inspect what was allowed, attempted and observed |
| Taint | A retained flag that an assumption broke | A later green status should not conceal a known earlier violation |
| Mortal memory | Crash and continuation records from the recorder | Work may be recoverable after a process stops; the label does not establish biological life |
| Proof or certificate | Evidence for a stated property under named assumptions and tests | Read its scope rather than extending it to every capability |

Think of the names as labels on a workshop's tools. A label tells you what a
tool is for; watching it work and measuring the result establishes what it can
do. The comparison concerns naming and evidence, rather than the full
organisation of Aura.

For values and units, read [the variable guide](VARIABLES_EXPLAINED.md).
For examples of mechanisms, read [the tour](PLAIN_ENGLISH_GUIDE.md).
For the scientific argument and its limits, read [ISC explained](ISC_EXPLAINED.md).

## Internal and engineering aliases

Two names for every mechanism. The internal one and the one that doesn't
sound like it's trying to convince you.

Both columns stay. The internal naming carries real design intent — a scar
is not a "record," and calling it one loses why it decays the way it does.
But a skeptical engineer should be able to read the same structure without
the vocabulary doing the persuading, so the sober label sits right next to
it.

If a claim only sounds impressive in the left column, it isn't a claim.
That's what this table is for.

| Internal name              | Sober label                                            |
|----------------------------|--------------------------------------------------------|
| qualia                     | phenomenal descriptor vector                           |
| soul                       | drive / value continuity model                         |
| consciousness              | integration / workspace / self-model proxy             |
| dream                      | offline consolidation                                  |
| dream cycle                | offline consolidation cycle                            |
| will                       | governed action arbitration                            |
| organism                   | self-maintaining autonomous runtime                    |
| heartbeat                  | autonomic background tick                              |
| scar                       | domain-typed avoidance memory with severity decay      |
| Eternal Record             | tier-3 narrative consolidation                         |
| narrative arc              | tier-2 narrative consolidation                         |
| episodic journal           | tier-1 narrative consolidation                         |
| substrate                  | shared homeostatic / affective state                   |
| substrate authority        | substrate-policy authority gate                        |
| lymphatic reaper           | low-affect record retention sweeper                    |
| black hole vault           | pinned high-sensitivity record store                   |
| volition                   | autonomous proposal generator                          |
| neural mesh                | distributed cognitive graph                            |
| mycelium                   | inter-subsystem topology graph                         |
| stem cell                  | immutable signed organ snapshot                        |
| somatic gate               | sensorimotor permission gate                           |
| oscillatory binding        | phase-locked subsystem binding                         |
| free energy                | model-dependent inference/control quantity; not thermodynamic energy without a physical model                         |
| phi / phi_s                | bounded integration estimator; formula and grain must be specified|
| HOT                        | higher-order representation of own state               |
| GWT ignition               | global broadcast event in the workspace                |
| latent-space bridge        | substrate-state-driven sampling parameter modulation   |
| residual stream offset     | activation-steering hidden-state addend                |
| conscience                 | irrevocable rule-based safety floor                    |
| play                       | non-utilitarian latent-space exploration               |
| viability state            | metabolic/operational state machine                    |
| capability token           | scoped, single-use authorization grant                 |
| receipt                    | structured audit record                                |
| felt thought               | token-level surprisal/entropy interoception signal     |
| flight recorder            | SIGKILL-survivable per-tick mind-moment ring buffer    |
| mortal memory              | death report + waking sequence from the flight recorder|
| source-body proprioception | boot-over-boot self-code diff + operating-on-me pulse  |
| Ulysses Covenant           | enforceable volitional self-binding (asymmetric ratchet)|
| calm witness               | fail-closed approver required to loosen a self-binding  |
| verifier foundry           | measured verifier reliability + self-training admission gate |
| frontier discovery engine  | verifier-gated generate→falsify→commit loop            |
| analogical leap            | out-of-distribution structure mapping with an OOD verdict|
| knowledge substrate        | offline reference corpus (FTS5/BM25) as a REFERENCE store|
| program DNA                | authorized clean-room behavioral reconstruction         |
| backpressure               | expected-timeout yield (not a degradation) under load   |
| reasoning amplifier        | mandatory verifier-gated hard-task cognition layer      |
| whole-system phi           | Gaussian stochastic-interaction estimate; exact search only at the declared small-grain limit |
| reality reach              | typed causal contract + reachability proof for a physical request |
| reality layer              | evidence scope: internal / effective / direct / ambient |
| limitation certificate     | typed machine-verifiable no-go verdict for an unreachable request |
| effect receipt             | independent post-action measurement (never transport success) |
| coupling class             | declared physical relation between a channel and an observable |
| faculty                    | a named cognitive capability with declared improvement metrics |
| binding constraint         | the faculty whose headroom most limits the rest of the stack |
| blind spot                 | a faculty no probe can currently measure (a self-knowledge gap) |
| stance                     | accumulated affective orientation toward a known entity |
| entity                     | content-addressed person/place/thing/org/concept record |
| taint register             | persistent record that the process is not the one its green verdict describes |
| lockdep                    | lock-order dependency tracking (ABBA detection without the deadlock) |
| shed ladder                | ordered load-shedding sequence under memory pressure    |
| telemetry dictionary       | declared channel ids with units and limits              |
| kept procedure             | model-written `solve()` kept only if it matches every known answer of its kind |
| procedure book             | per-kind routing table: word signatures, embedding centroids, kept procedures |
| words route / meaning route | lexical-signature vs embedding-centroid routing of a request to a kind |
| restatement                | model rewrite of a request into a kind's phrasing, checked to keep its numbers |
| met example                | answer stated in chat, or agreed by independent programs, stored as a solved example |
| learning in use            | idle-time procedure induction from met examples         |
| copy hold                  | DRY repetition penalty over her own earlier replies     |
| careful lookup             | paragraph retrieval from the articles a question names  |
| paragraph index            | per-paragraph BM25 index over the corpus (Tantivy)      |

This document is the machine-readable source of truth at
``docs/TERMINOLOGY.md``. No part of the UI reads it yet; an earlier version
of this page said a dashboard "Skeptic mode" did, and no such mode was ever
built.
