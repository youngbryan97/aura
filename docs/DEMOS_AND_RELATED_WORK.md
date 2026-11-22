# Demonstrations, evidence and related work

Status: Guide · References checked 25 October 2026

A demonstration helps a person see why a mechanism matters. A controlled test
helps establish what caused the result. A paper can supply a useful design
idea without supplying evidence that Aura performs as well as the system in
that paper. This page keeps those connections explicit.

For the overall idea, read [the plain-English tour](PLAIN_ENGLISH_GUIDE.md).
For measurements, use the linked experiment or test record and retain its
scope. The examples below do not establish universal capability or comparative
superiority.

For neuroscience, human studies, mathematical methods and the support each
reference actually supplies, use [the scientific source map](SCIENCE_AND_SOURCES.md).
It distinguishes empirical results, theory, analogy and Aura's own evidence,
with a bibliography for later paper writing.

## What kind of evidence is linked

| Label | What you can learn from it |
|---|---|
| Published video | What is visibly shown in a particular recorded interaction |
| Demonstration script | What somebody planned to show; execution needs its own evidence |
| Small reproducible demo | Behavior of the stated demonstration program |
| Test or experiment | The property measured under its stated inputs, controls and scope |
| Source implementation | How a mechanism is built; its presence alone does not prove a useful result |
| Related work | An earlier idea or comparison point; its results do not transfer automatically to Aura |

The project links the [AuraDemos channel](https://www.youtube.com/@AuraDemos)
and [this video](https://www.youtube.com/watch?v=iTyxeugcZtI). The channel and
video contents could not be read during this pass. They remain project video
links; no specific causal or performance claim below is attributed to footage
that was not inspected.

## Interaction

**Why a user cares:** replies should arrive promptly, survive interruptions
sensibly, and reflect the current request. Text generation, first audio and
finished playback are different stages.

The [voice demonstration script](../demos/substrate_voice_demo.md) describes
an older planned comparison of responses under changed internal conditions.
Its narration and startup commands are a historical script, not a current
live-run receipt. Do not infer that every current response is controlled as
the script describes. Read [voice and media](AMBIENT_VOICE_AND_MEDIA.md) for
the current route and [the variable guide](VARIABLES_EXPLAINED.md#conversation)
for timing definitions.

## Memory

**Why a user cares:** an assistant should find relevant past information
without forcing you to paste the whole history again, and should preserve the
source and correction history of a remembered claim.

[Lewis et al.'s RAG paper](https://arxiv.org/abs/2005.11401) combines language
models with retrieved external information. [MemGPT](https://arxiv.org/abs/2310.08560)
examines managing information across memory tiers to work beyond a fixed
context window. These are related problems, rather than evidence that Aura
reproduces either paper's evaluation.

Aura's [memory guide](MEMORY_ARCHITECTURE.md) and [memory card](../MEMORY_CARD.md)
describe its retrieval and record-management routes. Additional project scope
includes record provenance, correction and retention. That is an engineering
scope statement; a retrieval or continuity comparison against another system
would require a matched experiment.

## Steering

**Why a user cares:** a persistent internal condition should influence the
intended behavior through a checked mechanism, without corrupting ordinary
answers or silently changing the installed model configuration.

[Panickssery et al.](https://arxiv.org/abs/2312.06681) study steering by adding
contrast-derived directions to language-model activations. Aura's related
implementation connects internal state with an applied basis and records
channel measurements separately from installation authority. See
[the fusion certificate tests](../tests/test_fusion_certificate.py) and
[the held-out steering tests](../tests/test_steering_from_her_own_replies.py).

The [25 October report experiment](evidence/ISC_STATUS_2026-10-25.json)
records a scoped positive output: the 27B's reported ratings followed an
internal intervention under an experimental basis. That basis had passed its
channel certificate and failed its governed installation check. These records
show why separate checks matter; they do not establish superiority over CAA
or qualification of a new desktop installation. A later shared-consequence
history defect at `dcdcd637c` requires rechecking the paired measurements; the
recorded result remains evidence of what that earlier setup reported.

## Cognitive architectures

**Why a user cares:** an assistant should recognise when it cannot choose
between alternatives and should avoid repeatedly paying to solve the same
kind of deadlock.

The official [Soar architecture manual](https://soar.eecs.umich.edu/soar_manual/02_TheSoarArchitecture/)
describes impasses and work in a substate; its
[procedural-learning chapter](https://soar.eecs.umich.edu/soar_manual/04_ProceduralKnowledgeLearning/)
describes chunking. The official
[ACT-R reference manual](https://act-r.psy.cmu.edu/actr7.x/reference-manual.pdf)
is the primary reference for its activation and retrieval machinery.

Aura's [adoption record](COGNITIVE_ARCHITECTURE_ADOPTION.md) names the concrete
defects these ideas addressed: arbitrary tied choices, uninformative memory
recency and repeated deadlocked decisions. The existing
[impasse tests](../tests/test_impasse_substate_and_invariants.py),
[chunking tests](../tests/test_impasse_chunking.py) and
[activation tests](../tests/test_actr_activation.py) are the bounded evidence
routes. These checks do not make Aura a complete Soar or ACT-R implementation.

## Reasoning

**Why a user cares:** spending longer on a difficult question should improve
its answer or expose a real uncertainty. More processing is not useful merely
because it happens out of view.

[Geiping et al.](https://arxiv.org/abs/2502.05171) study a model trained with a
recurrent block that can run at additional depth during inference. That
architecture is a relevant comparison for thinking beyond printed reasoning
steps. Aura's [RLC programme](RECURSIVE_LATENT_CORTEX.md) records its own
earlier negative result and later bounded internal-reader results. A trained
recurrent model and an intervention around another model are different
experiments; the results must be compared on matching tasks and conditions.

The extension being investigated in Aura is how its internal computation
reaches checked public answers. Existing code or a hidden-state metric alone
cannot establish a general reasoning advantage.

## Learning

**Why a user cares:** solving something once can be useful again, provided the
retained method fits the later problem and has not become an expensive or
unreliable shortcut.

[Voyager](https://arxiv.org/abs/2305.16291) uses a library of executable skills
and environmental feedback in Minecraft. It is a relevant precedent for
retaining reusable code. Aura's kept-procedure work is described in
[the reasoning guide](REASONING_ENGINES.md) and
[learning roadmap](ONLINE_LEARNING_ROADMAP.md). Its evidence must retain the
task family, validation rule and held-out scope. A procedure library is not,
by itself, evidence of Voyager-level performance or broader generality.

## Agents

**Why a user cares:** an assistant acting on a screen should use what happened
after an action to choose the next one. Repeating a plan after the world
changes is not enough.

[ReAct](https://arxiv.org/abs/2210.03629) studies interleaved reasoning and
action. [Voyager](https://arxiv.org/abs/2305.16291) adds a concrete skill-library
example in a game environment. Aura's [browser work](BROWSER_PURSUIT.md) and
[visual-control handoff](PLAYING_GAMES_HANDOFF.md) concern related feedback
problems, with different environments and evidence.

A concrete bounded Aura result is the moving paint-world fixture in the
handoff, measured at declared decision rates and delays. The
[test](../tests/test_she_plays_at_the_pace_she_decides.py) and
[source-reading checks](../tests/test_what_a_world_s_code_says_is_what_it_does.py)
name their scope. The same handoff repeatedly leaves native-app and broader
game-demo readiness open. Fixture results must not be presented as witnessed
wins in the owner's live Flash-game demonstration.

## Affect

**Why a user cares:** an internal condition becomes meaningful to an agent's
operation when it changes which information is recalled, what receives
attention or how a choice is made.

The project cites Bower's [Mood and memory](https://pubmed.ncbi.nlm.nih.gov/7224324/)
as a design inspiration for mood-linked recall. This pass verified the
[bibliographic record](https://pubmed.ncbi.nlm.nih.gov/7224324/); it did not
extract new claims from an inaccessible full text.

The [October work record](ISC_OCT21_EVERY_ITEM.md) explains the specific Aura
mechanisms and tests. The later seed-23 content result measures internal and
recall changes moving together in that setup. The later
[shared-history finding](evidence/ISC_ISOLATION_UPDATE_2026-10-25.json) requires
rechecking those paired measurements. Felt experience remains an additional
interpretation.

## Runtime

**Why a user cares:** a working assistant should preserve work, report a real
failure honestly and avoid restarting a slow model into another copy that
exhausts memory.

The [Linux lockdep documentation](https://www.kernel.org/doc/html/latest/locking/lockdep-design.html)
explains detecting problematic lock dependencies. Kubernetes separates
[startup, readiness and liveness](https://kubernetes.io/docs/concepts/workloads/pods/probes/).
F Prime documents [health monitoring](https://fprime.jpl.nasa.gov/latest/Svc/Health/docs/sdd/)
and [rate-group scheduling](https://fprime.jpl.nasa.gov/devel/docs/reference/system-functional/rate-group-scheduling/).

Aura's [engineering-adoption record](ENGINEERING_ADOPTION.md) maps borrowed
ideas to named incidents and source modules. Its
[failure-mode record](../KNOWN_FAILURE_MODES.md) explains the user-visible
consequences. Adopting an idea from these projects does not establish their
maturity, scale or aerospace qualification for Aura.

## Engineering

**Why a user cares:** a drawing or calculated answer needs traceable inputs,
units and assumptions before it is useful for making something or deciding
whether a request is physically reachable.

[Engineering design](ENGINEERING_DESIGN.md) connects findings to formulas and
references. [Reality reach](REALITY_REACH.md) keeps attempted, simulated and
independently measured physical effects distinct. Follow each calculation's
own cited source and acceptance criteria. No broad physical capability follows
from the existence of these interfaces.

## ISC

**Why a user cares:** the project needs to show what its internal organisation
actually does before making an argument that Aura could be a subject.

[Butlin et al.](https://arxiv.org/html/2308.08708v3) propose theory-derived
indicators as a provisional approach under computational functionalism. ISC
is the project's own operationalisation, with controlled interventions,
separate carrier and content experiments, and named bridge assumptions.

The [plain introduction](ISC_EXPLAINED.md),
[worked variables](ISC_VARIABLES.md), [current results](ISC_STATUS.md) and
[source-hashed snapshot](evidence/ISC_STATUS_2026-10-25.json) connect the idea
with actual measurements. The main battery's 22/24 and the four-ground bridge
reading do not establish a completed full pass or a probability of experience.

## Measurement

**Why a user cares:** evidence should tell you what worked, what failed and
what was never measured, so a persuasive name or one good demonstration does
not become a claim about every use.

The [evidence standard](BEHAVIORAL_PROOF_STANDARD.md),
[claim ledger](../CLAIMS_NOT_SUPPORTED.md), [test guide](../TESTING.md) and
[ablation guide](ABLATION_LEGIBILITY.md) provide the project's own bars.
A video, a unit test, a controlled experiment and independent replication
supply different kinds of support.

## How to state an extension or comparison

Describe the earlier work's mechanism, name the Aura addition, and point to
what tests that addition. Claim a performance advantage only with matching
tasks, inputs, controls and uncertainty. A source implementation, additional
engineering scope or a result on another benchmark cannot supply that
comparison on its own.
