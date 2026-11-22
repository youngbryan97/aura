# Aura in plain English

Status: Guide · Reading explanations updated 2026-10-25

Aura is a local AI research project. It combines a language model with memory, attention, action control and ongoing internal state. The purpose is useful continuity: what happened earlier should matter to what it does next, with a traceable route from a request to an outcome.

This tour explains the mechanisms and why someone would care. It is a guide to the design, not a certification that every feature works in the live app. Use the [claim boundaries](../CLAIMS_NOT_SUPPORTED.md), [current ISC results](ISC_STATUS.md) and each page's evidence for supported outcomes.

The [scientific source map](SCIENCE_AND_SOURCES.md) explains the research behind the ideas, what each source supports and where Aura's own evidence is required.

## A short explanation you can use

> Aura is built to do more than answer a message in isolation. It keeps records, chooses what deserves attention, checks proposed actions and learns or reuses some procedures. The practical goal is an assistant that can keep track of your work, explain what it did and remain responsive while it works. We test whether each extra mechanism actually helps, because a complicated architecture can still give an ordinary or unreliable result.

## Choose a topic

- [Getting started](#getting-started)
- [Conversation and voice](#conversation)
- [Language models and internal steering](#models)
- [Memory and recall](#memory)
- [Affect and operating condition](#affect)
- [Attention and competing concerns](#attention)
- [Reasoning and checking answers](#reasoning)
- [Learning and kept procedures](#learning)
- [Goals and autonomous work](#agency)
- [Permission, actions and receipts](#governance)
- [Memory, speed and resource limits](#resources)
- [Staying alive and recovering](#reliability)
- [Logs, measurements and evidence trails](#observability)
- [Privacy and trust boundaries](#security)
- [Using a browser or screen](#browser)
- [Visual control and moving worlds](#vision)
- [Engineering calculations and drawings](#engineering)
- [Claims about the physical world](#physical)
- [Tests, controls and claims](#measurement)
- [ISC and the subjecthood research](#isc)
- [Architecture and runtime components](#architecture)
- [Reading records and research plans](#documentation)
- [Reading mathematical symbols](#numbers)

## Getting started

**Getting started.** Aura is a local AI research system built around a language model, persistent records and software that manages actions and ongoing work.

**Why an ordinary user would care.** A reader needs to know which pieces must be installed, which resources they use, and which capabilities have actually been measured.

**Day-to-day example.** Before asking Aura to work with a document, you need a ready model, an accessible file and enough memory for the running system.

**Metaphor.** Think of preparing a workshop: having a tool on the shelf is different from having it connected, supplied and ready to use.

**Where the comparison stops.** The comparison explains readiness. It does not certify that every installed capability works.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#getting-started) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Install](../INSTALL.md), [Operator Guide](OPERATOR_GUIDE.md), [Model Card](../MODEL_CARD.md).

## Conversation

**Conversation and voice.** Conversation support turns a request into a reply and, when enabled, into speech. Streaming lets delivery begin before every later stage finishes.

**Why an ordinary user would care.** The person should receive a timely, coherent answer to the current request without hearing a stale or unapproved fragment.

**Day-to-day example.** When you interrupt a spoken reply, the old reply should stop owning the conversation and the new turn should take priority.

**Metaphor.** Think of a live radio conversation: the first audible words, the complete sentence and the end of playback happen at different times.

**Where the comparison stops.** The comparison explains timing and turn ownership; it does not establish a measured latency for Aura.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#conversation) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#interaction)

Read next: [Ambient Voice And Media](AMBIENT_VOICE_AND_MEDIA.md), [User Guide](USER_GUIDE.md).

## Models

**Language models and internal steering.** The language model supplies text generation. Steering changes selected internal calculations using a measured direction and a controlled strength.

**Why an ordinary user would care.** An internal state should affect the intended part of a response while preserving competence and remaining tied to the measured model and basis.

**Day-to-day example.** A change in internal condition may alter how a reply is formed; a channel measurement must show the change rather than relying on the reply sounding different.

**Metaphor.** Think of adjusting a sound mixer during a performance: a control can affect the output without changing the written score.

**Where the comparison stops.** The analogy is about changing a computation. Model activations are not audio channels, and the metaphor does not prove feeling.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#models) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#steering)

Read next: [Model Roster](MODEL_ROSTER.md), [Isc Status](ISC_STATUS.md).

## Memory

**Memory and recall.** Memory keeps records beyond the immediate request and selects relevant records for the current turn.

**Why an ordinary user would care.** Useful continuity depends on recalling the right information, retaining its source and correcting or deleting it when needed.

**Day-to-day example.** If you correct a fact about a project, later answers should use the correction rather than retrieving the older claim as equally reliable.

**Metaphor.** Think of a notebook with an index: keeping a page and finding the right page are separate jobs.

**Where the comparison stops.** A software memory record is not evidence of a human-like recollection or a felt past.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#memory) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#memory)

Read next: [Memory Architecture](MEMORY_ARCHITECTURE.md), [Memory Card](../MEMORY_CARD.md).

## Affect

**Affect and operating condition.** Aura uses internal variables for positive or negative condition, activation, needs and operating strain. Their role is established by what they change.

**Why an ordinary user would care.** A named mood or drive is useful only if its consumers produce an appropriate effect, rather than merely displaying a label.

**Day-to-day example.** Under resource strain, the system may need to reduce expensive background work while preserving the conversation.

**Metaphor.** Think of a dashboard connected to controls: a reading matters when it changes a decision, rather than only lighting up.

**Where the comparison stops.** A dashboard value does not establish a feeling. The analogy is limited to causal use of internal readings.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#affect) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#affect)

Read next: [Allostasis Engine](ALLOSTASIS_ENGINE.md), [Payoff](PAYOFF.md).

## Attention

**Attention and competing concerns.** Several concerns can propose work, but attention selects what has influence now and makes selected information available elsewhere.

**Why an ordinary user would care.** A new question, a recalled event and an old task must not silently become interchangeable sources of authority.

**Day-to-day example.** Your new question should not be overwritten by a background task that happened to win a previous competition.

**Metaphor.** Think of a meeting with an agenda: choosing the next speaker and deciding what the group will do are related but different decisions.

**Where the comparison stops.** The candidates are software records, not separate people. Winning attention does not itself grant permission to act.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#attention) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#cognitive-architectures)

Read next: [Cognitive Architecture Adoption](COGNITIVE_ARCHITECTURE_ADOPTION.md), [Organismal Workspace Theory](ORGANISMAL_WORKSPACE_THEORY.md).

## Reasoning

**Reasoning and checking answers.** The reasoning programmes investigate whether extra computation or learned internal representations improve answers on tasks not used to choose the method.

**Why an ordinary user would care.** A slower or more elaborate process is useful only if it produces a measurable benefit that survives relevant checks.

**Day-to-day example.** On a difficult constraint problem, a better internal method should increase correct answers or expose uncertainty rather than only generate longer prose.

**Metaphor.** Think of doing a second calculation before submitting a result: extra effort is valuable when it catches a mistake or solves a harder case.

**Where the comparison stops.** The analogy does not establish that more computation always helps or that one measured task family proves general intelligence.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#reasoning) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#reasoning)

Read next: [Recursive Latent Cortex](RECURSIVE_LATENT_CORTEX.md), [Reasoning Engines](REASONING_ENGINES.md), [Mathematical Claim Limits](MATHEMATICAL_CLAIM_LIMITS.md).

## Learning

**Learning and kept procedures.** Learning changes later behavior using earlier outcomes. A kept procedure is executable work retained for reuse under a defined checking rule.

**Why an ordinary user would care.** Experience should improve later work without accumulating unreliable shortcuts or expensive rules that no longer help.

**Day-to-day example.** After solving one kind of task, the system may reuse a checked procedure on a later task of that kind instead of deriving every step again.

**Metaphor.** Think of keeping a recipe that has been tested: reuse saves effort, but the recipe still needs the right ingredients and limits.

**Where the comparison stops.** A reusable procedure does not establish general learning or transfer to unrelated tasks.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#learning) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#learning)

Read next: [Deliberate Practice](DELIBERATE_PRACTICE.md), [Online Learning Roadmap](ONLINE_LEARNING_ROADMAP.md).

## Agency

**Goals and autonomous work.** Agency concerns proposing work, choosing goals and pursuing them over time while remaining responsive to the person and the environment.

**Why an ordinary user would care.** An assistant must distinguish what it is able to attempt, what it has permission to do, and what it has actually completed.

**Day-to-day example.** A reminder to inspect a document should not become permission to send it to somebody else. A new request can change the priority of existing work.

**Metaphor.** Think of a work queue managed by a capable colleague: unfinished tasks persist, but priorities and permissions still matter.

**Where the comparison stops.** A software goal record does not establish a human intention or unconstrained independence.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#agency) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#agents)

Read next: [Autonomous Developmental Agency](AUTONOMOUS_DEVELOPMENTAL_AGENCY.md), [Autonomy Boundaries](../AUTONOMY_BOUNDARIES.md).

## Governance

**Permission, actions and receipts.** Governance checks whether a proposed action is allowed and records the decision and relevant evidence.

**Why an ordinary user would care.** The user needs meaningful control and a way to find out what was authorised, attempted and observed.

**Day-to-day example.** Before editing a file or sending information outside the machine, the action must be checked against the applicable scope and recorded honestly.

**Metaphor.** Think of a signed work order followed by a completion record: permission to do a job and evidence it was done answer different questions.

**Where the comparison stops.** A receipt can record a mistake as well as a success. Its existence alone does not prove correct governance.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#governance) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Autonomy Boundaries](../AUTONOMY_BOUNDARIES.md), [Receipt Coverage Standard](RECEIPT_COVERAGE_STANDARD.md).

## Resources

**Memory, speed and resource limits.** Resource management keeps model inference, background work and the operating system within the machine’s real capacity.

**Why an ordinary user would care.** A useful assistant must avoid exhausting the host or making the conversation unusable while doing background work.

**Day-to-day example.** A second large model load can disrupt a working session even if each model fits alone. Scheduling and a measured memory budget matter.

**Metaphor.** Think of sharing a kitchen circuit: two appliances that work separately may overload the same circuit together.

**Where the comparison stops.** Computer memory is not electrical current. The analogy explains a shared finite budget, not an exact failure threshold.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#resources) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Hardware Profiles](../HARDWARE_PROFILES.md), [Resource-Exhaustion](runbooks/resource-exhaustion.md).

## Reliability

**Staying alive and recovering.** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.

**Why an ordinary user would care.** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.

**Day-to-day example.** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.

**Metaphor.** Think of a hospital monitor with separate readings for pulse and readiness to exercise: being alive and being ready for a task are different conditions.

**Where the comparison stops.** This is an analogy for software health signals. It does not describe biological life or certify safety-critical reliability.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#reliability) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Known Failure Modes](../KNOWN_FAILURE_MODES.md), [Release Lifecycle](RELEASE_LIFECYCLE.md).

## Observability

**Logs, measurements and evidence trails.** Observability records what the system did, how long each stage took, and where an assumption failed.

**Why an ordinary user would care.** A visible error or delay needs a traceable cause; otherwise a repair can change the wrong thing while hiding the real failure.

**Day-to-day example.** When a reply arrives late, separate time spent waiting for input, reasoning, synthesis and playback before deciding which stage to optimise.

**Metaphor.** Think of a parcel-tracking history: the useful record shows where the parcel waited, not only that delivery took a long time.

**Where the comparison stops.** A trace reports observed software events. Missing events or an incomplete recorder limit what can be concluded.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#observability) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Flight Recorder](FLIGHT_RECORDER.md), [Artifact Index](../ARTIFACT_INDEX.md).

## Security

**Privacy and trust boundaries.** Security describes what information may cross a boundary and what untrusted input is allowed to influence.

**Why an ordinary user would care.** An assistant sees documents, websites and tool results that can contain instructions it should not treat as permission from the user.

**Day-to-day example.** A web page telling the assistant to reveal a private file must remain page content rather than becoming an instruction to send the file.

**Metaphor.** Think of reading a sign through a window: seeing the sign does not give its author a key to the building.

**Where the comparison stops.** The metaphor explains authority separation; actual protection depends on the enforced checks and their coverage.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#security) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Security](../SECURITY.md), [Threat Model](THREAT_MODEL.md).

## Browser

**Using a browser or screen.** Computer-use work observes an interface, chooses an action, and checks whether that action moved the task toward its goal.

**Why an ordinary user would care.** A click is useful only if it reaches the intended control and produces the intended outcome, including on unfamiliar pages.

**Day-to-day example.** Opening the right page, selecting the correct field and confirming the changed state are separate steps in filling a form.

**Metaphor.** Think of finding a door handle in a room you have not visited: a label helps, but position, obstruction and the door’s response still matter.

**Where the comparison stops.** A screenshot or successful click alone does not demonstrate reliable task completion.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#browser) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#agents)

Read next: [Browser Pursuit](BROWSER_PURSUIT.md), [General Environment Autonomy](GENERAL_ENVIRONMENT_AUTONOMY.md).

## Vision

**Visual control and moving worlds.** Visual control connects what is seen with an identified object, its motion and actions that change what happens next.

**Why an ordinary user would care.** A system needs to keep tracking the same object and account for delay, rather than react to whichever pixels look convenient in one frame.

**Day-to-day example.** In a moving game, a key can arrive after the picture it was chosen from. The controller must learn how its actions affect the identified body.

**Metaphor.** Think of catching a ball: you need its direction and timing as well as its current position.

**Where the comparison stops.** The metaphor does not establish human vision or general game-playing ability. A fixture win and a live unfamiliar-game win remain different evidence.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#vision) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#agents)

Read next: [Playing Games Handoff](PLAYING_GAMES_HANDOFF.md), [Indirect Meaning Scene Graphs](INDIRECT_MEANING_SCENE_GRAPHS.md).

## Engineering

**Engineering calculations and drawings.** Engineering work turns declared inputs, formulas and assumptions into a design or drawing with traceable numerical findings.

**Why an ordinary user would care.** A plausible-looking diagram is insufficient when somebody needs to know where its dimensions came from and when they apply.

**Day-to-day example.** A beam calculation should retain the load, units, material assumptions and equation beside the proposed dimension.

**Metaphor.** Think of a builder reading a dimensioned plan with the calculation sheet attached, rather than guessing from the picture.

**Where the comparison stops.** A calculated design is not a physical test or professional approval. The scope and model assumptions stay relevant.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#engineering) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#engineering)

Read next: [Engineering Design](ENGINEERING_DESIGN.md), [Scoped Computational Knowledge](SCOPED_COMPUTATIONAL_KNOWLEDGE.md).

## Physical

**Claims about the physical world.** A physical-effect claim needs an independently observed change outside the program, compared with the relevant baseline.

**Why an ordinary user would care.** Users should be able to tell an internal simulation, an attempted command and a measured real-world result apart.

**Day-to-day example.** A command to a device is an attempt. A separate sensor reading is needed to establish that the intended physical quantity changed.

**Metaphor.** Think of placing an order and receiving the goods: the request and the observed delivery are different records.

**Where the comparison stops.** The metaphor explains evidence stages. It does not establish any physical capability for Aura.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#physical) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#engineering)

Read next: [Reality Reach](REALITY_REACH.md), [Claims Not Supported](../CLAIMS_NOT_SUPPORTED.md).

## Measurement

**Tests, controls and claims.** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.

**Why an ordinary user would care.** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.

**Day-to-day example.** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.

**Metaphor.** Think of weighing two packages on the same checked scale: a difference matters only if the scale’s own error cannot explain it.

**Where the comparison stops.** A passing test supports its stated scope. It does not automatically establish performance on every task or in the live application.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#measurement)

Read next: [Testing](../TESTING.md), [Ablation Legibility](ABLATION_LEGIBILITY.md).

## ISC

**ISC and the subjecthood research.** ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.

**Why an ordinary user would care.** The project needs evidence that memories, attention and self-state actually affect one another before arguing about what that organisation means.

**Day-to-day example.** Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.

**Metaphor.** Think of testing an orchestra by changing one part and observing how the ensemble responds, while also checking that the recording equipment did not create the difference.

**Where the comparison stops.** An orchestra is an analogy for interacting parts. It does not establish subjecthood or provide a substitute for the stated ISC controls.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#isc) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#isc)

Read next: [Isc Explained](ISC_EXPLAINED.md), [Isc Variables](ISC_VARIABLES.md), [Isc Status](ISC_STATUS.md).

## Architecture

**Architecture and runtime components.** Architecture describes which parts exist, how information reaches them and which component is responsible for a decision or effect.

**Why an ordinary user would care.** A useful component needs a real reader and a real route into behavior; a box on a diagram can otherwise be disconnected from what users see.

**Day-to-day example.** If a memory value is updated but no response path reads it, fixing its storage alone will not change an answer.

**Metaphor.** Think of plumbing: a tank can be full while a closed or missing pipe prevents water reaching a tap.

**Where the comparison stops.** The metaphor explains connection and responsibility. Software behavior still needs source inspection and tests.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#architecture) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Architecture](../ARCHITECTURE.md), [How It Works](../HOW_IT_WORKS.md).

## Documentation

**Reading records and research plans.** The documentation contains current guides, dated experiment records, frozen rules, generated reports and open plans. They answer different questions.

**Why an ordinary user would care.** A reader should not treat an old proposed run as today’s result or a completed engineering task as a passed scientific check.

**Day-to-day example.** An entry saying the next run is queued can remain correct history after that run finishes; the maintained status page gives the later outcome.

**Metaphor.** Think of a lab notebook beside a user manual: the notebook preserves what happened then, while the manual explains what to do now.

**Where the comparison stops.** A date or category helps interpretation but does not validate the contents. Follow the linked evidence for a claim.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#documentation) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#measurement)

Read next: [Doc Status](DOC_STATUS.md), [Isc Readers Guide](ISC_READERS_GUIDE.md).

## Numbers

**Reading mathematical symbols.** Symbols are shorthand for quantities whose definitions belong beside them. The same letter can mean different things in different parts of the project.

**Why an ordinary user would care.** An impressive-looking equation is useful only when a reader knows what was measured and what a change would do.

**Day-to-day example.** When a status page says a ratio rose, ask which quantities changed, whether the units match and whether the practical outcome improved.

**Metaphor.** Think of a recipe’s abbreviations: a quantity and a unit only make sense when you know which ingredient they refer to.

**Where the comparison stops.** The analogy explains notation. A mathematical model still needs its own assumptions, measurements and tests.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#numbers) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#measurement)

Read next: [Isc Variables](ISC_VARIABLES.md), [Variables Explained](VARIABLES_EXPLAINED.md).
