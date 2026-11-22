# Aura in plain English

Aura is an experimental assistant for Apple Silicon Macs. It saves conversation records and can look them up when you return with a related question. The app can also make a decision about a tool request and keep a record of what happened.

[How a chat turn works](../HOW_IT_WORKS.md) · [Measured results](../CLAIMS_MATRIX.md)

## Topics

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

Aura is a local AI research system built around a language model, persistent records and software that manages actions and ongoing work.

Before asking Aura to work with a document, you need a ready model, an accessible file and enough memory for the running system.

[Install](../INSTALL.md) · [Operator Guide](OPERATOR_GUIDE.md) · [Model Card](../MODEL_CARD.md)

## Conversation

Conversation support turns a request into a reply and, when enabled, into speech. Streaming lets delivery begin before every later stage finishes.

When you interrupt a spoken reply, the old reply should stop owning the conversation and the new turn should take priority.

[Ambient Voice And Media](AMBIENT_VOICE_AND_MEDIA.md) · [User Guide](USER_GUIDE.md)

## Models

The language model supplies text generation. Steering changes selected internal calculations using a measured direction and a controlled strength.

A change in internal condition may alter how a reply is formed; a channel measurement must show the change rather than relying on the reply sounding different.

[Model Roster](MODEL_ROSTER.md) · [Isc Status](ISC_STATUS.md)

## Memory

Memory keeps records beyond the immediate request and selects relevant records for the current turn.

If you correct a fact about a project, later answers should use the correction rather than retrieving the older claim as equally reliable.

[Memory Architecture](MEMORY_ARCHITECTURE.md) · [Memory Card](../MEMORY_CARD.md)

## Affect

Aura uses internal variables for positive or negative condition, activation, needs and operating strain. Their role is established by what they change.

Under resource strain, the system may need to reduce expensive background work while preserving the conversation.

[Allostasis Engine](ALLOSTASIS_ENGINE.md) · [Payoff](PAYOFF.md)

## Attention

Several concerns can propose work, but attention selects what has influence now and makes selected information available elsewhere.

Your new question should not be overwritten by a background task that happened to win a previous competition.

[Cognitive Architecture Adoption](COGNITIVE_ARCHITECTURE_ADOPTION.md) · [Organismal Workspace Theory](ORGANISMAL_WORKSPACE_THEORY.md)

## Reasoning

The reasoning programmes investigate whether extra computation or learned internal representations improve answers on tasks not used to choose the method.

On a difficult constraint problem, a better internal method should increase correct answers or expose uncertainty rather than only generate longer prose.

[Recursive Latent Cortex](RECURSIVE_LATENT_CORTEX.md) · [Reasoning Engines](REASONING_ENGINES.md) · [Mathematical Claim Limits](MATHEMATICAL_CLAIM_LIMITS.md)

## Learning

Learning changes later behavior using earlier outcomes. A kept procedure is executable work retained for reuse under a defined checking rule.

After solving one kind of task, the system may reuse a checked procedure on a later task of that kind instead of deriving every step again.

[Deliberate Practice](DELIBERATE_PRACTICE.md) · [Online Learning Roadmap](ONLINE_LEARNING_ROADMAP.md)

## Agency

Agency concerns proposing work, choosing goals and pursuing them over time while remaining responsive to the person and the environment.

A reminder to inspect a document should not become permission to send it to somebody else. A new request can change the priority of existing work.

[Autonomous Developmental Agency](AUTONOMOUS_DEVELOPMENTAL_AGENCY.md) · [Autonomy Boundaries](../AUTONOMY_BOUNDARIES.md)

## Governance

Governance checks whether a proposed action is allowed and records the decision and relevant evidence.

Before editing a file or sending information outside the machine, the action must be checked against the applicable scope and recorded honestly.

[Autonomy Boundaries](../AUTONOMY_BOUNDARIES.md) · [Receipt Coverage Standard](RECEIPT_COVERAGE_STANDARD.md)

## Resources

Resource management keeps model inference, background work and the operating system within the machine’s real capacity.

A second large model load can disrupt a working session even if each model fits alone. Scheduling and a measured memory budget matter.

[Hardware Profiles](../HARDWARE_PROFILES.md) · [Resource-Exhaustion](runbooks/resource-exhaustion.md)

## Reliability

Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.

If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.

[Known Failure Modes](../KNOWN_FAILURE_MODES.md) · [Release Lifecycle](RELEASE_LIFECYCLE.md)

## Observability

Observability records what the system did, how long each stage took, and where an assumption failed.

When a reply arrives late, separate time spent waiting for input, reasoning, synthesis and playback before deciding which stage to optimise.

[Flight Recorder](FLIGHT_RECORDER.md) · [Artifact Index](../ARTIFACT_INDEX.md)

## Security

Security describes what information may cross a boundary and what untrusted input is allowed to influence.

A web page telling the assistant to reveal a private file must remain page content rather than becoming an instruction to send the file.

[Security](../SECURITY.md) · [Threat Model](THREAT_MODEL.md)

## Browser

Computer-use work observes an interface, chooses an action, and checks whether that action moved the task toward its goal.

Opening the right page, selecting the correct field and confirming the changed state are separate steps in filling a form.

[Browser Pursuit](BROWSER_PURSUIT.md) · [General Environment Autonomy](GENERAL_ENVIRONMENT_AUTONOMY.md)

## Vision

Visual control connects what is seen with an identified object, its motion and actions that change what happens next.

In a moving game, a key can arrive after the picture it was chosen from. The controller must learn how its actions affect the identified body.

[Playing Games Handoff](PLAYING_GAMES_HANDOFF.md) · [Indirect Meaning Scene Graphs](INDIRECT_MEANING_SCENE_GRAPHS.md)

## Engineering

Engineering work turns declared inputs, formulas and assumptions into a design or drawing with traceable numerical findings.

A beam calculation should retain the load, units, material assumptions and equation beside the proposed dimension.

[Engineering Design](ENGINEERING_DESIGN.md) · [Scoped Computational Knowledge](SCOPED_COMPUTATIONAL_KNOWLEDGE.md)

## Physical

A physical-effect claim needs an independently observed change outside the program, compared with the relevant baseline.

A command to a device is an attempt. A separate sensor reading is needed to establish that the intended physical quantity changed.

[Reality Reach](REALITY_REACH.md) · [Claims Not Supported](../CLAIMS_NOT_SUPPORTED.md)

## Measurement

A test makes a specific claim answerable by a stated measurement, comparison and decision rule.

To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.

[Testing](../TESTING.md) · [Ablation Legibility](ABLATION_LEGIBILITY.md)

## ISC

ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.

Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.

[Isc Explained](ISC_EXPLAINED.md) · [Isc Variables](ISC_VARIABLES.md) · [Isc Status](ISC_STATUS.md)

## Architecture

Architecture describes which parts exist, how information reaches them and which component is responsible for a decision or effect.

If a memory value is updated but no response path reads it, fixing its storage alone will not change an answer.

[Architecture](../ARCHITECTURE.md) · [How It Works](../HOW_IT_WORKS.md)

## Documentation

The documentation contains current guides, dated experiment records, frozen rules, generated reports and open plans. They answer different questions.

An entry saying the next run is queued can remain correct history after that run finishes; the maintained status page gives the later outcome.

[Doc Status](DOC_STATUS.md) · [Isc Readers Guide](ISC_READERS_GUIDE.md)

## Numbers

Symbols are shorthand for quantities whose definitions belong beside them. The same letter can mean different things in different parts of the project.

When a status page says a ratio rose, ask which quantities changed, whether the units match and whether the practical outcome improved.

[Isc Variables](ISC_VARIABLES.md) · [Variables Explained](VARIABLES_EXPLAINED.md)
