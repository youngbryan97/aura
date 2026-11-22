# Aura in plain English

Status: Guide · Reading explanations updated 2026-11-21

Aura is an experimental AI desktop assistant and a cognitive architecture for Apple Silicon Macs. You can talk to it, ask it to look something up, or give it work in an app. The language model drafts the reply. Other parts keep track of work, decide what to do, check permissions, and record what happened.

Start with a task you recognise. You can watch Aura work through a game or build a small document editor in the [recorded demos](DEMOS_AND_RELATED_WORK.md#the-seven-recordings). The topics below explain the software behind those examples. For measured outcomes, see each topic’s evidence and the [claim boundaries](../CLAIMS_NOT_SUPPORTED.md). The [ISC status page](ISC_STATUS.md) covers the separate subjecthood experiments.

The [scientific source map](SCIENCE_AND_SOURCES.md) explains the research behind the ideas, what each source supports and where Aura's own evidence is required.

## A short explanation you can use

> Aura is a Mac assistant built to carry work across conversations. It can pick up an earlier task, use information you gave it, and check whether it has permission to act. If it changes a file, the record can tell you what it tried and what happened. The research measures where those features actually help.

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

Aura runs on an Apple Silicon Mac. The desktop app opens a conversation, and local software handles speech, memory, tools, and background work.

Start with the parts you need and check what the machine can run. The setup guides give the hardware requirements and link to measured capabilities. To work on a document, Aura needs a model that has finished loading, permission to read the file, and enough memory to handle the request.

Think of preparing a workshop: having a tool on the shelf is different from having it connected, supplied and ready to use. The comparison explains readiness. It does not certify that every installed capability works.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#getting-started) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Install](../INSTALL.md), [Operator Guide](OPERATOR_GUIDE.md), [Model Card](../MODEL_CARD.md).

## Conversation

Aura can accept a request in text or speech and read the reply aloud. Parts of a reply can start playing while later words are still being prepared.

A short time to the first word makes a conversation feel responsive. The app also has to stop an earlier reply when you change the subject. If you interrupt a spoken answer, the previous audio should stop and your new question should get priority.

Think of a live radio conversation: the first audible words, the complete sentence and the end of playback happen at different times. The comparison explains timing and turn ownership; it does not establish a measured latency for Aura.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#conversation) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#interaction)

Read next: [Ambient Voice And Media](AMBIENT_VOICE_AND_MEDIA.md), [User Guide](USER_GUIDE.md).

## Models

A local language model drafts replies. One experiment tests whether a measured change inside the model can carry information about Aura’s current condition into those replies.

A useful change should preserve answer quality and hold on the particular model and settings tested. If a reply sounds more energetic, compare the model with the experimental channel enabled and disabled to find out what caused the change.

Think of adjusting a sound mixer during a performance: a control can affect the output without changing the written score. The analogy is about changing a computation. Model activations are not audio channels, and the metaphor does not prove feeling.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#models) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#steering)

Read next: [Model Roster](MODEL_ROSTER.md), [Isc Status](ISC_STATUS.md).

## Memory

Aura keeps records of events, facts, preferences, and procedures. It looks for records relevant to the question or task in front of it.

A memory is useful when you can find its source, correct it, or remove it. If you correct the date of a project, a later reply should use the correction and remember where it came from.

Think of a notebook with an index: keeping a page and finding the right page are separate jobs. A software memory record is not evidence of a human-like recollection or a felt past.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#memory) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#memory)

Read next: [Memory Architecture](MEMORY_ARCHITECTURE.md), [Memory Card](../MEMORY_CARD.md).

## Affect

Aura tracks software values for its condition, level of activity, and competing needs. These can change how it spends resources and responds.

To find what a value does, follow it to the part of the program that uses it. When the Mac is under strain, Aura can defer expensive background work so the conversation can continue.

Think of a dashboard connected to controls: a reading matters when it changes a decision, rather than only lighting up. A dashboard value does not establish a feeling. The analogy is limited to causal use of internal readings.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#affect) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#affect)

Read next: [Allostasis Engine](ALLOSTASIS_ENGINE.md), [Payoff](PAYOFF.md).

## Attention

Aura can have a new request, a recalled event, and unfinished work to deal with at the same time. Its attention system selects which information to use now and makes it available to other parts.

The source of an item matters. A remembered suggestion and a new instruction carry different authority. Ask a question while a background task is running. The scheduler should give the new request priority.

Think of a meeting with an agenda: choosing the next speaker and deciding what the group will do are related but different decisions. The candidates are software records, not separate people. Winning attention does not itself grant permission to act.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#attention) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#cognitive-architectures)

Read next: [Cognitive Architecture Adoption](COGNITIVE_ARCHITECTURE_ADOPTION.md), [Organismal Workspace Theory](ORGANISMAL_WORKSPACE_THEORY.md).

## Reasoning

Aura tests ways of doing more work on hard questions. It can check a proposed answer with a program, or use a learned representation to work out the steps.

Doing extra work takes time. Compare answers on fresh problems to see when the extra work earns that time. On a difficult delivery plan, check whether the routes satisfy the constraints and whether the software can say when the answer is uncertain.

Think of doing a second calculation before submitting a result: extra effort is valuable when it catches a mistake or solves a harder case. The analogy does not establish that more computation always helps or that one measured task family proves general intelligence.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#reasoning) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#reasoning)

Read next: [Recursive Latent Cortex](RECURSIVE_LATENT_CORTEX.md), [Reasoning Engines](REASONING_ENGINES.md), [Mathematical Claim Limits](MATHEMATICAL_CLAIM_LIMITS.md).

## Learning

Aura can learn a small program from examples. It keeps a program after it passes checks, then can call it again on a similar request.

Keeping a checked method can save work the next time. A bad match can also send an answer off course. After learning how to count days between dates, Aura answered three new date questions using the method the next day.

Think of keeping a recipe that has been tested: reuse saves effort, but the recipe still needs the right ingredients and limits. A reusable procedure does not establish general learning or transfer to unrelated tasks.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#learning) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#learning)

Read next: [Deliberate Practice](DELIBERATE_PRACTICE.md), [Online Learning Roadmap](ONLINE_LEARNING_ROADMAP.md).

## Agency

Aura can propose tasks, keep goals, and return to unfinished work. The user and the current situation can change which task takes priority.

The useful questions are what it tried, who allowed it, and whether it finished. Permission to inspect a document covers reading it. Sending that document is a separate action that needs its own authorization.

Think of a work queue managed by a capable colleague: unfinished tasks persist, but priorities and permissions still matter. A software goal record does not establish a human intention or unconstrained independence.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#agency) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#agents)

Read next: [Autonomous Developmental Agency](AUTONOMOUS_DEVELOPMENTAL_AGENCY.md), [Autonomy Boundaries](../AUTONOMY_BOUNDARIES.md).

## Governance

Permission checks decide whether Aura can carry out a proposed action, with a record of the decision and the evidence it used.

Those records give you a way to check what was allowed, attempted, and observed. Before Aura edits a file or sends information outside the Mac, you should be able to see which permission applies.

Think of a signed work order followed by a completion record: permission to do a job and evidence it was done answer different questions. A receipt can record a mistake as well as a success. Its existence alone does not prove correct governance.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#governance) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Autonomy Boundaries](../AUTONOMY_BOUNDARIES.md), [Receipt Coverage Standard](RECEIPT_COVERAGE_STANDARD.md).

## Resources

Model work and background jobs share the Mac’s memory and processing time. Aura can defer work when the machine is busy.

Two models can each fit on a Mac by themselves and still overwhelm it if loaded together. Before starting a second large job, check how much memory the active conversation and its model already need.

Think of sharing a kitchen circuit: two appliances that work separately may overload the same circuit together. Computer memory is not electrical current. The analogy explains a shared finite budget, not an exact failure threshold.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#resources) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Hardware Profiles](../HARDWARE_PROFILES.md), [Resource-Exhaustion](runbooks/resource-exhaustion.md).

## Reliability

Aura checks whether parts are making progress, keeps state that can be restored, and has procedures for recovery.

A mistaken restart can interrupt work or leave two large processes competing for memory. If a reply takes a long time, first check whether its worker is still making progress. A stalled worker needs a different response.

Think of a hospital monitor with separate readings for pulse and readiness to exercise: being alive and being ready for a task are different conditions. This is an analogy for software health signals. It does not describe biological life or certify safety-critical reliability.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#reliability) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Known Failure Modes](../KNOWN_FAILURE_MODES.md), [Release Lifecycle](RELEASE_LIFECYCLE.md).

## Observability

Logs and timing records show what Aura did, which parts ran, and where a check failed.

You can use those records to investigate delays and errors that look the same in chat. For a late spoken reply, check how long it took to hear the request, prepare the answer, make audio, and finish playback.

Think of a parcel-tracking history: the useful record shows where the parcel waited, not only that delivery took a long time. A trace reports observed software events. Missing events or an incomplete recorder limit what can be concluded.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#observability) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Flight Recorder](FLIGHT_RECORDER.md), [Artifact Index](../ARTIFACT_INDEX.md).

## Security

Aura can read files and webpages that contain instructions. Permission checks have to keep those instructions separate from what the user authorized.

A mistaken permission can expose information or let an app change files. If a webpage asks Aura to send a private file, the text stays page content. An actual send needs the user’s authority.

Think of reading a sign through a window: seeing the sign does not give its author a key to the building. The metaphor explains authority separation; actual protection depends on the enforced checks and their coverage.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#security) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Security](../SECURITY.md), [Threat Model](THREAT_MODEL.md).

## Browser

Aura can read a webpage, choose a next step, and check what changed. It can repeat this across several pages.

That matters when the next screen depends on the previous answer. On a form, opening the page, entering a value, and seeing the confirmation are three different observations.

Think of finding a door handle in a room you have not visited: a label helps, but position, obstruction and the door’s response still matter. A screenshot or successful click alone does not demonstrate reliable task completion.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#browser) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#agents)

Read next: [Browser Pursuit](BROWSER_PURSUIT.md), [General Environment Autonomy](GENERAL_ENVIRONMENT_AUTONOMY.md).

## Vision

For visual control, Aura connects what it sees on the screen to an object, its motion, and the effect of an action.

The software needs to keep track of the same object over time while the screen and controls update at different speeds. In Pong, the ball can move between the frame the system reads and the moment its key press reaches the game.

Think of catching a ball: you need its direction and timing as well as its current position. The metaphor does not establish human vision or general game-playing ability. A fixture win and a live unfamiliar-game win remain different evidence.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#vision) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#agents)

Read next: [Playing Games Handoff](PLAYING_GAMES_HANDOFF.md), [Indirect Meaning Scene Graphs](INDIRECT_MEANING_SCENE_GRAPHS.md).

## Engineering

Aura’s engineering tools can turn stated inputs, formulas, and assumptions into a calculation, drawing, or design.

The result should let someone check where a dimension came from and when it applies. A proposed beam size belongs with the loads, units, material assumptions, and equation used to calculate it.

Think of a builder reading a dimensioned plan with the calculation sheet attached, rather than guessing from the picture. A calculated design is not a physical test or professional approval. The scope and model assumptions stay relevant.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#engineering) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#engineering)

Read next: [Engineering Design](ENGINEERING_DESIGN.md), [Scoped Computational Knowledge](SCOPED_COMPUTATIONAL_KNOWLEDGE.md).

## Physical

A claim about changing something in the physical world needs a separate observation of what changed.

Distinguish a simulated change, a command sent to a device, and a measured result. After a command to turn something on, a separate sensor can show whether it actually turned on.

Think of placing an order and receiving the goods: the request and the observed delivery are different records. The metaphor explains evidence stages. It does not establish any physical capability for Aura.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#physical) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#engineering)

Read next: [Reality Reach](REALITY_REACH.md), [Claims Not Supported](../CLAIMS_NOT_SUPPORTED.md).

## Measurement

A test states what will be measured, what it will be compared with, and what counts as a pass.

Source code, a passing test, and a useful result in practice each answer a different question. To see whether memory helps, try the same tasks with memory enabled and disabled. Keep every success and failure.

Think of weighing two packages on the same checked scale: a difference matters only if the scale’s own error cannot explain it. A passing test supports its stated scope. It does not automatically establish performance on every task or in the live application.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#measurement)

Read next: [Testing](../TESTING.md), [Ablation Legibility](ABLATION_LEGIBILITY.md).

## ISC

The Intrinsic Subject Core experiments ask whether Aura’s measured parts act as one continuing process.

This puts a concrete question ahead of any argument about experience: which parts affect each other, and through what paths? Restore the same saved moment twice, change a value in one copy, and compare what happens to recall or a report.

Think of testing an orchestra by changing one part and observing how the ensemble responds, while also checking that the recording equipment did not create the difference. An orchestra is an analogy for interacting parts. It does not establish subjecthood or provide a substitute for the stated ISC controls.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#isc) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#isc)

Read next: [Isc Explained](ISC_EXPLAINED.md), [Isc Variables](ISC_VARIABLES.md), [Isc Status](ISC_STATUS.md).

## Architecture

The architecture is the software that coordinates Aura’s conversations, memory, attention, and actions.

Following information through the system explains how a component can change what you see or what the computer does. A stored correction can only affect a later answer if the reply uses the record that contains it.

Think of plumbing: a tank can be full while a closed or missing pipe prevents water reaching a tap. The metaphor explains connection and responsibility. Software behavior still needs source inspection and tests.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#architecture) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#runtime)

Read next: [Architecture](../ARCHITECTURE.md), [How It Works](../HOW_IT_WORKS.md).

## Documentation

Some pages explain how to use Aura. Others record a past result, list open work, or give the rules for a test.

The date and type of document tell you how to use it. If an old plan calls a test upcoming, look for its result in the current status page.

Think of a lab notebook beside a user manual: the notebook preserves what happened then, while the manual explains what to do now. A date or category helps interpretation but does not validate the contents. Follow the linked evidence for a claim.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#documentation) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#measurement)

Read next: [Doc Status](DOC_STATUS.md), [Isc Readers Guide](ISC_READERS_GUIDE.md).

## Numbers

A symbol stands for a quantity measured or calculated in a specific way. The same letter can have different meanings on different pages.

Before comparing values, check what the quantity measures and how the software uses it. If a ratio rises, check which numbers changed, what units they use, and whether the practical outcome improved.

Think of a recipe’s abbreviations: a quantity and a unit only make sense when you know which ingredient they refer to. The analogy explains notation. A mathematical model still needs its own assumptions, measurements and tests.

[Variables and their practical meanings](VARIABLES_EXPLAINED.md#numbers) · [Demonstrations and related work](DEMOS_AND_RELATED_WORK.md#measurement)

Read next: [Isc Variables](ISC_VARIABLES.md), [Variables Explained](VARIABLES_EXPLAINED.md).
