# Aura architecture

The desktop app connects a language model to stored context, background
goals, and authorized tools. This page maps those paths to the code.
[Walkthrough](HOW_IT_WORKS.md) · [Generated dependency map](docs/ARCHITECTURE_MAP.md) ·
[Runtime health contract](docs/RUNTIME_CONTRACT.md)

## 0. The Unified Will: decision authority

The Unified Will evaluates a proposed action and returns an authorization
decision. Execution gateways check authority where the action crosses their
boundary. The overt action loop selects a goal step, requests a registered
skill, checks its result, and records the outcome.
[Will](core/governance/will.py) · [Authority gateway](core/executive/authority_gateway.py) ·
[Overt action loop](core/runtime/overt_action_loop.py)

## 1. System model

The orchestrator coordinates foreground requests and periodic background work.
A tick runs phases that prepare changes to state. The committed AuraState
version is atomic. The repository admits and publishes that version in one
owner transaction. A failed phase leaves preceding changes in the candidate
state. Effects outside the state object, including tool calls and writes to
other stores, survive that transaction's refusal.
[Orchestrator](core/orchestrator/main.py) · [State repository](core/state/state_repository.py)

The inference router selects among configured local model roles. An unavailable
model can lead to a qualified local fallback or an explicit refusal. The
request contract disables cloud inference fallback.
[Request contract](core/brain/request_contract.py) · [Models](MODEL_CARD.md)

## 2. The tick

A healthy foreground turn is eleven phases in the priority tick's configured
set. A turn can also need conditional tool execution. The background tick
runs the full 29-phase blueprint.
[Phase blueprint](core/runtime/pipeline_blueprint.py)

Suppressed on a user-facing tick (18 phases, in pipeline order):
NativeMultimodalBridge, EternalMemoryPhase, PerfectEmotionPhase,
PhiConsciousnessPhase, CognitiveIntegrationPhase, ShadowExecutionPhase,
EternalGrowthEngine, TrueEvolutionPhase, InferencePhase, BondingPhase,
GodModeToolPhase, RepairPhase, MemoryConsolidationPhase,
IdentityReflectionPhase, InitiativeGenerationPhase, ConsciousnessPhase,
SelfReviewPhase, LearningPhase.

## 3. Bounded integration models inspired by IIT

The 16-node estimator builds a transition matrix from selected state
variables and uses a spectral partition search. The eight-node affective
subset has an exhaustive comparison. The whole-system estimator uses a
separate approximation. The tests measure the properties defined by each
estimator. [Estimator](core/consciousness/phi_core.py) ·
[Whole-system method](docs/WHOLE_SYSTEM_PHI.md)

## 4. Affective modulation

The affect update phase maintains state such as valence and arousal. The
inference bridge can use those values when it selects sampling settings
and assembles context. [Affect update](core/phases/affect_update.py) ·
[Inference bridge](core/brain/latent_bridge.py)

## 5. Activation steering

Selected directions can change model activations during decoding. The
runtime has separate qualification and serving checks for the direction
and the active model. The ISC report experiment used a fitted valence
direction under controlled conditions.
[Steering](core/consciousness/affective_steering.py) ·
[Certificate](core/consciousness/fusion_certificate.py) · [Report results](docs/ISC_STATUS.md)

## 6. Continuous state

The liquid substrate maintains a changing numerical state. It exposes
derived measures, including valence and arousal, for other parts of the
runtime to use. [Substrate](core/consciousness/liquid_substrate.py)

## 7. Timing-based plasticity

The timing rules adjust connections inside the project’s plasticity model.
The local language model's adaptation and promotion follow a separate
training path. [Timing rules](core/consciousness/stdp_learning.py) ·
[Model adaptation](core/adaptation/online_lora_governor.py)

<a id="8-memory-architecture-116-modules"></a>

## 8. Memory architecture

Working context holds the current request. Episodic records preserve
events, and semantic records supply facts and their provenance. Retrieval
can combine lexical matches with embedding scores. The memory panel reads
its own supported stores and exposes controls for individual records.
[Memory architecture](docs/MEMORY_ARCHITECTURE.md) · [Memory card](MEMORY_CARD.md)

## 9. Attention and the self-model

Attention, the self-model, and integration estimates have distinct code
paths. The project studies how these mechanisms interact under changes
to recorded state. [Subsystem guides](docs/README.md) ·
[ISC experiment](docs/ISC_EXPLAINED.md)

## 10. Response style

The model checkpoint, instructions, and retrieved conversation records
can shape the reply. The installed artifact determines which model and
training are in use. [Model card](MODEL_CARD.md) · [Memory card](MEMORY_CARD.md)

## 11. Model formats

The model artifacts can use different quantization, which affects disk
size and inference memory. Use a measured workload budget when comparing
models. [Model roles](MODEL_CARD.md) · [Hardware profiles](HARDWARE_PROFILES.md)

## 12. Limitations and mitigations

The fault registry records observed incidents and failure paths found in
the source. Each entry names the conditions, detection, and available
mitigations. [Fault registry](docs/FMEA.md) · [Runbooks](docs/runbooks/)

## 13. Open research program

The results pages list open comparisons for model recurrence, broader
reasoning performance, and the ISC control experiments.
[Recurrence results](docs/RECURSIVE_LATENT_CORTEX.md) ·
[ISC results](docs/ISC_STATUS.md) · [Open claims](CLAIMS_NOT_SUPPORTED.md)

## 14. Mechanism tests

This suite exercises specified state and mechanism checks in its test
environment. Read the individual conditions and inputs to understand what
a result covers. [Test source](tests/test_null_hypothesis_defeat.py)

## 15. The reasoning, self-model, and resilience layer

Reasoning combines model drafts with optional search and verification.
The runtime has separate health and recovery checks that can defer work
or record a fault. [Reasoning engines](docs/REASONING_ENGINES.md) ·
[Health checks](docs/RUNTIME_CONTRACT.md)

## 16. Proof, knowledge, and state checks

The project uses code-level checks for proofs, knowledge, and runtime
state. For individual contracts, follow the implementation and its
evidence. [Architecture adoption](docs/ENGINEERING_ADOPTION.md)

## 17. Layering and runtime controls

Layering rules are written in package `DEPS` files and checked with
`make layering`. The engineering adoption guide describes the lock,
resource, and lifecycle checks and the incidents behind them.
[Contributor guide](CONTRIBUTING.md) · [Engineering adoption](docs/ENGINEERING_ADOPTION.md)

## 18. The reality boundary

Physical requests carry a declared channel and evidence requirement.
The application records dispatch and observed effects separately.
Untrusted code execution has a distinct process and sandbox policy.
[Reachability and evidence](docs/REALITY_REACH.md) · [Threat model](docs/THREAT_MODEL.md)

## 19. Recursive Latent Cortex

The recurrence research tested frozen-model loops and learned readings
from hidden states. Their results have different scopes. The October
G09 comparison measures kept procedures on held-out math and planning
tasks. [Experiments](docs/RECURSIVE_LATENT_CORTEX.md) ·
[G09 result](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md)

## 20. Knowledge, kept procedures, and learning in use

The local corpus can search Wikipedia passages and pages saved from web
requests. The chat path also consults accepted procedures when it can
identify a task type. Some routes return the computed answer directly.
Others give the result to the model as context. During idle work a learner
can group solved examples and test a proposed procedure.
[Corpus](core/knowledge/) · [Chat routing](core/phases/kept_procedures_in_a_turn.py) ·
[Learning trial and its scope](docs/evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md)
