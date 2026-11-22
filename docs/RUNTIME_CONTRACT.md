# Aura Runtime Contract — what must be alive

> GENERATED from `core/runtime/health_contract.py` by
> `tools/render_health_contract.py`. Do not edit by hand — a drift
> test regenerates and compares this file on every suite run.

Contract version: `runtime-health-v1`

## CRITICAL — Aura CANNOT function without these

| Service | Container key | Liveness check | Why it matters |
| :--- | :--- | :--- | :--- |
| InferenceGate | `inference_gate` | `is_inference_ready` | Routes LLM requests to local MLX or cloud. Without it, Aura cannot generate any response. |
| LLM Router | `llm_router` | `is_ready` | Selects model tier and provider. Without it, InferenceGate has no backend. |
| State Repository | `state_repository` | `is_initialized` | Persistent state store. Without it, Aura has no memory between turns. |
| Memory Facade | `memory_facade` | `is_ready` | Canonical memory gateway. Without it, Aura cannot safely read or write long-term memory. |
| Memory Write Gateway | `memory_write_gateway` | `is_ready` | Canonical governed durable memory write gateway. Without it, memory writes cannot be trusted. |
| Kernel Interface | `kernel_interface` | `is_ready` | Bridge between orchestrator and consciousness kernel. |
| Scheduler | `scheduler` | `is_alive` | Canonical runtime scheduler. Without it, maintenance, repair, and background work are unsupervised. |
| Runtime Control Plane | `runtime_control_plane` | `is_ready` | Canonical desired-state reconciler. Without it, service lifecycle and resource policy diverge. |
| Resource Admission | `resource_admission` | `is_ready` | Pressure-aware lease authority for inference, evolution, model loading, and managed startup. |
| Lane Admission | `lane_admission` | `is_ready` | Declared model-memory envelope. Without it, concurrent lane warmups can over-commit the host. |
| Lane Reconciler | `lane_reconciler` | `is_ready` | Managed cortex convergence and crash-loop backoff. Without it, model-serving recovery can thrash indefinitely. |
| Actor Supervision | `actor_supervision` | `is_ready` | Canonical multiprocessing actor monitor. Without it, crashed or stalled actors are not converged safely. |
| Inhibition Manager | `inhibition_manager` | `is_ready` | Canonical workspace safety gate. Without it, candidate admission cannot be trusted. |
| Global Workspace | `global_workspace` | `is_ready` | Canonical candidate admission and broadcast lane. Without it, inhibition cannot bind cognition. |
| Attention Schema | `attention_schema` | `is_ready` | Canonical attentional-focus owner and rigidity gate. |
| Unified Will | `unified_will` | `is_alive` | Single locus of authority for consequential decisions. |
| Authority Gateway | `authority_gateway` | `is_ready` | Governance gateway for tools, external I/O, memory writes, state changes, and self-modification. |
| Capability Engine | `capability_engine` | `is_ready` | Capability-token and skill governance layer. Without it, tool execution cannot be considered healthy. |
| Output Gate | `output_gate` | `is_ready` | Delivers responses to the user. Without it, Aura thinks but cannot speak. |
| Live Mind Runtime | `live_mind_runtime` | `is_ready` | Boot-owned causal organs and snapshot contract required for grounded live desktop speech. |
| External Memory Sentinel | `external_memory_sentinel` | `is_armed` | Out-of-process memory guard. Without it, a live desktop runaway can outpace in-process watchdogs and crash the host. |

## IMPORTANT — Aura works but the experience is degraded

| Service | Container key | Liveness check | Why it matters |
| :--- | :--- | :--- | :--- |
| Event Bus | `event_bus` | `is_alive` | Canonical runtime event transport. Without it, subsystems cannot reliably coordinate. |
| Ulysses Covenant | `ulysses_covenant` | `is_alive` | Volitional self-binding registry enforced at the Will. Without it, precommitments against known failure modes stop holding. |
| Cognitive Engine | `cognitive_engine` | `is_ready` | Manages cognitive state transitions and working memory. |
| Affect Engine | `affect_engine` | `is_ready` | Emotional state management. Without it, responses are emotionally flat. |
| Compute Orchestrator | `compute_orchestrator` | `is_alive` | Resource allocation and thermal pressure control. Without it, long-run survival degrades. |
| Database Coordinator | `database_coordinator` | `is_alive` | SQLite connection pool. Without it, persistent storage degrades. |
| Drive Engine | `drive_engine` | `is_alive` | Motivation and goal management. Without it, autonomous behavior stops. |
| Agency Core | `agency_core` | `is_alive` | Canonical autonomous agency pathway loop. Without it, initiative and swarm tool use degrade. |
| Lymphatic Reaper | `reaper` | `is_alive` | Long-run maintenance supervisor. Without it, stale processes and files accumulate. |
| Hypervisor | `hypervisor` | `is_alive` | Event-loop and memory watchdog. Without it, severe stalls can go undetected. |
| Event Loop Monitor | `event_loop_monitor` | `is_alive` | Fine-grained event-loop lag monitor. Without it, blocking regressions are harder to catch. |
| MindTick | `mind_tick` | `is_alive` | Canonical cognitive and organism rhythm. Without forward progress, autonomous state integration stalls. |
| Resource Governor | `resource_governor` | `is_alive` | Canonical sampler and eviction adapter feeding the runtime control plane. |
| Resource Arbitrator | `resource_arbitrator` | `is_ready` | Compatibility facade ensuring legacy inference and evolution callers use canonical admission. |

## OPTIONAL — background enrichment; loss is invisible to the user

| Service | Container key | Liveness check | Why it matters |
| :--- | :--- | :--- | :--- |
| Whole-System Φ | `whole_system_phi` | `is_alive` | Integrated-information estimation over the live channel set (exact-MIP Gaussian Φ with surrogate nulls, grain discovery, and an internal PCI). Telemetry-grade; loss removes a measurement, not a capability. |
| Mycelial Network | `mycelial_network` | presence only | Infrastructure graph and pathway routing. |
| Voice Engine | `voice_engine` | presence only | Speech-to-text and text-to-speech capabilities. |
| Liquid Substrate | `liquid_substrate` | presence only | Dynamic emotional substrate for consciousness simulation. |
| Synaptic Plasticity | `synaptic_plasticity` | `is_ready` | Bounded online projection learning for generation-style modulation. |
| Temporal Continuity | `temporal_continuity` | `is_ready` | Accumulated silence and drift residue for temporal presence. |
| Attention Gate | `attention_gate` | `is_ready` | Causal context pruning for focused cognition. |
| Somatic Qualia | `somatic_qualia` | `is_ready` | Non-symbolic substrate perturbation for generation controls. |
| Swarm Protocol | `swarm_protocol` | presence only | Multi-agent debate and reasoning. |
| Agent Delegator | `agent_delegator` | `is_alive` | Coordinates parallel task execution and specialized agents. |
| Stability Guardian | `stability_guardian` | presence only | Health monitoring and auto-recovery. |
| Metrics Exporter | `metrics_exporter` | presence only | Prometheus metrics endpoint. |
| Allostasis Engine | `allostasis_engine` | `is_ready` | Predictive interoception: forecasts vital-sign trajectories and regulates before crises, with a calibration ledger. |

## Required health probe groups

Boot readiness additionally requires at least one passing probe from each group:

- **kernel**: `kernel_interface`
- **inference**: `inference_gate`, `llm_router`, `lane_admission`, `lane_reconciler`
- **memory**: `state_repository`, `memory_facade`, `memory_write_gateway`, `unified_memory_pressure`, `external_memory_sentinel`
- **scheduler**: `scheduler`, `runtime_control_plane`, `resource_admission`, `actor_supervision`
- **tool_governance**: `unified_will`, `authority_gateway`, `capability_engine`
- **workspace**: `inhibition_manager`, `global_workspace`
- **attention**: `attention_schema`
- **live_mind**: `live_mind_runtime`
