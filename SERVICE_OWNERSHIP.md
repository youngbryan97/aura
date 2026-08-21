# Aura Subsystem and Service Ownership Manifest

This file outlines every registered service, its source code location, registration origin, failure policy, and operational requirements.

| Service | Owner File | Registered By | Required For | Failure Policy |
|---|---|---|---|---|
| `absorbed_voices` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `abstract_thought_layer` | `core/brain/abstract_thought_layer.py` | `core/brain/abstract_thought_layer.py` | boot | `fail-closed` |
| `abstraction_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `active_inference_sampler` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `actor_bus` | `aura_main.py` | `aura_main.py` | boot | `degrade_with_receipt` |
| `adaptive_mood` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `aesthetic_critic` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | optional features | `degrade_with_receipt` |
| `aesthetic_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `affect_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `affect_engine` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `affect_facade` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `affect_manager` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `affective_circumplex` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `affective_steering` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `affective_steering_engine` | `core/consciousness/affective_steering.py` | `core/consciousness/affective_steering.py` | optional features | `degrade_with_receipt` |
| `agency_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `agency_core` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `agency_facade` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `agent_delegator` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `agent_workspace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `alignment` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `alignment_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `anomaly_detector` | `core/cybernetics/ice_layer.py` | `core/cybernetics/ice_layer.py` | boot | `fail-closed` |
| `api_adapter` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `app_registry` | `core/capabilities/app_registry.py` | `core/capabilities/app_registry.py` | optional features | `degrade_with_receipt` |
| `architecture_governor` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `architecture_index` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `attention_schema` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `attention_summarizer` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `audit` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `aura_kernel` | `core/kernel/kernel_interface.py` | `core/kernel/kernel_interface.py` | boot | `fail-closed` |
| `aura_now` | `core/being/runtime.py` | `core/being/runtime.py` | optional features | `degrade_with_receipt` |
| `aura_now_runtime` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `aura_protocol_server` | `core/consciousness/aura_protocol.py` | `core/consciousness/aura_protocol.py` | boot | `fail-closed` |
| `aura_runtime` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `aura_workspace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `authority_gateway` | `core/executive/authority_gateway.py` | `core/executive/authority_gateway.py` | optional features | `degrade_with_receipt` |
| `autonomic_core` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `autonomous_architecture_governor` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `autonomous_brain` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `autonomous_initiative_loop` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `autonomous_self_modification` | `core/autonomy/self_modification.py` | `core/autonomy/self_modification.py` | optional features | `degrade_with_receipt` |
| `autonomous_task_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `ava` | `core/fictional_ai_synthesis.py` | `core/fictional_ai_synthesis.py` | boot | `fail-closed` |
| `ava_social` | `core/fictional/ava.py` | `core/fictional/registry.py` | boot | `fail-closed` |
| `backup_manager` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `backup_system` | `core/safety/self_preservation_safe.py` | `core/safety/self_preservation_safe.py` | boot | `fail-closed` |
| `behavioral_proof` | `core/phenomenal_substrate/philosophical_stance.py` | `core/phenomenal_substrate/philosophical_stance.py` | optional features | `degrade_with_receipt` |
| `being_runtime` | `core/being/runtime.py` | `core/being/runtime.py` | optional features | `degrade_with_receipt` |
| `belief_authority` | `core/constitution.py` | `core/constitution.py` | optional features | `degrade_with_receipt` |
| `belief_challenger` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `belief_graph` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `belief_revision_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `belief_sync` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `blood_brain_barrier` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `branch_manager` | `core/consciousness/parallel_branches.py` | `core/consciousness/parallel_branches.py` | boot | `fail-closed` |
| `browser_controller` | `core/capabilities/browser_controller.py` | `core/capabilities/browser_controller.py` | optional features | `degrade_with_receipt` |
| `bryan_model` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `bryan_model_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `canonical_self` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `canonical_self_engine` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `capability_discovery` | `core/capabilities/capability_discovery.py` | `core/capabilities/capability_discovery.py` | optional features | `degrade_with_receipt` |
| `capability_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `causal_world_model` | `core/brain/causal_world_model.py` | `core/brain/causal_world_model.py` | boot | `fail-closed` |
| `cellular_turnover` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `circadian` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `clipboard_manager` | `core/capabilities/clipboard_manager.py` | `core/capabilities/clipboard_manager.py` | optional features | `degrade_with_receipt` |
| `closed_causal_loop` | `core/consciousness/closed_loop.py` | `core/consciousness/closed_loop.py` | boot | `fail-closed` |
| `code_refiner` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognition` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `cognitive_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognitive_integration` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `cognitive_kernel` | `core/cognition/cognitive_integration_layer.py` | `core/cognition/cognitive_integration_layer.py` | boot | `fail-closed` |
| `cognitive_ledger` | `core/kernel/aura_kernel.py` | `core/kernel/aura_kernel.py` | boot | `fail-closed` |
| `cognitive_loop` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `cognitive_manager` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `cognitive_router` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `commitment_engine` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `composer_node` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `compute_orchestrator` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `concept_bridge` | `core/brain/concept_vector_bridge.py` | `core/brain/concept_vector_bridge.py` | boot | `fail-closed` |
| `concept_linker` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `conscious_substrate` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `consciousness` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `consciousness_bridge` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `consciousness_core` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `consciousness_evidence` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `consciousness_integration` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `consciousness_system` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `constitutional_core` | `core/constitution.py` | `core/constitution.py` | optional features | `degrade_with_receipt` |
| `constitutional_gate` | `core/safety/constitutional_gate.py` | `core/safety/constitutional_gate.py` | optional features | `degrade_with_receipt` |
| `context_manager` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `continuity` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `continuous_cognition` | `core/continuous_cognition.py` | `core/continuous_cognition.py` | optional features | `degrade_with_receipt` |
| `continuous_experience_frame` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `continuous_experience_stream` | `core/consciousness/continuous_experience.py` | `core/consciousness/continuous_experience.py` | optional features | `degrade_with_receipt` |
| `continuous_learner` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `continuous_vision` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `conversation_reflector` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `conversational_momentum_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `cortana` | `core/fictional_ai_synthesis.py` | `core/fictional_ai_synthesis.py` | boot | `fail-closed` |
| `cortana_health` | `core/fictional/cortana.py` | `core/fictional/registry.py` | boot | `fail-closed` |
| `counterfactual_engine` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `credit_assignment` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `critic_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `crsm` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `crsm_lora_bridge` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `cryptolalia_decoder` | `core/brain/cryptolalia_decoder.py` | `core/brain/cryptolalia_decoder.py` | boot | `fail-closed` |
| `curiosity_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `curiosity_explorer` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `database_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `deliberator` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `diagnostics` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `dialectical_crucible` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `discourse_tracker` | `core/social/presence_integration.py` | `core/social/presence_integration.py` | boot | `fail-closed` |
| `dlq` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `document_service` | `core/capabilities/document_service.py` | `core/capabilities/document_service.py` | optional features | `degrade_with_receipt` |
| `dream_journal` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `dreamer_v2` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `drift_monitor` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `drive_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `dynamic_router` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `ears` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `edi` | `core/fictional_ai_synthesis.py` | `core/fictional_ai_synthesis.py` | boot | `fail-closed` |
| `edi_autonomy` | `core/fictional/edi.py` | `core/fictional/registry.py` | boot | `fail-closed` |
| `embodied_interoception` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `emergency_protocol` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `emergent_goal_engine` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `emotional_coloring` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `episodic_memory` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `epistemic_filter` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `epistemic_humility` | `core/adaptation/epistemic_humility.py` | `core/adaptation/epistemic_humility.py` | boot | `fail-closed` |
| `epistemic_state` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `epistemic_tracker` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `event_bus` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `event_loop_monitor` | `core/orchestrator/initializers/hardening.py` | `core/orchestrator/initializers/hardening.py` | boot | `degrade_with_receipt` |
| `evidence_mode` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `evolution_orchestrator` | `core/evolution/evolution_orchestrator.py` | `core/evolution/evolution_orchestrator.py` | optional features | `degrade_with_receipt` |
| `executive_authority` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `executive_closure` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `executive_core` | `core/executive/executive_core.py` | `core/executive/executive_core.py` | optional features | `degrade_with_receipt` |
| `experience_consolidator` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `external_chat` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `feedback_processor` | `core/somatic/action_feedback.py` | `core/somatic/action_feedback.py` | optional features | `degrade_with_receipt` |
| `file_broker` | `core/capabilities/file_broker.py` | `core/capabilities/file_broker.py` | optional features | `degrade_with_receipt` |
| `free_energy_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `global_workspace` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `goal_belief_manager` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `goal_drift_detector` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `goal_engine` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `goal_hierarchy` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | optional features | `degrade_with_receipt` |
| `goal_manager` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `goal_memory` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `growth_ladder` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `healing_swarm` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `health_monitor` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | boot | `fail-closed` |
| `hearing` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `heartstone_values` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `hedonic_gradient` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `hemispheric_split` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `hephaestus_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `heuristic_synthesizer` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `hierarchical_phi` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `hierarchical_planner` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `homeostasis` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `homeostatic_coupling` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `host_automation` | `core/capabilities/host_automation.py` | `core/capabilities/host_automation.py` | optional features | `degrade_with_receipt` |
| `hot_engine` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `hotfix_engine` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `hypervisor` | `core/orchestrator/initializers/hardening.py` | `core/orchestrator/initializers/hardening.py` | boot | `degrade_with_receipt` |
| `identity` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `identity_anchor` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `identity_chronicle` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `identity_guard` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `identity_service` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `immune_system` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `inference_gate` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `inhibition_manager` | `core/ops/resilient_boot.py` | `core/ops/resilient_boot.py` | boot | `fail-closed` |
| `initiative_arbiter` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `initiative_synthesizer` | `core/initiative_synthesis.py` | `core/initiative_synthesis.py` | optional features | `degrade_with_receipt` |
| `inner_monologue` | `core/cognition/cognitive_integration_layer.py` | `core/cognition/cognitive_integration_layer.py` | boot | `fail-closed` |
| `inquiry_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `insight_journal` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `integrity_guard` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `integrity_guardian` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `integrity_monitor` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `intent_router` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `interaction_signals` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `internal_simulator` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `jarvis` | `core/fictional_ai_synthesis.py` | `core/fictional_ai_synthesis.py` | boot | `fail-closed` |
| `jarvis_anticipation` | `core/fictional/jarvis.py` | `core/fictional/registry.py` | boot | `fail-closed` |
| `joy_social` | `skills/joy_social_integration.py` | `skills/joy_social_integration.py` | optional features | `degrade_with_receipt` |
| `keep_awake_controller` | `core/runtime/keep_awake.py` | `core/runtime/keep_awake.py` | optional features | `degrade_with_receipt` |
| `kernel_interface` | `core/kernel/kernel_interface.py` | `core/kernel/kernel_interface.py` | boot | `fail-closed` |
| `knowledge_graph` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `language_center` | `core/cognition/cognitive_integration_layer.py` | `core/cognition/cognitive_integration_layer.py` | boot | `fail-closed` |
| `lazarus` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `life_trace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `lineage_manager` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `liquid_neural_network` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `liquid_state` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `liquid_substrate` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `live_learner` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `llm_interface` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `llm_router` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `loop_monitor` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `markdown_workspace` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `memory` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | boot | `fail-closed` |
| `memory_coordinator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `memory_facade` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `memory_governor` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `memory_guard` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `memory_manager` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | boot | `fail-closed` |
| `memory_monitor` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `memory_subsystem` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `memory_synthesizer` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `memory_vector` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `memory_write_gateway` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `mental_simulator` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | optional features | `degrade_with_receipt` |
| `mesh_cognition` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `meta_cognition_loop` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `meta_cognition_shard` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `meta_evolution` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `metabolic_coordinator` | `core/orchestrator/mixins/boot/boot_background.py` | `core/orchestrator/mixins/boot/boot_background.py` | boot | `fail-closed` |
| `metabolic_monitor` | `core/orchestrator/mixins/boot/boot_background.py` | `core/orchestrator/mixins/boot/boot_background.py` | boot | `fail-closed` |
| `metabolism` | `core/orchestrator/mixins/boot/boot_background.py` | `core/orchestrator/mixins/boot/boot_background.py` | boot | `fail-closed` |
| `metabolism_state` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `metacognition` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `metacognitive_calibrator` | `core/final_engines.py` | `core/final_engines.py` | boot | `fail-closed` |
| `metrics` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `metrics_exporter` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `mhaf` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `mind_model` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `mind_moment` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `mind_state_exporter` | `core/self/mind_state_export.py` | `core/self/mind_state_export.py` | optional features | `degrade_with_receipt` |
| `minimal_selfhood` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `mission_state` | `core/planning/mission_state.py` | `core/planning/mission_state.py` | optional features | `degrade_with_receipt` |
| `mist` | `core/fictional_ai_synthesis.py` | `core/fictional_ai_synthesis.py` | boot | `fail-closed` |
| `mist_scheduler` | `core/fictional/mist.py` | `core/fictional/registry.py` | boot | `fail-closed` |
| `moral` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `moral_reasoning` | `core/morality/master_moral_integration.py` | `core/morality/master_moral_integration.py` | boot | `fail-closed` |
| `morphic_forking` | `core/brain/morphic_forking.py` | `core/brain/morphic_forking.py` | boot | `fail-closed` |
| `morphogenetic_runtime` | `core/morphogenesis/integration.py` | `core/morphogenesis/integration.py` | optional features | `degrade_with_receipt` |
| `motivation_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `motor_cortex` | `core/somatic/motor_cortex.py` | `core/somatic/motor_cortex.py` | optional features | `degrade_with_receipt` |
| `multimodal_orchestrator` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `mycelial_network` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `mycelium` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `narrative_engine` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `narrative_identity` | `core/final_engines.py` | `core/final_engines.py` | boot | `fail-closed` |
| `narrative_thread` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `narrator` | `core/brain/narrator.py` | `core/brain/narrator.py` | boot | `fail-closed` |
| `native_system2` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `neologism_engine` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `nethack_adapter` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `neural_intent_router` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `neural_mesh` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `neurochemical_system` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `nucleus` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `octopus_federation` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `ontology_genesis` | `core/brain/ontology_genesis.py` | `core/brain/ontology_genesis.py` | boot | `fail-closed` |
| `opinion_engine` | `core/social/presence_integration.py` | `core/social/presence_integration.py` | boot | `fail-closed` |
| `orchestrator` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `os_settings` | `core/capabilities/os_settings.py` | `core/capabilities/os_settings.py` | optional features | `degrade_with_receipt` |
| `oscillatory_binding` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `output_gate` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `paraconsistent_engine` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `perceptual_pump` | `core/perception/perceptual_pump.py` | `core/perception/perceptual_pump.py` | optional features | `degrade_with_receipt` |
| `permission_guard` | `core/security/permission_guard.py` | `core/security/permission_guard.py` | optional features | `degrade_with_receipt` |
| `permission_model` | `core/capabilities/permission_model.py` | `core/capabilities/permission_model.py` | optional features | `degrade_with_receipt` |
| `permission_setup` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `persistence` | `core/orchestrator/initializers/core_baseline.py` | `core/orchestrator/initializers/core_baseline.py` | boot | `fail-closed` |
| `persistent_state` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `persona_evolver` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `personality` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `personality_bridge` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `personality_engine` | `core/morality/master_moral_integration.py` | `core/morality/master_moral_integration.py` | boot | `fail-closed` |
| `phenomenal_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `phenomenological_experiencer` | `core/consciousness/integration.py` | `core/consciousness/integration.py` | boot | `fail-closed` |
| `phi_core` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `plasticity_controller` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `platform_root` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `pneuma` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `post_action_verifier` | `core/capabilities/post_action_verifier.py` | `core/capabilities/post_action_verifier.py` | optional features | `degrade_with_receipt` |
| `pre_linguistic` | `core/cognition/pre_linguistic.py` | `core/cognition/pre_linguistic.py` | optional features | `degrade_with_receipt` |
| `precognitive_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `predictive_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `proactive_comm` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `proactive_presence` | `core/social/presence_integration.py` | `core/social/presence_integration.py` | boot | `fail-closed` |
| `probe_manager` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `process_manager` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `prompt_compiler` | `core/brain/llm/compiler.py` | `core/brain/llm/compiler.py` | boot | `fail-closed` |
| `qualia_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `qualia_synthesizer` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `react_loop` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `reaper` | `core/orchestrator/initializers/hardening.py` | `core/orchestrator/initializers/hardening.py` | boot | `degrade_with_receipt` |
| `recovery_engine` | `core/planning/recovery_engine.py` | `core/planning/recovery_engine.py` | optional features | `degrade_with_receipt` |
| `recursive_self_improvement` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `recursive_tom` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `refusal_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `reimplementation_lab` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | optional features | `degrade_with_receipt` |
| `reliability_engine` | `core/reliability_engine.py` | `core/reliability_engine.py` | boot | `fail-closed` |
| `research_cycle` | `core/autonomy/research_cycle.py` | `core/autonomy/research_cycle.py` | boot | `fail-closed` |
| `resilience` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `resilience_engine` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `resource_stakes` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `rsi_lab` | `research/meta_learning_loop.py` | `research/meta_learning_loop.py` | boot | `fail-closed` |
| `runtime_hygiene` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `sandboxed_modifier` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `scar_formation` | `core/memory/scar_formation.py` | `core/memory/scar_formation.py` | optional features | `degrade_with_receipt` |
| `scheduler` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | optional features | `degrade_with_receipt` |
| `scratchpad_engine` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `screen_perception` | `core/perception/screen_perception.py` | `core/perception/screen_perception.py` | optional features | `degrade_with_receipt` |
| `self_awareness_suite` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `self_diagnostics` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `self_model` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `self_modification_engine` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `self_prediction` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `self_report_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `semantic_memory` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `sensory_motor_cortex` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `sensory_system` | `core/perception/sensory_integration.py` | `core/perception/sensory_integration.py` | boot | `fail-closed` |
| `sentience_engine` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `server` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `session_guardian` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `shadow_ast_healer` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `shared_ground` | `core/social/presence_integration.py` | `core/social/presence_integration.py` | boot | `fail-closed` |
| `shutdown_coordinator` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `simulation_well` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `singularity_loops` | `core/evolution/singularity_loops.py` | `core/evolution/singularity_loops.py` | optional features | `degrade_with_receipt` |
| `singularity_monitor` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | boot | `fail-closed` |
| `skill_evolution` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `skill_library` | `core/agency/skill_library.py` | `core/agency/skill_library.py` | boot | `fail-closed` |
| `skill_manager` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `skill_registry` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `skill_router` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `skill_synthesizer` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `skynet` | `core/fictional_ai_synthesis.py` | `core/fictional_ai_synthesis.py` | boot | `fail-closed` |
| `skynet_resilience` | `core/fictional/skynet.py` | `core/fictional/registry.py` | boot | `fail-closed` |
| `sleep_trigger` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `sme` | `core/providers/ops_provider.py` | `core/providers/ops_provider.py` | optional features | `degrade_with_receipt` |
| `snapshot_manager` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `social` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `social_memory` | `core/social/presence_integration.py` | `core/social/presence_integration.py` | boot | `fail-closed` |
| `soma` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `soma_subsystem` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `somatic_marker_gate` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `source_summarizer` | `core/capabilities/source_summarizer.py` | `core/capabilities/source_summarizer.py` | optional features | `degrade_with_receipt` |
| `sovereign_ears` | `core/ops/resilient_boot.py` | `core/ops/resilient_boot.py` | boot | `fail-closed` |
| `sovereign_pruner` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `sovereign_scanner` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `sovereign_watchdog` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `spine` | `core/orchestrator/mixins/boot/boot_identity.py` | `core/orchestrator/mixins/boot/boot_identity.py` | boot | `fail-closed` |
| `stability_guardian` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `star_reasoner` | `core/adaptation/star_reasoner.py` | `core/adaptation/star_reasoner.py` | optional features | `degrade_with_receipt` |
| `state_authority` | `core/state/state_authority.py` | `core/state/state_authority.py` | boot | `fail-closed` |
| `state_machine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `state_repo` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `state_repository` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `stream_of_being` | `core/consciousness/stream_of_being.py` | `core/consciousness/stream_of_being.py` | boot | `fail-closed` |
| `structural_improver` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `structural_mutator` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `subconscious_loop` | `core/consciousness/subconscious_loop.py` | `core/consciousness/subconscious_loop.py` | boot | `fail-closed` |
| `substrate_authority` | `core/consciousness/system.py` | `core/consciousness/system.py` | boot | `fail-closed` |
| `substrate_evolution` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `substrate_voice_engine` | `core/voice/substrate_voice_engine.py` | `core/voice/substrate_voice_engine.py` | boot | `fail-closed` |
| `subsystem_audit` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `supervisor` | `aura_main.py` | `aura_main.py` | boot | `fail-closed` |
| `swarm_protocol` | `core/orchestrator/boot.py` | `core/orchestrator/boot.py` | boot | `fail-closed` |
| `system2_search` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | optional features | `degrade_with_receipt` |
| `system_governor` | `core/orchestrator/mixins/boot/boot_resilience.py` | `core/orchestrator/mixins/boot/boot_resilience.py` | boot | `fail-closed` |
| `system_monitor` | `core/providers/cognitive_provider.py` | `core/providers/cognitive_provider.py` | boot | `fail-closed` |
| `task_decomposer` | `core/planning/task_decomposer.py` | `core/planning/task_decomposer.py` | optional features | `degrade_with_receipt` |
| `task_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `task_supervisor` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `task_tracker` | `aura_main.py` | `aura_main.py` | optional features | `degrade_with_receipt` |
| `temporal_atlas_factory` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `temporal_binding` | `core/orchestrator/mixins/boot/boot_cognitive.py` | `core/orchestrator/mixins/boot/boot_cognitive.py` | boot | `fail-closed` |
| `tension_engine` | `core/service_registration.py` | `core/service_registration.py` | boot | `fail-closed` |
| `terminal_fallback` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `terminal_monitor` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `terminal_watchdog` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `theory_arbitration` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `theory_of_mind` | `core/morality/master_moral_integration.py` | `core/morality/master_moral_integration.py` | boot | `fail-closed` |
| `time_dilation` | `core/providers/consciousness_provider.py` | `core/providers/consciousness_provider.py` | optional features | `degrade_with_receipt` |
| `tool_orchestrator` | `core/service_registration.py` | `core/service_registration.py` | optional features | `degrade_with_receipt` |
| `trust_engine` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `tts_stream` | `core/providers/sensory_provider.py` | `core/providers/sensory_provider.py` | optional features | `degrade_with_receipt` |
| `unified_field` | `core/consciousness/consciousness_bridge.py` | `core/consciousness/consciousness_bridge.py` | boot | `fail-closed` |
| `unified_self` | `core/consciousness/coordinator.py` | `core/consciousness/coordinator.py` | boot | `fail-closed` |
| `unified_will` | `core/governance/will.py` | `core/governance/will.py` | optional features | `degrade_with_receipt` |
| `unity_draft_set` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `unity_fragmentation_report` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `unity_repair_plan` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `unity_runtime` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `unity_state` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `unity_workspace_frame` | `core/unity/runtime.py` | `core/unity/runtime.py` | optional features | `degrade_with_receipt` |
| `user_recognizer` | `core/orchestrator/main.py` | `core/orchestrator/main.py` | boot | `fail-closed` |
| `value_autopoiesis` | `core/adaptation/value_autopoiesis.py` | `core/adaptation/value_autopoiesis.py` | optional features | `degrade_with_receipt` |
| `value_system` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `values_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `vector_memory` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `vector_memory_engine` | `core/providers/memory_provider.py` | `core/providers/memory_provider.py` | optional features | `degrade_with_receipt` |
| `vision` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `vision_engine` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `voice_engine` | `core/orchestrator/mixins/boot/boot_sensory.py` | `core/orchestrator/mixins/boot/boot_sensory.py` | boot | `fail-closed` |
| `voice_session` | `core/voice/voice_session.py` | `core/voice/voice_session.py` | optional features | `degrade_with_receipt` |
| `volition_engine` | `core/orchestrator/mixins/boot/boot_autonomy.py` | `core/orchestrator/mixins/boot/boot_autonomy.py` | boot | `fail-closed` |
| `wake_word` | `core/voice/wake_word.py` | `core/voice/wake_word.py` | optional features | `degrade_with_receipt` |
| `web_asset_handler` | `core/capabilities/web_asset_handler.py` | `core/capabilities/web_asset_handler.py` | optional features | `degrade_with_receipt` |
| `world_model` | `core/final_engines.py` | `core/final_engines.py` | boot | `fail-closed` |
| `world_state` | `core/world_state.py` | `core/world_state.py` | optional features | `degrade_with_receipt` |
