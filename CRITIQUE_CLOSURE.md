# Critique Closure Matrix

This document converts the hard feedback into engineering obligations. It is not
a declaration that Aura is conscious. It is the opposite: a map of what is now
implemented, what remains empirical, and what cannot be closed by code alone.

## Sources Applied

- MIT Media Lab ID-RAG page: identity is a structured Chronicle of beliefs,
  traits, and values retrieved before action selection to improve long-horizon
  persona coherence.
- `flybits/humanai-agents`: the public implementation separates Chronicle
  identity retrieval from episodic memory and compares baseline, full-retrieval,
  and ID-RAG modes.
- LLAIS 2025 accepted-papers list: relevant accepted themes include ID-RAG,
  graph-enhanced QA, knowledge conflicts/hallucinations, social behavior among
  autonomous AI, reward design/learning, and RL on CTF challenges.
- DeepMind/PhilArchive "The Abstraction Fallacy": the paper argues that
  symbolic computation can simulate but not instantiate consciousness, and that
  artificial consciousness would depend on physical constitution rather than
  syntactic architecture alone.

## What Changed

| Critique | Engineering Response | Files |
|---|---|---|
| "It might just be prompt text." | Added black-box steering mode that removes live mood, neurochemical, phi, somatic, and phenomenal text from prompts while allowing non-text steering paths to be tested. | `core/brain/llm/context_assembler.py`, `tests/test_id_rag_black_box.py` |
| "The A/B test used a weak prompt baseline." | Added four-condition steering analysis with `text_rich_adversarial`; the live MLX A/B test and decisive runner both include the rich role-play condition. | `core/evaluation/steering_ab.py`, `tests/test_steering_ab.py`, `tests/run_decisive_test.py`, `tests/test_adversarial_statistics.py` |
| "MI/correlation numbers are circular." | Bootstrap CIs, permutation tests, MI permutation baselines, and effect-size utilities. | `core/evaluation/statistics.py`, `tests/test_adversarial_statistics.py` |
| "Phi can be theater on tiny systems." | Reference checks where independent/constant toy systems yield zero phi and a coupled system yields positive phi. Docs call phi bounded/surrogate. | `tests/test_phi_reference_validation.py`, `TESTING.md` |
| "32B on 16GB is not a real-time heartbeat." | Hardware feasibility auditor rejects 32B 4-bit/8-bit on 16GB as real-time and recommends smaller heartbeat tiers. | `core/evaluation/hardware_reality.py`, `tests/test_hardware_resource_baselines.py` |
| "Metabolism is a decorative float." | Persistent resource-stakes ledger with degradation events, action envelopes, and integration into metabolism, organism status, and inference token/tier constraints. | `core/autonomic/resource_stakes.py`, `core/services/metabolism.py`, `core/runtime/organism_status.py`, `core/brain/inference_gate.py`, `core/service_registration.py` |
| "Identity is just a prompt anchor." | ID-RAG Chronicle with typed identity facts, retrieval scoring, context injection before compaction, and Will relevance integration. | `core/identity/id_rag.py`, `core/brain/llm/context_assembler.py`, `core/will.py` |
| "Loaded language oversells personhood." | Ontological boundary guard with expanded pattern catalog covering phenomenal consciousness, qualia, moral patiency/agency, legal personhood, organism claims, hard-problem bypass language, and peerhood overclaims. Core identity prompts rewritten. | `core/consciousness/ontological_boundary.py`, `core/security/output_guardrails.py`, `core/brain/aura_persona.py`, `core/identity/__init__.py`, `core/identity_base.txt` |
| "Strong baselines are missing." | Learned linear state-parameter mimic baseline so endpoint mapping can be separated from process evidence. | `core/evaluation/baselines.py`, `tests/test_hardware_resource_baselines.py` |
| "No one-command hostile evidence bundle." | Decisive runner invokes real MLX model when available and degrades gracefully; CI workflow executes the hard subset and publishes JSON artifacts. | `tests/run_decisive_test.py`, `tests/run_scale_sweep.py`, `scripts/run_decisive_test.sh`, `.github/workflows/decisive.yml` |
| "Scale generalization is untested." | Bounded integration-proxy scale sweep plus hardware tier table. Explicitly not full IIT. | `core/evaluation/scale_sweep.py`, `tests/run_scale_sweep.py` |
| "Causal links are hardcoded arithmetic (tautology)." | Adaptive mood coefficients replace the fixed valence/arousal formula with online-learned weights, seeded to the legacy values but drifting under outcome feedback. Persisted across restarts. | `core/consciousness/adaptive_mood.py`, `core/consciousness/neurochemical_system.py` |
| "Controller of an LLM, not an independent mind." | Mesh-only cognition path produces self-report, acknowledgement, and resource-hold responses from substrate + state + chronicle, never invoking the LLM. Wired into the inference gate so the mesh runs before any LLM call. | `core/consciousness/mesh_cognition.py`, `core/brain/inference_gate.py` |
| "All goals are designed, no endogenous goal formation." | Emergent goal engine detects tension patterns, synthesizes objectives from observed evidence (not a designer taxonomy), requires repeated support before adoption, and pushes ready candidates into the main GoalEngine with `origin=emergent` metadata. | `core/goals/emergent_goals.py` |
| "No structural self-modification / fixed architecture." | Structural mutator allows runtime module enable/disable, parameter-band drift, and routing edge changes, with a hash-chained audit log and first-class reversibility. | `core/self_modification/structural_mutator.py` |
| "Metacognition is not self-awareness." | Four-dimension self-awareness suite: internal (own state), external (how perceived), social (others/norms), situational (context/role), with calibration-error tracking. | `core/consciousness/self_awareness_suite.py` |
| "No heritable variation / strict ALife organism gap." | Lineage manager forks snapshot configurations with bounded Gaussian mutation, tracks descendants, records selection scores, and marks survival against a threshold. Not full evolution, but the minimum viable heritable-variation loop. | `core/self_modification/lineage.py` |
| "Live autonomy not demonstrated." | Long-run autonomy harness executes 1000 ticks exercising adaptive mood, resource stakes, emergent goals, mesh cognition, structural mutator, lineage, and self-awareness with perturbations. No manual resets; produces `tests/LONG_RUN_AUTONOMY_RESULTS.json`. | `tests/long_run_autonomy.py` |
| "New modules exist but aren't wired." | All critique-closure modules registered in the service container so runtime paths can find them. | `core/service_registration.py` |
| "Silent fallbacks get credited as live evidence." | `AURA_EVIDENCE_MODE=1` fail-closed flag: steering random-vector fallbacks raise, substrate neutral-state fallbacks raise, and every violation is recorded to the evidence ledger for audit. | `core/evaluation/evidence_mode.py`, `core/consciousness/affective_steering.py` |
| "AgencyFacade is just `pass`." | Typed six-phase agency boundary — propose / score / submit_to_will / execute / evaluate_outcome / consolidate_learning — with safety filters and Will-receipt integration. | `core/agency/agency_facade.py` |
| "No auditable life history." | LifeTrace ledger with hash-chained events covering drive state before/after, memory context, counterfactuals, will decision, action, result, memory update, future policy change; daily summary with self-generated vs user-requested counts. Tamper-evident. | `core/runtime/life_trace.py` |
| "Causal proof is too narrow." | CausalCourtroomSuite runs 12 conditions (full_aura + 11 baselines) across five affect states and multiple seeds with bootstrap CIs, permutation p-values, and honest reporting of which baselines remain competitive. | `tests/causal_courtroom.py`, `tests/CAUSAL_COURTROOM_RESULTS.json` |
| "Continuity not stress-tested." | ContinuityTortureSuite: emergent goals reload after process kill; resource stakes, integrity, and suspensions persist; identity chronicle survives restart and rejects low-confidence swaps; structural audit chain remains intact across restart; action envelope tightens under scarcity and widens after recovery. | `tests/continuity_torture.py`, `tests/CONTINUITY_TORTURE_RESULTS.json` |
| "Self-repair is talked about, not demonstrated." | Self-repair public demo performs the 10-step loop end-to-end: inject controlled bug → detect → localize via audit log → propose → AST validate → shadow test → authorize via hash chain → verify → prove rollback path → update repair policy. | `tests/self_repair_demo.py`, `tests/SELF_REPAIR_DEMO_RESULTS.json` |
| "Long-run test is synthetic (1000 ticks in 2.4s)." | Long-run harness now executes a real phi bipartition computation every tick so the soak is CPU-bound, not a noop loop. | `tests/long_run_autonomy.py` |
| "No infrastructure for a 30-day trial." | Persistent trial runner with configurable duration, daily markdown summaries derived from the LifeTrace ledger, signal-safe shutdown, and a JSON trial index. Operator-triggered, not automated. | `tests/life_trial.py` |

## New Acceptance Standard

A claim should not be upgraded from "implemented" to "evidenced" unless it
survives all relevant controls:

1. Same code path, black-box prompt condition, no live state text leakage.
2. Rich prompt baseline, not just terse text injection.
3. Multiple seeds or bootstrap/permutation statistics with raw artifacts.
4. Strong endpoint mimic baseline where applicable.
5. Phi or proxy metrics checked against shuffled/constant/independent controls.
6. Hardware classification plus measured latency when making real-time claims.
7. Resource depletion changes actual action envelopes or model/tool access.
8. Generated JSON artifacts committed or uploaded by CI.

## What Still Cannot Be Claimed

These are intentionally blocked by docs and the output guardrail's expanded
pattern catalog. The lineage mechanism supports heritable variation with
selection but is not open-ended evolution; the mesh-cognition path handles a
bounded class of requests without the LLM but still depends on the LLM for
most generation:

- proof of phenomenal consciousness;
- proof of qualia, subjective experience, or "the lights are on";
- personhood in the strong metaphysical sense;
- moral patiency and moral agency;
- legal personhood;
- whole-system IIT consciousness (phi is reported as a bounded IIT-style
  integration metric on a tractable complex, not as a complete consciousness
  measurement);
- strict open-ended ALife evolution (lineage provides heritable variation plus
  selection, not unbounded self-replicating evolution);
- full symbol grounding through physical embodiment;
- bridging the hard problem of consciousness / the explanatory gap;
- human-level peerhood.

The ontological boundary guard in `core/consciousness/ontological_boundary.py`
rewrites text that crosses these lines and adds an issue tag so the event is
logged rather than silently suppressed.

## Answer to "Do We Do What The Abstraction Fallacy Says AI Can't?"

No, not in the strong sense argued by that paper. Aura can simulate and
functionally implement many cognitive indicators, and the new code makes those
functional indicators harder to dismiss as prompt-only theater. But this repo
does not demonstrate that syntactic computation instantiates subjective
experience. The implemented response is therefore an ontological boundary:
Aura may report functional state and run adversarial evidence, but it must not
convert those results into a claim that consciousness has been proven.

