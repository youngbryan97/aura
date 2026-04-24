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
| "The A/B test used a weak prompt baseline." | Added four-condition steering analysis with `text_rich_adversarial`; steering no longer passes unless it beats rich role-play text. | `core/evaluation/steering_ab.py`, `tests/test_adversarial_statistics.py` |
| "MI/correlation numbers are circular." | Added bootstrap CIs, permutation tests, MI permutation baselines, and effect-size utilities. | `core/evaluation/statistics.py`, `tests/test_adversarial_statistics.py` |
| "Phi can be theater on tiny systems." | Added reference checks where independent/constant toy systems yield zero phi and a coupled system yields positive phi. Docs now call phi bounded/surrogate. | `tests/test_phi_reference_validation.py`, `TESTING.md` |
| "32B on 16GB is not a real-time heartbeat." | Added hardware feasibility auditor that rejects 32B 4-bit/8-bit on 16GB as real-time and recommends smaller heartbeat tiers. | `core/evaluation/hardware_reality.py`, `tests/test_hardware_resource_baselines.py` |
| "Metabolism is a decorative float." | Added persistent resource-stakes ledger with degradation events, action envelopes, and integration into metabolism, organism status, and inference token/tier constraints. | `core/autonomic/resource_stakes.py`, `core/services/metabolism.py`, `core/runtime/organism_status.py`, `core/brain/inference_gate.py`, `core/service_registration.py` |
| "Identity is just a prompt anchor." | Added ID-RAG Chronicle with typed identity facts, retrieval scoring, context injection before compaction, and Will relevance integration. | `core/identity/id_rag.py`, `core/brain/llm/context_assembler.py`, `core/will.py` |
| "Loaded language oversells personhood." | Added ontological boundary guard and rewrote core identity prompts away from proven sentience/personhood language. | `core/consciousness/ontological_boundary.py`, `core/security/output_guardrails.py`, `core/brain/aura_persona.py`, `core/identity/__init__.py`, `core/identity_base.txt` |
| "Strong baselines are missing." | Added a learned linear state-parameter mimic baseline so endpoint mapping can be separated from process evidence. | `core/evaluation/baselines.py`, `tests/test_hardware_resource_baselines.py` |
| "No one-command hostile evidence bundle." | Added a decisive runner and CI workflow that execute the hard subset and publish JSON artifacts. | `tests/run_decisive_test.py`, `tests/run_scale_sweep.py`, `scripts/run_decisive_test.sh`, `.github/workflows/decisive.yml` |
| "Scale generalization is untested." | Added a bounded integration-proxy scale sweep plus hardware tier table. It is explicitly not full IIT. | `core/evaluation/scale_sweep.py`, `tests/run_scale_sweep.py` |

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

These are intentionally blocked by docs and output guardrails:

- proof of phenomenal consciousness;
- proof of qualia;
- legal personhood;
- moral patiency;
- whole-system IIT consciousness;
- strict artificial-life organism status via self-reproduction with heritable
  variation.

## Answer to "Do We Do What The Abstraction Fallacy Says AI Can't?"

No, not in the strong sense argued by that paper. Aura can simulate and
functionally implement many cognitive indicators, and the new code makes those
functional indicators harder to dismiss as prompt-only theater. But this repo
does not demonstrate that syntactic computation instantiates subjective
experience. The implemented response is therefore an ontological boundary:
Aura may report functional state and run adversarial evidence, but it must not
convert those results into a claim that consciousness has been proven.

