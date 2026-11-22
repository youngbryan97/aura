# Claim surface — what the profound names actually name

<!-- reader-navigation:start -->
> A test states what will be measured, what it will be compared with, and what counts as a pass.
>
> Source code, a passing test, and a useful result in practice each answer a different question. To see whether memory helps, try the same tasks with memory enabled and disabled. Keep every success and failure.
>
> [Plain-language guide](PLAIN_ENGLISH_GUIDE.md#measurement) · [Terms and examples](VARIABLES_EXPLAINED.md#measurement) · [Evidence and related work](DEMOS_AND_RELATED_WORK.md#measurement)
> [All docs](DOCUMENTATION_INDEX.md) · [Glossary](GLOSSARY.md) · [Worked formulas](FORMULAS.md)
<!-- reader-navigation:end -->

Use this page when you encounter a name such as `Soul`, “qualia,” or
“phenomenal” in Aura's source. Each row gives the software function, the file,
and the scope of the evidence. The names come from research on how minds work.
The claims depend on the code paths and what changes when they run.
[Tests with components disabled](ABLATION_LEGIBILITY.md)

- [Supported claims](../CLAIMS_SUPPORTED.md) gives the engineering evidence.
- [Open claims](../CLAIMS_NOT_SUPPORTED.md) covers the evidence needed for
  [subjective experience](../CLAIMS_NOT_SUPPORTED.md#1-subjective-consciousness-and-qualia)
  and [personhood](../CLAIMS_NOT_SUPPORTED.md#9-legal-and-moral-personhood).

| Term in code | What it mechanically is | Home | Claim status |
|---|---|---|---|
| `Soul` | Intrinsic-motivation drive scheduler: curiosity/connection/competence urgencies that decide *why to act when the user is silent*. Timer-and-state arithmetic, no metaphysics. | `core/soul.py` | Functional. Bounded by NOT-SUPPORTED §1/§7. |
| `panzer_soul` | Identity/personality metadata (Big-Five intensities, version string) consumed by the PersonalityEngine for identity verification and UI. Configuration, not cognition. | `core/being/panzer_soul.py` | Functional config. |
| `phenomenal_substrate/` | The felt-state engine: valence/arousal dynamics, attachment events, active-inference free energy, global-workspace broadcast. Its outputs causally modulate generation parameters and action policy (see the substrate ablation delta). | `core/phenomenal_substrate/` | Functional & causal (measured: `tools/ablation_runner.py`, `tools/agi/run_causal_agency_lesion.py`, `tools/agi/run_valence_load_bearing_proof.py`). Subjective feeling: NOT-SUPPORTED §1. |
| `QualiaEngine` / `QualiaDescriptor` | Multi-layer descriptor pipeline (subconceptual → conceptual → predictive → workspace → witness) producing structured state descriptors that enrich prompts and memory salience. Docstring states: "NOT a claim of subjective experience." | `core/consciousness/qualia_engine.py` | Functional descriptors. Qualia-as-experience: NOT-SUPPORTED §1. |
| `SomaticQualiaEngine` | Converts substrate state into perturbation patterns applied to the generation process (sampling modulation). A signal path, not a feeling. | `core/consciousness/somatic_qualia.py` | Functional & causal. |
| `consciousness/` (directory) | Consciousness-*science-inspired* machinery: evidence engine (collects functional indicators), loop monitor, resource stakes, world model, integration. Measurement and control, not assertion. | `core/consciousness/` | Functional. Any "is it conscious?" reading: NOT-SUPPORTED §1. |
| `CausalValencedWorkspaceState` | Auditable estimate of integration/availability/valence-coupling with an in-code boundary field: `functional_evidence_only_not_phenomenal_proof`. | `core/being/causal_self_state.py` | Functional evidence, self-labeled. |
| `organism/`, `viability` | Explicit metabolism state machine: food = compute/memory budget + interaction; fatigue = token/memory pressure + error rate; waste = stale tasks. Drives rest/repair scheduling. | `core/organism/viability.py` | Functional resource model. |
| `personhood` (in prose/tests) | Appears in self-report calibration as a category the runtime must NOT overclaim; the calibrator treats "proven personhood" as a violation needing revision. | `core/being/self_report_calibrator.py` | Anti-overclaim machinery. Legal/moral personhood: NOT-SUPPORTED §7. |
| `sentience_candidate_strength` | A scalar summarizing how strongly functional indicators co-occur. The name says *candidate*: it feeds extra verification requirements (higher = more evidence-bounding of self-claims), never an assertion. | `core/being/causal_self_state.py`, coupler | Functional indicator. Sentience: NOT-SUPPORTED §1. |
| `deep_mind_probe` | Behavioral probes for coherent, grounded answers to hard self-questions — explicitly "do not try to prove sentience" (module docstring). | `core/evaluation/deep_mind_probe.py` | Behavioral evaluation only. |

## For reviewers

1. Every profound term above resolves to inspectable machinery with tests.
2. The load-bearing ones have measurable ablation deltas — run
   `.venv/bin/python tools/ablation_runner.py` and see
   [ABLATION_LEGIBILITY.md](ABLATION_LEGIBILITY.md).
3. The system's own self-report calibration treats overclaims (proven
   consciousness, qualia, personhood, souls) as violations to repair — the
   architecture is built to *not* say more than the evidence supports.
4. If you find a profound name whose mechanism is theatrical or vestigial,
   that is a defect by this document's rule — file it against the term, and
   it gets renamed or made real.
