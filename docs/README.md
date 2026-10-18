# docs/

This folder contains 164 documentation files at its top level, with dated evidence under [evidence/](evidence/). Here is an overview of what you will find here.

If you need to check *whether a document is up to date*, see [DOC_STATUS.md](DOC_STATUS.md). It categorizes files into current docs, historical records, auto-generated files, and standards. Check there first if something looks unexpected.

## Running it

| | |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | How to use the app day-to-day |
| [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md) | Running Aura on your own machine — requirements, diagnostics, settings, and troubleshooting |
| [runbooks/](runbooks/) | Step-by-step guides for fixing problems, based on output from `aura doctor --bundle` |
| [SLO.md](SLO.md) | Runtime performance and reliability targets (Service Level Objectives), tested in CI |
| [PLATFORM_POSTURE.md](PLATFORM_POSTURE.md) | Architectural choices (no RBAC, no SSO, single-user only, manual disaster recovery, approved plugins) and how code enforces them |

## Understanding it

| | |
|---|---|
| [ARCHITECTURE_MAP.md](ARCHITECTURE_MAP.md) | System dependency map — components, connections, and startup order (auto-generated) |
| [ONTOLOGY.md](ONTOLOGY.md) | Core concepts and definitions, verified by automated tests |
| [TERMINOLOGY.md](TERMINOLOGY.md) | Glossary translating internal project names to standard engineering terms |
| [RUNTIME_CONTRACT.md](RUNTIME_CONTRACT.md) | System health and behavior contracts, generated from `health_contract.py` |
| [FMEA.md](FMEA.md) | Failure Mode and Effects Analysis — a registry of potential failure modes (auto-generated) |
| [ENGINEERING_ADOPTION.md](ENGINEERING_ADOPTION.md) | The seven design phases and why each was implemented |
| [COGNITIVE_ARCHITECTURE_ADOPTION.md](COGNITIVE_ARCHITECTURE_ADOPTION.md) | Ideas adapted from cognitive science architectures (Soar and ACT-R), including what worked and what didn't |
| [MODEL_ROSTER.md](MODEL_ROSTER.md) | Every AI model used (large language models, speech-to-text, embeddings) and benchmark data justifying its use |
| [BROWSER_PURSUIT.md](BROWSER_PURSUIT.md) | How the autonomous browser loop works: permissions, state carried across steps, and stop conditions |
| [WRITING_RULES.md](WRITING_RULES.md) | Writing guidelines to keep documentation clear and avoid machine-sounding text, checked by `make writing` |

## The claim surface

This section tracks project claims and verification standards to keep evaluations honest.

| | |
|---|---|
| [REALITY_REACH.md](REALITY_REACH.md) | Real-world physical interaction claims: hardware contracts, proofs of reachability, and test records. Read "Current Evidence" before accepting any physical claim |
| [CLAIM_BOUNDARIES.md](CLAIM_BOUNDARIES.md) · [CLAIM_SURFACE.md](CLAIM_SURFACE.md) | What the project claims to do and what falls outside its scope |
| [ABLATION_LEGIBILITY.md](ABLATION_LEGIBILITY.md) | Ablation tests (running with components disabled) showing measured performance changes — including cases with no difference |
| [BEHAVIORAL_PROOF_STANDARD.md](BEHAVIORAL_PROOF_STANDARD.md) | Evidence required to claim autonomous behavior or novel model output |
| `*_STANDARD.md` (11 files) | Evidence standards for each capability area. These remain fixed and only update when standards change, not when code changes |

## Research programmes

Long-running research initiatives and their progress records.

- **Recursive latent cortex** — the core model research track. Start with
  [RECURSIVE_LATENT_CORTEX.md](RECURSIVE_LATENT_CORTEX.md), the overview page
  explaining the research milestones. Supporting documents include:
  [INTRINSIC_RECURRENCE.md](INTRINSIC_RECURRENCE.md) (current training experiments),
  [RLC_RECONCILIATION.md](RLC_RECONCILIATION.md) (analysis of earlier negative results),
  [RLC_WIRING_HANDOFF.md](RLC_WIRING_HANDOFF.md),
  [RLC_SPARK_EXECUTION_LEDGER.md](RLC_SPARK_EXECUTION_LEDGER.md),
  [RLC_COMMITMENT_SEARCH.md](RLC_COMMITMENT_SEARCH.md),
  [RLC_SPARK_LITERATURE.md](RLC_SPARK_LITERATURE.md),
  [RLC_KNOWLEDGE_SOURCE_MATRIX.md](RLC_KNOWLEDGE_SOURCE_MATRIX.md), and
  [SPARK_PRETRAINING_LEGS.md](SPARK_PRETRAINING_LEGS.md).
- **General reasoning and kept procedures** — whether what she learns and keeps
  makes her reason better, measured on held-out tasks under plans committed
  before the tasks are drawn. The plan is
  [evidence/GENERAL_REASONING_PROGRAM_2026-10-14.md](evidence/GENERAL_REASONING_PROGRAM_2026-10-14.md)
  and its first round
  [evidence/GENERAL_REASONING_DEVELOPMENT_RESULT_2026-10-16.md](evidence/GENERAL_REASONING_DEVELOPMENT_RESULT_2026-10-16.md);
  kept procedures (G09) closed with
  [evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md](evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md),
  after [evidence/G09_PROCEDURES_RESULT_5_2026-10-17.md](evidence/G09_PROCEDURES_RESULT_5_2026-10-17.md).
  Every item's state is in [AURA_1_0_MASTER_TODO.md](AURA_1_0_MASTER_TODO.md).
- **Language substrate and generality** — research on how the system forms,
  represents, and combines new concepts.
  [LANGUAGE_SUBSTRATE_AND_GENERALITY.md](LANGUAGE_SUBSTRATE_AND_GENERALITY.md)
  is the main overview; related files include
  [GENERALITY_ANALYSIS.md](GENERALITY_ANALYSIS.md) (problem breakdown and literature review),
  [GENERALITY_TODO.md](GENERALITY_TODO.md) (the development backlog),
  [ENDOGENOUS_LANGUAGE_PATHWAY.md](ENDOGENOUS_LANGUAGE_PATHWAY.md) (turning internal states into language),
  [FRONTIER_GENERAL_ARC.md](FRONTIER_GENERAL_ARC.md) (generalization goals), and
  [METALANGUAGE_MY_OWN_ATTEMPT.md](METALANGUAGE_MY_OWN_ATTEMPT.md).
- **Consciousness and integration** — exploration of system integration and self-modeling:
  [WHOLE_SYSTEM_PHI.md](WHOLE_SYSTEM_PHI.md),
  [ORGANISMAL_WORKSPACE_THEORY.md](ORGANISMAL_WORKSPACE_THEORY.md),
  [INNER_LIGHT_TEST.md](INNER_LIGHT_TEST.md), and
  [GHOST_SUBSTRATE.md](GHOST_SUBSTRATE.md).
- **Autonomy and self-modification** — safety boundaries and mechanisms for self-directed operation:
  [ULYSSES_COVENANT.md](ULYSSES_COVENANT.md),
  [ALLOSTASIS_ENGINE.md](ALLOSTASIS_ENGINE.md),
  [AUTONOMOUS_ARCHITECTURE_GOVERNOR.md](AUTONOMOUS_ARCHITECTURE_GOVERNOR.md),
  [GENERAL_ENVIRONMENT_AUTONOMY.md](GENERAL_ENVIRONMENT_AUTONOMY.md),
  [RSI_VALIDATION.md](RSI_VALIDATION.md), and
  [RSI_ARCHITECTURE.md](RSI_ARCHITECTURE.md).
- **Subsystem deep dives** —
  [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) (architecture of the 116-module memory system), and
  [REASONING_ENGINES.md](REASONING_ENGINES.md) (rule-based reasoning, sandboxed code execution, and operations kept outside the neural network).
- **Capability maps** — evaluations of capabilities against benchmarks and external frameworks:
  [OMEGA_CAPABILITY_MATRIX.md](OMEGA_CAPABILITY_MATRIX.md),
  [FICTIONAL_AI_CAPABILITY_MAP.md](FICTIONAL_AI_CAPABILITY_MAP.md),
  [FLIGHT_RECORDER.md](FLIGHT_RECORDER.md), and
  [DELIBERATE_PRACTICE.md](DELIBERATE_PRACTICE.md).

## Records

`evidence/` and all files with a date in their filename are historical snapshots. Each includes a header marking it as an archive. **These records are preserved as originally recorded** — for example, an evaluation verdict from July is not altered in August.

## Generated — do not hand-edit

Four files here are rendered from code. A manual edit survives until the
next build and then vanishes.

```bash
make architecture-map                          # ARCHITECTURE_MAP.md
make fmea-doc                                  # FMEA.md
make contract-doc                              # RUNTIME_CONTRACT.md
python tools/reqproof/progress.py              # AURA_PROGRESS.md
```

Check for drift without rewriting:

```bash
python tools/render_fmea.py --check
python tools/render_health_contract.py --check
```
