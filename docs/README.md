# docs/

102 files live here. This is what's in them.

If what you actually need to know is *whether a given file is still true*,
that's [DOC_STATUS.md](DOC_STATUS.md) — it splits everything into current,
historical, generated, and standards. Start there when a document surprises
you.

## Running it

| | |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Using the app |
| [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md) | Running it on your own hardware — gates, diagnostics, tuning, debugging entry points |
| [runbooks/](runbooks/) | 39 incident procedures, one per known failure mode. Written against `aura doctor --bundle` fields |
| [SLO.md](SLO.md) | What the runtime promises, measured by `slo/` and gated in CI |
| [PLATFORM_POSTURE.md](PLATFORM_POSTURE.md) | The five deliberate platform decisions (no RBAC, no SSO, single-tenant, manual DR, hash-allowlist plugins) and what enforces each |

## Understanding it

| | |
|---|---|
| [ARCHITECTURE_MAP.md](ARCHITECTURE_MAP.md) | Generated dependency map — subsystems, edges, boot contract |
| [ONTOLOGY.md](ONTOLOGY.md) | The formal ontology: continuants, occurrents, axioms rendered as test invariants |
| [TERMINOLOGY.md](TERMINOLOGY.md) | Internal poetic name ↔ sober technical label |
| [RUNTIME_CONTRACT.md](RUNTIME_CONTRACT.md) | Generated from `health_contract.py` |
| [FMEA.md](FMEA.md) | Generated failure-mode registry |
| [ENGINEERING_ADOPTION.md](ENGINEERING_ADOPTION.md) | The seven clean-room adoption waves and why each landed |
| [COGNITIVE_ARCHITECTURE_ADOPTION.md](COGNITIVE_ARCHITECTURE_ADOPTION.md) | What was taken from Soar and ACT-R, the equation that fitted, and the one that didn't |
| [MODEL_ROSTER.md](MODEL_ROSTER.md) | Every model lane — LLM, ASR, embeddings — and the measurement that put it there |
| [BROWSER_PURSUIT.md](BROWSER_PURSUIT.md) | The closed observe-decide-act loop: who may drive a browser, what the loop carries between rounds, and what ends a run |
| [WRITING_RULES.md](WRITING_RULES.md) | The eighteen patterns that read as machine-written, and the `make writing` gate that checks for them |

## The claim surface

This is the part of the repo that exists to keep it honest.

| | |
|---|---|
| [REALITY_REACH.md](REALITY_REACH.md) | Physical claims: typed contracts, reachability proof, and an open ledger. Read "Current Evidence" before crediting any physical result |
| [CLAIM_BOUNDARIES.md](CLAIM_BOUNDARIES.md) · [CLAIM_SURFACE.md](CLAIM_SURFACE.md) | What may be claimed and where the edges are |
| [ABLATION_LEGIBILITY.md](ABLATION_LEGIBILITY.md) | Run it with pieces switched off and see the measured delta — including the no-delta results |
| [BEHAVIORAL_PROOF_STANDARD.md](BEHAVIORAL_PROOF_STANDARD.md) | The bar for autonomy and novel-output claims |
| `*_STANDARD.md` (11 files) | One evidence bar each. Evergreen — they change when the bar changes, not when the code does |

## Research programmes

Long-running lines of work, each with its own ledger.

- **Recursive latent cortex** — the flagship programme. Start at
  [RECURSIVE_LATENT_CORTEX.md](RECURSIVE_LATENT_CORTEX.md), which is the
  landing page and carries the claims ladder; then
  [INTRINSIC_RECURRENCE.md](INTRINSIC_RECURRENCE.md) (the live training front),
  [RLC_RECONCILIATION.md](RLC_RECONCILIATION.md) (why two negative results were
  void), [RLC_WIRING_HANDOFF.md](RLC_WIRING_HANDOFF.md),
  [RLC_SPARK_EXECUTION_LEDGER.md](RLC_SPARK_EXECUTION_LEDGER.md),
  [RLC_COMMITMENT_SEARCH.md](RLC_COMMITMENT_SEARCH.md),
  [RLC_SPARK_LITERATURE.md](RLC_SPARK_LITERATURE.md),
  [RLC_KNOWLEDGE_SOURCE_MATRIX.md](RLC_KNOWLEDGE_SOURCE_MATRIX.md),
  [SPARK_PRETRAINING_LEGS.md](SPARK_PRETRAINING_LEGS.md)
- **Language substrate and generality** — the programme for learning,
  representing, and composing new abstractions.
  [LANGUAGE_SUBSTRATE_AND_GENERALITY.md](LANGUAGE_SUBSTRATE_AND_GENERALITY.md)
  is the landing page; then
  [GENERALITY_ANALYSIS.md](GENERALITY_ANALYSIS.md) (decomposition and
  literature), [GENERALITY_TODO.md](GENERALITY_TODO.md) (the full backlog),
  [ENDOGENOUS_LANGUAGE_PATHWAY.md](ENDOGENOUS_LANGUAGE_PATHWAY.md) (state →
  language), [FRONTIER_GENERAL_ARC.md](FRONTIER_GENERAL_ARC.md) (the
  frontier-general reframe),
  [METALANGUAGE_MY_OWN_ATTEMPT.md](METALANGUAGE_MY_OWN_ATTEMPT.md)
- **Consciousness and integration** — [WHOLE_SYSTEM_PHI.md](WHOLE_SYSTEM_PHI.md),
  [ORGANISMAL_WORKSPACE_THEORY.md](ORGANISMAL_WORKSPACE_THEORY.md),
  [INNER_LIGHT_TEST.md](INNER_LIGHT_TEST.md),
  [GHOST_SUBSTRATE.md](GHOST_SUBSTRATE.md)
- **Autonomy and self-modification** — [ULYSSES_COVENANT.md](ULYSSES_COVENANT.md),
  [ALLOSTASIS_ENGINE.md](ALLOSTASIS_ENGINE.md),
  [AUTONOMOUS_ARCHITECTURE_GOVERNOR.md](AUTONOMOUS_ARCHITECTURE_GOVERNOR.md),
  [GENERAL_ENVIRONMENT_AUTONOMY.md](GENERAL_ENVIRONMENT_AUTONOMY.md),
  [RSI_VALIDATION.md](RSI_VALIDATION.md)
- **Capability maps** — [OMEGA_CAPABILITY_MATRIX.md](OMEGA_CAPABILITY_MATRIX.md),
  [FICTIONAL_AI_CAPABILITY_MAP.md](FICTIONAL_AI_CAPABILITY_MAP.md),
  [FLIGHT_RECORDER.md](FLIGHT_RECORDER.md),
  [DELIBERATE_PRACTICE.md](DELIBERATE_PRACTICE.md)

## Records

`evidence/` and every file with a date in its name are point-in-time
records. They carry a banner saying so. **They are not updated to match the
present** — editing a July verdict in August is editing the record, not
correcting it.

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
