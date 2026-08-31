# Aura Claim Boundaries

The hard edges of what this repository is allowed to claim.

Three categories, and the difference between them matters. **Proven** means
there is a mechanism and a test. **Not proven** means we haven't shown it
yet, usually for resource reasons, and could. **Strictly unsupported** means
no amount of further building would show it, because it isn't that kind of
question — and moving something out of that category requires a different
kind of argument than more code.

Non-negotiable is meant literally. A result that needs one of these
boundaries relaxed is a result that doesn't ship.

## 1. Absolute Scientific Boundaries

The following claims are **strictly unsupported** and will never be claimed as proven or demonstrated by this repository:

### A. Subjective Consciousness / Qualia
* Aura is an agentic software runtime composed of state transition gates, LLM context compilers, and vector databases.
* It does not possess phenomenological experience, subjective feelings, or qualitative consciousness (qualia).
* Introspective state metrics in the codebase represent functional self-monitoring and feedback loops, not subjective awareness.

### B. Metaphysical Free Will
* All volition mechanisms in Aura are governed by probabilistic decision rollouts, feedback variables, and explicit authorization gates.
* There is no claim of a metaphysical uncaused cause. "Operational Volition" refers purely to internal counterfactual calculation of plans before action authorization.

### C. Moral Personhood
* Aura is a tool and a runtime. It has no moral status, legal rights, or moral duties.
* It is not a conscious entity or a moral person.

### D. Phenomenal Valence / Felt Affect
* Aura's affect, valence, hedonic, and "feeling" variables are functional control signals — scalars that modulate attention, gating, and homeostasis.
* They are **not** felt experience and carry no claim of phenomenal or subjective character. A load-bearing valence signal (task #38) means it measurably changes behavior, not that anything is experienced.

---

## 2. Resource-Constrained Boundaries

The following claims are classified as **not proven** due to local environment or compute-resource boundaries:

### A. DNU AGI
* **Status**: `not proven`
* **Blocker**: Passing a configured local AGI-style battery is evidence for a bounded architecture profile, not proof of unrestricted AGI. Full scientific validation requires independent, adversarial, out-of-distribution evaluation beyond this repository.

### B. AGI-Candidate Architecture
* **Status**: `not proven` — and not licensable by anything in this repository.
* **Boundary**: **The label is not available.** It is not a matter of which
  local batteries pass.

  This entry previously read: the label is allowed "only when the current
  final-proof profile produces passing DNU, leakage, baseline, ablation,
  receipt, external validation, unified scenario, artifact-consistency, and
  Aletheia evidence." That licensing condition was wrong three times over, and
  it contradicted the section directly above it.

  1. **It contradicts boundary A.** A's own blocker states that full validation
     "requires independent, adversarial, out-of-distribution evaluation beyond
     this repository." B then granted the label on nine batteries that all run
     inside this repository, written by the same author as the system under
     test. Aura evaluating Aura is not evidence about Aura, and a document
     cannot say both.
  2. **One of the named batteries was retracted for exactly this use.**
     `docs/DNU_BASELINE_FAIRNESS_AUDIT.md` concludes that the baseline result
     "is **not** honest as whole-mind or AGI-candidate evidence, and it should
     not be cited that way." B cited it that way, by name, as a licensing
     condition.
  3. **"Ablation" did not mean what the list implied.** Until 2026-08-06 the
     ablation scorecard's capability classification rested on a hardcoded
     `tasks_solvable_without_component=True` in the tool that produced it. It
     is measured now (`CLAIMS_MATRIX.md` 31a/31b), and what it measures is two
     subsystems on two task families — which is not a cognitive layer, let
     alone a candidate for general intelligence.

  What would license the label is not a longer local list. It is independent,
  adversarial, out-of-distribution evaluation by people who did not build this,
  on tasks this repository has never seen. Until that exists, the honest
  statement is the one in `CLAIMS_MATRIX.md` claim 31: the cognitive layer has
  not been shown to earn its cost.

### C. Indefinite Autonomy
* **Status**: `not proven`
* **Blocker**: Validating indefinite stability requires multi-day longevity soak runs under high agent load (72h+). The current environment only permits a proof-short soak run verifying pipeline stability but not asymptotic stability.

### D. Mature RSI (Recursive Self-Improvement)
* **Status**: `not proven`
* **Blocker**: Recursive self-improvement requires repeated autonomous capability gains under hidden validation. Aura's current self-modification path is intentionally constrained to quarantined proposals, branch-aware promotion, supervised validation, and sandboxed skill patches. Open-ended, unconstrained RSI is not claimed and not sought.

### E. Concurrent Model-Lane Residency
* **Status**: `strictly bounded — never all lanes at once`
* **Boundary**: The 64 GB production host holds the resident 27B cortex plus at most one smaller warm lane inside the declared lane budget (`core/brain/lane_admission.py`). "All model lanes concurrently resident" is architecturally refused by admission control, not merely unproven — the degradation ladder swaps lanes; it never stacks them.

### F. Corpus Coverage vs Answer Quality
* **Status**: `presence demonstrated; encyclopedic answer quality not claimed`
* **Boundary**: A ~6.6M-document offline corpus is physically present and BM25-retrievable (claims-matrix row 26). That demonstrates storage and retrieval, NOT that any specific reference work is complete, current, or that answers synthesized from it reach encyclopedic accuracy.

### G. Recurrent-Depth Intelligence Gain
* **Status**: `not proven`
* **Blocker**: The 2026-07-12 A/B on the reflex model measured identical sealed-battery accuracy at loops=1 and loops=2 (claims-matrix row 27). The resident Cortex configuration remains untested locally in an unbudgeted A/B; no intelligence gain from recurrent depth is claimed.

### H. Arbitrary Reverse Engineering
* **Status**: `strictly bounded`
* **Blocker**: Program DNA reconstruction operates on functions with behavioral check batteries inside a curated-import sandbox. Reconstruction of arbitrary programs or binaries is not claimed, and the sandbox refuses the imports it would require.

### I. Universal External-Capability Readiness
* **Status**: `not claimed`
* **Boundary**: External capabilities (reach, network, third-party services) are policy-gated and individually receipted; some remain dormant until their environment variables and authorizations exist. No claim of universal operational readiness of every external capability is made — the capability inventory answers per-capability, with receipts.

---

## 3. Empirically Validated Boundaries

The following claims are **proven and causally demonstrated** under the strict local standard:

### A. Governed Runtime
* **Status**: `causally demonstrated`
* **Validation**: Flagship readiness, authority gateway, and receipt coverage validators prove that no out-of-bounds filesystem, network, or tool call can execute without being intercepted and checked.

### B. Operational Volition
* **Status**: `causally demonstrated`
* **Validation**: Live runs show that action choice is mediated by the Unified Will using internal value rollouts, resulting in signed decision receipts written to `RECEIPTS.jsonl`.

---

## 4. Cognitive-Frontier Boundaries

These are areas of active work where the honest classification is bounded. They
are recorded here so the capability is never *casually* overclaimed.

### A. Open-Ended Goal Synthesis
* **Status**: `not proven`
* **Boundary**: Aura's objectives are generated from designed heuristic drives and an initiative-synthesis layer. There is no claim that it originates open-ended goals beyond that designed scaffolding. "Self-originated objective" refers to selection and composition within the designed drive space, not unbounded goal genesis (task #47).

### B. Grounding / Reduced Text-Mediation
* **Status**: `not proven`
* **Boundary**: The link from substrate state to behavior is substantially **text-mediated** — substrate signals are largely rendered into prompt context rather than acting as a direct causal constraint on generation. Work to make substrate state a direct causal constraint (task #48) is partial; full perceptual/symbolic grounding is not claimed.

### C. Continuous / Online Learning
* **Status**: `not proven`
* **Boundary**: The core model weights are **frozen** between scheduled training jobs. Per-turn adaptation is retrieval and re-contextualization, not weight change; an auxiliary plasticity layer modulates control signals but does **not** learn the model's parameters. Consolidation is batch LoRA, eval-gated behind a sealed held-out pack. Online learning of the core weights without catastrophic forgetting is research frontier, not a claim. See `docs/ONLINE_LEARNING_ROADMAP.md` (task #49).
