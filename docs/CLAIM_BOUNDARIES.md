# Aura Claim Boundaries

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

The hard edges of what this repository is allowed to claim.

A mechanism, a passing test and a scientific conclusion have different scopes.
**Demonstrated** refers to the conditions in a cited result. **Not proven**
means the evidence here does not establish the claim. **Strictly unsupported**
means the project makes no direct, assumption-free inference of that kind
from its software tests. It does not establish that the opposite is true or
that a theory-dependent scientific argument is impossible.

Non-negotiable is meant literally. A result that needs one of these
boundaries relaxed is a result that doesn't ship.

## 1. Absolute Scientific Boundaries

The following conclusions are not established by the project’s current evidence:

### A. Subjective Consciousness / Qualia
* Aura is an agentic software runtime composed of state transition gates, LLM context compilers, and vector databases.
* The project has not established phenomenological experience, subjective feelings or qualia. Software composition alone does not establish their absence either.
* Introspective state metrics in the codebase represent functional self-monitoring and feedback loops, not subjective awareness.

### B. Metaphysical Free Will
* All volition mechanisms in Aura are governed by probabilistic decision rollouts, feedback variables, and explicit authorization gates.
* There is no claim of a metaphysical uncaused cause. "Operational Volition" refers purely to internal counterfactual calculation of plans before action authorization.

### C. Moral Personhood
* The project’s experiments do not establish moral personhood or confer legal rights or duties.
* Moral status needs an ethical argument; legal status needs the applicable legal framework. A software label does not decide either question.

### D. Phenomenal Valence / Felt Affect
* Aura's affect, valence, hedonic, and "feeling" variables are functional control signals — scalars that modulate attention, gating, and homeostasis.
* Their measured effects do not establish felt experience. A load-bearing valence signal (task #38) measurably changes behavior; whether that supports a phenomenal interpretation requires further assumptions and evidence.

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
     test. Project-authored evaluations supply bounded internal evidence;
     they cannot substitute for the independent evaluation that A requires.
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
* **Evidence required**: State the duration, workload, failures and resource measurements of each soak. A short run does not establish days of continuity; even a 72-hour run cannot establish indefinite or asymptotic stability. Capacity to run a campaign depends on the host’s current workloads, not a permanent hardware prohibition.

### D. Mature RSI (Recursive Self-Improvement)
* **Status**: `not proven`
* **Blocker**: Recursive self-improvement requires repeated autonomous capability gains under hidden validation. Aura's current self-modification path is intentionally constrained to quarantined proposals, branch-aware promotion, supervised validation, and sandboxed skill patches. Open-ended, unconstrained RSI is not claimed and not sought.

### E. Concurrent Model-Lane Residency
* **Status**: `bounded by admission and serving policy`
* **Boundary**: The 64 GB host has a declared lane-memory budget and durable admission
  controls (`core/brain/lane_admission.py`, `core/runtime/model_lane_control.py`).
  A list of model roles is not permission to load them together. The arithmetic
  module alone does not establish a universal one-auxiliary-lane limit; the
  actual serving policy, reservations and observed workloads decide admission.
  Do not duplicate the resident Cortex or bypass the live-instance rules.

### F. Corpus Coverage vs Answer Quality
* **Status**: `presence demonstrated; encyclopedic answer quality not claimed`
* **Boundary**: A ~6.6M-document offline corpus is physically present and BM25-retrievable (claims-matrix row 26). That demonstrates storage and retrieval, NOT that any specific reference work is complete, current, or that answers synthesized from it reach encyclopedic accuracy.

### G. Recurrent-Depth Intelligence Gain
* **Status**: `not proven`
* **Evidence**: The July reflex A/B found identical accuracy at loops=1 and loops=2. The later frozen-loop campaigns found no gain at 1.5B and a negative point estimate at 32B. Typed semantic-machine and kept-procedure gains are separate mechanisms; G09’s closure does not establish gain from repeated backbone layers. [Current programme summary](RECURSIVE_LATENT_CORTEX.md).

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
* **Validation scope**: The authority gateway and receipt validators test declared routes and failure cases. The required contract is to intercept unauthorised effects; passing these checks is not a proof against every possible bypass or a substitute for independent security review.

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

## 5. Current research outcomes are separate claims

G09 passed its registered broad-reasoning-gain criterion on held-out math and
planning kinds: after row review, 29 gained and 0 lost on 444 fresh requests
([attempt 6](evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md)). This positive
claim must not be grouped into a blanket “broad gain not claimed” statement.
Unrestricted reasoning, competition-mathematics transfer, static fusion of
RLC into the backbone and frontier performance are separate claims.

L09’s live teach–induce–recall loop was demonstrated, but reliable improvement
over ordinary use remains open. Its later comparisons were underpowered,
not met, or stopped at the registered baseline-error step. The
[unsupported-claims ledger](../CLAIMS_NOT_SUPPORTED.md#22-learning-from-conversation-that-improves-ordinary-use)
links each result. G12’s named frontier comparison remains open in the
[master status list](AURA_1_0_MASTER_TODO.md).
