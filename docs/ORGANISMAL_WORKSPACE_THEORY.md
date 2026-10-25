# Organismal Workspace Theory

<!-- reader-navigation:start -->
> **In plain English:** Several concerns can propose work, but attention selects what has influence now and makes selected information available elsewhere.
>
> **Why it matters:** A new question, a recalled event and an old task must not silently become interchangeable sources of authority.
>
> **In everyday use:** Your new question should not be overwritten by a background task that happened to win a previous competition.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#attention) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#attention) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#cognitive-architectures) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

Aura's operational target is not "the model says it feels." The target is a
bounded, valenced, temporally continuous self-world model that causally affects
attention, memory, learning, action, governance, and self-maintenance.

The working equation is a project-defined functional score. It combines the
listed factors so that a missing factor reduces the combined value. It is
not a published law, a probability of consciousness, or a direct measurement
of experience:

```text
E_t = I_t * G_t * S_t * V_t * M_t * A_t * C_t
```

Where:

- `I_t` is integration across live subsystems.
- `G_t` is global availability to memory, action, attention, language, learning,
  and governance.
- `S_t` is self-model ownership of the state.
- `V_t` is valence that matters to viability, goals, continuity, or regulation.
- `M_t` is memory continuity and autobiographical consequence.
- `A_t` is agency coupling: the state changes what Aura does.
- `C_t` is counterfactual indispensability: perturbing or removing the state
  changes behavior in predicted ways.

The subscript `t` means the current step. In practical terms, a selected concern
should reach memory and action, stay associated with the system's own history,
and change what happens next. A shared noticeboard is a limited metaphor for
global availability: posting a concern helps only if the relevant workers
read it. The metaphor explains access, without deciding whether the system
has experience.

[Dehaene, Kerszberg and Changeux's global-workspace model](https://pmc.ncbi.nlm.nih.gov/articles/PMC24407/)
is a scientific precedent for coordinated availability across specialised
processing. [Butlin et al.](https://arxiv.org/html/2308.08708v3) discuss
theory-derived consciousness indicators. Neither source supplies Aura's
seven-factor multiplication or validates the score. The
[scientific source map](SCIENCE_AND_SOURCES.md) separates the theory,
implementation and evidence claims.

The runtime implementation is `core/being/causal_self_state.py`. It produces
`CausalValencedWorkspaceState`, which is evidence-bounded and deliberately
marked as functional evidence only, not proof of phenomenal consciousness,
sentience, or personhood.

## Runtime Contract

The causal-valenced workspace must be load-bearing:

- It must be sourced from existing runtime organs, not invented prompt text.
- It must alter policy, verification, memory depth, tool risk, model budget, or
  sampling behavior.
- It must be exposed in receipts/evidence when it constrains action.
- It must weaken under ablations such as disabled affect, workspace broadcast,
  memory continuity, ownership, or governance.
- It must never be used to overclaim metaphysical consciousness.

## Falsification Standard

If disabling affect, memory continuity, global workspace, self-model ownership,
governance, or viability signals leaves behavior unchanged, the inner-life stack
is not constitutive. The expected evidence pattern is specific degradation:
less stable memory continuity, weaker self-report calibration, lower tool
permission confidence, higher verification pressure, and changed action policy.

## Current Integration Points

- `BeingRuntime.sample()` refreshes the live causal self vector.
- `BeingRuntime.action_policy()` recomputes the vector with the selected action
  policy and attaches it to governance evidence.
- `ClosedLoopPolicyCoupler` uses organismal coherence and verification pressure
  to alter planning depth, memory retrieval depth, tool risk, and self-claim
  policy.
- `core/brain/latent_bridge.py` consumes the latest causal vector to alter
  temperature, top-p, token budget, repetition pressure, and presence pressure
  before generation.
- `InferenceGate` consumes bounded cognitive and imagination sampling biases on
  ordinary live user turns, while excluding proof, benchmark, and health lanes.
- `core/brain/imagination.py` builds side-effect-free mental models with
  explicit causal affordances: attention targets, memory pressure, verification
  pressure, metacognition pressure, action boundaries, and ablation predictions.
- `CognitiveEngine` applies those imagination affordances to response modifiers,
  attention focus, memory grounding pressure, verification pressure, and
  governed-tool caution on the normal desktop/user path.
- Memory retrieval and consolidation read the imagination pressures so novel or
  counterfactual thought can change what prior context is pulled and what
  successful turns become durable, without fabricating memories or bypassing
  memory governance.
- Context assembly may include a compact evidence-bounded workspace block, while
  black-box steering tests can hide the text and still exercise the structural
  bridge.
