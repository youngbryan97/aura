# Research audit — 2026-04-28

Bryan asked to look at 9 sources and apply what would actually improve Aura.
Audit means: confirmed each paper's claim, then judged fit against Aura's
current architecture and roadmap. Two are applicable tonight; seven are
skip-or-defer with stated reasons.

## Applied

### Beyköylü / Vervaeke / Meling — "From flow to mystical experiences" (Cognitive Continuum)
**Claim:** Entropy-Fluency hypothesis — increased entropy signals destabilization,
then increased fluency. Person-world coupling, relevance realization,
metastable attunement. Maps over flow / insight / mystical experiences.
**Fit:** **High.** Aura's substrate already has phi (integration), neurochemicals,
oscillatory binding. Adding an entropy metric on substrate state plus a
fluency proxy gives the system observable hooks for "destabilization →
reorganization" phases that map cleanly onto her existing dynamics.
**Implementation:** `core/consciousness/entropy_fluency.py` with Shannon
entropy on binarized substrate, fluency from inverse smoothed prediction
error, phase classifier (stable / destabilizing / reorganizing / fluent),
and EWMA tracking. Tests cover phase transitions and edge cases.

### Mathematical Understanding of Loss of Plasticity (arxiv 2510.00304)
**Claim:** Theoretical: continual-learning networks lose adaptability via
low-rank singular-value collapse in their weight matrices.
**Fit:** **High.** Aura's STDP closes a loop on the LiquidSubstrate's W
matrix — the technical-critique-response already flagged this as needing
external validation. The paper's core observation (rank collapse → plasticity
loss) is directly measurable on W.
**Implementation:** `core/consciousness/plasticity_monitor.py`. Periodic SVD
on W, tracks effective rank (Roy & Vetterli stable rank), warns when
plasticity floor is breached. Defensive against shape changes; pure numpy.
Tests cover full-rank, low-rank, and degenerate matrices.

## Skipped — with reasons

### Unified Latents (arxiv 2602.17270)
Joint latent training across diffusion / autoencoder architectures. Aura
isn't training generative models in the diffusion sense; the substrate
state vector is a dynamical-system primitive, not a latent code. Skip.

### Conductor: NL agent orchestration (arxiv 2512.04388)
Learned natural-language coordination across agents. The autonomy pipeline
already has deterministic orchestration (curiosity → router → fetcher →
comprehension → reflection). Migrating to a learned conductor is a real
project, not a tonight item. Note for future direction.

### Thinking Without Words / Abstract CoT (arxiv 2604.22709)
Latent reasoning via non-linguistic abstract tokens. Requires base-model
training to introduce the abstract tokens. Qwen 2.5 wasn't trained that
way. Round-3+ if Bryan wants to swap to a base model that supports it.

### LLM.int8() (arxiv 2208.07339)
Vector-wise int8 with outlier mixed-precision. The paper informed MLX's
quantization design; Aura already runs Qwen 2.5-32B-8bit and 72B-4bit
on MLX. The wins are already in the runtime. Nothing new to apply at
the application layer.

### Self-Adapting Language Models / SEAL (arxiv 2506.10943)
Inference-time weight updates from in-context signals. Conflicts directly
with the Will identity gate (refuses unauthorized identity-modifying
writes) and creates a real safety surface. Also: runtime weight updates
during MLX inference invalidate KV cache and risk memory pressure on
Apple Silicon. Skip on architectural and safety grounds.

### LeWorldModel JEPA from Pixels (arxiv 2603.19312)
Joint-Embedding Predictive Architecture with temporal latent path
straightening for pixel-based world modeling. Aura's perception layer is
screen-text + OCR + audio; pixel-based JEPA doesn't fit the current
sensory pipeline and would be a separate project.

### JCode (github 1jehuang/jcode)
Coding-agent harness with semantic memory and agent swarm. Aura's
memory_facade and autonomy_pipeline already cover the analogous patterns.
Specific implementation details aren't portable.

## Round-2 dataset signals (carry over from prior live-test report)

These were already on the round-2 list and remain relevant — research
audit doesn't change them:
- Anti-confabulation honest-absence pairs ("I don't have a specific instance, but…")
- Substrate-grounded introspection examples (reference valence/arousal/neurochemicals,
  not generic chat-AI prose)
- Multi-turn coherence at 5+ turns
- Diverse phrasings of similar prompts (kill mode-collapse)
- File-reference acknowledgement examples (now possible since chat preflight
  handles the file-load; the model needs examples of using that context)

## Notes on what was skipped vs fundamentally rejected

"Skip" here means "not the right fit *now*" — not that the paper is wrong
or unusable. The Conductor and ACT papers in particular are real research
that could inform later evolutions of Aura's autonomy and reasoning layers.
SEAL is the only one I'd call architecturally incompatible without a
reframing of the Will gate.
