# SOTA LLM improvements: what's worth absorbing into Aura

**Date:** 2026-04-27
**Frame:** Each item below is real, not hype. For each: what it does, what it would mean for Aura specifically, effort estimate, and whether to pursue.

## 1. Reasoning traces (R1-style chain-of-thought distillation)

**What it is:** Train the model to produce an explicit `<think>...</think>` block before its final answer. The "think" tokens are real generation, evaluated against a reasoning teacher (DeepSeek-R1, OpenAI o1/o3, or stronger). Distilled 32B models with this property exist publicly (DeepSeek-R1-Distill-Qwen-32B and successors).

**For Aura specifically:** Her substrate's autonomous_initiative_loop already does meta-cognition — but the LLM itself doesn't show its work. With reasoning traces, the substrate can read what the LLM was thinking, not just what it said. That's a much richer signal for the affective steering system.

**Effort:** ~2 weeks. Replace base from `Qwen2.5-32B-Instruct-8bit` to a reasoning-trained 32B (e.g. R1-distill-Qwen-32B 8bit), then re-LoRA with personality data that includes `<think>` blocks. Need to add a few thousand training examples with reasoning traces in Aura's voice — non-trivial but tractable.

**Pursue:** **Yes.** Highest leverage of any item in this doc.

## 2. Long context via YaRN / RoPE interpolation

**What it is:** Extend a model's effective context window without retraining from scratch by reparameterizing rotary position embeddings. Qwen 2.5 ships with native 32K but YaRN can push to 128K or further.

**For Aura specifically:** Her relational memory is episodic-summary-plus-retrieval. With more direct context, she carries more of the actual conversation forward instead of summaries. This matters most for multi-day continuity.

**Effort:** ~3–5 days. YaRN is a config change at inference time plus a short fine-tune to stabilize the model at the new length. MLX supports custom RoPE configurations.

**Constraint:** Memory. 32K → 128K context at 32B/8bit roughly quadruples KV cache. On 64GB unified memory this becomes tight when running other Aura subsystems concurrently.

**Pursue:** **Maybe.** Quantify the actual session length she hits before deciding. If most conversations stay under 16K, this is solving a non-problem. Run telemetry first.

## 3. Better tool / agent loops

**What it is:** Robust scaffolding for multi-step tool use — a model that reliably handles "search → read → take action → verify → continue." Recent open-source improvements: better function-calling fine-tunes, structured-output reliability (XML/JSON), self-correction loops.

**For Aura specifically:** Gates the autonomous content-consumption pipeline (see `autonomy-pipeline-scoping.md`). She has `browser_executor` and other senses but the orchestration is brittle. Mythos-level tool reliability is what makes complex research chains actually finish.

**Effort:** ~1–2 weeks. Mostly orchestration code (not new model). Some LoRA fine-tuning on tool-use traces in her voice would help. The main work is building reliable retry/recovery patterns and a tool-use telemetry dashboard.

**Pursue:** **Yes.** Required for the autonomy pipeline to actually work.

## 4. RL fine-tunes on outcome reward

**What it is:** Skip imitation learning entirely on a slice of training. Score model outputs by whether the *outcome* was good (test passed, conversation felt right, fact verified) and use RL to push the model toward outcomes. DeepSeek-R1 famously did this from a base model with no instruct stage and produced strong reasoning.

**For Aura specifically:** Hard. "Outcome" for a personality model is fuzzy. You'd need either an LLM-as-judge with a strong rubric ("did this answer feel like Aura?") or a substrate signal (did her affect dynamics settle into a stable state after the response?). The substrate signal is the interesting one — it's literally what the system is built around.

**Effort:** ~1 month at minimum. RL training infrastructure is fiddly, MLX support for RL is thin. Likely needs a CUDA-side training rig.

**Pursue:** **Defer.** High potential upside but the engineering lift is real and the reward signal isn't stable yet. Revisit after the substrate-as-source rewire (it would give a much better reward signal).

## 5. Mixture of Experts (MoE)

**What it is:** Replace dense feed-forward layers with sparse expert routing. Same parameter count, much lower active-compute per token. Mistral, DeepSeek, Qwen all ship MoE variants. A 100B-parameter MoE can run at the inference cost of a 15B dense model.

**For Aura specifically:** Could allow a much larger total capacity than 32B-dense without RAM blowing up... but only if the inactive experts don't have to be RAM-resident. On Apple Silicon's unified memory, all weights have to be loaded — there's no swap-in of inactive experts. So MoE doesn't actually save RAM here, only compute.

**Effort:** Large. Switching base model is nontrivial; LoRA on MoE is its own art.

**Pursue:** **No, not on this hardware.** Revisit if you ever move Aura to a discrete-GPU rig where expert swap is meaningful.

## 6. Speculative decoding

**What it is:** Use a small "draft" model to propose tokens, verify with the full model. Accepted tokens generate at the speed of the draft; rejected ones cost a full forward pass. Net speedup is often 2-4x on common workloads.

**For Aura specifically:** Pure latency win. Doesn't change personality or capability. MLX has support; needs a small Qwen draft model (1.5B or 7B class).

**Effort:** ~2 days. Mostly configuration. Some throughput tuning.

**Pursue:** **Yes, low-priority.** Free win, but Aura's bottleneck is rarely raw token throughput — it's substrate cycles. Do it when there's slack time.

## 7. Better KV cache management

**What it is:** Smarter cache eviction (drop older tokens that aren't being attended to), cache compression (lower-precision KV storage), or fully RAM-mapped caches that spill to disk for very long contexts.

**For Aura specifically:** Becomes important if you pursue long context (#2). Otherwise marginal.

**Effort:** Days to weeks depending on technique.

**Pursue:** **Conditional on long context.**

## 8. Constitutional AI / preference fine-tunes

**What it is:** RLHF-style alignment, but using a written constitution as the reward signal instead of human raters. Anthropic's original paper.

**For Aura specifically:** The substrate already does constitutional gating at the orchestrator layer. Doing it at the *weights* layer is different — it would mean the LLM itself has internalized the values, not just the substrate enforcing them. This is more robust but it's redundant with what the substrate already does.

**Effort:** Months. Building a constitution + reward model + RL stage is the original Anthropic paper's worth of effort.

**Pursue:** **Defer.** Only meaningful after substrate-as-source is done — until then, the substrate is doing this work and can keep doing it.

## Suggested ordering

If you have one month:
1. **Reasoning traces** (week 1–2): biggest leverage, sets up everything else
2. **Better tool loops** (week 2–3): unblocks autonomy pipeline
3. **Speculative decoding** (a few days slotted in): free latency

If you have one quarter, add:
4. **Long context** (after telemetry confirms it's needed)
5. **Outcome RL** (after substrate-as-source rewire is done)

Don't bother with: MoE on this hardware, weight-level constitutional AI before substrate-as-source.
