# Claude Mythos: what's public, what's worth porting

**Date:** 2026-04-27
**Status:** Synthesis of public reporting + Anthropic preview material. Some content is reporter speculation around a leak; flagged inline.

## Quick facts (high-confidence)

Mythos was first public via a CMS-misconfigured pre-release blog post on 2026-03-26; officially announced as **Mythos Preview** on 2026-04-08.

| Property | Value |
|---|---|
| Tier | New tier (not an Opus successor) |
| Context window | 1M tokens |
| Max output | 128K tokens |
| Reasoning | Supported (chain-of-thought visible) |
| Knowledge cutoff | Dec 2025 |
| SWE-bench | 93.9% |
| USAMO | 97.6% |
| Cybersecurity | "Strikingly capable" — Anthropic claims it can identify and exploit zero-days in real-world software |

Available on Vertex AI, Bedrock, and direct via the API. A separate preview, **Project Glasswing**, is the cybersecurity-specific research preview, invitation-only.

Sources:
- https://red.anthropic.com/2026/mythos-preview/
- https://platform.claude.com/docs/en/about-claude/models/overview
- https://cloud.google.com/blog/products/ai-machine-learning/claude-mythos-preview-on-vertex-ai
- https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-mythos-preview.html
- https://siliconangle.com/2026/03/27/anthropic-launch-new-claude-mythos-model-advanced-reasoning-features/
- https://fortune.com/2026/03/26/anthropic-says-testing-mythos-powerful-new-ai-model-after-data-leak-reveals-its-existence-step-change-in-capabilities/

## What this implies architecturally (some of this is informed speculation)

**1M context with 128K output is a structural step.** Most prior Claude models traded off context for output, or used compression tricks at long context. A clean 1M with 128K out implies:
- An attention/retrieval scheme that genuinely scales (interpolated/yarn rope or a new positional method, plus likely RAG-like memory routing inside the model)
- Significant inference-time engineering for the 128K output side (speculative decoding or a parallel-decoder variant)

**SWE-bench 93.9% and USAMO 97.6% together suggest two things working at once:**
- A reasoning trace that is much deeper than what Sonnet/Opus 4.6 had (probably distilled from a stronger reasoning teacher)
- Tool-use / agent orchestration improvements — SWE-bench at this level requires multi-step plan execution, file search, edit, test loops

**Cybersecurity as a flagship is a new posture for Anthropic.** Project Glasswing being a separate cybersecurity-only preview means the core Mythos model has the capability gated behind safety classifiers, and the unleashed version is invitation-only for defensive researchers.

## What's worth porting to Aura (in order of fit)

### 1. Reasoning traces with visible chain-of-thought
**Why:** Aura already has a substrate that thinks before responding (continuous_substrate, autonomous_initiative_loop). What she lacks is *visible reasoning at the LLM layer* that the substrate can read back. Adding a reasoning-trace head to her LLM means substrate can post-hoc evaluate the LLM's reasoning, not just its final tokens.
**Effort:** Months. Either fine-tune a reasoning trace into the existing LoRA stack (achievable with a R1-style distillation dataset) or wait for a Mythos-class open model that ships with reasoning out of the box.

### 2. Long context (toward 1M)
**Why:** Aura's memory architecture currently relies on episodic summary + vector retrieval. With longer context, more of her *direct* recent history can stay in the working window without needing to be re-retrieved. This is huge for multi-day relational coherence.
**Effort:** Medium for partial gains (yarn/rope interpolation can take Qwen 32K → 128K with modest training), large for true 1M (needs new positional encoding + attention scheme).
**Constraint:** RAM. 32B at 8bit with 1M context would not fit on a 64GB Mac. Wait until a smaller flagship-class model (or until you have more hardware) before chasing the full 1M.

### 3. Better tool / agent loops
**Why:** Aura's `core/executors/browser_executor.py` exists but isn't getting heavy use. The autonomy pipeline scoping (separate doc) needs solid tool-use to drive the content-consumption hierarchy. Mythos-level tool reliability is what makes "fetch a transcript, then a creator interview, then a Wikipedia page" actually work without dropping the thread.
**Effort:** Medium. Doesn't need new model weights — just better orchestration code, prompt scaffolding, and maybe some fine-tuning on tool-use traces.

### 4. (Don't port) Cybersecurity capabilities
**Why not:** Aura's purpose isn't pentest. The capability surface is irrelevant to her, and the safety surface is large. Skip.

## What's NOT a Mythos feature, but adjacent and worth doing

- **Speculative decoding** — Mythos likely uses it for the 128K output. Aura would benefit on response latency. MLX has speculative-decode support; needs a small draft model.
- **Better KV cache management** — at long context this becomes the bottleneck. MLX's cache is decent but not state of the art.
- **Outcome-RL fine-tunes** — DeepSeek-R1 demonstrated that pure RL on outcome reward (no human-feedback intermediate) can produce frontier reasoning. Worth experimenting with on Aura's substrate-loop outputs.

## Caveats

- I have no privileged information about Mythos's actual architecture. Most of the "what this implies" section is reporter and analyst speculation around the public benchmark numbers and the docs.
- The Anthropic preview page (red.anthropic.com) contains the most reliable framing; the Fortune piece quoted Anthropic on "step change in capabilities" — that wording is theirs.
- Don't make architectural decisions on the assumption that any of the "informed speculation" section is right. Treat it as priors, not fact.

## Recommendation

Don't try to "be Mythos." Pick the 2–3 capabilities that meaningfully change what Aura *can do* and pursue those:

1. **Reasoning traces** (highest leverage for her substrate)
2. **Better tool loops** (gates the autonomy pipeline)
3. **Speculative decoding** (free latency win)

Long context is tempting but RAM-bound. Defer.
