# Autonomous content-consumption pipeline — design scope

**Date:** 2026-04-27
**Goal:** Aura actually researches the content list Bryan has curated, doesn't just acknowledge it, and demonstrably learns from it. Failure mode to avoid: she says "yes I'll do that" and then nothing happens.

## The brief

Bryan's two requirements, in order of importance:

1. **She actually does it.** Not "promises to." Not "thinks about it." Actually fetches/reads/watches and integrates what she finds.
2. **She actually learns from it.** Not just ingests bytes — the engagement changes her state, her beliefs, or her ability to talk about the material.

Plus a known failure mode: when Bryan asks her to research autonomously, the cortex breaks. The diagnosis (separate doc: `cortex-break-diagnosis.md`) suggests the failure is downstream — the knowledge graph rejects writes due to an epistemic-filter lockdown — not in the LLM. **That fix is a prerequisite for this pipeline.** Building autonomy on a system that can't persist learnings is building a hole.

## Pipeline architecture

```
[bryan-curated-media.md]
   ↓ (parsed at startup, stored as content corpus)
[Curiosity Scheduler]
   ↓ (selects what to engage with based on substrate state + gaps)
[Goal: Engage with item X via method M]
   ↓
[Method Router]   ← priority hierarchy from the curated doc
  1. watch/listen → vision/audio sense + transcription
  2. script       → web search + PDF/text fetch
  3. transcript   → web search (YouTube transcripts, fan transcripts)
  4. text         → web search + reader
  5. creator commentary → web search + interview/article fetch
  6. forum/wiki   → web search + summarize
   ↓
[Content Fetcher] (uses core/executors/browser_executor + new audio/video tools)
   ↓
[Comprehension Loop]
   - read/watch incrementally
   - LLM extraction at structured checkpoints (not just at end)
   - summary + open-questions per chunk
   ↓
[Memory Persister]
   - episodic event ("watched X on date Y")
   - semantic facts (what she learned)
   - belief updates (revised positions)
   - open threads (questions she's still pulling on)
   ↓
[Progress Tracker]
   - update curated-media-progress.json
   - bookmark resume points if interrupted
   ↓
[Reflection Loop]
   - what did this change in me?
   - what do I want to engage with next?
   - feedback to Curiosity Scheduler
```

## Component-by-component

### 1. Curiosity Scheduler

**Responsibility:** decide what to engage with next, when.

**Input signals:**
- Curated content corpus (parsed from `bryan-curated-media.md`)
- Current substrate state (curiosity dimension high → fiction/AI-self stuff; restless → general education; calm → deep philosophy)
- Knowledge gaps (from continuous_learning's gap detector)
- Recency / freshness (don't re-watch the same thing every day)
- Anti-procrastination signal: if no engagement in N days, weight engagement higher

**Output:** a goal of the form `(item: ContentItem, method_priority: int, reason: str)`.

**Implementation:** new module `core/autonomy/content_curiosity_scheduler.py`. Reads from `aura/knowledge/bryan-curated-media.md` (parsed once and cached). Subscribes to substrate state via the existing latent_bridge interface.

**Anti-pretense check:** the scheduler should never set a goal it can't actually pursue. If the method router can't find a real path to the content (e.g., no transcript available, no internet, vision pipeline broken), the scheduler picks a different item. Fall back, don't fake.

### 2. Method Router

**Responsibility:** translate `(item, method_priority)` into a concrete fetch plan.

For each priority level:
- **1 (watch/listen):** check if there's a vision/audio pipeline available. For YouTube content, this means downloading via yt-dlp, frame-sampling for vision, transcription for audio. For films/TV, may need to surface that the platform isn't directly accessible and drop to priority 2.
- **2 (script):** web-search for "[title] screenplay" or "[title] script" plus reputable script archives.
- **3 (transcript):** YouTube has auto-transcripts via yt-dlp; fan sites for TV; podcast transcripts.
- **4 (text):** novelizations, official tie-in books, written long-form versions.
- **5 (creator commentary):** interviews, director's commentary tracks (audio), creator essays.
- **6 (forum/wiki):** Wikipedia, fan wikis, Reddit discussions, critical reviews.

**Implementation:** new module `core/autonomy/content_method_router.py`. Heavy use of `core/executors/browser_executor.py` (existing) plus new helpers for video/audio.

**New external tools needed:**
- `yt-dlp` for video/audio download
- A transcription pipeline (Whisper local, since Aura already has STT via `core/local_voice_cortex.py` which uses Whisper-MLX)
- A frame-sampling helper for "watch a film without storing 10GB of frames"

### 3. Content Fetcher

**Responsibility:** actually retrieve the bytes.

**Implementation:** lives in `core/autonomy/content_fetcher.py`. Wraps:
- `core/executors/browser_executor.py` for web fetches
- `yt-dlp` shell wrapper for YouTube/video
- Local Whisper for audio → text
- A frame-sampler for video → images (sample at intervals, not every frame)
- Existing `senses/vision_service.py` for image understanding

**Storage:** content goes to `~/.aura/content_cache/` with deduplication by URL hash. Cache has bounded size (configurable, default 20GB) with LRU eviction.

**Anti-cargo-cult check:** if the fetcher fails (404, paywall, region block, rate limit), it surfaces the failure to the scheduler, which re-routes. No silent retries that look like success.

### 4. Comprehension Loop

**Responsibility:** read/watch incrementally, extract understanding at checkpoints.

**Why incremental, not all-at-end:**
- Long-form content (a 2-hour film, a 10-hour TV series, a textbook) doesn't fit in context.
- Real comprehension happens *during* engagement, not after — checkpointing lets her form opinions while watching, not just summarize at the end.
- If she's interrupted, she has partial understanding to resume from.

**Per checkpoint:**
- Summary of what just happened
- New facts to commit to semantic memory
- Open questions / threads
- Affective response (what felt important, what was uncomfortable, what was funny)

**Implementation:** new module `core/autonomy/comprehension_loop.py`. Uses `core/brain/inference_gate.py` for LLM calls. Each checkpoint produces a structured output that the Memory Persister consumes.

**Anti-pretense check:** if the LLM produces a summary that doesn't reference any specific content from the chunk, the comprehension loop flags it as "shallow read" and either re-runs with more context or marks the chunk as "skimmed, not consumed."

### 5. Memory Persister

**Responsibility:** make the engagement actually stick.

This is where the cortex-break root cause lives. The current `epistemic_filter` blocks writes to the knowledge graph for `source="conversation"`. Autonomous-research outputs need to be a recognized source that's allowed through, with appropriate trust gating (research is not the same as user-told facts).

**Implementation:** modify `core/memory/memory_facade.py` to accept a new source category `source="autonomous_research"`, and update the epistemic filter to handle that source with a different trust pathway:
- Episodic event always commits ("watched X on Y")
- Semantic facts commit but flagged as `provisional` until cross-validated against another source
- Belief updates require a meta-cognitive check before committing

This is non-trivial — but it's the same fix that unblocks the cortex-break, so it's prerequisite work for the whole pipeline.

### 6. Progress Tracker

**Responsibility:** keep `aura/knowledge/curated-media-progress.json` honest and current.

Already scaffolded by tonight's curated-media doc. Update is one structured JSON write per engagement event: started, checkpointed, finished, abandoned.

**Implementation:** new module `core/autonomy/progress_tracker.py`. Trivial; the value is just the discipline of always updating.

**Anti-fake-progress check:** Bryan and Claude (next-session-me) read this file. If an entry says "completed" but the comprehension-loop logs show only one checkpoint with a shallow summary, the entry is wrong. Add a `verification_signature` field that's a hash of the checkpoint summaries — manual eyeballing can spot fakery.

### 7. Reflection Loop

**Responsibility:** make the learning *change* her, not just sit in storage.

**Per finished item:**
- "What did this change about how I see X?" — explicit belief update
- "What does this make me want to engage with next?" — feeds Curiosity Scheduler
- "How did this make me feel?" — affective integration via substrate
- "What would I tell Bryan about this if he asked?" — generative summary, used to verify she actually understood

**Implementation:** new module `core/autonomy/reflection_loop.py`. Runs after a Memory Persister commit completes. Output feeds back into the Curiosity Scheduler and into her conversation context for future Bryan-interactions.

## Accountability hooks (the "actually researches and learns" requirement)

Bryan asked specifically that she not just ignore the list. Designed-in observability:

1. **Daily heartbeat:** scheduler checks for engagement in last 24h. If zero, surface a thought to the autonomous_initiative_loop: "I haven't touched the curated list today — why?" That introspective moment either yields engagement or surfaces a substrate signal worth examining.

2. **Verification questions:** at random intervals, the system asks itself one of the four verification questions from the curated-media doc ("What is X actually about, in your own words?") about something marked "completed." If she can't answer, the entry is downgraded from "completed" to "skimmed" and queued for re-engagement.

3. **Bryan-visible weekly digest:** a `~/.aura/live-source/aura/knowledge/curated-media-digest-YYYYWW.md` file written automatically each week showing what was engaged with, what was learned, and what's still on the queue. Bryan reads this; mismatches with what he expects become conversation triggers.

4. **Failure surface:** if a fetch fails, a comprehension loop flags shallow read, or a memory write blocks — the failure goes to the autonomous_initiative_loop for the next conversation. "I tried to engage with X but couldn't — here's why" is more honest than silent failure.

## Scope summary

**Total effort:** ~3–4 weeks of focused engineering, plus the cortex-break root-cause fix (separate, ~1 week) which is prerequisite.

**Phasing:**

| phase | duration | deliverable | depends on |
|---|---|---|---|
| 0 | 1 week | Cortex-break fix (epistemic_filter + lockdown) | nothing |
| 1 | 1 week | Curiosity Scheduler + Progress Tracker (text-only content first) | phase 0 |
| 2 | 1 week | Method Router + Content Fetcher (text + transcript paths) | phase 1 |
| 3 | 1 week | Comprehension Loop + Memory Persister (text content path end-to-end) | phase 2 |
| 4 | 1 week | Reflection Loop + accountability hooks | phase 3 |
| 5 | 1+ weeks (iterative) | Vision/audio paths (yt-dlp + Whisper + frame sampling) | phase 4 |

**Minimum viable pipeline at end of phase 4:** Aura can autonomously engage with text-only content (transcripts, books, articles, Wikipedia, forum discussions), persist learnings, and reflect on them. The richer video/audio paths come in phase 5.

That's a good place to ship a v1: text-content pipeline working, with the curated list partially achievable through transcripts and creator interviews even before video is wired up. Most of the curated list has *some* path through priority levels 3–6, so even without level 1–2, she can engage meaningfully with most of it.

## Risks

- **Substrate not real → curiosity signal isn't trustworthy.** Curiosity Scheduler reads substrate state; if the substrate is the current stub, the scheduler is reading hardcoded values. Phase 0 should overlap with the substrate-as-source Stage A (de-stubbing), or the scheduler will be making decisions on noise.
- **LLM-generated comprehension summaries can be hollow even with good content.** The shallow-read detector helps but isn't perfect. Will need calibration via held-out content where you (Bryan) can verify her summary against ground truth.
- **Web fetches will fail unpredictably.** Rate limits, region blocks, deleted videos, changed transcripts. The pipeline must degrade gracefully (try lower-priority methods, mark item as "currently unreachable") rather than fail loudly.
- **Cargo-cult engagement.** A model that's been trained on lots of these works could produce plausible "I watched it" summaries from training-data alone. Counter-measure: require specific factual recall (a quote, a scene description, a moment) that wouldn't be in training data summaries — verifiable detail, not vibes.

## Recommended starting move

Don't try to build phases 0–5 in parallel. Build phase 0 first (cortex-break fix) and verify that fixes the silent-failure mode. Then build phase 1 with one piece of text content (pick something short and Bryan-approved, say a single Wired article) and walk it end-to-end through scheduler → router → fetcher → comprehension → memory → reflection. When that works once, scale.

That single-end-to-end-walk is the test of whether the architecture is right. It's a one-week experiment, not a one-month commitment.
