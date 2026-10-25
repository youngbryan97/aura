# Aura Memory Card

<!-- reader-navigation:start -->
> **In plain English:** Memory keeps records beyond the immediate request and selects relevant records for the current turn.
>
> **Why it matters:** Useful continuity depends on recalling the right information, retaining its source and correcting or deleting it when needed.
>
> **In everyday use:** If you correct a fact about a project, later answers should use the correction rather than retrieving the older claim as equally reliable.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#memory) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#memory) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#memory) · [Document map](docs/DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

## Purpose

How Aura's memory works, how what she remembers changes her actions, and how you can control it.

Memory here isn't just a stored transcript. It actively shapes how she feels and responds. Something she remembers about you today can change how she answers you next week. You should be able to see those memories and delete them if you want.

## Memory Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Memory Hierarchy                     │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │  Working   │  │  Episodic  │  │   Semantic    │  │
│  │  Memory    │  │  Memory    │  │   Memory      │  │
│  │ (session)  │  │ (convos)   │  │ (knowledge)   │  │
│  └─────┬──────┘  └─────┬──────┘  └───────┬───────┘  │
│        │               │                 │           │
│        └───────────┬───┴─────────────────┘           │
│                    │                                  │
│           ┌────────▼─────────┐                       │
│           │   ColdStore      │                       │
│           │  (long-term)     │                       │
│           └────────┬─────────┘                       │
│                    │                                  │
│           ┌────────▼─────────┐                       │
│           │ State Snapshots  │                       │
│           │  (backup/audit)  │                       │
│           └──────────────────┘                       │
└──────────────────────────────────────────────────────┘
```

## Memory-Behavior Causality

Memories actively change Aura's behavior in five ways:

1. **Context Assembly**: Relevant memories are pulled into the current conversation, directly shaping how she responds.

2. **Preference Learning**: Remembering what you like changes her response style, the tools she uses, and how proactive she is.

3. **Procedural Memory**: If she learns a process (like how to do a specific task for you), she remembers and follows it next time.

4. **Identity Continuity**: Her core personality (`CanonicalSelf`) stays consistent across different sessions, keeping your relationship with her stable.

5. **Error Memory**: She remembers past mistakes so she doesn't repeat them.

## Engram Dynamics (encoding, recall, reconsolidation)

Event memories (episodic memories) aren't just static recordings. Each memory trace (or "engram") has a lifecycle inspired by human neuroscience. You can find this code in `core/memory/episodic_memory.py`, `reconsolidation.py`, and `hippocampus.py`:

- **Encoding strength**: A memory forms more strongly if it involves emotion, a mistake, relationship importance, or something unexpected (novelty). When Aura is surprised, it triggers a signal (`on_novelty`) that makes the memory stick better.
- **Hippocampal index**: Every memory is linked to a few key triggers (cues). If you bring up just part of a memory, Aura can recall the rest of it (**pattern completion**). This works alongside standard text and keyword searches.
- **Reconsolidation**: When a memory is recalled, it becomes flexible (labile). Her current mood and context can blend into it. This means the memory's emotional tone updates to match the present, but its strict accuracy (**fidelity**) drops. A chemical signal called **plasticity** controls how much the memory can change, while strong, emotional memories resist changing. A cooldown period stops memories from changing too fast if recalled repeatedly.
- **Vividness ≠ accuracy**: Remembering something often makes the memory feel stronger and more vivid, but less perfectly accurate. If a memory changes too much, Aura flags it before using it in conversation.
- **Sleep replay**: While Aura "sleeps," important memories lock in (consolidate). If a memory is stressful or highly active, it goes through a safe softening process (**therapeutic reconsolidation**)—similar to revisiting a bad memory in a safe space to make it less intense.

Whenever a memory changes, it goes through the same strict checks as a new memory and triggers `memory.encoded`, `memory.reconsolidated`, or `memory.consolidated` events.

## Synaptic Plasticity Substrate (voltage-dependent STDP + homeostasis + competition)

Memory recall and storage rely on a neuroscience model called **voltage-based STDP with homeostasis** (found in `core/consciousness/voltage_plasticity.py`). This adds three key features that basic timing models (`stdp_learning.py`) miss:

- **Voltage-dependence**: A memory connection only changes if there is enough activity. Weak signals are ignored, and strengthening a connection requires a strong signal.
- **Homeostasis (stability)**: The system naturally balances itself so it doesn't get overloaded. If a memory connection gets too active, a built-in brake kicks in to lower it, preventing runaway feedback loops.
- **Competition**: Slightly stronger memories automatically override weaker ones.

In `core/memory/engram_plasticity.py`, this rules how Aura recalls events. When you ask a question (`recall_similar`), memories compete to answer it. The best match wins. Weakly related memories are blocked (voltage-gating), and the stability brake (homeostasis) prevents one loud memory from hijacking questions it doesn't belong to. This acts as an **anti-confabulation** mechanism to stop her from making things up. 

When Aura is emotional (aroused or positive), it's easier to trigger memories. Winning memories get a small, safe strength boost (**LTP**) in `_register_recall`. If one memory tries to dominate too much, the system flags it in its metrics (`engram_homeostatic_breach_total`).

**Learned associations**: In `core/memory/engram_association.py`, Aura links concepts together over time. If two memories are recalled at the same time, their connection gets stronger (*neurons that fire together, wire together*). These links are saved across sessions. When Aura tries to remember something, these learned links help her connect the dots better than just matching keywords.

**Positional recall**: If you ask "what did I first ask?", a content search won't work well. Instead, `core/conversation/grounded_recall.py` grabs the exact earliest or most recent message from your current chat. Aura reads the real quote back to you, guaranteeing she doesn't invent a response. Between memory competition, learned links, and positional grounding, Aura covers the three things memory needs: *what* was said, *what it connects to*, and *when* it happened.

## Symbolic Deduction (belief consistency)

Aura uses a strict logic checker (**natural-deduction proof search**) in `core/reasoning/natural_deduction.py`. It uses standard rules of logic to check if a set of statements makes sense or contradicts itself. It can provide a step-by-step proof or show exactly where the logic fails.

This is directly wired into her beliefs. In `core/reasoning/belief_consistency.py`, every plain-English belief is translated into formal logic. Whenever Aura learns something new (`process_new_claim`), `belief_revision.check_belief_consistency()` checks it. It catches direct contradictions (believing X and not-X) and indirect ones (believing X, knowing X leads to Y, but believing not-Y). 

If Aura catches a contradiction, she doesn't just flag it. In `core/reasoning/deduction_governance.py`, `_resolve_logical_conflicts` steps in and weakens the belief she is less confident in. This actively fixes her understanding of the world. The logic engine is also available to other systems via `SymbolicBridge.prove_logic` and `inference_audit.verify`.

This logic checker also monitors her **active reasoning**. `core/reasoning/inference_audit.py` reads her text to find logical arguments ("X, therefore Y") and proves them. Before a reply is saved (`_record_recent_response`), a background check (`_audit_recent_response_reasoning_sync()`) runs. If she confidently makes a logical leap that doesn't make sense, it's flagged as an error (`reasoning_non_sequitur_total`). The checker flags arguments it can disprove within the representation it supports. It records the finding without rewriting the reply; it does not establish that every natural-language argument was understood or checked.

## Substrate↔LLM integration audits (φ, CRSM-LoRA, CAA, integrity)

Aura runs background checks to ensure her deeper systems and language models are working together correctly. Instead of failing silently, these systems report their status clearly:

- **Grassmann φ on the transformer** (`core/consciousness/grassmann_phi.py`): This improves how Aura analyzes her own internal thought streams. Instead of crushing a massive amount of internal data (a ~5000-dimension vector) down to a simple average, it groups data into geometric "anchor modes" to better track her internal states. `phi_core.compute_grassmann_residual_phi()` uses this method to calculate a metric (`grassmann_phi_s`) that represents how integrated her thoughts are.
- **CRSM→LoRA loop** (`core/consciousness/crsm_loop_monitor.py`): This checks if the experiences she collects are actually being permanently learned (fine-tuned) into her neural network weights. It reports if the loop is CLOSED, OPEN, or IDLE, and warns if she is saving experiences but not learning from them. Currently, it is OPEN, meaning memories are saved but not yet fully learned.
- **CAA readiness** (`core/consciousness/caa/readiness_report.py`): This verifies the origin of the "steering vectors" used to guide her behavior. It checks if they were extracted from her core model or generated on the fly. Currently, it's in a BOOTSTRAP state, meaning her steering capacity is reduced because the vectors are generated on the fly.
- **System integrity audit** (`core/runtime/integrity_audit.py`): This combines all these health checks into a single report. It's available at `/api/health/heartbeat` so you can easily spot system failures without reading logs manually. It is strictly enforced if `AURA_STRICT_RUNTIME=1` is set.

The `SymbolicBridge` is also active: `audit_reasoning()` automatically checks every chat reply (`_record_recent_response`), sending logical arguments to the logic checker and math problems to a safe evaluator.

## Memory Governance

All memory updates follow a strict approval process:

```
Candidate Write → Will Decision → Receipt → Storage → Verification
```

- No memory is saved without formal approval (`WillReceipt`).
- Every memory is securely hashed to prevent tampering.
- The system logs exactly why each memory was written (its provenance).
- All memories can be audited, exported, or deleted.

## User Controls

You can manage large batches of memories using terminal `make` commands. To browse, edit, or delete individual memories, use the app's memory panel, which is powered by `interface/routes/memory.py`.

| Action | Command / mechanism | Effect |
|--------|---------------------|--------|
| List memories | App memory panel (`GET /api/memory/recent`, `/episodic`, `/semantic`) | Browse stored memories by store |
| Search / inspect | App memory panel | Find and open a specific memory |
| Delete one | App memory panel (`POST /api/memory/delete`) | Remove a specific memory |
| Export all | `make memory-export` | JSON export of all memory |
| Delete all | `make memory-purge` | Wipe all memories |
| Backup | `make backup` | Full state backup |
| Restore | `make restore BACKUP=<path>` | Restore from backup |
