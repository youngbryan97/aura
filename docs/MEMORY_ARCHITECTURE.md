# Aura Memory Architecture

## Overview
Aura's memory system is not a passive transcript but an active cognitive substrate. It shapes generation, preference, identity, and affective context. Memories are ingested, consolidated, and reconsolidated in a multi-tiered architecture that strictly enforces write governance and local encryption.

## Multi-Tiered Architecture

The memory hierarchy is stratified into several distinct tiers (`core/memory/stratified/`):
- **Working Memory** (`working.py`): The active conversational session context.
- **Episodic Memory** (`episodic.py`, `episodic_memory.py`): Autobiographical conversation traces (engrams) and specific events.
- **Semantic Memory** (`semantic.py`): Distilled facts and conceptual knowledge.
- **Strategic / Procedural Memory** (`strategic.py`): Learned preferences and strategies.

Older or inactive memories are moved to **ColdStore**, while **State Snapshots** provide system audibility and backups.

## Vector Memory Engine & SQLite Vector Store

Underneath retrieval lies the **Vector Memory Engine** (`vector_memory_engine.py`) and a custom **SQLite Vector Store** (`sqlite_vector_store.py`). This allows rapid dense cosine similarity matching and exact text recall without a heavyweight external database, embedding directly into the runtime's memory blocks.

## Entity Graph & Knowledge Graph

Semantic relations are modeled in the **Knowledge Graph** (`knowledge_graph.py`) and **Associative Entity Memory** (`associative_entity_memory.py`). Concepts and entities introduced in conversations are extracted, linked, and maintained in a navigable graph structure to provide cross-session situational awareness.

## Black Hole Quarantine & Scar Formation

Memory writes that trigger constitutional governance or represent unresolved high-conflict states can be sent to **Quarantine** (`black_hole.py` / `memory_write_gateway.py`). 
**BlackHole** implements local AES-256-GCM authenticated encryption for all persistent data. If encryption keys are unavailable, the system fails closed and refuses to write plaintext to disk.
High-arousal or traumatic conversation moments result in **Scar Formation** (`scar_formation.py`), bounding future plasticity.

## Memory Write Gateway & Retention Policies

All memory writes pass through a single unified authority: the **Memory Write Gateway** (`memory_write_gateway.py`).
- Records governance authorization in a `MemoryWriteReceipt`.
- Uses atomic schema-versioned writes.
- Manages quarantine.
Retention policies dictate what is kept, compacted, or discarded over time to manage context limits.

## Hippocampal Indexing & Consolidation

Mirroring human neuroscience, `hippocampus.py` manages indexing. Episodic memories are bound to sparse associative cues. Recalling part of a cue reinstates the entire memory (pattern completion). During consolidation, salient engrams are stabilized and undergo governed "therapeutic reconsolidation" driven by voltage-based STDP and homeostasis.

## Embedding Runtime

The core embedding model is **Qwen3-Embedding-0.6B** (`embedding_model.py`).
- Operating at exactly **384 dimensions**.
- Maximum input tokens: **32,768** (drastically reducing the need for arbitrary chunking).
- Explicit instructions differentiate tasks (`memory_recall`, `evidence`, `document`) rather than using generic web-search prefixes.

## Conceptual Gravitation & Reconsolidation

Recalling an episodic memory places it back into a labile state (`reconsolidation.py`). Its emotional tone and qualitative snapshot drift toward the present context (Conceptual Gravitation). The level of allowed change is gated by a neurochemical plasticity signal, ensuring fidelity drops only under governed parameters.

## Memory Physics & Substrate Allocation

Memory follows physical allocation limits (`physics.py`, `substrate_allocation.py`). Working memory and engram retrieval compete under homeostatic constraints where slightly stronger representations naturally suppress weaker ones, preventing confabulation.

## Evidence Boundaries: What Memory DOESN'T Do
- **Memory does NOT write to disk without a governance check.** The `MemoryWriteGateway` fails closed if no authority is wired.
- **Memory does NOT write plaintext.** If Horcrux/BlackHole encryption fails, the system crashes rather than persisting unencrypted text.
- **Memory does NOT chunk arbitrarily.** With the 32K token window in Qwen3-Embedding, chunking is purely an overflow mechanism, preventing the splitting of semantic context.
- **Memory does NOT statically record.** Repeated recall changes the vividness and fidelity of a trace, meaning the model reads a *reconsolidated* memory, not a permanent transcript.

---
*Module count: ~116 as of 2026-09-15.*
