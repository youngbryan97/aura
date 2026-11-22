# Aura Memory Architecture

<!-- reader-navigation:start -->
> Aura keeps records of events, facts, preferences, and procedures. It looks for records relevant to the question or task in front of it.
>
> A memory is useful when you can find its source, correct it, or remove it. If you correct the date of a project, a later reply should use the correction and remember where it came from.
>
> [Plain-language guide](PLAIN_ENGLISH_GUIDE.md#memory) · [Terms and examples](VARIABLES_EXPLAINED.md#memory) · [Evidence and related work](DEMOS_AND_RELATED_WORK.md#memory)
> [All docs](DOCUMENTATION_INDEX.md) · [Glossary](GLOSSARY.md) · [Worked formulas](FORMULAS.md)
<!-- reader-navigation:end -->

## Overview
Suppose you correct a fact about a project on Monday and ask about it on
Friday. Aura has to find the right record, know where it came from, and use it
when producing the answer. Different paths keep the current conversation,
past events, facts, and learned procedures. What they retrieve can affect
generation, preferences, identity records, and affect values.

The stores have different rules. The quarantine store can encrypt its own
records. For other stores, check the storage and retention rules that apply
to each one.

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
**BlackHole** implements authenticated encryption for the data that its own
store persists. This is not an encryption claim about every application store. If encryption keys are unavailable, the system fails closed and refuses to write plaintext to disk.
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
- Native model window: **32,768 tokens**. Safe live work uses **1,024-token**
  operational views and a **2,048-token** padded-work budget; longer text is
  covered through overlapping views, not one unbounded encode.
- Explicit instructions differentiate tasks (`memory_recall`, `evidence`, `document`) rather than using generic web-search prefixes.

## Conceptual Gravitation & Reconsolidation

Recalling an episodic memory places it back into a labile state (`reconsolidation.py`). Its emotional tone and qualitative snapshot drift toward the present context (Conceptual Gravitation). How much it may change is gated by a neurochemical plasticity signal, so fidelity drops only within governed parameters.

## Memory Physics & Substrate Allocation

Memory follows physical allocation limits (`physics.py`, `substrate_allocation.py`). Working memory and engram retrieval compete under homeostatic constraints where slightly stronger representations naturally suppress weaker ones, reducing one source of confabulation.

## Her Corpus and What Her Searches Kept

Beside episodic and semantic memory sits her corpus (`core/knowledge/local_corpus.py`): an offline copy of Wikipedia, dated by when the copy was taken, and the pages her web searches kept, stored as their own shelf (`web_retained`) with the date each was read and its address. A search she kept answers the same question after a restart without going online ([L05](evidence/L05_KNOWLEDGE_ACROSS_TURNS_2026-10-05.md)). Every paragraph is also indexed on its own (`core/knowledge/passage_index.py`, Tantivy), so a paragraph in an article the question never names can still be found.

## Evidence Boundaries: What Memory DOESN'T Do
- **Memory does NOT write to disk without a governance check.** The `MemoryWriteGateway` fails closed if no authority is wired.
- **Encrypted-store writes must fail closed when their keys are unavailable.**
  This requirement applies to the named encrypted store, not every application
  database. See [the data card](../DATA_CARD.md) for storage scope.
- **Embedding work is bounded below the native window.** Operational views
  enforce the live padded-work budget even when the text fits in 32K tokens.
- **Memory does NOT statically record.** Repeated recall changes the vividness and fidelity of a trace, meaning the model reads a *reconsolidated* memory, not a permanent transcript.

---
*Module count: ~116 as of 2026-09-15.*
