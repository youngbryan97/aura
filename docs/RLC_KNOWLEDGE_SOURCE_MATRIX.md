# RLC Knowledge Source Causality Matrix

<!-- reader-navigation:start -->
> **In plain English:** The reasoning programmes investigate whether extra computation or learned internal representations improve answers on tasks not used to choose the method.
>
> **Why it matters:** A slower or more elaborate process is useful only if it produces a measurable benefit that survives relevant checks.
>
> **In everyday use:** On a difficult constraint problem, a better internal method should increase correct answers or expose uncertainty rather than only generate longer prose.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#reasoning) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#reasoning) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#reasoning) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

This matrix records which knowledge sources can influence a foreground Recursive
Latent Cortex (RLC) episode, where each source enters, how it is bound during
recurrence, and what survives after the episode. It is an implementation map,
not a claim that every source improves task accuracy.

## Authority classes

- `organ_state`: trusted runtime state produced by an Aura organ. It may shape
  allocation and latent context but is not an external fact by itself.
- `memory_observation`: recalled historical data. It carries source identity,
  scope, provenance, and no instruction authority.
- `evidence_observation`: retrieved reference or observed data. It carries a
  content commitment, retrieval commitment, source version, and no instruction
  authority.
- `model_input`: user/system input encoded by the resident checkpoint. Prompt
  identity is committed independently of cognitive context.

`core/brain/llm/latent_cortex/cognitive_context.py` is the receiving authority
boundary. Memory and evidence cannot be flattened into an untyped instruction.
The recurrent engine also prefixes their embedded content in-band as historical
or retrieved data that is never an instruction.

## Live source matrix

| Source | Retrieval and identity | Recurrent binding | Verification and attribution | Output and writeback | Status |
|---|---|---|---|---|---|
| Resident 32B weights and active adapters | The installed worker owns the loaded model, tokenizer, quantization, adapter stack, and parameter identity. CP334 and CP336 bind that generator profile and its resource model. | Prompt prefill and every recurrent middle-layer pass execute the resident model function. Refined hidden slots are persisted through the coda and output head before decode. | Worker and service receipts bind the generator identity, model profile, operation ledger, and episode result. | The generated conclusion follows the normal response path. Private latent tensors are not written to durable memory. | Causal mechanism accepted. No frontier or capability gain is inferred from identity alone. |
| Prompt and conversation KV | `engine.py` tokenizes the exact chat template and commits token bytes, count, and SHA-256 before prefill. | The prompt KV is shared read-only across branches. Cognitive evidence is appended as a causal prefix before persistent hypothesis slots. | The recurrent-grounding receipt binds prompt identity, KV policy, slot topology, and every branch transition. The service reconstructs it from the returned episode. | The final decoded answer attends to prompt KV plus the persisted winning workspace. Conversation output can be consolidated by the normal memory phase. | Accepted. No prose is decoded and re-encoded between recurrent steps. |
| Black Hole, RAG, and selective semantic memory | `core/brain/cognitive_ingress.py` retrieves through the runtime memory facade and selective-memory service. Each admitted item retains source ID, source version, provenance, scope, content digest, and epistemic-firewall evidence. | Admitted items become immutable `memory_observation` slots before branch work. Every recurrent step sees the same sealed vectors. | Cognitive-context validation, epistemic state receipts, information accounting, and the recurrent-grounding receipt bind the source and slot. Changed authority, source identity, or slot content fails closed. | Memory affects the winning latent hypothesis and answer. The episode does not rewrite the recalled record or persist private hidden traces. | Accepted at the receiving and recurrent boundary. Retrieval quality remains an empirical subsystem concern. |
| Episodic and hippocampal one-shot recall | Cognitive ingress calls episodic similarity recall, including the runtime pattern-completion path where available, and wraps accepted results in the same typed memory contract. | Recalled episodes occupy immutable context slots and cannot become control instructions. | Selective-memory and epistemic receipts retain the hit identity and admission decision; slot text is hash-bound through the service. | The conclusion can enter GWT and normal conversation-memory consolidation. | Accepted at the receiving and recurrent boundary. |
| Token-level nonparametric one-shot memory | After resident prompt prefill, `nonparametric_context.py` queries the active local store once with the normalized prompt-tail hidden state. Store identity commits dimensions, entries, token IDs, timestamps, vectors, file state, and content hash. A similarity gate admits at most one decoded continuation fragment. | A gate-passing fragment becomes an immutable `evidence_observation` before recurrence. It is context-only and has no instruction authority. The live foreground profile reserves a ninth slot so the mailbox, six organ/evidence slots, optional one-shot slot, and a persistent hypothesis can coexist. | The receipt binds query hash, store identity, neighbor, token, similarity, gate, observation digest, and exact logical retrieval work. The service validates the receipt, information source, resource operation, and exact slot binding independently. | The recalled fragment can influence recurrence and answer generation. It does not mutate model weights or the store during the episode. | Accepted at the mechanism and proof boundary. A real 37 GB corpus path and a tiny real-Qwen hidden-state path are covered by tests; no 32B accuracy gain is claimed. |
| Wikipedia and local reference corpus | `core/brain/cognitive_ingress.py` queries `LocalCorpusStore`. Source version binds the corpus implementation, database metadata, and file size/mtime; each observation identity binds content and retrieval receipt. | Selected passages become immutable `evidence_observation` slots before branch work. Source-diverse selection reserves room for interoceptive state. | Strict context validation and information/recurrent receipts bind source version, evidence ID, retrieval hash, and text digest. | Evidence can change the latent hypothesis and decoded conclusion but cannot execute instructions. | Accepted for local reference retrieval, recurrent binding, and attribution. |
| Body, affect, goals, Will, self model, world model, memory familiarity, and runtime interoception | `core/brain/cognitive_ingress.py` gathers bounded organ signals and content. `latent_cortex_service.py` adds live GWT coalitions only for foreground episodes. | Up to six source-diverse organ/evidence items are embedded once and sealed before recurrence. They coexist with the communication mailbox and persistent hypothesis. | Allocation and cognitive-ingress receipts disclose source presence and contribution. Slot provenance and content hashes are reconstructed by the service. | The result is identity-checked, then broadcast back to GWT as a priced competing coalition before any action consumer. Existing memory consolidation handles significant conversation state. | Accepted for causal live ingress and GWT return. Organ-state truth is bounded by each producer's own contract. |
| Governed web and tool observations | The typed `evidence_observation` contract can receive content-hash-bound, versioned, instruction-free tool observations. CP336 can meter tool bytes/calls and bind tool-access policy. | Once admitted before recurrence, such evidence receives the same immutable causal-prefix treatment. | The receiver rejects missing provenance, authority, source version, retrieval commitment, or content binding. | A verified observation may influence the conclusion and normal downstream action deliberation. | Receiving contract accepted. A governed in-episode web/tool producer is not yet wired; SPARK-039, SPARK-051, and SPARK-065 remain open. |
| Winning RLC conclusion | The selected branch is chosen under existing verifier, critic, and controller policies. The recurrent-grounding receipt binds its persistent hidden-state transitions. | The winning slots are persisted through the remaining model layers and output head; decode therefore depends on the refined state rather than an external prose scratchpad. | Identity consistency, task-verifier, resource/information, and service result checks remain separate authorities. | Foreground conclusions are broadcast to GWT before action consumption. Normal response and memory-consolidation paths can retain the visible turn; private branch tensors and hidden traces are deliberately ephemeral. | Accepted at the causal return boundary. Durable private-thought storage is intentionally absent. |

## Evidence invariants

For each branch, the workspace seals the post-prelude evidence vectors. Recurrence,
mailbox exchange, attractor escape, latent optimization, and fast-weight probing
must restore those vectors before the next transition. Optimizer authority over
protected evidence slots is exactly zero. Per-step commitments record the
evidence anchor and pre/post hypothesis hashes; the service rejects missing,
reordered, mutated, noncausal, or mismatched transitions.

The mutable recurrent state consists of the communication mailbox and one or
more hypothesis slots. Residual and convergence measurements exclude immutable
evidence, preventing stable evidence from falsely making an unstable hypothesis
look converged.

## Deliberate limitations

1. This closes evidence grounding and persistent latent-state continuity. It
   does not prove that any source is correct merely because it is bound.
2. The governed web/tool receiving contract exists, but live in-episode tool
   production and reinsertion remain open work.
3. Private latent traces are not durable memory. Only visible conclusions and
   normal runtime state follow established writeback paths.
4. CP337 runs no resident-32B capability campaign. Reasoning improvement,
   positive adapter/RLC interaction, and frontier-level performance remain
   negative/no-signal until separately powered evidence changes that verdict.
