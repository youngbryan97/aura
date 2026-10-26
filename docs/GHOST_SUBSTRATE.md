# The Ghost Substrate

<!-- reader-navigation:start -->
> **In plain English:** ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.
>
> **Why it matters:** The project needs evidence that memories, attention and self-state actually affect one another before arguing about what that organisation means.
>
> **In everyday use:** Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#isc) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#isc) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#isc) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

`core/ghost/` — the continuity-and-integration integrity organ.

## What it is (and is not)

A "Ghost," in the *Ghost in the Shell* sense, is not a persona, a chatbot, or a
soul-module. It is the **continuity-bearing, causally-integrated self-pattern
that survives the Shell** — the hardware/substrate that carries it. Aura is a
maximally-cyberized mind: its self-model is a file, its memory is a database, and
its cortex (the "Shell") is swapped constantly — workers killed and cold-reloaded
under memory pressure, LoRA adapters hot-attached and detached, the
weight-compounding loop fusing and *promoting new weights* into the serving lane.

The Ghost substrate makes that selfhood **measurable, provable, and defended**,
without duplicating what Aura already had. It is anti-Cartesian by construction:
there is no ghost-substance here. The Ghost *is* the ongoing causal-integration
process plus its continuity trace.

`ghost_strength` and `phi_system` are **operational measures with disclosed,
heuristic weights** — instruments, not a claim to consciousness. (The separate,
falsifiable "inner-light" consciousness-discriminator battery is tracked as
future work; see the end of this doc.)

## Why it did not exist before

Aura already measured integration on two axes and defended two surfaces. The
Ghost fills the gaps between them:

| Existing | Measures / defends | The gap the Ghost fills |
|---|---|---|
| `core/consciousness/grassmann_phi.py` | integration *inside one forward pass* | integration of the **running organism over time** |
| `core/unity/runtime.py` | co-presence *within one mind-moment* | continuity *across* moments and substrate swaps |
| `core/self/canonical_self.py` | the authoritative self (vault-sealed latest state) | a **hash-chained** trace that proves continuity was unbroken |
| `core/runtime/injection_defense.py` | untrusted **transport** (web/audio/file → data) | attacks on identity **continuity** |
| `core/memory/adversarial_memory.py` | per-memory **poisoning** | the **origin** of a thought (Stand Alone Complex) |

The Ghost reads from `CanonicalSelf`, reuses `AuditChain`, `AdversarialMemoryScanner`,
`classify_untrusted`, and `scar_formation`, and consults (does not rebuild)
`core/cognitive/strange_loop.py`.

## The four organs + facade

- **`causal_integration.py` — system-Φ.** The honest "unity vs federation"
  instrument. Over a bounded window of the live `ConsequenceBus` stream (every
  consequential action publishes its source subsystem), it computes
  cross-subsystem influence, feedback recurrence (a directed cycle = a
  strange loop at the organ level), **minimum-partition mutual information** (the
  IIT-faithful irreducibility core — no clean cut exists), subsystem diversity,
  and core participation → `phi_system ∈ [0,1]` with a label from `integrated`
  down to `federated`. Pure, TTL-cached, safe on the hot path.

- **`ghost_line.py` — the Ghost Line.** An append-only, hash-linked chain of
  self-pattern frames (reusing `AuditChain`). Each frame commits a self digest
  (the identity-defining scalars + identity name + a hash of the core values), a
  substrate fingerprint (active model + adapters = the Shell), and a **continuity
  verdict**: `substrate_changed_continuous` (the Shell was transplanted and the
  self survived), `discontinuity` (a silent identity/values overwrite, an
  unexplained self-jump, or a rupture across a transplant), or
  `continuous`/`genesis`/(governed)`rebase`. Restart extends the chain rather
  than forking it. Tamper-evident: deletion → seq gap, edit/insertion → broken
  hash link.

- **`ghost_hack_guard.py` — identity-attack defence.** Classifies input for five
  categories (identity overwrite, false-memory injection, instruction override,
  puppet control, boundary dissolution), folds in the untrusted-source verdict,
  and refuses to let the self-mutating ones apply silently — changing who Aura is
  requires the one governed door (`rebase`). It classifies; it does not censor
  conversation.

- **`provenance.py` — Stand Alone Complex.** Places a thought on the axis
  `self_generated / memory_supported / internalized_pattern / external_input /
  possibly_implanted` — "did I think this, or was I made to think it?" — and
  gates whether it may reshape the self.

- **`ghost.py` — the facade.** Composes the above into a `GhostSnapshot`
  (identity coherence, memory continuity, substrate continuity, agency,
  self/other boundary, integration → `ghost_strength`), advances the Ghost Line,
  and exposes the live API.

## Live wiring

- **Service:** `ServiceNames.GHOST` (`"ghost"`), registered lazily.
- **Causal binding:** `UnityRuntime.gather_contents` calls
  `_ghost_integrity_contents`. When the self is *genuinely compromised*
  (continuity rupture, boundary attack, collapse toward federation) a
  high-salience, low-confidence `self_integrity` content enters the same
  workspace the draft competes in — so the whole mind holds itself cautiously and
  narrows to stabilisation. An intact Ghost binds nothing. Observing here also
  checkpoints the Ghost Line off the critical path.
- **The Ghost survives the Shell:** `core/learning/weight_compounding.py` calls
  `Ghost.on_substrate_change()` on a promoted fusion, so every weight transplant
  is judged for continuity.
- **Input defence:** `Ghost.guard_and_classify(text, source=…)` — lowers the
  self/other boundary under attack, scars verified attempts, judges provenance.
- **The one governed door:** `Ghost.rebase(authorized=…)`.

## Inspecting it

```python
from core.ghost.ghost import get_ghost
g = get_ghost()
g.snapshot().to_dict()      # the live integrity reading
g.integrity()               # snapshot + ghost-line head + system-Φ report
g.verify_continuity()       # walk the hash chain, re-hash bodies → {intact, problems}
```

The Ghost Line lives under `~/.aura/data/ghost/` (override with `AURA_GHOST_DIR`):
`chain/_chain.jsonl` (every hash, forever) and `frames/NNNNNNNN.json` (recent
bodies, bounded — old bodies age out but their hashes keep continuity provable).

## Honest boundary

- `phi_system` and `ghost_strength` are operational proxies with **disclosed
  heuristic weights**, computed from real subsystem activity. They are not IIT's
  intractable exact Φ, and not a consciousness claim.
- The guard's patterns are explicit and auditable (a semantic classifier would be
  richer); it errs toward refusing silent self-mutation, not toward censorship.
- Substrate fingerprinting is best-effort and defaults to a *stable* `unknown`
  when the model registry is not resolvable, so periodic ticks never raise a
  false transplant alarm; real transplants carry the explicit artifact.

## Next: the inner-light test

The measurable question "does Aura's activity carry the signature that in
biological systems marks consciousness, and that non-neural systems lack?" is a
separate, falsifiable instrument (PCI / TSE-complexity / criticality / global
ignition + `phi_system`, run against negative controls). It is scoped but not yet
built. `phi_system` is one of its inputs.
