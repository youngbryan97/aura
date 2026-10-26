# Aura Morphogenesis Patch

<!-- reader-navigation:start -->
> **In plain English:** Architecture describes which parts exist, how information reaches them and which component is responsible for a decision or effect.
>
> **Why it matters:** A useful component needs a real reader and a real route into behavior; a box on a diagram can otherwise be disconnected from what users see.
>
> **In everyday use:** If a memory value is updated but no response path reads it, fixing its storage alone will not change an answer.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#architecture) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#architecture) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

This patch adds a bounded morphogenetic runtime:

- `core/morphogenesis/types.py` — schemas for signals, cells, manifests, lifecycle states and config.
- `core/morphogenesis/field.py` — generalized tissue/morphogen field with diffusion and decay.
- `core/morphogenesis/cell.py` — local-rule self-organising cells.
- `core/morphogenesis/metabolism.py` — CPU/memory/energy budget manager.
- `core/morphogenesis/organs.py` — co-activation graph and organ formalization.
- `core/morphogenesis/registry.py` — persistent registry using AtomicWriter when present and receipts when present.
- `core/morphogenesis/runtime.py` — task-tracked runtime loop, adaptive-immunity bridge, episodic memory bridge.
- `core/morphogenesis/integration.py` — default Aura cells and ServiceContainer registration.
- `tests/test_morphogenesis_runtime.py` — minimal regression tests.

## Boot

Already wired. `aura_main.py` awaits
`core.morphogenesis.integration.start_morphogenesis_runtime()` after core
services are registered, and `core/runtime/flagship_readiness.py` warns
(`MORPHOGENESIS_NOT_BOOTED`) if that hook goes missing.

The standalone-patch install instructions this section used to carry — copy
the directory, apply a snippet from a `patches/` tree — described a portability
path that does not exist in this repository.

```bash
pytest tests/test_morphogenesis_runtime.py
```

## Design rules

This layer does **not** directly apply source-code patches. It turns repeated
signals into cell/organ structure and routes dangerous conditions into existing
immunity/resilience systems.

Actual source modification should remain inside Aura's existing governed patch,
sandbox, test, receipt and rollback pipeline.
