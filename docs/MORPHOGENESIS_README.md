# Aura Morphogenesis Patch

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

## Install

Copy the `core/morphogenesis` directory into your repo and add the boot snippet from
`patches/bootstrap_morphogenesis_snippet.py`.

Then run:

```bash
pytest tests/test_morphogenesis_runtime.py
```

## Design rules

This layer does **not** directly apply source-code patches. It turns repeated
signals into cell/organ structure and routes dangerous conditions into existing
immunity/resilience systems.

Actual source modification should remain inside Aura's existing governed patch,
sandbox, test, receipt and rollback pipeline.
