# Aura Closure Patch 05

Patch 05 adds persistence ownership and evidence collection.

## Adds

- `core/runtime/persistence_ownership.py`
- `scripts/aura_persistence_audit.py`
- `scripts/aura_collect_flagship_evidence.py`
- `tests/test_closure_patch_05.py`

## Apply

```bash
python scripts/aura_apply_closure_patch_05.py /path/to/aura
cd /path/to/aura
python -m pytest tests/test_closure_patch_05.py -q
python scripts/aura_persistence_audit.py . --json
python scripts/aura_collect_flagship_evidence.py . --out flagship_evidence
```

## Why this exists

The remaining A+/flagship path is not just feature additions. It is evidence discipline:

- durable writes should use one canonical ownership path
- direct persistence bypasses should be discoverable
- flagship claims should have a reproducible evidence bundle
- logs and gates should be packaged for outside review

This patch does not automatically rewrite every persistence write. It gives you a scanner plus a canonical helper so the migration can be done safely.
