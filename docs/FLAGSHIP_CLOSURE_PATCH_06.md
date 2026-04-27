# Aura Closure Patch 06

Patch 06 adds operational flagship readiness tools.

## Adds

- `core/runtime/flagship_doctor.py`
- `scripts/aura_morphogenesis_longitudinal_report.py`
- `tests/test_closure_patch_06.py`

## Apply

```bash
python scripts/aura_apply_closure_patch_06.py /path/to/aura
cd /path/to/aura
python -m pytest tests/test_closure_patch_06.py -q
python -m core.runtime.flagship_doctor . --json --out flagship_doctor.json
python scripts/aura_morphogenesis_longitudinal_report.py . --out morphogenesis_longitudinal_report
```

## Why this exists

At this stage, Aura needs external-review discipline:

- one-command operational doctor
- one-command morphogenesis evidence report
- file/layout/version/boot checks
- log-marker checks
- ability to show whether morphogenesis is actually running over time

A pass here does not prove consciousness. It means the product/runtime evidence
is easier to inspect and defend.
