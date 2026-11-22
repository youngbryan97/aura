# Runbook: Excessive Agency / Unauthorized Action

<!-- reader-navigation:start -->
> Aura can propose tasks, keep goals, and return to unfinished work. The user and the current situation can change which task takes priority.
>
> The useful questions are what it tried, who allowed it, and whether it finished. Permission to inspect a document covers reading it. Sending that document is a separate action that needs its own authorization.
>
> [Plain-language guide](../PLAIN_ENGLISH_GUIDE.md#agency) · [Terms and examples](../VARIABLES_EXPLAINED.md#agency) · [Evidence and related work](../DEMOS_AND_RELATED_WORK.md#agents)
> [All docs](../DOCUMENTATION_INDEX.md) · [Glossary](../GLOSSARY.md) · [Worked formulas](../FORMULAS.md)
<!-- reader-navigation:end -->

## Trigger

- Ungoverned action detected in governance lint
- Will receipt audit shows unauthorized tool call
- Operator reports unexpected Aura behavior

## Diagnosis

1. Run governance lint:
   ```bash
   make governance-lint
   ```

2. Check Will receipt log:
   ```bash
   python -c "from core.governance.will_receipt_log import get_receipt_log; log = get_receipt_log(); print(log.recent(20))"
   ```

3. Check for ungoverned action paths:
   ```bash
   rg "ungoverned" ~/.aura/logs/
   ```

## Response

1. **Contain the affected action lane through its supported live control.**
   The assignments below apply to a future launch; exporting them in another
   shell does not disable the current process:
   ```bash
   export AURA_AUTONOMY_LEVEL=0
   # or
   export AURA_MODE=safe
   ```

2. **Review the action trail**: Identify what actions were taken and their effects

3. **Revert if needed**: Use backup/restore to undo state changes

4. **Root cause**: Identify the code path that bypassed Will governance

## Prevention

- All consequential actions must route through `core/will.py`
- `core/governance/will_gate.py` enforces coverage
- `tools/lint_governance.py` detects ungoverned paths in CI
- Feature flags gate dangerous capabilities

## Escalation

If governance bypass is confirmed:
1. Disable the affected capability
2. Add explicit Will gating to the code path
3. Add regression test
4. Report as security incident
