# Runbook: Resource Exhaustion

<!-- reader-navigation:start -->
> Model work and background jobs share the Mac’s memory and processing time. Aura can defer work when the machine is busy.
>
> Two models can each fit on a Mac by themselves and still overwhelm it if loaded together. Before starting a second large job, check how much memory the active conversation and its model already need.
>
> [Plain-language guide](../PLAIN_ENGLISH_GUIDE.md#resources) · [Terms and examples](../VARIABLES_EXPLAINED.md#resources) · [Evidence and related work](../DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](../DOCUMENTATION_INDEX.md) · [Glossary](../GLOSSARY.md) · [Worked formulas](../FORMULAS.md)
<!-- reader-navigation:end -->

## Trigger

- RAM usage exceeds 80% of available
- GPU memory pressure detected
- Model inference timeouts increasing
- Process killed by OOM killer

## Diagnosis

1. Check current resource state:
   ```bash
   python -c "from core.ops.metabolic_monitor import get_metabolic_state; print(get_metabolic_state())"
   ```

2. Check resource governor status:
   ```bash
   grep "resource_governor" ~/.aura/logs/*.log | tail -20
   ```

3. Check token budget utilization:
   ```bash
   grep "token_budget" ~/.aura/logs/*.log | tail -10
   ```

## Response

1. **Reduce admitted work through its owning scheduler.** Identify the
   resident model and campaign owners; do not kill unrelated processes.
   The following setting applies to a future launch, not another running shell:
   ```bash
   export AURA_FOREGROUND_ONLY=1
   ```

2. **Choose a smaller validated profile for a later authorised restart.**
   Preserve state and logs; do not create a competing model load.

3. **If a process was OOM-killed**, retain the host’s pressure/crash records.
   On macOS use the memory incident records and unified logs; `dmesg` is not
   the reference macOS diagnostic. Verify the old owner has exited before
   an authorised replacement starts.

## Prevention

- Resource governor monitors RAM/GPU pressure
- Token budgets cap generation length
- Metal semaphore serializes GPU access
- Tier demotion under pressure is automatic
- `HARDWARE_PROFILES.md` documents expected resource usage per tier
