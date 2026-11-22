# Runbook: Resource Exhaustion

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
