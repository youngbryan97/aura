# Runbook: Resource Exhaustion

<!-- reader-navigation:start -->
> **In plain English:** Resource management keeps model inference, background work and the operating system within the machine’s real capacity.
>
> **Why it matters:** A useful assistant must avoid exhausting the host or making the conversation unusable while doing background work.
>
> **In everyday use:** A second large model load can disrupt a working session even if each model fits alone. Scheduling and a measured memory budget matter.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#resources) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#resources) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
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
   grep "resource_governor" logs/aura.log | tail -20
   ```

3. Check token budget utilization:
   ```bash
   grep "token_budget" logs/aura.log | tail -10
   ```

## Response

1. **Reduce load**: Kill background tasks
   ```bash
   export AURA_FOREGROUND_ONLY=1
   ```

2. **Restart with lower resource profile**: Reduce model tier

3. **If OOM killed**: Check `dmesg` for OOM details; restart Aura

## Prevention

- Resource governor monitors RAM/GPU pressure
- Token budgets cap generation length
- Metal semaphore serializes GPU access
- Tier demotion under pressure is automatic
- `HARDWARE_PROFILES.md` documents expected resource usage per tier
