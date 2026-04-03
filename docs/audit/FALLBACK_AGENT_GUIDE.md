# Aura: Fallback Agent Guide

> For Gemini 3 Flash or any fallback agent resuming work on Aura.

## System Architecture Quick Reference

| Component | File | Role |
|---|---|---|
| Orchestrator | `core/orchestrator.py` | Main event loop, heartbeat coordinator |
| Cognitive Engine | `core/brain/cognitive_engine.py` | LLM routing (3-tier fallback) |
| Capability Engine | `core/capability_engine.py` | Skill registry & execution |
| Memory Manager | `core/managers/memory_manager.py` | Episodic + vector memory facade |
| Metabolic Coordinator | `core/coordinators/metabolic_coordinator.py` | Database hygiene, resource monitoring |
| Service Registration | `core/service_registration.py` | DI container wiring (~40 services) |
| Subsystem Audit | `core/subsystem_audit.py` | Heartbeat health monitoring |
| Mycelial Network | `core/mycelium.py` | Physarum-inspired subsystem connectivity |
| Liquid Substrate | `core/liquid_substrate.py` | Async state update layer |

## Common Failure Modes & Fixes

### 1. Heartbeat NEVER SEEN / STALE
**Symptom:** `/api/health` shows subsystem as `NEVER SEEN` or `STALE`.
**Root Cause:** Heartbeat only fires on success, not on attempt. Or an async method is called synchronously.
**Fix Pattern:**
```python
# BAD: heartbeat only on success
try:
    do_work()
    audit.heartbeat("subsystem")  # ← never reached if do_work() fails

# GOOD: heartbeat on attempt
audit.heartbeat("subsystem")  # ← always fires
try:
    do_work()
except Exception as e:
    audit.report_failure("subsystem", str(e))
```

### 2. Unawaited Coroutines (async called as sync)
**Symptom:** Silent failures, `RuntimeWarning: coroutine was never awaited`.
**Root Cause:** `async def` method called without `await`.
**Fix Pattern:**
```python
# BAD
liquid_state.update(data)  # creates coroutine object, discards it

# GOOD
asyncio.get_running_loop().create_task(liquid_state.update(data))
```

### 3. LLM Timeouts
**Symptom:** Cognitive engine hangs, Cycle=0.
**Root Cause:** Ollama or LLM backend unresponsive.
**Fix:** `cognitive_engine.py` has 90s hard timeout + 3-tier fallback (Cloud → Local → Reflex).

### 4. Screenshot / Display Errors
**Symptom:** `could not create image from display`.
**Root Cause:** macOS screen recording permission not granted.
**Fix:** System Preferences → Privacy & Security → Screen Recording → enable for Terminal/Python.

### 5. Import Errors at Boot
**Diagnosis:**
```bash
cd autonomy_engine
find core -name "*.py" | xargs -I{} python3 -m py_compile {} 2>&1 | grep -i error
```

## Key Commands

```bash
# Launch Aura
cd /Users/bryan/.gemini/antigravity/scratch/autonomy_engine
python3 desktop_viewer.py

# Run tests
python3 -m pytest tests/ -v

# Compile check a file
python3 -m py_compile core/path/to/file.py

# Check health
curl http://localhost:8080/api/health | python3 -m json.tool
```

## Critical Files to Never Break
- `core/orchestrator.py` — main loop
- `core/service_registration.py` — boot wiring
- `core/config.py` — all paths and model configs
- `core/container.py` — DI container

## Mycelial Network Integration Pattern
When adding a new subsystem, follow this pattern (from `voice_engine.py`):
```python
def _get_mycelium(self):
    if self._mycelium is None:
        self._mycelium = ServiceContainer.get("mycelial_network", default=None)
    return self._mycelium

def _pulse_hypha(self, source, target, success=True):
    m = self._get_mycelium()
    if m:
        hypha = m.get_hypha(source, target)
        if hypha:
            hypha.pulse(success=success)
```
Then establish connections in `service_registration.py`:
```python
mycelial.establish_connection("new_subsystem", "cognition", priority=0.9)
```
