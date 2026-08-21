# Runbook: Disaster Recovery

## Recovery Scenarios

### 1. Dirty Shutdown Recovery

**Symptoms**: Aura was killed (SIGKILL, power loss, crash)

```bash
# Boot will automatically recover via WAL replay
make run
# Verify state integrity
make doctor
```

### 2. Corrupted Memory Database

**Symptoms**: SQLite errors on boot, memory retrieval fails

```bash
# Check integrity
python -c "
import sqlite3
conn = sqlite3.connect('data/aura_memory.db')
result = conn.execute('PRAGMA integrity_check').fetchone()
print(f'Integrity: {result[0]}')
"

# If corrupted: restore from backup
make restore BACKUP=$(ls -t ~/.aura/backups/*.tar.gz | head -1)

# If no backup: rebuild from WAL
python -c "
import sqlite3
conn = sqlite3.connect('data/aura_memory.db')
conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
print('WAL checkpoint complete')
"
```

### 3. Partial Model Download

**Symptoms**: Model fails to load, checksum mismatch

```bash
# Verify model files
python -c "
from core.brain.llm.model_registry import verify_model_integrity
result = verify_model_integrity()
print(f'Model integrity: {result}')
"

# Re-download whatever is missing (bounded, resumable)
python -c "
from core.brain.llm.model_lifecycle import get_model_lifecycle_manager
print(get_model_lifecycle_manager().ensure_present())
"
```

`ensure_present` checks free disk first, downloads only the models with a
known source, and resumes a partial fetch rather than restarting it.

### 4. Failed Self-Repair

**Symptoms**: Self-repair left system in inconsistent state

```bash
# Refuse self-modification for this run. `production`, `live`, `safe`, `test`,
# `simulated`, and `research` all set allows_self_modification=False; only
# `dev` permits it (core/runtime/mode.py).
export AURA_MODE=production

# Check repair registry
cat data/selfmod/pending_patch_registry.jsonl

# If registry is corrupt: remove it. AURA_PENDING_PATCH_REGISTRY overrides
# this location; without it the path is data_dir/selfmod/ from core/config.py.
rm data/selfmod/pending_patch_registry.jsonl

# Restart
make run
```

### 5. Disk Full

**Symptoms**: Write failures, log rotation stops

```bash
# Check disk usage
df -h

# Purge logs
make log-purge

# Purge old backups (keep last 3)
ls -t ~/.aura/backups/*.tar.gz | tail -n +4 | xargs rm -f

# Purge bytecode caches
find . -name __pycache__ -exec rm -rf {} +

# Restart
make run
```

### 6. Version Downgrade

**Symptoms**: Need to roll back to previous version

```bash
# Check current version
git log -1 --oneline

# Restore previous version
git checkout <previous-commit>

# Restore state from backup taken at that version
make restore BACKUP=<matching-backup>

# Verify
make doctor
make run
```

### 7. Full Disaster Recovery (Fresh Machine)

```bash
# 1. Clone repository
git clone https://github.com/youngbryan97/aura.git
cd aura

# 2. Setup
make setup-prod

# 3. Restore state from backup
make restore BACKUP=<backup-path>

# 4. Verify
make doctor

# 5. Run
make run
```

## Recovery Verification Checklist

After any recovery:

- [ ] `make doctor` passes
- [ ] `make compile` succeeds
- [ ] Boot completes without errors
- [ ] Memory retrieval returns expected results
- [ ] Model loads successfully
- [ ] Will receipt chain is intact
- [ ] No orphaned background tasks
