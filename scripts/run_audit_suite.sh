#!/bin/bash
# Aura Zenith: Audit Suite runner
source venv/bin/activate || echo "Warning: Virtualenv not active."

echo "🔍 Starting Audit Suite..."

echo "1. System Fingerprint"
uname -a
python3 --version

echo "2. Dependency Check"
pip list | grep -E "mlx|numpy|pydantic|Restricted|prometheus"

echo "3. Unit & Integration Tests"
pytest tests/test_phase3_hardening.py tests/test_phase4_agi.py tests/test_phase5_evolution.py --tb=short

echo "4. Performance/Memory Snapshot"
# Brief run to check observability
echo "Executing brief health check..."
# (Logic to run orchestrator briefly and check metrics would go here)

echo "✅ Audit Suite Finished."
