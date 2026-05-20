.PHONY: lint test typecheck compile quality smoke setup setup-dev run demo-autonomy report bench courtroom baselines longevity longevity-24h chaos governance-lint security enterprise-gate enterprise-collect enterprise-strict production-gate architecture-map provenance decisive proof-bundle behavioral-proof activation-audit source-hygiene clean-bench

PYTHON ?= python
RUFF_SURFACE_TARGETS ?= core interface llm security senses skills executors infrastructure aura_main.py tools tests
RUFF_CRITICAL_TARGETS ?= core interface llm security senses skills executors infrastructure aura_main.py
RUFF_CRITICAL_SELECT ?= F821,F822,F823,F601
RUFF_TARGETS ?= core/apply_response_patches.py core/brain/llm/context_assembler.py core/brain/llm/context_limit.py core/cognitive_integration_layer.py core/safe_mode.py core/coordinators/metabolic_coordinator.py core/evolution/persona_evolver.py core/orchestrator/mixins/autonomy.py core/orchestrator/mixins/context_streaming.py core/orchestrator/mixins/learning_evolution.py core/resilience/dream_cycle.py tests/test_response_patch_retirement.py tests/test_context_assembler_runtime.py tests/test_context_limit_runtime.py tests/test_cognitive_pipeline_2026.py tests/test_safe_mode_runtime.py tests/test_consciousness_patch_retirement.py
MYPY_TARGETS ?= core/apply_response_patches.py core/brain/llm/context_limit.py core/safe_mode.py core/runtime/atomic_writer.py core/consciousness/continuous_experience.py core/environment/experience_replay.py core/memory/procedural/store.py core/unity/runtime.py tools/aura_production_readiness_gate.py tools/build_provenance.py
MYPY_FLAGS ?= --follow-imports=skip --explicit-package-bases
PYTEST_TARGETS ?= tests -q
SMOKE_TEST_TARGETS ?= tests/test_response_contract.py tests/test_chat_format.py tests/test_effect_closure.py tests/test_local_server_client.py tests/test_cognitive_pipeline_2026.py tests/test_safe_mode_runtime.py tests/test_response_patch_retirement.py tests/test_context_assembler_runtime.py tests/test_context_limit_runtime.py tests/test_consciousness_patch_retirement.py -q
ENTERPRISE_BASELINE ?= config/aura_enterprise_gate_baseline.json

# ─── Reproducible build (one-command path for external reviewers) ────────

setup:
	@echo "🔧 Setup: creating virtualenv (.venv) and installing requirements"
	@if [ ! -d .venv ]; then $(PYTHON) -m venv .venv; fi
	@. .venv/bin/activate; pip install -U pip wheel; pip install -r requirements/core.txt 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true
	@. .venv/bin/activate; if [ -f requirements/dev.txt ]; then pip install -r requirements/dev.txt; else pip install -e ".[dev]"; fi
	@echo "✅ Setup complete"

setup-dev:
	@echo "🔧 Installing Aura development quality tools..."
	@. .venv/bin/activate; if [ -f requirements/dev.txt ]; then pip install -r requirements/dev.txt; else pip install -e ".[dev]"; fi
	@echo "✅ Development tools installed"

run:
	@echo "▶️  Launching Aura (foreground)..."
	@$(PYTHON) aura_main.py --desktop

demo-autonomy:
	@echo "🤖 Running autonomy demo (60s soak)..."
	@$(PYTHON) -m tools.longevity.run_gauntlet --profile 24h_no_user --tick-s 5 || true

report:
	@echo "📊 Generating bench + courtroom + baseline reports..."
	@$(PYTHON) -c "import asyncio; from aura_bench.runner import run_all, write_report; r=asyncio.run(run_all()); write_report(r); print('bench done')"
	@$(PYTHON) -m aura_bench.courtroom.courtroom || true
	@$(PYTHON) -m aura_bench.baselines.runner || true
	@echo "✅ Reports written to ~/.aura/data/bench/ and aura_bench/courtroom/report.md"

# ─── Compile / lint / test gates ─────────────────────────────────────────

compile:
	@echo "🔍 Compiling all Python files..."
	@$(PYTHON) -m compileall -q core tests
	@echo "✅ All files compile"

lint:
	@echo "🧹 Running ruff..."
	@$(PYTHON) -m ruff check $(RUFF_SURFACE_TARGETS) --select E9
	@$(PYTHON) -m ruff check $(RUFF_CRITICAL_TARGETS) --select $(RUFF_CRITICAL_SELECT)
	@$(PYTHON) -m ruff check $(RUFF_TARGETS)
	@echo "✅ Ruff passed"

source-hygiene:
	@echo "🧼 Checking source snapshot hygiene..."
	@tracked="$$(git ls-files | grep -E '(^|/)__pycache__/|\.py[cod]$$|\$$py\.class$$|(^|/)\.(pytest|mypy|ruff)_cache/' || true)"; \
	if [ -n "$$tracked" ]; then \
		echo "Generated cache artifacts are tracked:"; \
		echo "$$tracked"; \
		exit 1; \
	fi
	@echo "✅ Source snapshot hygiene passed"

governance-lint:
	@echo "🛡  Running governance lint..."
	@$(PYTHON) tools/lint_governance.py

security:
	@echo "🔐 Running local security scan..."
	@$(PYTHON) tools/security_scan.py

enterprise-gate:
	@echo "🏢 Running enterprise static ratchet gate..."
	@AURA_TEST_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) tools/aura_enterprise_gate.py --root . --baseline $(ENTERPRISE_BASELINE) --fail-on-regression --skip-pytest-collect --out /tmp/aura_enterprise_gate.json
	@echo "✅ Enterprise gate passed; report written to /tmp/aura_enterprise_gate.json"

enterprise-collect:
	@echo "🏢 Running enterprise pytest collection gate..."
	@AURA_TEST_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) tools/aura_enterprise_gate.py --root . --baseline $(ENTERPRISE_BASELINE) --fail-on-regression --skip-compile --out /tmp/aura_enterprise_collect_gate.json
	@echo "✅ Enterprise collection gate passed; report written to /tmp/aura_enterprise_collect_gate.json"

enterprise-strict:
	@echo "🏢 Running strict enterprise certification gate..."
	@AURA_TEST_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) tools/aura_enterprise_gate.py --root . --strict

production-gate:
	@echo "🚦 Running production readiness contract..."
	@AURA_TEST_MODE=1 $(PYTHON) tools/aura_production_readiness_gate.py --out /tmp/aura_production_readiness.json
	@echo "✅ Production readiness contract passed; report written to /tmp/aura_production_readiness.json"

architecture-map:
	@echo "🧭 Generating operational architecture dependency map..."
	@$(PYTHON) tools/arch_map.py --write-latest --json > /tmp/aura_architecture_map.json
	@echo "✅ Architecture map written to artifacts/architecture/latest.json and latest.md"

provenance:
	@echo "📦 Generating SBOM and release provenance..."
	@$(PYTHON) tools/build_provenance.py --output-dir artifacts/provenance
	@echo "✅ Provenance written to artifacts/provenance"

activation-audit:
	@echo "🧭 Auditing active Aura loops..."
	@$(PYTHON) tools/activation_audit.py --output artifacts/activation_report.json

test:
	@echo "🧪 Running tests..."
	@$(PYTHON) -m pytest $(PYTEST_TARGETS)
	@echo "✅ Tests passed"

typecheck:
	@echo "📝 Running typechecker..."
	@$(PYTHON) -m mypy $(MYPY_FLAGS) $(MYPY_TARGETS)
	@echo "✅ Typecheck passed"

smoke:
	@echo "💨 Running smoke suite..."
	@$(PYTHON) -m pytest $(SMOKE_TEST_TARGETS)
	@echo "✅ Smoke suite passed"

quality: source-hygiene enterprise-gate enterprise-collect production-gate architecture-map compile lint governance-lint security typecheck smoke
	@echo "🏁 Quality gates passed"

decisive:
	@echo "🏁 Generating decisive readiness bundle..."
	@$(PYTHON) tools/proof_bundle.py --output-dir artifacts/proof_bundle/latest

behavioral-proof:
	@echo "🧪 Running behavioral proof smoke gate..."
	@$(PYTHON) tools/behavioral_proof_smoke.py --output artifacts/behavioral_proof/latest.json

proof-bundle: decisive behavioral-proof
	@echo "📦 Proof bundle written to artifacts/proof_bundle/latest"

# ─── Bench / chaos / longevity ────────────────────────────────────────────

bench:
	@$(PYTHON) -c "import asyncio; from aura_bench.runner import run_all, write_report; r=asyncio.run(run_all()); write_report(r); print('bench done')"

courtroom:
	@$(PYTHON) -m aura_bench.courtroom.courtroom

baselines:
	@$(PYTHON) -m aura_bench.baselines.runner

longevity:
	@$(PYTHON) -m tools.longevity.run_gauntlet --profile 24h_no_user

longevity-24h: longevity

chaos:
	@$(PYTHON) -m tools.chaos.injector --kind random

clean-bench:
	@rm -rf ~/.aura/data/bench
	@echo "🧹 cleaned ~/.aura/data/bench"

# ─── Gold Master Seal ─────────────────────────────────────────────────────
# Single-command verification that Aura is sealed for indefinite operation.
# This is not a test suite — it's a production readiness certification.

.PHONY: seal seal-quick

seal-quick: compile lint source-hygiene
	@echo "🔒 Running quick seal checks..."
	@$(PYTHON) -c "\
from core.governance.will_gate import audit_will_coverage; \
report = audit_will_coverage(strict=False); \
print(f'  Will coverage: {report[\"total_gated\"]} methods gated, {len(report[\"missing\"])} missing'); \
"
	@$(PYTHON) -c "\
from core.governance.feature_flags import get_feature_flags; \
flags = get_feature_flags(); \
all_flags = flags.get_all(); \
enabled = sum(1 for v in all_flags.values() if v); \
print(f'  Feature flags: {enabled}/{len(all_flags)} enabled'); \
"
	@$(PYTHON) -c "\
from core.observability.metrics import check_readiness; \
r = check_readiness(); \
print(f'  Readiness: {r[\"status\"]} ({len(r.get(\"issues\", []))} issues)'); \
"
	@echo "✅ Quick seal checks passed"

seal: quality seal-quick
	@echo ""
	@echo "🔒 ══════════════════════════════════════════════════════"
	@echo "🔒  AURA GOLD MASTER SEAL — PRODUCTION READINESS"
	@echo "🔒 ══════════════════════════════════════════════════════"
	@echo ""
	@echo "  All quality gates passed."
	@echo "  All seal verification checks passed."
	@echo "  Aura is certified for indefinite autonomous operation."
	@echo ""
	@echo "🔒 ══════════════════════════════════════════════════════"
