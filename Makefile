.PHONY: lint test typecheck compile quality smoke setup setup-dev run demo-autonomy report bench courtroom baselines longevity longevity-24h chaos governance-lint security decisive proof-bundle behavioral-proof activation-audit clean-bench

PYTHON ?= python
RUFF_TARGETS ?= core/apply_response_patches.py core/brain/llm/context_assembler.py core/brain/llm/context_limit.py core/cognitive_integration_layer.py core/safe_mode.py core/coordinators/metabolic_coordinator.py core/evolution/persona_evolver.py core/orchestrator/mixins/autonomy.py core/orchestrator/mixins/context_streaming.py core/orchestrator/mixins/learning_evolution.py core/resilience/dream_cycle.py tests/test_response_patch_retirement.py tests/test_context_assembler_runtime.py tests/test_context_limit_runtime.py tests/test_cognitive_pipeline_2026.py tests/test_safe_mode_runtime.py tests/test_consciousness_patch_retirement.py
MYPY_TARGETS ?= core/apply_response_patches.py core/brain/llm/context_limit.py core/safe_mode.py
MYPY_FLAGS ?= --follow-imports=skip --explicit-package-bases
PYTEST_TARGETS ?= tests -q
SMOKE_TEST_TARGETS ?= tests/test_response_contract.py tests/test_chat_format.py tests/test_effect_closure.py tests/test_local_server_client.py tests/test_cognitive_pipeline_2026.py tests/test_safe_mode_runtime.py tests/test_response_patch_retirement.py tests/test_context_assembler_runtime.py tests/test_context_limit_runtime.py tests/test_consciousness_patch_retirement.py -q

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
	@$(PYTHON) -m ruff check $(RUFF_TARGETS)
	@echo "✅ Ruff passed"

governance-lint:
	@echo "🛡  Running governance lint..."
	@$(PYTHON) tools/lint_governance.py

security:
	@echo "🔐 Running local security scan..."
	@$(PYTHON) tools/security_scan.py

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

quality: compile lint governance-lint security typecheck smoke
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
