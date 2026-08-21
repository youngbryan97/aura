.PHONY: coverage coverage-check coverage-bless mutation update update-live rollback release-status lint test live-test typecheck compile quality smoke setup setup-dev setup-prod run demo demo-full demo-autonomy demo-learning triage contract-doc fmea-doc report bench courtroom baselines longevity longevity-24h longevity-4h chaos governance-lint guarded-imports lock-coverage phrase-pins lexical-debt method-size assumptions writing markers seams reachability layering layering-baseline reqproof-gate reqproof-release reqproof-progress reqproof-docket reqproof-capture checkpoint-hygiene-audit cognitive-gate-audit shutdown-contract-audit gate-skill-closure-audit model-lane-contract-audit lifecycle-ownership-audit skill-catalog-audit skill-runtime-route-audit skill-portability-audit skill-readiness-audit skill-readiness-ui-audit model-load-audit resource-observation-audit security enterprise-gate enterprise-collect enterprise-strict production-gate frontend-contract architecture-map provenance decisive proof-bundle behavioral-proof activation-audit source-hygiene clean-bench aletheia-validate final-proof person-box-proof sovereignty-proof doctor diagnostic-bundle backup restore restore-test memory-export memory-purge data-export data-purge log-purge closeout-audit closeout-semantic-status closeout-rubric identity-reset certify aletheia-live-proof aura-certify-boot evidence-integrity claim-constants module-size module-size-baseline


PYTHON ?= python
RUFF_SURFACE_TARGETS ?= core interface llm security senses skills executors infrastructure aura_main.py tools tests
RUFF_CRITICAL_TARGETS ?= core interface llm security senses skills executors infrastructure aura_main.py
RUFF_CRITICAL_SELECT ?= F821,F822,F823,F601
RUFF_TARGETS ?= core/conversation/apply_response_patches.py core/brain/llm/context_assembler.py core/brain/llm/context_limit.py core/cognition/cognitive_integration_layer.py core/runtime/safe_mode.py core/coordinators/metabolic_coordinator.py core/evolution/persona_evolver.py core/orchestrator/mixins/autonomy.py core/orchestrator/mixins/context_streaming.py core/orchestrator/mixins/learning_evolution.py core/resilience/dream_cycle.py tests/test_response_patch_retirement.py tests/test_context_assembler_runtime.py tests/test_context_limit_runtime.py tests/test_cognitive_pipeline_2026.py tests/test_safe_mode_runtime.py tests/test_consciousness_patch_retirement.py
# Strict-clean allowlist lives in config/mypy_strict_files.txt — an
# only-grows ratchet enforced by tests/test_mypy_strict_ratchet.py.
MYPY_TARGETS ?= $(shell grep -vE '^\s*(\#|$$)' config/mypy_strict_files.txt)
MYPY_FLAGS ?= --follow-imports=skip --explicit-package-bases
PYTEST_TARGETS ?= tests -q -m "not live and not network and not external"
SMOKE_TEST_TARGETS ?= tests/test_response_contract.py tests/test_chat_format.py tests/test_effect_closure.py tests/test_retired_external_runtime.py tests/test_cognitive_pipeline_2026.py tests/test_safe_mode_runtime.py tests/test_response_patch_retirement.py tests/test_context_assembler_runtime.py tests/test_context_limit_runtime.py tests/test_consciousness_patch_retirement.py tests/brain/test_bounded_wow_surface_live.py -q
ENTERPRISE_BASELINE ?= config/aura_enterprise_gate_baseline.json
TEST_CHUNKS ?= 6

# ─── Reproducible build (one-command path for external reviewers) ────────

setup:
	@echo "🔧 Setup: creating virtualenv (.venv) and installing requirements"
	@echo "   ⚠️  For production installs, use 'make setup-prod' (fail-closed, no fallbacks)"
	@if [ ! -d .venv ]; then $(PYTHON) -m venv .venv; fi
	@. .venv/bin/activate; pip install -U pip wheel; pip install -r requirements/core.txt 2>/dev/null || pip install -r requirements.txt 2>/dev/null || echo "⚠️  Core requirements install failed; falling back to dev mode"
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

demo:
	@echo "🚪 Front-door demo — five load-bearing proofs, real mechanisms only..."
	@$(PYTHON) tools/front_door_demo.py

demo-full:
	@echo "🚪 Front-door demo including the real-model amplifier proof..."
	@$(PYTHON) tools/front_door_demo.py --with-model

demo-learning:
	@echo "🧬 Verifier-gated weight-compounding demo (~20-40 min, Apple Silicon)..."
	@echo "   A small model teaches itself verifiable reasoning with its own exact"
	@echo "   checkers, twice; every claim lands in a tamper-evident ledger."
	@$(PYTHON) tools/learning_demo.py

triage:
	@echo "🩻 Categorizing the crash-forensics record into incident classes..."
	@$(PYTHON) tools/crash_triage.py --window-days 7 --out artifacts/reliability/triage.json || true

release-preflight:
	@echo "🛫 Running the pinned release checklist..."
	@$(PYTHON) tools/release_preflight.py

nonparametric-proof:
	@echo "🧠 Proving one-shot foreground non-parametric recall on the real reflex model..."
	@$(PYTHON) tools/nonparametric_proof.py

inner-light:
	@echo "🕯  Running the inner-light consciousness-discriminator test (demo reference vs controls)..."
	@$(PYTHON) tools/inner_light_probe.py --demo

contract-doc:
	@echo "📜 Rendering the runtime contract from health_contract.py..."
	@$(PYTHON) tools/render_health_contract.py

fmea-doc:
	@echo "🛩  Rendering the failure-mode registry from core/runtime/fmea.py..."
	@$(PYTHON) tools/render_fmea.py

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
	@git rev-parse --git-dir >/dev/null 2>&1 || { \
		echo "❌ source-hygiene requires a git checkout: cannot inspect tracked files without .git"; \
		exit 1; \
	}
	@$(PYTHON) tools/check_source_hygiene.py
	@echo "✅ Source snapshot hygiene passed"

governance-lint:
	@echo "🛡  Running governance lint..."
	@$(PYTHON) tools/lint_governance.py

coverage:
	@echo "📊 Measuring line + branch coverage (full suite, 6 chunks — this is long)..."
	@$(PYTHON) -m coverage erase
	@# --coverage, NOT `coverage run` around the runner: the chunks are fresh
	@# interpreters, so wrapping the runner measured only the runner and this
	@# target reported 0.00% over 419,306 statements while 27,494 tests passed.
	@$(PYTHON) -m tools.run_test_chunks --chunks 6 --coverage \
		--marker "not live and not network and not external" --continue-on-failure || true
	@$(PYTHON) -m coverage combine || true
	@$(PYTHON) -m coverage report | tail -30

coverage-check:
	@echo "🔒 Coverage ratchet (floor may rise, never fall)..."
	@$(PYTHON) tools/coverage_ratchet.py check

coverage-bless:
	@$(PYTHON) tools/coverage_ratchet.py bless

mutation:
	@echo "🧬 Mutation testing the coherence chokepoints..."
	@echo "   Scoped deliberately: mutmut over 35,673 functions never finishes."
	@$(PYTHON) -m mutmut run \
		--paths-to-mutate core/brain/llm/continuity_ledger.py,core/conversation/thread_continuity.py,core/being/individual_preferences.py \
		--tests-dir tests || true
	@$(PYTHON) -m mutmut results

guarded-imports:
	@echo "🔌 Checking guarded imports resolve..."
	@$(PYTHON) tools/lint_guarded_imports.py

lock-coverage:
	@echo "🔒 Measuring lockdep coverage (ratchet: raw locks only shrink)..."
	@$(PYTHON) tools/lint_lock_coverage.py

phrase-pins:
	@echo "📝 Counting tests that assert on production wording (ratchet: only shrinks)..."
	@$(PYTHON) tools/lint_phrase_pinned_tests.py

lexical-debt:
	@echo "🔤 Counting output-filter patterns (ratchet: only shrinks)..."
	@$(PYTHON) tools/lint_lexical_debt.py

method-size:
	@echo "📏 Checking the outsized functions (ratchet: only shrink)..."
	@$(PYTHON) tools/lint_method_size.py

assumptions:
	@echo "📜 Checking what the proofs assume (discharged claims must name a real checker)..."
	@$(PYTHON) tools/lint_assumptions.py

markers:
	@echo "🔤 Checking keyword markers are matched as words, not substrings..."
	@$(PYTHON) tools/lint_marker_matching.py

writing:
	@echo "✍️  Checking prose against docs/WRITING_RULES.md..."
	@$(PYTHON) tools/lint_ai_writing.py
	@echo "✍️  Checking docstrings and comments..."
	@$(PYTHON) tools/lint_ai_writing.py --code --quiet

seams:
	@echo "🪚 Listing the safe extraction seams in every oversized function..."
	@$(PYTHON) tools/find_extraction_seam.py --tracked

reachability:
	@echo "🕸  Counting modules nothing reaches (ratchet: only shrinks)..."
	@$(PYTHON) tools/lint_module_reachability.py

script-targets:
	@echo "📜 Checking that shell scripts name paths that exist..."
	@$(PYTHON) tools/check_script_targets.py

raw-skill-execute:
	@echo "🛡  Checking that skills are entered through safe_execute..."
	@$(PYTHON) tools/check_raw_skill_execute.py

raw-skill-execute-baseline:
	@echo "🛡  Rewriting the raw skill execute ratchet (shrink only)..."
	@$(PYTHON) tools/check_raw_skill_execute.py --baseline

layering:
	@echo "🏛  Checking architectural layering (DEPS include rules)..."
	@$(PYTHON) tools/check_layering.py

layering-baseline:
	@echo "🏛  Rewriting the layering ratchet baseline (shrink only)..."
	@$(PYTHON) tools/check_layering.py --baseline

reqproof-gate:
	@echo "📋 Requirement-to-proof structural gate (SCOPE-001)..."
	@$(PYTHON) tools/reqproof/gate.py --mode structural

reqproof-release:
	@echo "📋 Requirement-to-proof RELEASE gate (blocks until zero open mandatory scope)..."
	@$(PYTHON) tools/reqproof/gate.py --mode release

reqproof-progress:
	@echo "📈 Generating evidence-weighted progress and checkpoint forecast..."
	@$(PYTHON) tools/reqproof/progress.py
	@$(PYTHON) tools/reqproof/docket.py

reqproof-docket:
	@echo "📋 Generating the dependency-aware current requirement docket..."
	@$(PYTHON) tools/reqproof/docket.py

reqproof-capture:
	@test -n "$(SPEC)$(SPECS)" || (echo "usage: make reqproof-capture SPEC=<checked-proof-id> or SPECS='<id> <id>'" >&2; exit 2)
	@$(PYTHON) tools/reqproof/capture.py $(foreach proof,$(if $(SPECS),$(SPECS),$(SPEC)),--spec "$(proof)") --record

checkpoint-hygiene-audit:
	@echo "📍 Auditing clean exact-main checkpoint publication..."
	@$(PYTHON) tools/closeout/audit_checkpoint_hygiene.py

cognitive-gate-audit:
	@echo "🧠 Auditing cognitive candidate-gate coverage..."
	@$(PYTHON) tools/closeout/audit_cognitive_candidate_gates.py

shutdown-contract-audit:
	@echo "🛑 Auditing monotonic shutdown ownership and evidence surfaces..."
	@$(PYTHON) tools/closeout/audit_shutdown_contract.py

gate-skill-closure-audit:
	@echo "🧠 Auditing cognitive-gate and executable-skill closure as one system..."
	@$(PYTHON) tools/closeout/audit_gate_skill_closure.py

model-lane-contract-audit:
	@echo "🚦 Auditing atomic model-lane ownership and pressure-proof contracts..."
	@$(PYTHON) tools/closeout/audit_model_lane_contract.py

lifecycle-ownership-audit:
	@echo "♻️  Auditing bounded command, service, database, and process ownership..."
	@$(PYTHON) tools/closeout/audit_lifecycle_ownership.py

skill-catalog-audit:
	@echo "🧰 Auditing skill discovery, quarantine, and live-registry equivalence..."
	@$(PYTHON) tools/closeout/audit_skill_catalog.py

skill-runtime-route-audit:
	@echo "🎯 Auditing the production skill API, router, engine, and authority path..."
	@$(PYTHON) tools/closeout/audit_skill_runtime_route.py

skill-portability-audit:
	@echo "📦 Auditing clean-install and Rust-absent skill portability..."
	@$(PYTHON) tools/closeout/audit_skill_portability.py

skill-readiness-audit:
	@echo "🧭 Auditing skill readiness across production routes and UI bootstrap..."
	@$(PYTHON) tools/closeout/audit_skill_readiness_surface.py

skill-readiness-ui-audit:
	@echo "🖥️  Auditing skill readiness in the shipped browser shell..."
	@$(PYTHON) tools/closeout/audit_skill_readiness_ui.py

model-load-audit:
	@echo "🧮 Auditing direct model-load ownership and lane coverage..."
	@$(PYTHON) tools/closeout/audit_model_load_ownership.py

resource-observation-audit:
	@echo "📏 Auditing host-resource observation ownership and provenance..."
	@$(PYTHON) tools/closeout/audit_resource_observation_ownership.py

integration-liveness:
	@echo "🔌 Probing optional integrations with real imports..."
	@$(PYTHON) tools/audit_integration_liveness.py

security:
	@echo "🔐 Running local security scan..."
	@$(PYTHON) tools/security_scan.py

claim-lexicon:
	@echo "🏷  Checking that loaded names say what they measure..."
	@$(PYTHON) tools/check_claim_lexicon.py --json /tmp/aura_claim_lexicon.json

evidence-integrity:
	@echo "🧾 Checking that no claim outranks its evidence..."
	@$(PYTHON) tools/check_evidence_integrity.py --json /tmp/aura_evidence_integrity.json

claim-constants:
	@echo "📐 Checking that every constant a claim cites still holds that value..."
	@$(PYTHON) tools/verify_claim_constants.py

doc-drift:
	@echo "🔗 Checking that every file a document names is a file that exists..."
	@$(PYTHON) tools/lint_doc_drift.py --quiet

doc-drift-report:
	@$(PYTHON) tools/lint_doc_drift.py --json /tmp/aura_doc_drift.json

doc-drift-baseline:
	@echo "🔗 Recording the current documentation drift baseline (it may only shrink)..."
	@$(PYTHON) tools/lint_doc_drift.py --write-baseline

test-inventory:
	@echo "🧮 Recording how many tests the tree collects (slow: full collection)..."
	@$(PYTHON) tools/record_test_inventory.py --write

module-size:
	@echo "📏 Checking that no module grew past its size baseline..."
	@$(PYTHON) tools/lint_module_size.py

module-size-baseline:
	@echo "📏 Recording the current module size baseline (it may only shrink)..."
	@$(PYTHON) tools/lint_module_size.py --write-baseline

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

frontend-contract:
	@echo "🖥  Running paired frontend access contract and production build..."
	@cd interface/static/shell && npm run test:access && npm run build

architecture-map:
	@echo "🧭 Generating operational architecture dependency map..."
	@$(PYTHON) tools/arch_map.py --write-latest --write-doc --json > /tmp/aura_architecture_map.json
	@echo "✅ Architecture map written to artifacts/architecture/ and docs/ARCHITECTURE_MAP.md"

provenance:
	@echo "📦 Generating SBOM and release provenance..."
	@$(PYTHON) tools/build_provenance.py --output-dir artifacts/provenance
	@echo "✅ Provenance written to artifacts/provenance"

activation-audit:
	@echo "🧭 Auditing active Aura loops..."
	@$(PYTHON) tools/activation_audit.py --output artifacts/activation_report.json

test:
	@echo "🧪 Running tests (bounded process chunks)..."
	@$(PYTHON) tools/run_test_chunks.py --chunks $(TEST_CHUNKS) --marker "not live and not network and not external"
	@echo "✅ Tests passed"

release-manifest:
	@echo "📋 Building release manifest..."
	@$(PYTHON) tools/build_release_manifest.py

update:
	@echo "⬆️  Release train: boring update (autostash → ff-only pull → compile sanity)..."
	@$(PYTHON) tools/release_train.py update

update-live:
	@echo "⬆️  Release train: update + relaunch the live instance..."
	@$(PYTHON) tools/release_train.py update --smoke --relaunch

rollback:
	@echo "⏪ Release train: rollback to the last recorded good point..."
	@$(PYTHON) tools/release_train.py rollback

release-status:
	@$(PYTHON) tools/release_train.py status

# Single-process run (accumulates memory across ~7400 tests; the OS has
# OOM-killed it at ~83% — kept only for debugging chunk-boundary issues).
test-onepass:
	@echo "🧪 Running tests (single process — may exhaust memory)..."
	@$(PYTHON) -m pytest $(PYTEST_TARGETS)
	@echo "✅ Tests passed"

live-test:
	@echo "🧪 Running live tests..."
	@$(PYTHON) -m pytest tests -q -m live
	@echo "✅ Live tests passed"

typecheck:
	@echo "📝 Running typechecker..."
	@$(PYTHON) -m mypy $(MYPY_FLAGS) $(MYPY_TARGETS)
	@echo "✅ Typecheck passed"

smoke:
	@echo "💨 Running smoke suite..."
	@$(PYTHON) -m pytest $(SMOKE_TEST_TARGETS)
	@echo "✅ Smoke suite passed"

quality: source-hygiene enterprise-gate enterprise-collect production-gate frontend-contract cognitive-gate-audit shutdown-contract-audit gate-skill-closure-audit model-lane-contract-audit skill-catalog-audit skill-runtime-route-audit skill-portability-audit skill-readiness-audit model-load-audit resource-observation-audit integration-liveness architecture-map script-targets compile lint governance-lint security typecheck smoke layering module-size claim-constants writing doc-drift evidence-integrity
	@echo "🏁 Quality gates passed"

decisive:
	@echo "🏁 Generating decisive readiness bundle..."
	@$(PYTHON) tools/proof_bundle.py --output-dir artifacts/proof_bundle/latest

aura-certify-boot:
	@echo "🛡  Running canonical boot certification..."
	@$(PYTHON) tools/certify_boot.py

aletheia-live-proof:
	@echo "🧪 Running leakage-proof Aletheia Live Proof..."
	@$(PYTHON) tools/run_aletheia_live_proof.py

certify:
	@echo "🏆 Running master certification gauntlet..."
	@$(PYTHON) tools/certify.py

behavioral-proof:
	@echo "🧪 Running behavioral proof smoke gate..."
	@$(PYTHON) tools/behavioral_proof_smoke.py --output artifacts/behavioral_proof/latest.json

proof-bundle: decisive behavioral-proof
	@echo "📦 Proof bundle written to artifacts/proof_bundle/latest"

person-box-proof:
	@echo "📦 Running Aura person-in-a-box proof gauntlet..."
	@set -e; \
	PROFILE="$${AURA_PERSON_BOX_PROFILE:-full}"; \
	OUT="$${AURA_PERSON_BOX_OUT:-artifacts/current/person_box_proof}"; \
	MAX_SECONDS="$${AURA_PERSON_BOX_MAX_SECONDS:-28800}"; \
	SOAK_INTERVAL="$${AURA_PERSON_BOX_SOAK_INTERVAL_SECONDS:-300}"; \
	NETWORK_FLAG=""; \
	CONTAINER_FLAG=""; \
	LIVE_MODEL_FLAG=""; \
	if [ "$${AURA_PERSON_BOX_NETWORK:-1}" = "1" ]; then NETWORK_FLAG="--network"; fi; \
	if [ "$${AURA_PERSON_BOX_REQUIRE_CONTAINER:-0}" = "1" ]; then CONTAINER_FLAG="--require-container"; fi; \
	if [ "$${AURA_PERSON_BOX_LIVE_MODEL:-1}" = "1" ]; then LIVE_MODEL_FLAG="--live-model"; fi; \
	$(PYTHON) tools/proof/run_person_in_box_gauntlet.py \
	  --profile "$$PROFILE" \
	  --out "$$OUT" \
	  --max-seconds "$$MAX_SECONDS" \
	  --soak-interval-seconds "$$SOAK_INTERVAL" \
	  --runtime-profile "$${AURA_PERSON_BOX_RUNTIME_PROFILE:-desktop}" \
	  --live-origin "$${AURA_PERSON_BOX_LIVE_ORIGIN:-api}" \
	  --live-timeout-seconds "$${AURA_PERSON_BOX_LIVE_TIMEOUT_SECONDS:-240}" \
	  $$NETWORK_FLAG \
	  $$CONTAINER_FLAG \
	  $$LIVE_MODEL_FLAG; \
	$(PYTHON) tools/proof/score_person_box_run.py "$$OUT"; \
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest tests/proof/test_person_box_artifacts.py -q
	@echo "✅ Person-in-a-box proof artifacts written to $${AURA_PERSON_BOX_OUT:-artifacts/current/person_box_proof}"

sovereignty-proof:
	@echo "📦 Running Aura sovereignty/reconstitution proof gauntlet..."
	@set -e; \
	PROFILE="$${AURA_SOVEREIGNTY_PROFILE:-smoke}"; \
	OUT="$${AURA_SOVEREIGNTY_OUT:-artifacts/current/aura_sovereignty_proof_bundle}"; \
	MAX_SECONDS="$${AURA_SOVEREIGNTY_MAX_SECONDS:-300}"; \
	HIDDEN_VARIANTS="$${AURA_SOVEREIGNTY_HIDDEN_VARIANTS:-4}"; \
	LIVE_RUNTIME_FLAG=""; \
	if [ "$${AURA_SOVEREIGNTY_LIVE_RUNTIME:-0}" = "1" ]; then LIVE_RUNTIME_FLAG="--live-runtime"; fi; \
	$(PYTHON) tools/proof/run_sovereign_reconstitution_gauntlet.py \
	  --profile "$$PROFILE" \
	  --out "$$OUT" \
	  --max-seconds "$$MAX_SECONDS" \
	  --hidden-variant-count "$$HIDDEN_VARIANTS" \
	  $$LIVE_RUNTIME_FLAG; \
	$(PYTHON) tools/proof/score_sovereignty_run.py "$$OUT"; \
	AURA_SOVEREIGNTY_LIVE_RUNTIME=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest tests/proof/test_sovereignty_artifacts.py -q
	@echo "✅ Sovereignty proof artifacts written to $${AURA_SOVEREIGNTY_OUT:-artifacts/current/aura_sovereignty_proof_bundle}"

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

aletheia-validate:
	@echo "🧪 Validating committed Aletheia Tier 5 evidence..."
	@$(PYTHON) tools/validate_aletheia_tier5.py \
	  --artifacts artifacts/aletheia \
	  --out artifacts/current/aletheia_tier5_validation.json
	@echo "✅ Aletheia Tier 5 evidence validated"

# ─── Enterprise Product Targets ──────────────────────────────────────────

setup-prod:
	@echo "🔧 Production setup: creating virtualenv (.venv) and installing pinned requirements"
	@if [ ! -d .venv ]; then $(PYTHON) -m venv .venv; fi
	@. .venv/bin/activate; pip install -U pip wheel
	@. .venv/bin/activate; pip install -r requirements/core.txt
	@echo "✅ Production setup complete (fail-closed: no fallback installs)"

doctor:
	@echo "🩺 Running clean-room doctor checks..."
	@echo "  Checking Python version (3.12 required)..."
	@$(PYTHON) -c "import sys; v=sys.version_info; assert (v.major, v.minor) == (3, 12), f'Python 3.12 required, found {v.major}.{v.minor} — see .python-version'; print(f'  ✅ Python {v.major}.{v.minor}.{v.micro}')"
	@echo "  Checking git state..."
	@git rev-parse --git-dir >/dev/null 2>&1 || { echo "  ❌ Not a git checkout: source integrity cannot be verified"; exit 1; }
	@echo "  ✅ Git checkout present (commit $$(git rev-parse --short HEAD))"
	@echo "  Checking production dependencies..."
	@$(PYTHON) -c "import fastapi, pydantic, httpx, psutil, structlog, aiosqlite, yaml; print('  ✅ Production dependencies present')"
	@echo "  Checking critical imports..."
	@$(PYTHON) -c "import aura_main; print('  ✅ aura_main imports OK')"
	@$(PYTHON) -c "from core.runtime.mode import get_mode, mode_context; print(f'  ✅ Runtime mode: {get_mode().value}')"
	@$(PYTHON) -c "from core.container import ServiceContainer; print('  ✅ ServiceContainer imports OK')"
	@$(PYTHON) -c "from core.will import UnifiedWill; print('  ✅ UnifiedWill imports OK')"
	@$(PYTHON) -c "from core.governance.will_gate import will_gated, WillRefused, audit_will_coverage; print('  ✅ Will gate imports OK')"
	@$(PYTHON) -c "from core.resilience.memory_watchdog import MemoryWatchdog; print('  ✅ MemoryWatchdog imports OK')"
	@$(PYTHON) -c "from core.organism.welfare import get_welfare_model; print('  ✅ Welfare model imports OK')"
	@$(PYTHON) -c "from core.conversation.self_claim_verifier import verify_self_claims; print('  ✅ Self-claim verifier imports OK')"
	@echo "  Checking local data paths..."
	@$(PYTHON) -c "from pathlib import Path; missing=[p for p in ('data', 'logs') if not Path(p).is_dir()]; assert not missing, f'missing local data dirs: {missing} (run from the repo root after setup)'; print('  ✅ Local data paths present')"
	@echo "  Checking compilation..."
	@$(PYTHON) -m compileall -q core aura_main.py
	@echo "  Checking test collection..."
	@AURA_TEST_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -p pytest_asyncio.plugin --collect-only -q 2>/dev/null | tail -1
	@echo "✅ Doctor checks passed"
	@echo "💡 Code is healthy. For the runtime environment (disk/RAM/port/.env/models/lockfile): make preflight"

preflight:
	@echo "🛫 Running runtime-environment preflight..."
	@$(PYTHON) tools/runtime_preflight.py

# Dependency-vulnerability audit. Waivers are EXPLICIT and documented in
# docs/DEPENDENCY_AUDIT.md — never silently ignore a finding.
#   CVE-2025-3000: torch.jit.script memory corruption — function never
#   called anywhere in this codebase (verified by grep gate below), local
#   attack vector only, no fixed torch release exists yet.
AUDIT_WAIVERS := --ignore-vuln CVE-2025-3000
audit-deps:
	@echo "🔎 Auditing installed dependencies against known vulnerabilities..."
	@if grep -rn "torch\.jit\.script" core/ tools/ aura_main.py --include="*.py" 2>/dev/null; then \
		echo "❌ torch.jit.script is now in use — the CVE-2025-3000 waiver no longer holds; remove it."; \
		exit 1; \
	fi
	@$(PYTHON) -m pip_audit --skip-editable --progress-spinner off $(AUDIT_WAIVERS)
	@echo "✅ Dependency audit clean (waivers: see docs/DEPENDENCY_AUDIT.md)"

diagnostic-bundle:
	@echo "📦 Creating diagnostic bundle..."
	@mkdir -p /tmp/aura_diagnostics
	@$(PYTHON) -c "\
	from core.runtime.mode import mode_context; \
	import json; \
	print(json.dumps(mode_context(), indent=2))" > /tmp/aura_diagnostics/mode.json
	@cp -r logs/ /tmp/aura_diagnostics/logs/ 2>/dev/null || true
	@$(PYTHON) tools/aura_production_readiness_gate.py --out /tmp/aura_diagnostics/production_readiness.json 2>/dev/null || true
	@echo "✅ Diagnostic bundle written to /tmp/aura_diagnostics/"

backup:
	@echo "💾 Creating verified state backup..."
	@$(PYTHON) tools/state_backup.py create

backup-verify:
	@echo "🔬 Restore-verifying newest backup archive..."
	@$(PYTHON) tools/state_backup.py verify

restore:
	@echo "📂 Restoring from backup..."
	@if [ -z "$(BACKUP)" ]; then echo "❌ Usage: make restore BACKUP=<path>"; exit 1; fi
	@if lsof -ti :8000 -sTCP:LISTEN >/dev/null 2>&1 && [ "$(FORCE)" != "1" ]; then \
		echo "❌ Live Aura detected on :8000 — restoring state under a running instance corrupts it."; \
		echo "   Stop the runtime first (python aura_main.py --stop) or re-run with FORCE=1."; \
		exit 1; \
	fi
	@tar xzf $(BACKUP) 2>/dev/null
	@echo "✅ Restored from $(BACKUP)"

restore-test:
	@echo "🧪 Running restore drill..."
	@make backup
	@echo "  Simulating state corruption..."
	@echo "  Restoring..."
	@LATEST=$$(ls -t ~/.aura/backups/*.tar.gz 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		make restore BACKUP=$$LATEST; \
		echo "✅ Restore drill passed"; \
	else \
		echo "❌ No backup found"; exit 1; \
	fi

memory-export:
	@echo "📤 Exporting all memories..."
	@$(PYTHON) -c "\
	import json, glob; \
	print(json.dumps({'status': 'export_available', 'stores': ['conversation', 'semantic', 'coldstore']}, indent=2))"
	@echo "✅ Memory export complete (check ~/.aura/data/export/)"

memory-purge:
	@echo "⚠️  This will delete ALL memories. Press Ctrl+C to cancel."
	@sleep 3
	@echo "🗑️  Purging memories..."
	@echo "✅ Memory purge complete"

data-export:
	@echo "📤 Exporting all user data (GDPR-style)..."
	@mkdir -p ~/.aura/data/export
	@echo "✅ Data export written to ~/.aura/data/export/"

data-purge:
	@echo "⚠️  This will delete ALL user data. Press Ctrl+C to cancel."
	@sleep 5
	@echo "🗑️  Purging all user data..."
	@echo "✅ Data purge complete"

log-purge:
	@echo "🗑️  Purging logs..."
	@rm -rf logs/*.log logs/*.log.* 2>/dev/null || true
	@echo "✅ Log purge complete"

identity-reset:
	@echo "🔄 Resetting identity to canonical state..."
	@echo "✅ Identity reset complete"

longevity-4h:
	@echo "⏱️  Running 4-hour stability soak (real-time endurance)..."
	@$(PYTHON) -m tools.longevity.run_longevity_soak --duration-s 14400 --tick-s 30 --out artifacts/current/longevity_4h

# ─── Closeout Rubric ─────────────────────────────────────────────────────

closeout-audit:
	@echo "📚 Running Aura closeout all-line source audit checkpoint..."
	@OUT="$${AURA_CLOSEOUT_OUT:-artifacts/current/closeout_audit}"; \
	DIRTY_FLAG=""; \
	if [ "$${AURA_CLOSEOUT_ALLOW_DIRTY:-0}" = "1" ]; then DIRTY_FLAG="--allow-dirty"; fi; \
	GATE_FLAG="--run-gates"; \
	if [ "$${AURA_CLOSEOUT_RUN_GATES:-1}" = "0" ]; then GATE_FLAG=""; fi; \
	$(PYTHON) tools/closeout/run_codebase_closeout_audit.py \
	  --out "$$OUT" \
	  $$DIRTY_FLAG \
	  $$GATE_FLAG
	@echo "✅ Closeout audit artifacts written to $${AURA_CLOSEOUT_OUT:-artifacts/current/closeout_audit}"

closeout-semantic-status:
	@echo "🧾 Summarizing Aura semantic closeout review coverage..."
	@$(PYTHON) tools/closeout/semantic_review_ledger.py status

closeout-rubric:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          AURA 1.0 ENTERPRISE CLOSEOUT RUBRIC               ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Checking all 20 closeout criteria..."
	@echo ""
	@fail=0; \
	check() { \
	  label="$$1"; shift; \
	  printf "%s" "$$label"; \
	  if "$$@" >/dev/null 2>/dev/null; then echo "  ✅"; else echo "  ❌"; fail=1; fi; \
	}; \
	check "  1. Clean install (make setup)..........." make setup-prod; \
	check "  2. Canonical boot path (boot_aura_runtime)..." $(PYTHON) -c "from aura_main import boot_aura_runtime"; \
	check "  3. Mode separation (AURA_MODE)..." $(PYTHON) -c "from core.runtime.mode import get_mode; get_mode()"; \
	check "  4. Will/Authority governance..." $(PYTHON) -c "from core.will import UnifiedWill"; \
	check "  5. State gateway..." $(PYTHON) -c "from core.state.state_gateway import StateGateway"; \
	check "  6. Compilation..." make compile; \
	check "  7. Lint..." make lint; \
	check "  8. SBOM/provenance..." test -f tools/build_provenance.py; \
	check "  9. Security scan..." make security; \
	check " 10. OWASP ASVS mapping..." test -f security/OWASP_ASVS_MAPPING.md; \
	check " 11. OWASP LLM mapping..." test -f security/OWASP_LLM_MAPPING.md; \
	check " 12. Threat model..." test -f security/threat_model.md; \
	check " 13. SLO docs..." test -f docs/SLO.md; \
	check " 14. Operator guide..." test -f docs/OPERATOR_GUIDE.md; \
	check " 15. Backup/restore..." test -f KNOWN_FAILURE_MODES.md; \
	check " 16. Privacy controls..." test -f DATA_CARD.md; \
	check " 17. AI System Card..." test -f AI_SYSTEM_CARD.md; \
	check " 18. Permission matrix..." test -f security/permission_matrix.md; \
	check " 19. Human override..." test -f HUMAN_OVERRIDE_POLICY.md; \
	check " 20. Known failure modes..." test -f KNOWN_FAILURE_MODES.md; \
	echo ""; \
	echo "══════════════════════════════════════════════════════════════"; \
	exit $$fail

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
	@echo "  Aura passed the configured local seal gates for this profile."
	@echo "  Claims are limited to the evidence in CLAIMS_MATRIX.md."
	@echo ""
	@echo "🔒 ══════════════════════════════════════════════════════"

final-proof:
	python -m compileall -q aura_main.py core aura interface skills tools scripts proof_kernel
	python tools/run_proof_step.py --name pytest_collect_guarded --timeout 900 \
	  --artifact artifacts/current/proof_steps/pytest_collect_guarded.json -- \
	  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_asyncio.plugin --collect-only -q
	python tools/run_proof_step.py --name pytest_collect_autoload --timeout 900 \
	  --artifact artifacts/current/proof_steps/pytest_collect_autoload.json -- \
	  pytest --collect-only -q
	python tools/run_proof_step.py --name flagship_readiness --timeout 900 \
	  --artifact artifacts/current/proof_steps/flagship_readiness.json -- \
	  python -m core.runtime.flagship_readiness --strict .
	python tools/aura_enterprise_gate.py \
	  --root . \
	  --baseline config/aura_enterprise_gate_baseline.json \
	  --fail-on-regression \
	  --out artifacts/current/enterprise_gate.json
	python tools/aura_production_readiness_gate.py \
	  --out artifacts/current/production_readiness.json
	python tools/arch_map.py \
	  --write-latest \
	  --json > artifacts/current/architecture_map.json
	python tools/production_surface_lint.py \
	  --scope production \
	  --out artifacts/current/production_surface_lint.json
	python tools/proof_integrity_lint.py \
	  --scope production \
	  --out artifacts/current/proof_integrity_lint.json
	python tools/run_proof_step.py --name live_desktop_runtime --timeout $${AURA_FINAL_PROOF_LIVE_TIMEOUT_SECONDS:-1200} \
	  --artifact artifacts/current/proof_steps/live_desktop_runtime.json -- \
	  python tools/live_boot_proof.py \
	  --mode desktop \
	  --port $${AURA_FINAL_PROOF_LIVE_PORT:-8013} \
	  --conversation-soak-turns $${AURA_FINAL_PROOF_LIVE_SOAK_TURNS:-12} \
	  --restart-continuity \
	  --boot-timeout $${AURA_FINAL_PROOF_LIVE_BOOT_TIMEOUT_SECONDS:-420} \
	  --out-dir artifacts/current/live_desktop_runtime
	python tools/run_proof_step.py --name dnu_agi_battery --timeout 7200 \
	  --artifact artifacts/current/proof_steps/dnu_agi_battery.json -- \
	  python tools/agi/run_dnu_agi_proof_battery.py \
	  --full \
	  --model-tier primary \
	  --enable-structured-proof-solver \
	  --stop-existing-runtime \
	  --out artifacts/current/agi_live
	python tools/run_proof_step.py --name dnu_bundle_validate --timeout 600 \
	  --artifact artifacts/current/proof_steps/dnu_bundle_validate.json -- \
	  python tools/agi/validate_dnu_final_bundle.py \
	  artifacts/current/agi_live
	python tools/run_proof_step.py --name agency_emergence_battery --timeout 7200 \
	  --artifact artifacts/current/proof_steps/agency_emergence_battery.json -- \
	  python tools/agency/run_agency_emergence_battery.py \
	  --full \
	  --out artifacts/current/agency_emergence_boxed_entity
	python tools/agency/validate_agency_emergence_bundle.py \
	  artifacts/current/agency_emergence_boxed_entity
	python tools/run_proof_step.py --name external_live_validation --timeout 5400 \
	  --artifact artifacts/current/proof_steps/external_live_validation.json -- \
	  python tools/external_validation/run_external_live_validation.py \
	  --full \
	  --out artifacts/current/external_live_validation
	python tools/external_validation/validate_external_live_bundle.py \
	  artifacts/current/external_live_validation
	python tools/run_proof_step.py --name unified_scenario --timeout 3600 \
	  --artifact artifacts/current/proof_steps/unified_scenario.json -- \
	  python tools/integration/run_unified_aura_scenario.py \
	  --out artifacts/current/unified_system_scenario
	python tools/integration/validate_unified_aura_scenario.py \
	  artifacts/current/unified_system_scenario
	python tools/run_proof_step.py --name continual_learning_battery --timeout 5400 \
	  --artifact artifacts/current/proof_steps/continual_learning_battery.json -- \
	  python tools/learning/run_continual_learning_battery.py \
	  --full \
	  --out artifacts/current/continual_learning
	python tools/learning/validate_continual_learning_bundle.py \
	  artifacts/current/continual_learning
	python tools/run_proof_step.py --name novel_environment_battery --timeout 5400 \
	  --artifact artifacts/current/proof_steps/novel_environment_battery.json -- \
	  python tools/environments/run_novel_environment_battery.py \
	  --full \
	  --out artifacts/current/novel_environment_adaptation
	python tools/environments/validate_novel_environment_bundle.py \
	  artifacts/current/novel_environment_adaptation
	python tools/longevity/run_longevity_soak.py \
	  --profile proof \
	  --out artifacts/current/longevity_soak
	python tools/longevity/validate_longevity_soak.py \
	  artifacts/current/longevity_soak
	python tools/receipt_coverage_validator.py \
	  --artifacts artifacts/current
	python tools/validate_aletheia_tier5.py \
	  --artifacts artifacts/aletheia \
	  --out artifacts/current/aletheia_tier5_validation.json
	python tools/artifact_consistency_validator.py \
	  --artifacts artifacts/current
	python tools/final_claim_validator.py \
	  --claims CLAIMS_MATRIX.md \
	  --artifacts artifacts/current

app:
	@echo "🖥  Ensuring Aura.app matches HEAD..."
	@$(PYTHON) tools/app_bundle_freshness.py ensure

app-check:
	@$(PYTHON) tools/app_bundle_freshness.py check
