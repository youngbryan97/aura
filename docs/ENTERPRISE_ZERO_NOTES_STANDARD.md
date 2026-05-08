# Aura Enterprise Zero-Notes Standard

This is the acceptance standard for moving Aura from impressive research code to an enterprise-grade, indefinitely runnable autonomous cognitive system.

## Non-Negotiable Release Gates

1. **Full test collection completes in under 90 seconds** with `AURA_TEST_MODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Collection must not load models, open sockets, start background loops, spawn processes, or initialize UI frameworks.
2. **No production import side effects.** Importing any production module may define classes/functions/constants only. Boot belongs in explicit `start()` / `run()` / lifecycle methods.
3. **No unbounded loops without cancellation, sleep/yield, deadline, and health heartbeat.** Every loop must have a named supervisor, stop signal, backoff, metric, and dead-letter path.
4. **Every consequential action goes through one authority spine.** File writes, shell/process execution, network calls, browser actions, social posting, memory/state mutation, model modification, hot reload, promotion, and self-repair must require Will/AuthorityGateway receipts.
5. **No swallowed broad exceptions in production.** Exceptions may be broad only at process/task boundaries, and they must record degradation, preserve traceback, update health, and route recovery.
6. **No placeholder production behavior.** Stubs, mocks, dummy success objects, fake receipts, fake metrics, and test-conditioned branches are forbidden outside tests/fixtures.
7. **No unsafe dynamic execution outside reviewed sandboxes.** `eval`, `exec`, `compile`, pickle/dill loading, subprocess, and shell entry points require allowlists, containment, receipts, timeouts, and audit logs.
8. **Full-code quality gates, not curated slices.** Ruff, typecheck, security, compile, governance lint, smoke, integration, property, fuzz, and longevity gates must target the whole production surface or document a temporary quarantine.
9. **Proof bundle is generated from live code, not copied artifacts.** CAA, STDP, governance coverage, longevity, mutation safety, boot health, security, repair lineage, and behavioral proof must be regenerated in CI/release and include environment metadata.
10. **Runtime claims degrade honestly.** Missing hardware, missing models, missing data, unavailable sensors, or disabled network/tools must lower claimed capability rather than returning fake success.

## Production Claim Surface

Production code currently means `aura_main.py`, `core/`, `executors/`, `infrastructure/`, `interface/`, `llm/`, `security/`, `senses/`, and `skills/`. Experimental, archived, generated, benchmark, proof-artifact, training-data, scratch, and fixture surfaces may be scanned for inventory, but release claims must not depend on them unless they graduate into the production claim surface.

## Ratchet Policy

The repository carries historical enterprise-hardening debt. `config/aura_enterprise_gate_baseline.json` captures that debt as a maximum, not a permission slip. New high-risk counts may not rise. Every cleanup pass should lower the baseline alongside the code change that removed the debt.

For release certification, run the gate without baseline forgiveness:

```bash
AURA_TEST_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/aura_enterprise_gate.py --root . --strict
```

For regression control while debt is being retired:

```bash
AURA_TEST_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/aura_enterprise_gate.py --root . --baseline config/aura_enterprise_gate_baseline.json --fail-on-regression
```

## Definition Of Done For NetHack-General Agency

Aura does not need NetHack-specific scripting. It needs a general environment kernel that can:

- compile raw observations into typed state with confidence and uncertainty;
- maintain a persistent world model and topological/causal map;
- generate affordances and legal action candidates;
- simulate short-horizon consequences and long-horizon strategy;
- choose actions under resource, risk, and goal constraints;
- execute through the same action gateway used for every real-world tool;
- attribute outcomes, learn procedural skills, and replay failures;
- recover from partial observability, delayed rewards, hunger/resource clocks, irreversible mistakes, and mode/context switches.

Repeated deep NetHack runs would be evidence of general planning, perception-action grounding, procedural learning, and long-horizon self-correction. It would not by itself prove AGI or consciousness.
