# Aura SLO Contract

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

What the runtime promises an operator, and what happens when it stops
keeping the promise.

Every SLO here is **measured by code in `slo/`** and **gated in CI**
(`.github/workflows/slo-gate.yml`). Regress a measurement past tolerance or
break a hard limit and the release gate fails. These are not aspirations
written down next to a number — nothing in this file is enforced by
somebody remembering it.

The list is deliberately narrow. It covers load-bearing infrastructure and
excludes anything needing GPU or model loads, because a number that swings
with model availability isn't a promise, it's weather. Those live in the
benchmark harness under `aura_bench/` instead.

## Format

Each SLO has:

| Field           | Meaning                                                     |
|-----------------|-------------------------------------------------------------|
| `value`         | The recorded baseline measurement                           |
| `tolerance_pct` | Soft regression tolerance as a percent of baseline          |
| `hard_limit`    | Absolute ceiling: violation fails CI even if within tolerance |
| `unit`          | The measurement's unit (`ms`, `us`, `score`)                |

A measurement is **failing** when:

```
measurement > baseline.value * (1 + tolerance_pct/100)   # soft
OR
measurement > slo.hard_limit                             # hard
```

For correctness scores (Brier loss on a synthetic perfect predictor)
the inequality flips: lower is better, so failing means
`measurement > slo.hard_limit`.

## Current SLOs

| ID                                    | Surface                  | Why it matters                                                                 |
|---------------------------------------|--------------------------|--------------------------------------------------------------------------------|
| `audit_chain_append_p95_ms`           | `core/runtime/audit_chain.py` | Receipt emission must not become a hot path bottleneck                |
| `audit_chain_verify_per_entry_us`     | `core/runtime/audit_chain.py` | Tamper verification is run on every diagnostics bundle                |
| `prediction_ledger_register_p95_ms`   | `core/runtime/prediction_ledger.py` | Predictions register in the hot loop; budget is tight             |
| `prediction_ledger_brier_correctness` | `core/runtime/prediction_ledger.py` | Brier on a perfect predictor must be exactly 0; this is correctness, not perf |
| `mutation_eval_passed_p95_ms`         | `core/self_modification/mutation_safety.py` | Self-mod gating cannot stall the autonomy loop              |
| `doctor_bundle_p95_ms`                | `core/runtime/diagnostics_bundle.py`   | Operators need a bundle in seconds, not minutes                |

## Out of scope

* Anything requiring the resident Cortex fused model: those run under
  `aura_bench/` against a recorded ablation matrix, not this gate.
* End-to-end latency from user message to streamed first token: that
  belongs in a separate UX-focused SLO once telemetry stabilises.
* Disk space and OS-level resource pressure: these are detected by
  `aura doctor` (pre-bundle checks) and the runbooks in
  `docs/runbooks/`.

## Updating the baseline

Bump baselines deliberately:

```
python -m slo.measure --emit > slo/baseline.candidate.json
# review, then:
mv slo/baseline.candidate.json slo/baseline.json
```

The PR diff for a baseline change should explain *why* the regression
is acceptable.  CI will accept the new numbers because the comparison
runs against the file in the PR, but reviewers should resist baseline
inflation that is not justified by intentional code changes.
