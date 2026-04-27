# Flagship Closure Patch 07 – Remediation Planner

This closure patch introduces a **remediation planner** to Aura's flagship
tooling.  As the system matures, multiple readiness gates, doctor scans,
task‑ownership audits and persistence audits produce JSON reports
describing outstanding issues.  Patch 07 provides a single tool to collect
those reports and assemble a consolidated remediation plan.

## What’s included?

1. **`scripts/aura_remediation_planner.py`** – a command‑line tool that
   accepts a directory of JSON reports and aggregates problem lists across
   categories (`issues`, `errors`, `tasks`, `persistence`, etc.), deduplicating
   entries and emitting a single JSON document.  The planner is resilient to
   unknown report fields and gracefully ignores malformed data.
2. **New Makefile target** (added via the apply script) called
   `flagship-remediation`, which runs the planner against the current
   directory and writes `remediation_plan.json`.  This provides a
   simple one‑command remediation aggregation.
3. **`tests/test_closure_patch_07.py`** – unit tests covering basic
   aggregation, deduplication, nested structures and non‑list inputs.
4. **Documentation** (this file) explaining the purpose and usage of the
   remediation planner.

## Usage

After applying this patch to your Aura repository, generate your flagship
reports as usual (e.g. run the readiness gate, doctor, task ownership audit
and persistence audit).  Then run the following command from the repo root:

```bash
make flagship-remediation
```

This will invoke `python scripts/aura_remediation_planner.py . --out remediation_plan.json`.
The resulting `remediation_plan.json` contains all issues grouped by
category.  Use this file to prioritise fixes and to track progress toward
flagship‑level polish.

You can also run the planner manually:

```bash
python scripts/aura_remediation_planner.py path/to/report/dir --out plan.json
```

where `path/to/report/dir` contains the JSON reports you wish to aggregate.

## Why this matters

Aura’s increasingly comprehensive tooling surfaces a wide variety of issues
across different domains.  Without a unified view, it is easy to miss
important items or to be overwhelmed by disparate reports.  The
remediation planner provides a coherent summary, enabling developers to
systematically address outstanding problems and move Aura toward
Chrome‑grade stability and polish.