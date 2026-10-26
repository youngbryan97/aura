# Quality gates

Status: Reference · Enforcement paths reconciled 25 October 2026

A rule, an installed local hook and a passing CI run are different evidence.
The executable definitions are [the Makefile](../Makefile),
[pre-commit configuration](../.pre-commit-config.yaml),
[project configuration](../pyproject.toml) and workflows under `.github/workflows/`.
A documented rule is not proof that every contributor installed the hook.

## Local hooks

The pre-commit configuration includes whitespace and file checks, YAML/JSON
checks, merge-conflict and large-added-file checks, Ruff, changed-file Mypy
and method-size checks. It does not run every test or import every core module
before every commit. The configured hook owns its file-size threshold;
there is no universal one-megabyte limit established by this page.

## Change validation

Follow [CONTRIBUTING](../CONTRIBUTING.md) and [TESTING](../TESTING.md).
Run `make smoke` after changes, then the focused checks appropriate to them.
Use the bounded chunk runner for the full offline suite; a single
`pytest tests/ -q` process can exhaust memory beside the resident model.
Set a separate `AURA_LOG_DIR`. Do not launch chunks while editing Python.

A live acceptance test needs explicit runtime ownership, source identity and
independent outcome evidence. Unit checks, HTTP success, a health response
or the absence of an exception do not supply live task success.

`make doc-drift`, `make claim-constants` and `make evidence-integrity`
protect named documentation facts, cited constants and retraction boundaries.
Their passing status does not certify every natural-language sentence.

## Response quality

Style indicators are diagnostics, not semantic correctness. First-person
pronouns, a lack of hedging or a confident tone cannot establish a correct
answer or stable identity. Evaluate whether the answer satisfies the request,
uses available evidence, acknowledges material uncertainty and makes no
unsupported action or capability claim. Personality examples in
[the personality specification](PERSONALITY.md) are writing guidance,
not an empirical consciousness finding.

## Release acceptance

`make quality` aggregates the targets declared in the Makefile. It is a broad
engineering sweep, not a harmless low-resource command or a scientific result.
Choose a host and budget before executing it. An install test must use an
independent environment and its declared dependency set; it must not mutate
the resident runtime’s environment.

A release claim needs the actual revision-bound CI or local artifacts and
any required live proof. [The release lifecycle](../docs/RELEASE_LIFECYCLE.md)
describes the producers, and [claim boundaries](../docs/CLAIM_BOUNDARIES.md)
state what their results can establish.
