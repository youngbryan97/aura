# Aura Flagship Readiness Gate

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

A conservative release gate. Conservative meaning it only flags things that
are wrong in every case, so a hit is a bug rather than a discussion:

```bash
python -m core.runtime.flagship_readiness --strict .
```

Each pattern below is here because it caused a real incident in this
runtime, not because a style guide dislikes it:

- raw production `asyncio.create_task` / `asyncio.ensure_future`
- direct `Path.write_text` calls that may bypass durable persistence policy
- `sys.exit(...)` inside async functions
- import-time `asyncio.Lock/Event/Semaphore/Queue`
- missing morphogenesis boot wiring
- missing global asyncio task supervision patch
- missing direct morphogenesis lifecycle counters

This is a fast red/green gate for the kinds of issues that most often stop a large local AI runtime from feeling flagship-grade. It does not prove the runtime is correct.

## Run

```bash
python -m core.runtime.flagship_readiness --strict .
```

Exit non-zero (in `--strict`) on any finding, so it drops straight into a CI
gate. The historical one-shot "closure patch" bootstrap that originally
installed this gate has been merged into the tree and removed.
