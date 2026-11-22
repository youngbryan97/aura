# Third-Party Replication Protocol for Aura

<!-- reader-navigation:start -->
> The Intrinsic Subject Core experiments ask whether Aura’s measured parts act as one continuing process.
>
> This puts a concrete question ahead of any argument about experience: which parts affect each other, and through what paths? Restore the same saved moment twice, change a value in one copy, and compare what happens to recall or a report.
>
> [Plain-language guide](../../docs/PLAIN_ENGLISH_GUIDE.md#isc) · [Terms and examples](../../docs/VARIABLES_EXPLAINED.md#isc) · [Evidence and related work](../../docs/DEMOS_AND_RELATED_WORK.md#isc)
> [All docs](../../docs/DOCUMENTATION_INDEX.md) · [Glossary](../../docs/GLOSSARY.md) · [Worked formulas](../../docs/FORMULAS.md)
<!-- reader-navigation:end -->

This document describes how external auditors can run the software introspection and behavioral-adaptation suites. They check their
declared mechanisms, not subjective experience or a complete ISC pass. For
the larger experiment use [ISC reproduction](../../docs/ISC_REPRODUCTION.md).

## Test Environment Setup

1. Choose a bounded offline test environment. A 16 GB test setup does not
   qualify the resident 27B desktop; follow [hardware profiles](../../HARDWARE_PROFILES.md).
2. Initialize requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Introspection and Lesion Suites

Auditors can run the automated tests using the following command:

```bash
pytest tests/test_introspection_and_lesions.py
```

This verifies:
- **Blind Introspection**: Checks the suite's specified telemetry/report relationship under its controls; accuracy does not generalise to every state or model by passing this test.
- **Lesion Adaptations**: Checks if disabling or forcing welfare metrics (e.g. locking energy low) forces policy engines to restrict risk level of actions.
- **Self-Model Explanations**: Checks if the generated narrative explanations correlate with active preference parameters.
