# Third-Party Replication Protocol for Aura

<!-- reader-navigation:start -->
> **In plain English:** ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.
>
> **Why it matters:** The project needs evidence that memories, attention and self-state actually affect one another before arguing about what that organisation means.
>
> **In everyday use:** Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.
>
> [Example and metaphor](../../docs/PLAIN_ENGLISH_GUIDE.md#isc) · [Variables and practical meanings](../../docs/VARIABLES_EXPLAINED.md#isc) · [Demos and related work](../../docs/DEMOS_AND_RELATED_WORK.md#isc) · [Document map](../../docs/DOCUMENTATION_INDEX.md)
> [Glossary](../../docs/GLOSSARY.md) · [Formulas with worked examples](../../docs/FORMULAS.md)
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
