# Condition ablation in shared procedural learning

The previous lesion loop intersected all successful traces, then searched
those same traces for a missing member of that intersection. Its removal
branch was unreachable. Tests that varied time of day only tested the
intersection; they did not exercise condition removal.

`ProceduralGeneralizer.derive` now accepts an explicit proposed condition set.
It checks proposed widening against measured examples outside that set and
searches the wider scope for counterevidence. It tests each removal against
the current condition set, so separate successes do not imply a safe joint
removal. Unjudged examples cannot support widening. Support is counted only
inside the final scope. Existing callers retain conservative intersection
behavior when no explicit proposal is supplied.

This is observational condition ablation, not proof of causal necessity or
general transfer. The native System2 integration test additionally records a
failure after successful rule reuse and verifies that a subsequent new request
does not reuse the invalidated rule. It runs the production deliberation code
offline; it is not desktop-session evidence.

This implements part of the human questionnaire's principle formation and
falsification requirements. It does not close G03 or the rest of the G ledger.
