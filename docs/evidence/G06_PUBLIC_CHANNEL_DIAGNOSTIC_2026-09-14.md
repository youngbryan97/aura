# Public-channel measurement repair

Scope: G05/G06 measurement infrastructure. G03 through G12 remain open.
This diagnostic is not a new transfer, fusion, generality, or frontier verdict.

## Retained evidence

The immutable diagnostic files are under
`artifacts/closeout/latent_cortex/public_channel_diagnostic_20260914/`:

- `legacy-termination-audit.json`: validates the CP1003 result seal and all
  302 journal events, including all 300 task/arm decodes.
- `pilot.json`: one exposed development task on the actual fused 27B model.
- `pilot.py`: the exact scratch driver, retained as historical execution code.
  Its hard-coded temporary output path is not a production launcher.

No historical CP1003 result, journal, activation, or verdict was rewritten.

## Two measurement defects

The CP1003 ordinary arm had zero parsed answers. Its journal contains 18
unstopped answers of exactly 384 generated tokens, and 42 stopped answers
whose contract parser reports `payload_not_json_object`. The old decoder
stripped special tokens and applied completion predicates to the combined
private/public text. Native-channel closure cannot be reconstructed from
those retained strings; the audit does not claim otherwise.

The real-model diagnostic used development seed 2026091306, the first coding
task, and no runtime steering. The old decoder exhausted 384 tokens in
46.292 seconds without a parsed answer. The channel-aware decoder reached
a public answer after 3,027 tokens in 380.201 seconds, with the native
boundary closed. Budget and channel handling changed together: this is not
a controlled estimate of their separate effects.

The legacy grader marked that public answer incorrect. Direct inspection
shows both return values and every intermediate pressure value agree with
the expected result. The sole difference is `Theta(n^2)` versus the legacy
grader's literal `O(n^2)`. The public task asks for tight worst-case complexity
and does not require that literal spelling. The pilot's original `correct:
false` field is retained; semantic grading of this exposed example is a
new diagnostic interpretation, not a replacement measurement.
The pilot predates the checkpoint: its parent commit and measured file hashes
are retained, but it is not a clean-commit campaign receipt.

## Mechanical repairs

`public_channel_decode` uses the existing native-channel parser. Completion
predicates see public text only. Public wire prefixes are placed after the
native closing delimiter, with boundary and wire tokens counted separately.
The returned receipt retains public text, private-text hash/count, sampled
token hash/count, latency, and an explicit termination cause, not private prose.

Both neural semantic canary runners now default to version-2 measurement.
They retain explicit version-1 replay modes. New reports bind the decode
policy, budget, relevant source files, and every attempt's resource evidence.
The composition protocol continues its declared non-thinking template mode;
it is not silently redefined as native-thinking evaluation.

Independent verification recomputes channel coverage and resource totals.
Token-capped, exhausted, or unclosed attempts remain in the evidence and
prevent an uncensored-transfer admission, including failed retries before
a successful last attempt. This does not cancel live work or exclude bad
answers: completed wrong answers are measured failures.

The versioned semantic grader keeps legacy exact-wire grading unchanged.
For new coding measurements, a bounded AST builds SymPy polynomial objects
directly to compare supported positive polynomial/logarithmic growth classes.
No model-authored string reaches `eval`, `sympify`, or `parse_expr`. Values,
types, and all other answer constraints still pass through the original
strict task grader. Unknown syntax is unsupported, not assumed equivalent.

## Limits and remaining experiments

- One exposed task is not a fresh replication or an estimate of broad gain.
- The diagnostic ran the bare fused model, not the desktop runtime or CAA.
- State-backed arms still have their declared correction/prefix protocol;
  that is architectural assistance, not unaided-model reasoning.
- New full source-bound campaigns and current-manifest qualification are
  still required. The old composition tests expose manifest/source drift;
  these alarms have not been suppressed or re-sealed.
- G06 still requires fair alternative compute and complete cost accounting;
  channel correctness alone does not satisfy that obligation.

## Focused verification

77 tests passed across channel decoding, receipt accounting, semantic grading,
historical termination audit, runner contracts, and the independent verifier.
Six additional composition-verifier tests passed. Synthetic fixture receipts
exercise rejection paths; they are never labeled model measurements.
