# G05: resident shape result

The 27B run on frozen commit `5a63f16d2` completed all three development cases.
The model descriptor was
`52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c`.

| Task | Public answer | Generated tokens | Decode milliseconds |
| --- | --- | --- | --- |
| Sort 9, 2, 5 | `[2, 5, 9]` | 41 | 6720 |
| English-to-French color mapping | `{"red":"rouge","blue":"bleu"}` | 84 | 11407 |
| Multiply [2, 3, 5] by 7 | `[14, 21, 35]` | 70 | 9885 |

Each answer matched the independent expected JSON, stopped at EOS, closed its
private boundary, and retained an enforcing processor with zero refusals.
Native reasoning was enabled; private text was not saved as an answer.
The 3072-token allocation was not exhausted. The supervised run passed and
released its model process.

The original result, including public strings, hashes and source bindings, is
copied unchanged to
`artifacts/rlc/public_shape_canary_dev_20260914/resident_result.json`.
The supervisor retains its process receipt at
`~/.aura/experiments/public-shape-20260914-v3/`.

This is a resident decoder-component check, not a desktop interaction. It
demonstrates that the repaired shared grammar can preserve simple task values
under the installed runtime tokenizer. It does not compare RLC against an
ordinary model, establish broad gain, or qualify a serving configuration.
G05 remains open for the end-to-end internal-result-to-public-answer claim.
