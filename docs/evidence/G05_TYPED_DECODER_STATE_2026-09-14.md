# G05: preserve the decoder's actual output boundary

Claude's shared JSON decoder is already connected to the worker's output-shape
request. Inspection of the installed MLX generation loop found that its first
processor call carries prompt tokens, before any output has been sampled. The
shape processor treated those tokens as the answer. Prompt prose could therefore
disable the shape at the first call. A historical private-channel close in a
prompt could also enable the shape for the wrong turn.

The processor now latches that initial prefix and parses only generated tokens.
It compares the active token history before reusing incremental parser state,
so a speculative rewind or same-length draft replacement reparses the actual
suffix. It does not constrain private reasoning or change a prompt.

Three parser/cache defects were also reproduced and repaired:

- A cache key containing only the top bracket conflated one open array with
  two. A single vocabulary token may close several containers, so the full
  stack determines its validity.
- An exponent awaiting an optional sign shared a mask with an exponent that
  already contained a sign. Their allowed next tokens differ.
- Unicode escapes did not require four hexadecimal digits. Invalid escapes
  could finish as supposedly valid JSON. The corrected transitions follow
  [RFC 8259 section 7](https://www.rfc-editor.org/rfc/rfc8259.html#section-7).

A combined test run then exposed an order-dependent tokenizer cache defect:
the cache retained only Python object ids, which can be reused after collection.
It now retains the two cached tokenizer owners. Per-generation mask caching is
bounded to 32 MiB with eviction and recomputation, without pruning legal tokens.

The new suite checks numeric strings against Python's JSON parser, whole-token
cache equivalence, split Unicode escapes, prompt/private boundaries, speculative
rewinds, vocabulary ownership and cache eviction. A small deterministic model
is run through the installed `mlx_lm.generate.generate_step`, not a replacement
generation loop; its preferred prose is constrained to a parseable object.
Together with the existing real-tokenizer and request-propagation suites,
52 tests pass. This is worker-adapter evidence, not a resident 27B decode or a
broad reasoning-gain measurement. G05 remains open.

Validation on the final patch: 52 focused tests passed in 45.65 seconds.
Lint, compile, governance, layering and writing passed. Smoke recorded 163
passed, one skipped and the existing live-activation failure in 216.46 seconds:
`semantic_neural_activation_invalid:resident_manifest_drift`, with no drifted
bound source files. The decoder repair does not renew that qualification.
