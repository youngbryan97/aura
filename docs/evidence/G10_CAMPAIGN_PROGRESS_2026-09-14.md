# G10: interrupted campaigns retain completed samples

The trained-vector campaign disappeared without a final result after reporting
seven completed conditions. The runner held those outputs in memory and wrote
only after its final condition. No retained artifact can turn that partial run
into a measured result.

The runner now commits each completed sample through the file gateway. Its
progress record binds the plan, model descriptor, actual vector-file hashes,
generation metadata, implementation files, dependency versions, prompts,
sampling settings and condition order. Resume requires an explicit flag and an
identical identity. The record must be an ordered prefix; corruption, skipped
conditions and excess samples are errors. Final output remains separate and is
written only after all conditions complete and input identity is rechecked.

Each sample also retains the hook composites. Restoring only output strings
would change the next decode because steering uses exponential smoothing.
Control-vector RNG setup and per-sample seeds follow the same sequence on
resume. The reusable progress class receives persistence from its caller; the
campaign runner owns the output paths and governed writes.

Thirty-three focused tests passed in 18.49 seconds. A full runner test injects
an interruption inside the random-vector arm and obtains the same 54 outputs
after resume, with no repeated completed samples. A real MLX hook test restores
a nonzero composite and obtains bit-identical subsequent composite updates.
The model in the full runner test is substituted; no resident-model output
equivalence is claimed from that test.

Lint, compile, governance, layering and writing passed. Smoke: 163 passed, one
skipped and the existing resident-manifest activation alarm failed, in 58.62
seconds. No admission predicates or historical evidence changed. The 216-sample
resident campaign still needs to finish; G10 and G11 are not closed.
