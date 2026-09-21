# Deduplicate fit evidence storage

The first full-retention archive captured 25,038 comparisons and 286,731
parameters. Its file SHA-256 is
`7303b5b8361583662baa309ea26bf992e064af7feb0039323d659d191a3ba8ae`.
It completed in 520.2 seconds, including source mining, and occupies 1.4 GiB.
A process sample during serialization found compression consuming the CPU.

The writer now stores equal arrays once by dtype, shape and SHA-256 content.
Distinct source arrays remain distinct objects on load; repeated references
retain their original aliasing. This changes storage, not numerical evidence.
The existing archive remains intact. No compressed-size or runtime improvement
is claimed without a new full-size measurement.

Nineteen focused replay/checkpoint tests pass. Smoke: 164 passed, one skipped.
Lint, compile, governance, layering and writing gates pass. G03 remains open.
