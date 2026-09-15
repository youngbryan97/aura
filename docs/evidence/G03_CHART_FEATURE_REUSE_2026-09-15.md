# Decode-local chart features

The contrast-only joint experiment was stopped after 3,558.17 seconds.
Its checkpoint retained 500 incumbent rows and 216 candidate rows, so no
complete paired verdict exists. The supervisor reports `stopped`,
`passed=false`, and return code -15. The source-error trajectory already
recorded in the retention note remains the reason to replace that objective.

Graph search repeatedly formed the same span vectors and definition-pointer
scores for overlapping candidate charts. The decoder now shares those values
within one decode call. Operation, reference and definition spans use the same
cache. A new decode allocates a new cache; no features cross requests or model
identities. The candidate grammar, scores and search bounds are unchanged.

The regression checks compare cached and uncached assignments and full decoder
outputs, count span computations, and verify request-local cache ownership.
This optimization changes computation reuse, not semantic capacity. It grants
no serving authority or G03 closure.

Checks: 40 focused tests pass; smoke reports 164 passed and one skipped.
Ruff, compile, governance and layering pass. Governance initially reported two
unrecorded async trace-writer calls from the inherited trace repair. Both use
the named internal scope and the async gateway; seven trace tests pass. The
inventory records exactly those two additions, without a canonical-owner
exemption or a change to runtime permissions.

Stopped-run receipt:
`~/.aura/rlc-evidence/semantic-joint-graphs-supervisor-20260915/detached_receipt.json`.
Partial comparison:
`~/.aura/rlc-evidence/semantic-joint-graphs-dev-20260915/validation.checkpoint.json`,
receipt hash `2d0000928b71eb0bebe0399d812dcc290c8faf4ef44b23d8fdc4b567dfc7faf2`.
