# Observation identity and operation-feature overlap

The v1 observation feasibility audit omitted individual observations from its
receipt when they had no duplicate. Different cohorts with the same counts could
therefore share a receipt. V2 records the entire development population, including
observation, basis, source, split, target and input anchors, and rejects empty
populations. Historical v1 evidence remains unchanged.

The corrected full development audit contains 1,264 observations and 1,264
distinct observation identities. There are no exact duplicate groups. This
rules out witnessed conflicting labels for identical complete observations in
this population; it does not prove learnability or scorer capacity.

- Artifact: `~/.aura/rlc-evidence/semantic-observation-feasibility-v2-20260921.json`
- Receipt: `6400b1b033b554f0c2cd3647aabb33b4575a2e62d6472233e0da47a58ca88bd1`
- Population: `685d895be070c0f228fbaa3539b7b81961c9ac5e24af4a2aa190c759baa0a5b0`

An additional diagnostic hashes the actual operation-head feature arrays at
annotated operation spans. The head uses `contextual_mean`. Among 2,912 annotated
operations there are 1,630 distinct feature vectors and no exact-feature groups
with conflicting operation labels. Of 1,192 validation operations, 64 have an
exact training-feature match. This overlap must remain visible when interpreting
operation-level validation. Absence of collisions does not prove linear
separability, correct boundary recovery or general transfer.

- Artifact: `~/.aura/rlc-evidence/semantic-operation-feature-overlap-20260921.json`
- Receipt: `3200950ca98ebc7657e1c51aff8a2c833c461675d5044a7cfe5d69d18d947201`

Both diagnostics exclude sealed test examples. Neither trains parameters,
promotes a candidate, grants serving authority or closes a G checkpoint.
