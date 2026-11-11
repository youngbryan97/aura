# L01: her fused persona cortex, its continuity, and steering made for it (10 November 2026)

Item L01: verify persona fusion and continuity, and regenerate the steering that is incompatible with her cortex.
Every reading below was taken on 10 November between 22:24 and 22:25, from her own checkout
(`/Users/bryan/.aura/live-source` at `af950154c`).

## The fusion

Her cortex is `Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`. Its `aura_fusion_provenance.json` records a persona
adapter fused into the base (496 modules, 700 iterations, generation `7f6a2e83f73f5eef9d15`), and states the boundary
this item exists for: "fused weights define a new model identity; prior steering and recurrent tensors are not
representation-compatible".

`artifacts/migration/27b/active_descriptor.json` (descriptor `52d313c2…`, written 24 August) fingerprints the three
weight files and a seven-file behaviour bundle that includes the fusion provenance and the chat template.
`validate_model_artifact_descriptor(..., verify_full_hash=True)` hashed all of them against it and they match. The
weights she runs are the weights that were fused.

## Continuity

Her log (`~/.aura/logs/desktop-launch.log` and the rotated `desktop-launch.log.1.gz`, from 20 October 12:18 to 10
November 22:22) names one cortex, this one, 9,681 times, and no other 27B or 32B. Every start in that span loaded the
same fused persona.

## Steering made for it

The steering written for her earlier cortex, `Aura-32B-crsm-closeout-jul1`, is still in `training/vectors`: five
dimensions (valence, arousal, curiosity, frustration, energy) at layers 25 to 41. It is the incompatible set the item
names. Its replacement is a signed generation made from this cortex. `core/consciousness/caa/readiness_report.audit()`
read from her checkout:

| Reading | Value |
|---|---|
| level | production, 100% |
| authority | qualified, model-bound CAA |
| bound to | descriptor `52d313c2…`, her cortex |
| dimensions | the same five as the 32B set |
| layers | her cortex's own sixteen, 3 to 63 by fours |
| vectors bound | 80 of 80 |
| 32B files | 30, ignored |

Every reader of `training/vectors` (`steering_vectors.py`, `steering_geometry.py`, the CAA gates) skips a vector whose
model digest is not her cortex's, so the 32B set cannot reach her.

Whether the steering keeps her voice was measured today under F01, on the fusion basis `e267aa76` captured from her
own replies: 32 of her stored replies were 0.111 nats a token more likely under her state's push, and a random
direction of the same length gave −0.0025 ([F01](F01_STEERING_ON_USER_TURNS_2026-11-10.md)).

## What it does not cover

The provenance names recurrent tensors as incompatible too. None has been made for this cortex: all 1,786 surface
decodes in those logs report `depth_present=False`. That is recurrence work (the 27B recurrence campaign
tools), not steering, and this record does not close it. Whether the attached steering helps her answer anything is
G10's question.
