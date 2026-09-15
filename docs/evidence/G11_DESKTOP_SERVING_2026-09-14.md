# G11: qualified desktop serving

## Measured

Four fresh tasks were entered through the live desktop chat input and Send
button on 2026-09-14. The displayed answers and neural feed were read after
each turn. Coding, calibration, premise auditing, and causal inference each
returned the exact expected JSON answer. Each durable delivery is completed,
fresh rather than replayed, and reports `cognitive_engine_qualified_recurrent`.
The qualified mechanism owns the terminal bytes; these turns did not consume
resident model generation.

The runtime loaded commit `c9a826b9b5da581eb91a34b5099ed5eb70a191aa` from
the frozen `codex-g11-live-20260914` worktree. Runtime source identity is
verified, current, and stable. The resident model was
`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`.

| Family | Turn | Exact |
| --- | --- | --- |
| Coding | `9a5e9b468c2a405b8007948f312f20d6` | yes |
| Calibration | `6694692f2d614b3588f5bfd465de621e` | yes |
| Premise audit | `888de7a7f05f47a486b1faa97c1235cf` | yes |
| Causal inference | `bd2b44d02488460c8b0dff952212e043` | yes |

## Replay

The archive retains the prompts, independently checked expected answers,
desktop-observed text, exact journal JSON and its hash, and runtime identity.
The quality projection references the underlying result receipt hash; it is
not a separately sealed result receipt. The full result receipt is not in the
public journal projection. The earlier in-process 120-task replay retains the
full mechanism and lesion evidence; this archive establishes desktop serving.

```sh
python tools/verify_qualified_desktop_delivery.py artifacts/migration/27b/recovery/desktop-serving-20260914/delivery.json
python -m pytest tests/test_qualified_desktop_delivery_evidence.py -q
```

The archive verifier passes all four deliveries. Fifteen tests cover the
retained positive and changed prompt, answer, model, route, source revision,
journal seal, delivery state, duplicate turn, and missing family controls.

## Limits and defects retained

G11's eligible-live-serving obligation is complete. This does not establish
unseen-family transfer, broad reasoning gain, native decoder recurrence, or
frontier performance. G03-G10 and G12 retain their own requirements.

Runtime health remains unresolved: the neural feed reported multi-second
event-loop lag and stale health snapshots. The boot observation in the
archive reports that condition without converting it into successful health.
The generic prose assessor also labeled the correct coding JSON answer
`unanswered_question_part`; deferred learning rejected that answer on the same
basis. Delivery preserved the correct bytes. Those assessment consumers still
need the typed completion evidence, rather than lexical overlap, to judge it.
