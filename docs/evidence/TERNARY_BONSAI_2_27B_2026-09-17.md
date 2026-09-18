# Ternary Bonsai 2 27B on this host, 2026-09-17

A 27B-class model in the brainstem's footprint class, measured against the
brainstem it would replace. Same host, same harness, greedy decoding, a
160-token cap, with the live instance running.

## What it is

Qwen3.5 27B at two bits, group 128, MLX affine — `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`,
8.6GB on disk. Two bits alone destroys a model this size: the damage
concentrates in a few activation channels whose magnitude dwarfs the rest,
and a group scale sized for the outlier quantises everything else to
nothing. The pack rotates each block with a Hadamard transform before
quantising, so no group is sized by one outlier, and folds the inverse
rotation into the weights on the other side. MLX has both primitives —
`mx.hadamard_transform` and a two-bit `quantized_matmul` — so the decode
path is the ordinary one.

`mlx==0.32.0` and `mlx-lm==0.31.3` are what the pack asks for and what this
venv has.

## The loader

The runtime published with the pack reads schema 1. The pack is schema 2:
it refuses the config it ships with, and looks for the sign vectors among
the tensors, where schema 2 keeps them in `hadamard.json` instead.
`core/brain/llm/prism_hadamard.py` reads schema 2. The pack also ships the
vision tower, which its README says is not included — true of the runtime,
not of the weights — so everything outside the text tower stays on disk.

## Measured

| case | Bonsai tokens | tok/s | seconds | right | 9B tokens | tok/s | seconds | right |
|---|---|---|---|---|---|---|---|---|
| all-but-9 sheep | 49 | 8.7 | 5.6 | yes | 160 | 21.4 | 7.5 | yes |
| bat and ball | 85 | 8.1 | 10.5 | yes | 160 | 24.2 | 6.6 | **no** |
| 100 days ago | 120 | 8.1 | 14.8 | yes | 160 | 26.0 | 6.2 | yes |
| r's in strawberry | 124 | 8.6 | 14.4 | yes | 160 | 26.3 | 6.1 | yes |
| tallest/shortest | 29 | 8.5 | 3.4 | yes | 160 | 25.8 | 6.2 | yes |
| JSON only | 73 | 5.7 | 12.8 | yes | 160 | 26.4 | 6.1 | yes |
| not bordering France | 143 | 8.2 | 17.4 | — | 160 | 26.2 | 6.1 | — |
| Python one-liner | 160 | 8.6 | 18.6 | yes | 160 | 26.0 | 6.2 | yes |

- **Right: 7 of 7 checkable, against 6 of 7.** The 9B fails bat-and-ball.
- **Decode: 8.3 tok/s against 25.7 — three times slower per token.** That is
  the rotation's cost, paid per layer, and it is real.
- **Finished inside the budget: 7 of 8, against 0 of 8.** The 9B opens every
  answer with "Thinking Process:" and spends the whole budget narrating. It
  never reached an answer on any of the eight.
- Active memory 7.68GB against 5.04GB; peak 8.70 against 5.27. Load 1.6-2.5s.

The budget is what makes this matter. Per token the 9B is three times
faster; per *answer* it did not produce one. The fallback ladder runs under
a deadline, which is exactly this shape.

## What this is not

Eight prompts, greedy, one run. A smoke comparison, not a benchmark, and
not the published retention table — which is a claim about other
benchmarks and is not evidence here. The next thing this needs is Aura's
own battery.

## Not the cortex

The cortex is fused: persona and CRSM adapters are merged into the weights.
A LoRA trained on the unrotated base does not compose with weights carrying
the inverse rotation, so fusing onto a pack is not a matter of trying it —
`mlx_worker` refuses an adapter path on a pack and says why. Fusing first
and folding after would need a rotation-aware fusion path, which is L01.
Its two weakest published columns, agentic tool calling and knowledge, are
the cortex's two main jobs.

Reproduce: `tests/test_a_folded_pack_loads_like_any_checkpoint.py`.
