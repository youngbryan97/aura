# Ternary Bonsai 2.5 27B on this host, 2026-11-24

The brainstem's replacement, measured against the Bonsai 2 it replaces. Same host, same harness, one model loaded at a
time with her runtime down, greedy decoding, a 400-token cap; the 160-token budget of the 17 September comparison is
counted as well.

## What it is

`prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit`, released 24 November 2026 under Apache 2.0: Qwen3.8 27B at two bits, group
128, MLX affine, with the Hadamard rotation folded in, 8.88 GB on disk. The pack's model type is the same as Bonsai 2's
(`prism_hadamard_qwen35`, 64 layers, `hadamard.json` beside the weights), and `core/brain/llm/prism_hadamard.py` loads it
unchanged. It also ships a multi-token-prediction head (`mtp/`), which nothing here uses yet. PrismML's release names
5.9 GB as its footprint; the MLX pack on disk is 8.88 GB, and its peak below is what this host measured.

## Measured

| case | 2.5 tokens | tok/s | right | 2 tokens | tok/s | right |
|---|---|---|---|---|---|---|
| all-but-9 sheep | 31 | 17.7 | yes | 91 | 20.6 | yes |
| bat and ball | 347 | 22.0 | yes | 196 | 21.3 | yes |
| 100 days ago | 227 | 21.2 | yes | 174 | 20.7 | yes |
| r's in strawberry | 156 | 20.9 | yes | 147 | 20.6 | yes |
| tallest/shortest | 72 | 19.4 | yes | 49 | 18.7 | yes |
| JSON only | 5 | 7.1 | yes | 5 | 7.7 | yes |
| not bordering France | 227 | 21.2 | — | 400 | 21.7 | — |
| Python one-liner | 9 | 10.6 | yes | 9 | 10.6 | yes |

- **Right: 7 of 7 checkable for both.** The prompts are the 17 September ones in kind, written again; that comparison
  did not keep its harness.
- **Decode: 20.8 tok/s against 20.7.** Peak memory 8.42 GB for both; load 0.9 s against 0.8 s.
- **Ended inside 400 tokens: 8 of 8 against 7 of 8.** Asked for a European country that does not border France,
  Bonsai 2 wrote "Let's pick **Germany**... no." again and again until the cap. Bonsai 2.5 named Poland.
- **Inside the brainstem's 160-token budget: 5 of 8 for both.** Bonsai 2.5 is longer on the bat-and-ball question
  (347 tokens against 196).

## What this is not

Eight prompts, greedy, one run each: it shows the pack loads and runs here at the old one's cost, and answers at
least as well on these. PrismML's published gains are in agentic tool calling and coding (tau3-bench 73.9 against
69.1, SWE-bench Verified 75.6 against 60.8); those are the vendor's numbers on its own harness, not evidence here.

Reproduce: the harness is a load of each pack with `load_prism_hadamard_pack` and `mlx_lm.generate` on the eight
prompts above, chat template with thinking off.
