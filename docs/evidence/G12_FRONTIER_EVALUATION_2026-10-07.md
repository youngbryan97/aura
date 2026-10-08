# G12: her performance against named current baselines, with resources stated

G12 asks for her performance on independent broad tasks against named
current baselines, with resource and tool access reported fairly. This page
fixes the protocol before the numbers; the numbers follow as the runs finish.

## Systems

* **Her**: `Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`, the model
  her runtime serves, 4-bit MLX, on this host (Apple silicon, 64 GB).
* **Her base model**: `Qwen3.8-27B-4bit-3e6447f082e8`
  (mlx-community conversion of `Qwen/Qwen3.8-27B`), same host.
* **Published current systems**, as their authors measured them on their own
  protocols (context, not matched experiments): Qwen3.8-27B, Qwen3.6-27B,
  Qwen3.7-Plus, Muse Glimmer-30B and Opus 4.6 Max from the
  [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) (August
  2026; temperature 1.0, top-p 0.95, thinking on; read 7 October 2026);
  GPT-5 (high) and o3 on MATH-500 from the public aggregation at
  [benchlm.ai](https://www.benchlm.ai/benchmarks/math500) (July 2026; read
  7 October 2026).
* **Systems measured on the same questions under the same protocol**: the
  [BBEH leaderboard](https://github.com/google-deepmind/bbeh/blob/main/leaderboard.md)
  (read 7 October 2026) reports o3-mini (high), DeepSeek R1, Gemini 2.0
  Flash, GPT-4o and Gemma 3 27B on the 460 questions of BBEH Mini with the
  paper's instruction ([Kazemi et al. 2025](https://arxiv.org/pdf/2502.19187)).

## One protocol for both local systems

Each request is the benchmark's own text in one user turn through the
model's chat template, private reasoning on at her serving effort, one
greedy decode, up to 32,768 tokens, no tools, no retrieval, no retries; a
reply stopped at its budget has no answer. Decodes run eight at a time
(`tools/g12_batched.py`), the same for both systems. Rows are written once.

| Benchmark | Requests | Grader |
| --- | ---: | --- |
| MATH-500 | 500 | PRM800K's `grading` package on the last `\boxed{}` |
| IFBench | 300 | AllenAI's own `run_eval`, in its own environment |
| BIG-Bench Extra Hard Mini | 460 (20 from each of 23 tasks) | the authors' evaluator after "The answer is:" |

## Resources and access, stated

Both local systems: 4-bit weights, one machine, no network at answer time,
no tool calls, greedy decoding, the token allowance above. The published
systems ran on their providers' hardware at full precision, sampling at
temperature 1.0 where the card says so, with their own harnesses. Their
numbers are reported beside hers, not as matched results; the matched
comparison is her against her base model under one protocol.

## Results

Appended here, dated, as the runs finish. The page above was committed
when 72 of her 500 MATH-500 rows were written and no other run had started.
Paired comparisons and intervals come from `tools/g12_analysis.py`.
