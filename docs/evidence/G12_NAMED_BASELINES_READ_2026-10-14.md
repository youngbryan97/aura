# G12: the named systems' published numbers, as read on 14 October 2026

The G12 protocol (G12_FRONTIER_EVALUATION_2026-10-07.md) names the systems her
three runs are set beside and where their numbers come from. It names the
sources but does not record their numbers. This page records them as read on
14 October 2026, before any of her three runs is complete, so the result
quotes figures with a date and a source.

## IFBench, from the Qwen3.8-27B model card

Source: https://huggingface.co/Qwen/Qwen3.8-27B (citation "August 2026",
collection "Updated Aug 13"), Text Performance table.

| System | IFBench |
| --- | ---: |
| Qwen3.8-27B | 79.5 |
| Qwen3.7-Plus | 79.1 |
| Muse Glimmer-30B | 77.0 |
| Qwen3.6-27B | 69.1 |
| Opus 4.6 Max | 62.5 |

The card's recommended thinking-mode sampling is temperature 1.0, top-p 0.95,
top-k 20. It does not state the settings or token limits used for IFBench.
The card has no MATH-500 or BBEH row.

## MATH-500, from benchlm.ai

Source: https://www.benchlm.ai/benchmarks/math500 ("Last updated: October 14,
2026"). The page gives no settings, no measurement dates, and does not say
whether scores are self-reported.

The protocol of 7 October named GPT-5 (high) and o3 from this page. Neither
is listed on 14 October. These are the listed systems nearest her in kind
and size, and the strongest:

| System | MATH-500 (%) |
| --- | ---: |
| Ternary Bonsai 2 27B | 98.8 |
| QwQ-32B | 98.0 |
| Qwen3 235B-A22B (Reasoning) | 98.0 |
| o4-mini | 97.5 |
| DeepSeek-R1 | 97.3 |
| Qwen3 32B (Reasoning) | 97.2 |
| Claude 3.7 Sonnet Thinking | 96.2 |
| DeepSeek V3 | 90.2 |
| o1-mini | 90.0 |

## BBEH Mini, from the BBEH leaderboard

Source: https://github.com/google-deepmind/bbeh/blob/main/leaderboard.md, the
copy saved on this host at `~/.aura/benchmarks/g09/bbeh-leaderboard/leaderboard.md`.
The same 460 questions and the paper's instruction; micro average.

| System | BBEH Mini (micro avg) |
| --- | ---: |
| o3-mini (high) | 56.7 |
| DeepSeek R1 | 37.2 |
| Gemini 2.0 Flash | 27.0 |
| GPT-4o | 23.5 |
| Gemini 2.0 Flash-Lite | 22.2 |
| Gemma 3 27B | 17.4 |
| Distill R1 Qwen 32B | 15.4 |
| Gemma 2 27B IT | 15.0 |

## Note, 14 October 2026, 23:55: the commit G12's runs resume at

G12's runs go on at ff696bc2d, which is 19b9c7e6d with one change: a model a
standalone tool loads is held to the host's MLX memory envelope (ed8d6fae6 on
main), after a GPU out-of-memory freeze at 22:43 forced a reset. The envelope
bounds how much memory MLX keeps; it changes no decoded token. Each run
directory's source_commit.txt lists every commit it ran at.
