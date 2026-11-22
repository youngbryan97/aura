# Supported Aura claims

| Result | Where it was measured |
|---|---|
| Retrieval helps find an earlier answer outside a fixed context window. | In the August 32B test, retrieval solved 40 of 40 and recency solved 0 of 40 with a twelve-turn budget. The full-transcript control solved 39 of 40. [Run record](artifacts/ablation/capability_scorecard_32b_longhistory.json) |
| Saved programs recover some missed math and planning answers. | The October G09 review found 29 gains and no losses across 444 fresh requests. It covered the registered math and planning tasks. [Result](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md) |
| Answers taught in chat can become a procedure used later. | In one desktop trial, Aura used a learned program to answer three fresh date questions the next day. The run included a manual reset of the learning group. [Live trial](docs/evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md) |
| Experimental steering can change a model's output. | The August 32B experiment measured a shift in a vocabulary score at the tested injection strength. The October ISC study used a separate valence channel and a different report task. [August scorecard](artifacts/ablation/affect_causality_scorecard_n160.json) · [ISC results](docs/ISC_STATUS.md) |
| A small model training run can use the prior cycle's published weights. | The July 1.5B demonstration completed two cycles. Its held-out scores were 0.667 and 0.625. [Run bundle](artifacts/learning_compounding/2026-07-07-1p5b-2cycle/) |

## Engineering checks

The repository has tests for memory storage, action authorization, and
sandbox restrictions. The [claim matrix](CLAIMS_MATRIX.md) names each
check and what would cause it to fail. To check a running installation,
use its actual request and action records.
[Memory tests](tests/test_continuous_experience_stream.py) ·
[Authorization tests](tests/test_executive_authority.py) ·
[Sandbox tests](tests/test_sandbox_hardening.py)
