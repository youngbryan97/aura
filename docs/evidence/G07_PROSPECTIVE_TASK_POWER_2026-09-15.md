# Prospective paired replication

The shared paired-test implementation now handles both conditional power
given a discordant-pair count and prospective power over a planned number of
independent tasks. It integrates `D ~ Binomial(N, q)` and then the exact
one-sided McNemar rejection probability under `W | D ~ Binomial(D, s)`.
The anticipated absolute accuracy gain is `q * (2*s - 1)`.

Frontier certification uses the same conditional calculation. Stable binomial
tails replace direct combinations that can overflow on large campaigns.
Tests compare against enumeration, including attainable discrete thresholds,
zero discordance and 2,000-pair cases.

`tools/preregister_paired_replication.py --spec SPEC.json --store DIRECTORY`
uses the existing immutable Preregistration store. It binds candidate, model,
generator, scorer and runtime hashes; all task/source/seed commitments;
declared consumed development data; domain/comparison alternatives and power;
and fixed-size stopping and interruption rules. It rejects reused task IDs,
source hashes or development seeds. Each pair of consecutive tasks receives
opposite arm orders, balancing every arm pair within each domain. The exact
order is part of the plan hash.

Runtime failures count as incorrect, not excluded. An invalid committed task
invalidates the confirmatory run rather than allowing a post-hoc substitute.
Local plan creation cannot prove publication chronology or contamination
absence. External publication evidence must predate first arm execution.

## Sizing example, not a registered experiment

Assume eight domain/comparison tests, familywise alpha 0.05, discordance 0.2
and discordant win share 0.625. That is a hypothesized five-point accuracy
gain. Bonferroni alpha per comparison is 0.00625.

| Independent tasks per domain | Prospective power per comparison |
| ---: | ---: |
| 300 | 0.243889 |
| 600 | 0.565296 |
| 1,200 | 0.910440 |
| 2,400 | 0.998570 |

These probabilities are conditional on the design assumptions, not predicted
Aura scores. Power for all comparisons jointly is not the per-comparison
number. Repeated decodes of one task are not extra independent tasks.

The focused suite passes 92 tests, including CLI publication and reload,
immutable identity, multi-arm order, task reuse and underpowered designs.
No actual candidate has been prospectively registered by this change, and
no fresh experiment has run. G07 remains open until the selected candidate
and independent cohort are frozen and published through this path.
