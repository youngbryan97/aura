# G03 policy and coefficient controls

## Completed measurements

The two-round function-coordinate trial repaired all four selected training
errors: 28/32 became 32/32, with no training regression. The 100 exposed
validation rows stayed at 91/100. The training retained 256 source rows;
validation did not enter fitting. Wall time was 897.73 seconds.

The follow-up separates the selection-policy change from coefficient learning
on those same 100 validation observations:

| Coefficients | Operation selection | Equivalent | Gains vs incumbent | Regressions |
| --- | --- | ---: | ---: | ---: |
| Incumbent | First feasible | 99/100 | 0 | 0 |
| Incumbent | Joint factor score | 91/100 | 0 | 8 |
| Fitted | Joint factor score | 91/100 | 0 | 8 |
| Fitted | First feasible | 99/100 | 0 | 0 |

The fitted first-feasible arm changes only the declared selection policy after
training. It is a diagnostic intervention, not a promoted model. The control
replay took 158.52 seconds. These are exposed development measurements, not
fresh transfer, a 500-row result, public-generation accuracy, or broad gain.

The joint-score policy causes the eight measured regressions on this cohort.
The fitted coefficients neither repair nor worsen the incumbent's remaining
error. A single frozen chart margin did not control all moving runtime
competitors; re-mining repaired the training cases, but did not create a
validation gain. The incumbent remains the selected serving candidate.

## Retained evidence

- `~/.aura/rlc-evidence/semantic-functional-iterative-20260919.json`:
  `766883ef329c02e137ad52e9dd5ead12975dd904c42c7ac86824eed647c40269`
- `~/.aura/rlc-evidence/semantic-functional-policy-controls-20260919.json`:
  `cdf80b867b7eb396f6c700b295ec22fd668cb3ec7cfca714daf1e1806f932b78`
- Candidate and optimizer checkpoints remain beside the iterative receipt.
  Neither result changes a qualification certificate or grants serving authority.

G03 stays open. Candidate inspection must distinguish missing interpretations,
incomplete search, and misranking before another coefficient trial.
