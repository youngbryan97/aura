# Diagnose Reachability Without Enumerating Past Its Witness

The full development audit preserved 170 rows at this observation: 168
semantically equivalent programs and two witnessed selection errors. It is
not a completed cohort result. In both failures, comparison index one (the
first alternative after the selected program) was equivalent to the target.
The banks nevertheless contained 38 and 24 distinct compared programs.

The audit process spent minutes in the HiGHS optimizer between completed
rows. Native sampling confirms optimizer work; it does not identify which
Python phase consumed every interval. The old process continues in its frozen
checkout while the revised diagnostic is tested.

Two changes reduce diagnostic work without fitting coefficients or choosing
a different serving answer:

1. Ordinary decode accepts an optional search allowance. The remaining
   allowance reaches the argument-chart solver after chart construction.
   Serving defaults remain unchanged. The diagnostic passes its declared
   allowance through ordinary decode and annotated scoring as well as the
   alternative search. Incomplete search remains a refusal/unknown, never
   infeasibility or correctness. This is a solver allowance, not a hard
   process-level wall-clock deadline for every preprocessing operation.
2. The cohort diagnostic first completes a one-chart, one-alternative bank.
   Only afterward does the diagnostic compare it against the target. A
   witnessed equivalent alternative establishes reachability; enumeration
   beyond that witness is unnecessary for this diagnosis. Unresolved cases
   expand to the declared full bank. Exhaustion remains required for an
   absence claim, and unknown equivalence stays unknown.

Target comparison controls only whether the offline diagnosis buys another
completed bank. It never changes the bank's scoring, its selected answer, or
the live decoder. The version-two report explicitly records that diagnostic
budget dependency. It must not be used as an answer-blind runtime search-cost
or latency measurement.

Each failed row now retains complete bank receipts, including candidate
programs, followed by their diagnoses. Earlier version-one records contain
the diagnosis and bank hash but not those replay inputs. They remain partial
historical evidence; the new source/policy identity cannot reuse them as new
observations.

Verification: 134 focused checks passed for allowance propagation, solver
contracts, cohort auditing and candidate banks. After adding staged diagnosis,
69 focused checks passed, including source identity and graph-trial consumers.
The complete patch passes smoke (164 passed, one skipped), lint, compile,
governance lint, layering, and writing. No G item closes, no new candidate is promoted, and no full
cohort accuracy is claimed here.
