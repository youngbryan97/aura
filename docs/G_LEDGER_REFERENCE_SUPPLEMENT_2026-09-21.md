# Additional reference-package disposition

These attachments supplement the existing review queue; they do not replace
the master ledger or provide Aura measurements. The PDF explicitly describes
an independent review without inspection of Aura internals. Its 22 pages were
read as extracted text; the Python package, Markdown report and JSON run log
were inspected separately. The attached test package passed five tests locally.

| Source | SHA-256 |
|---|---|
| test_reference_package.py | 090bd921286d9c73aeb25bb8b5952d92ecd9c43478c9244fb142abd057520221 |
| aura_reasoning_reference.py | 61191338f2e1319d6dc50f76f4af2cc9a895658b2838678c556e7b6d2431cced |
| Aura_General_Reasoning_Ledger_Closeout_Report_2026-09-21.md | b320358fa57a39589e55bedea18062298ecbe2aa01e8c3e14864161afe0fc0a8 |
| reference_run_log.json | 844a6de8625f4d51f2cfbe4832cc4c6475a14b3aaa595d44f1212bd9cb00750a |
| AURA_G_LEDGER_REVIEW_2026-09-21.pdf | 6bc38f4f805803ad943e3c51b0d5faea39c63c911835c561b75ef9a5b2cb559d |

## Mechanism mapping

- Decoder-consistent selection: reuse semantic_joint_graph_learning and
  semantic_argument_chart. First-feasible decoding already compares operation
  charts before bindings; normalized joint scoring is a different experimental
  policy, not a guaranteed repair. Fixed-bank and end-to-end comparisons remain
  separate obligations (Q03, Q06-Q10 in the original disposition).
- Reachability ledger: reuse semantic_failure_diagnosis and candidate-bank
  exhaustion receipts. A found witness establishes reachability; incomplete
  enumeration or unknown equivalence cannot establish absence (Q01-Q04).
- Capacity LP: reuse score_capacity and exact affine certificates. The scratch
  linear separator cannot certify capacity of the complete nonlinear decoder
  (Q05).
- Trace verifier: reuse typed procedure execution and independent semantic
  comparison. Executing a proposed program correctly does not prove that it
  represents the request (Q11-Q13, Q19).
- Calibrated arbitration: retain as a calibration experiment, not a zero-
  regression guarantee. A probability threshold needs independent observations
  and shift evaluation. Baseline correctness is not known at runtime (Q22-Q23).
- Neural-guided enumeration and reusable macros: extend existing floor search
  and retained operators, with bounded completeness and independent held-out
  benefit. Shared sequence reach now repairs one missing consumer (Q17-Q21).
- Oracle alignment, frozen-support reranking, increased-budget reachability,
  and competitor co-sampling are diagnostic interventions, not serving access
  to answers. Use current full-cohort receipts before repeating stale pilots.
- Fresh transfer, free public decoding, lesions, independent verification,
  current-model qualification and matched frontier studies remain required
  under G04-G10/G12. Proposed sample sizes depend on effect, clustering and
  preregistered analysis; they are not universal constants.

## Corrections and limits

The PDF's claim that all monitored margins improve in its toy example conflicts
with the JSON values: 1.0 to 0.8, 1.5 to 1.4, and 2.0 to 1.5. The general lesson
about unmonitored competitors can still hold; that particular improvement claim
cannot be used as evidence. Finite-probe behavior deduplication is only probe
equivalence, not unrestricted program equivalence. Toy power calculations and
probabilistic replacement simulations are not Aura accuracy or safety results.

No master G item closes from this supplement or its five passing toy tests.
