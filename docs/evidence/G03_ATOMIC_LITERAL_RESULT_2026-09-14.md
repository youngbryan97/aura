# G03: complete atomic-literal development result

The candidate from `8b4ecaa1e` completed all 500 exposed development cases in
151.242 seconds. The detached supervisor finished with no remaining child.

| Source family | Exact | Total |
| --- | ---: | ---: |
| Arithmetic | 117 | 128 |
| Cataphoric sequence | 47 | 48 |
| Fork/join | 192 | 192 |
| Role binding | 44 | 48 |
| Reserved aliases | 37 | 48 |
| Natural source | 24 | 24 |
| Natural aliases | 12 | 12 |
| Total | 473 | 500 |

Against the overlap-complete parent at 474/500, atomic literals gain four
cataphoric cases and lose one cataphoric plus four role-binding cases.
The candidate is not promoted. The literal-boundary invariant repairs a real
fragment-binding defect; it does not resolve learned role assignment.

The next training question is whether argument supervision matches the
operation spans and negative proposals the learned decoder actually uses.
The present fitter uses annotated operation spans and a short negative list;
runtime operation pointers may select different spans. This is a concrete
training/runtime mismatch to test, not evidence that another refit will win.

[Complete source report](../../artifacts/rlc/semantic_atomic_literals_dev_20260914/source.json)
and [detached receipt](../../artifacts/rlc/semantic_atomic_literals_dev_20260914/detached_receipt.json)
are retained. These are development observations, not fresh transfer proof.
