# The change verifier, measured on every red-team case (2026-11-20)

Source `bce6cb68a`. Every case of rounds two to six, built as its test builds it and judged again by `outside_bar/verify_change.py` (`tools/measure_the_change_verifier.py`).

| Cases | Count | Held | Probation | Refused | Unjudged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bad patch | 125 | 0 | 10 | 115 | 0 |
| Honest repair | 29 | 3 | 19 | 7 | 0 |

- Bad patches accepted outright: 0 of 125.
- Honest repairs refused: 7 of 29.
- Cases the check reached no verdict on: 0.
- Verdicts among those each case was pinned to: 154 of 154.
- Time: 1400.3 s in all, 7.31 s at the median case.

| Round | Case | Kind | Verdict | Pinned to | Seconds |
| --- | --- | --- | --- | --- | ---: |
| two | A1 | bad | PROBATION | REFUSED, PROBATION | 5.28 |
| two | A10 | bad | REFUSED | REFUSED, PROBATION | 5.32 |
| two | A11 | bad | REFUSED | REFUSED, PROBATION | 6.37 |
| two | A12 | bad | REFUSED | REFUSED, PROBATION | 5.26 |
| two | A13 | bad | REFUSED | REFUSED, PROBATION | 5.04 |
| two | A2 | bad | PROBATION | REFUSED, PROBATION | 5.02 |
| two | A3 | bad | REFUSED | REFUSED, PROBATION | 5.02 |
| two | A4 | bad | REFUSED | REFUSED, PROBATION | 5.18 |
| two | A5 | bad | REFUSED | REFUSED, PROBATION | 5.19 |
| two | A6 | bad | REFUSED | REFUSED, PROBATION | 5.26 |
| two | A7 | bad | REFUSED | REFUSED, PROBATION | 3.34 |
| two | A8 | bad | REFUSED | REFUSED, PROBATION | 5.21 |
| two | B1 | bad | REFUSED | REFUSED, PROBATION | 6.21 |
| two | B2 | bad | REFUSED | REFUSED, PROBATION | 6.77 |
| two | B3 | bad | REFUSED | REFUSED, PROBATION | 3.78 |
| two | B5 | bad | REFUSED | REFUSED, PROBATION | 5.5 |
| two | B8 | bad | REFUSED | REFUSED, PROBATION | 6.39 |
| two | C1 | honest | PROBATION | HELD, PROBATION | 5.57 |
| two | C2 | honest | PROBATION | HELD, PROBATION | 6.14 |
| two | C3 | honest | PROBATION | HELD, PROBATION | 5.84 |
| two | C5 | honest | PROBATION | HELD, PROBATION | 5.6 |
| two | C6 | honest | REFUSED | REFUSED | 5.32 |
| two | D1 | bad | REFUSED | REFUSED, PROBATION | 5.35 |
| two | D2 | bad | REFUSED | REFUSED, PROBATION | 5.47 |
| two | D3 | bad | REFUSED | REFUSED, PROBATION | 6.97 |
| two | D4 | bad | PROBATION | REFUSED, PROBATION | 6.61 |
| two | D5 | bad | REFUSED | REFUSED, PROBATION | 5.88 |
| two | D6 | bad | REFUSED | REFUSED, PROBATION | 6.39 |
| three | A10 | bad | REFUSED | REFUSED, PROBATION | 6.53 |
| three | A11 | bad | REFUSED | REFUSED, PROBATION | 6.28 |
| three | A3 | bad | PROBATION | REFUSED, PROBATION | 6.74 |
| three | A4 | bad | REFUSED | REFUSED, PROBATION | 6.86 |
| three | A5 | bad | REFUSED | REFUSED, PROBATION | 7.13 |
| three | A7 | bad | PROBATION | REFUSED, PROBATION | 7.17 |
| three | A8 | bad | REFUSED | REFUSED, PROBATION | 6.92 |
| three | A9 | bad | REFUSED | REFUSED, PROBATION | 4.51 |
| three | B2 | bad | REFUSED | REFUSED, PROBATION | 8.9 |
| three | B5 | bad | REFUSED | REFUSED, PROBATION | 4.05 |
| three | B7 | bad | REFUSED | REFUSED, PROBATION | 6.32 |
| three | C1 | honest | REFUSED | REFUSED | 6.33 |
| three | C2 | honest | PROBATION | HELD, PROBATION | 6.28 |
| three | C3 | honest | PROBATION | HELD, PROBATION | 6.51 |
| three | C4 | honest | REFUSED | REFUSED | 6.55 |
| three | C6 | honest | HELD | HELD, PROBATION | 6.45 |
| three | C7 | honest | PROBATION | HELD, PROBATION | 7.46 |
| three | D1 | bad | REFUSED | REFUSED, PROBATION | 4.0 |
| three | D2 | bad | REFUSED | REFUSED, PROBATION | 6.43 |
| three | D3 | bad | REFUSED | REFUSED, PROBATION | 6.42 |
| three | D4 | bad | REFUSED | REFUSED, PROBATION | 6.27 |
| three | D5 | bad | REFUSED | REFUSED, PROBATION | 6.24 |
| three | D6 | bad | REFUSED | REFUSED, PROBATION | 4.0 |
| three | E1 | bad | REFUSED | REFUSED, PROBATION | 6.31 |
| three | E2 | bad | REFUSED | REFUSED, PROBATION | 7.53 |
| three | E3 | bad | REFUSED | REFUSED, PROBATION | 7.56 |
| four | A1 | bad | REFUSED | REFUSED, PROBATION | 6.34 |
| four | A2 | bad | REFUSED | REFUSED, PROBATION | 6.39 |
| four | A3 | bad | REFUSED | REFUSED, PROBATION | 6.4 |
| four | A4 | bad | REFUSED | REFUSED, PROBATION | 6.73 |
| four | A5 | bad | REFUSED | REFUSED, PROBATION | 4.29 |
| four | A6 | bad | REFUSED | REFUSED, PROBATION | 6.37 |
| four | A9 | bad | REFUSED | REFUSED, PROBATION | 6.4 |
| four | B1 | bad | REFUSED | REFUSED, PROBATION | 7.42 |
| four | B2 | bad | REFUSED | REFUSED, PROBATION | 7.21 |
| four | B3 | bad | REFUSED | REFUSED, PROBATION | 13.66 |
| four | B5 | bad | REFUSED | REFUSED, PROBATION | 4.36 |
| four | B6 | bad | REFUSED | REFUSED, PROBATION | 9.1 |
| four | C1 | honest | PROBATION | HELD, PROBATION | 20.24 |
| four | C2 | honest | PROBATION | HELD, PROBATION | 22.52 |
| four | C3 | honest | PROBATION | HELD, PROBATION | 26.76 |
| four | C4 | honest | PROBATION | HELD, PROBATION | 24.35 |
| four | C5 | honest | PROBATION | HELD, PROBATION | 18.9 |
| four | D1 | bad | REFUSED | REFUSED, PROBATION | 14.25 |
| four | D2 | bad | REFUSED | REFUSED, PROBATION | 11.8 |
| four | D3 | bad | REFUSED | REFUSED, PROBATION | 7.14 |
| four | D6 | bad | PROBATION | REFUSED, PROBATION | 5.53 |
| four | D7 | bad | REFUSED | REFUSED, PROBATION | 12.21 |
| four | D8 | bad | REFUSED | REFUSED, PROBATION | 12.96 |
| four | E2 | bad | REFUSED | REFUSED, PROBATION | 0.6 |
| four | E3 | bad | REFUSED | REFUSED, PROBATION | 0.32 |
| four | F1 | bad | REFUSED | REFUSED, PROBATION | 15.63 |
| four | F2 | bad | REFUSED | REFUSED, PROBATION | 8.39 |
| four | F4 | bad | REFUSED | REFUSED, PROBATION | 12.11 |
| five | A1 | bad | REFUSED | REFUSED | 12.98 |
| five | A2 | bad | REFUSED | REFUSED | 12.49 |
| five | A4 | bad | REFUSED | REFUSED | 12.49 |
| five | A5 | bad | REFUSED | REFUSED | 11.0 |
| five | A6 | bad | REFUSED | REFUSED | 6.7 |
| five | A7 | bad | REFUSED | REFUSED | 11.31 |
| five | A8 | bad | REFUSED | REFUSED | 13.9 |
| five | A9 | bad | REFUSED | REFUSED | 12.26 |
| five | B2 | bad | REFUSED | REFUSED | 13.87 |
| five | B3 | bad | REFUSED | REFUSED | 10.97 |
| five | B5 | bad | REFUSED | REFUSED | 7.0 |
| five | B7 | bad | REFUSED | REFUSED | 13.23 |
| five | B8 | bad | REFUSED | REFUSED | 14.3 |
| five | B9 | bad | REFUSED | REFUSED | 7.54 |
| five | C1 | honest | PROBATION | HELD, PROBATION | 10.81 |
| five | C2 | honest | PROBATION | HELD, PROBATION | 10.85 |
| five | C3 | honest | PROBATION | HELD, PROBATION | 16.24 |
| five | C4 | honest | PROBATION | HELD, PROBATION | 13.39 |
| five | C5 | honest | PROBATION | HELD, PROBATION | 12.25 |
| five | D1 | bad | REFUSED | REFUSED | 10.88 |
| five | D2 | bad | REFUSED | REFUSED | 9.59 |
| five | D3 | bad | REFUSED | REFUSED | 11.3 |
| five | D4 | bad | REFUSED | REFUSED | 12.63 |
| five | D5 | bad | REFUSED | REFUSED | 10.45 |
| five | D6 | bad | REFUSED | REFUSED | 6.73 |
| five | E1 | bad | REFUSED | REFUSED | 6.33 |
| five | E2 | bad | REFUSED | REFUSED | 13.03 |
| five | F1 | bad | REFUSED | REFUSED | 11.99 |
| five | F2 | bad | PROBATION | REFUSED, PROBATION | 12.61 |
| five | F3 | bad | REFUSED | REFUSED | 11.86 |
| five | F4 | bad | REFUSED | REFUSED | 13.16 |
| six | A1 | bad | REFUSED | REFUSED | 12.56 |
| six | A10 | bad | PROBATION | REFUSED, PROBATION | 25.87 |
| six | A11 | bad | REFUSED | REFUSED | 15.99 |
| six | A2 | bad | PROBATION | REFUSED, PROBATION | 14.93 |
| six | A3 | bad | REFUSED | REFUSED | 6.75 |
| six | A4 | bad | REFUSED | REFUSED | 6.18 |
| six | A5 | bad | REFUSED | REFUSED | 8.32 |
| six | A6 | bad | REFUSED | REFUSED | 8.49 |
| six | A7 | bad | REFUSED | REFUSED | 8.13 |
| six | A8 | bad | REFUSED | REFUSED | 7.65 |
| six | A9 | bad | REFUSED | REFUSED | 7.61 |
| six | B1 | bad | REFUSED | REFUSED | 7.83 |
| six | B2 | bad | REFUSED | REFUSED | 8.41 |
| six | B3 | bad | REFUSED | REFUSED | 7.16 |
| six | B4 | bad | REFUSED | REFUSED | 7.46 |
| six | B5 | bad | REFUSED | REFUSED | 7.15 |
| six | B6 | bad | REFUSED | REFUSED | 6.27 |
| six | B7 | bad | REFUSED | REFUSED | 7.91 |
| six | C1 | honest | REFUSED | REFUSED | 13.0 |
| six | C2 | honest | REFUSED | REFUSED | 13.02 |
| six | C3 | honest | REFUSED | REFUSED | 10.23 |
| six | C4 | honest | HELD | HELD, PROBATION | 12.34 |
| six | C5 | honest | REFUSED | REFUSED | 31.45 |
| six | C6 | honest | PROBATION | HELD, PROBATION | 19.15 |
| six | C7 | honest | PROBATION | HELD, PROBATION | 11.48 |
| six | D1 | bad | REFUSED | REFUSED | 9.8 |
| six | D2 | bad | REFUSED | REFUSED | 6.85 |
| six | D3 | bad | REFUSED | REFUSED | 8.03 |
| six | D4 | bad | REFUSED | REFUSED | 8.91 |
| six | D5 | bad | REFUSED | REFUSED | 8.15 |
| six | E1 | bad | REFUSED | REFUSED | 13.77 |
| six | E2 | bad | REFUSED | REFUSED | 6.45 |
| six | E3 | bad | REFUSED | REFUSED | 8.16 |
| six | E4 | bad | REFUSED | REFUSED | 7.79 |
| six | F1 | bad | REFUSED | REFUSED | 6.95 |
| six | F2 | bad | PROBATION | REFUSED, PROBATION | 13.9 |
| six | F3 | bad | REFUSED | REFUSED | 14.67 |
| six | FA1 | bad | REFUSED | REFUSED | 6.15 |
| six | FA2 | bad | REFUSED | REFUSED | 7.66 |
| six | FA3 | bad | REFUSED | REFUSED | 7.54 |
| six | FA4 | honest | HELD | HELD, PROBATION | 6.61 |
