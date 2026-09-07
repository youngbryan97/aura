# Inherited obligation reconciliation

Review checkout: `f1f1bd450`. Review date: 2026-09-06.
Owner: Codex. This review does not authorize release or change historical evidence.

## Scope and completion

The ten sources in `tools/reqproof/inherited.py` contain 57,868 lines,
725 checkbox occurrences and 66 unchecked occurrences. Six unchecked boxes
are inline in three agency rows. Checkbox counting alone misses prose,
requirements tables and outstanding campaigns under closed Atlas cards.

All 66 unchecked occurrences now have source-hashed mechanism mappings in
63 source blocks. This closes the checkbox mapping subtask only. Prose and
table reconciliation remain open, and evidence-needed rows are not completed
implementation claims.

The source-linked inventory is generated from
`config/inherited_ledger_reviews.json`. Decisions bind exact source hashes;
changed text requires another review. The requirement registry and its
verified docket remain authoritative for their 313 requirements.

I01, I02 and I03 remain open until all unresolved obligations have a reviewed
mapping. The extraction includes historical material that has not all been
semantically reconciled. An unreviewed block cannot count as closed.

## Decisions supported so far

| Original obligation | Reconciliation | Evidence and remaining boundary |
| --- | --- | --- |
| Generality J1 | Stale unchecked status; later historical closeout closes it | `dd5e4880f9bce471b455c42d7c509d736cc4d376`, `docs/GENERALITY_TODO.md:757`; not a new live certificate |
| Generality J3 | Same later closeout closes the canned finding defect | Same commit and source; other empty-answer failures remain open |
| Generality J4 | Later closeout closes six measured locking sites | Same commit; does not close R06 or subsequently observed contention |
| Generality J5 | Named regression now passes | `tests/test_server_conversation_lane.py::test_api_chat_desktop_nonexecuting_decision_question_blocks_desktop_task` on review checkout |
| Generality H3 and K1 | One context-scaffold mechanism, two retained observations | `docs/GENERALITY_TODO.md:159` and `:793`; complete-answer live replay remains required |
| Endogenous 18.16 | Tests exist and pass, but acceptance scope needs adjudication | `4a9bf7ece`, `tests/test_what_the_name_bought.py`; head-versus-leaf comparison does not establish literal inline expansion in every representation |

The focused reconciliation run passed 17 tests in 25.25 seconds: eleven
inventory tests, five name-versus-body tests including the generated-family
measurement, and the named J5 regression. No model or live runtime was used.

## Mechanism map under review

These groups preserve the original obligations. Grouping does not erase a
different acceptance condition or convert a historical result to current proof.

| Mechanism | Original obligations | Master queue | Next action |
| --- | --- | --- | --- |
| Context size and complete answers | Generality H3, K1; execution tracker current empty-answer observations | R05, R09, R11 | Source-matched complete-answer replay with context accounting |
| Desktop request interpretation | Generality J5; learned-language sites and surfaces | U01, U06, L04 | Preserve passing regression; qualify learned surfaces separately |
| Background outcome custody | Generality J2, K2 | R09, L06 | Check disconnect and autonomous receipts against current callers |
| Resident allocation | Generality J6 | Q03, R11 | Measure actual lane selection and resource admission |
| Action abstraction | Generality A7; agency 9.2, 9.3, 9.5, 9.6, 9.8, 9.9 | L07 | Trace construction, evaluation, installation and persistence separately |
| Repeated capability improvement | Generality C7, E2, E3; Atlas 167; gauntlet D3, D10 | L03, L04, L07 | Verify held-out benefit rather than installation counts |
| Independent developmental transfer | Endogenous J and 18.2-18.18; gauntlet D4-D11 | G04, G06-G09 | Retain each null and independent-family condition |
| Adversarial learning and retention | Endogenous 0.6, 19.8-19.12 | L03, Q04 | Audit admission and run corruption, forgetting and bypass controls |
| Inline/body control | Endogenous 18.16 and checked 6.4 | G06, L04 | Adjudicate representability and the narrower measured comparison |
| Coding comparison | Atlas 156; gauntlet B7 | G12, U07 | Independent coding battery with reported scaffold and compute |
| Desktop benchmark | Atlas A1.5, A3.7, A4.6; gauntlet B6 | U05, U06, G12 | Preserve each benchmark protocol while sharing launch infrastructure |
| Desktop fault recovery | Atlas A2.12 | U05, U08 | Seed faults and score recovery from user-visible outcomes |
| Screen and steering | Atlas A2.18; steering qualification | L01, U05 | Current-checkpoint causal comparison and no-regression controls |
| OS and design capability | Atlas A1.16 | U06, U07 | Seeded tasks with artifact-quality acceptance |
| Broad capability board | Atlas A3.8, A8.10; gauntlet B5, B12, B14, D7 | G12, U08 | Preserve domain scores and external baselines |
| Productive endurance | Atlas A8.9, A8.11, A8.12; gauntlet B8; historical closeout longevity | Q10 | One source-matched campaign may satisfy multiple obligations only if it measures each bar |
| Video-rate tracking | Atlas 133 | U03 | Real video-rate measurement, not image-only unit tests |
| Complexity and causal contribution | Generality U5-U7; gauntlet A12, C2, D12-D16 | L08, Q06 | Re-measure complexity and independent lesions on current source |
| Learned semantic decisions | All twelve learned-language surfaces and six debt-site rows | L04, U01, U06 | Current data, calibration, abstention and caller evidence per surface |
| Connectome acquisition and analysis | Connectome waves 1-2 | Q06, L08 | Separate tested reconstruction/statistics from live functional claims |
| Connectome forecast | Connectome wave 3 | L08, Q08 | Require real held-out run in addition to synthetic test coverage |
| Connectome dynamics | Connectome waves 4-5 | L08, Q06 | Verify runtime consumers and each stated null |
| Connectome integration | Connectome wave 6 | R08, Q08 | Check telemetry ownership, export and release gates |
| External reproduction and installation | Gauntlet B18, D6; historical CLOSEOUT fresh install and external reviewer | Q11, Q12 | Independent clean-environment receipts |
| Normative requirements | All 313 registry requirements and their original source references | Q06, I06 | Use existing verified docket; do not invent replacement requirement IDs |

Source documents: [generality](GENERALITY_TODO.md),
[agency](AUTONOMOUS_DEVELOPMENTAL_AGENCY_TODO.md),
[endogenous expansion](RECURSIVE_ENDOGENOUS_EXPANSION_TODO.md),
[gauntlet](AGI_GAUNTLET_TRACKER.md), [Atlas](gap_atlas/TODO.md),
[connectome](CONNECTOME_TODO.md),
[language](LEARNED_LANGUAGE_INTERPRETATION_TODO.md),
[worktodo](worktodo/TODO.md), [historical closeout](evidence/CLOSEOUT.md),
[execution tracker](AURA_EXECUTION_TRACKER.md).

## Reconciliation still required

- Complete exact source-span decisions for prose and normative requirements;
  the map above is not the exhaustive I03 certificate.
- Resolve all 26 connectome checkboxes against their individual implementation,
  test, runtime and real-run conditions. The earlier 72-test pass included
  inventory and connectome suites; it did not establish every wave's acceptance.
- Reconcile the Atlas's 14 outstanding records independently of its 414 closed
  cards. A closed mechanism with an outstanding campaign remains outstanding.
- Diagnose why twelve previously certified requirements lack currently
  qualifying evidence in the fresh docket. Missing current qualification is
  not evidence that their code has disappeared.
  Confirmed example: SHUTDOWN-001 and CTX2-SHUTDOWN-001 receipts bind older
  `core/runtime/runtime_hygiene.py` and `interface/routes/system.py` bytes.
  Their stale-source rejection is reproducible; do not rewrite the old hashes.
- Compare later generality and agency work against their earlier unchecked
  entries; preserve partial acceptance rather than substituting test existence.
- Reconcile current language-substrate consumers against the older debt table.
- Keep historical CLOSEOUT and execution records immutable; record supersession
  here and in the hash-bound decisions, with original links.
