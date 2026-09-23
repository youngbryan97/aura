# G03 search interruption and source-trained ranker, 2026-09-23

Status: development evidence only. No serving authority, fresh transfer claim,
or G03 closure.

## Causal repair

The diagnostic candidate bank previously resumed search only after the exact
`decode_search_budget_exhausted` refusal. Other argument-solver interruptions
returned no diagnostic alternatives even though the same target-blind grammar
could find them. `SemanticTransductionOutcome.search_interrupted` now carries
that *typed* condition from the decoder. The public refusal remains unchanged.
Candidate search may continue after an interrupted ordinary decode and can skip
one interrupted chart without discarding later charts. A pre-search grounding
failure still does not enter candidate search. Tests cover these distinctions.

Source-bank acquisition used the frozen 764 training examples, current 27B
representation, four charts, two graphs per chart, and one second per graph.
The old bank reached 518/764 verified-equivalent programs, with 246 rows
unavailable to alternative search. The revised bank reached 764/764. This is
bounded candidate reach on the training source population, not public-answer
correctness, search completeness, or proof of a universally correct compiler.
The ordinary selected program was equivalent on 104/764 in the revised replay;
the earlier bank recorded 168/764. Timed ordinary search can vary, so those
selected counts are not a paired quality comparison of the repair.

Artifacts:

- Old bank: `~/.aura/rlc-evidence/semantic-identity-evidence-bank-full-4x2-20260923/report.json`
- Revised bank: `~/.aura/rlc-evidence/semantic-interruption-recovery-bank-full-4x2-20260923/report.json`
- Stratified 42-row repair pilot: `~/.aura/rlc-evidence/semantic-search-interruption-recovery-42-20260923/report.json`

## Source construction folds

The three frozen construction folds each held out disjoint source groups.
Training used only the other folds. Both arms saw the same retained candidate
programs and evidence-path multiplicity; the control did not use the argument
evidence values. Candidate labels entered only after target-blind bank
generation. No validation or test labels trained either arm.

| Fold | Population | Correct reachable | Ordinary selected | Matched control | Argument evidence | Same-checkpoint evidence lesion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 256 | 256 | 85 | 186 | 198 | 102 |
| 1 | 254 | 254 | 11 | 203 | 212 | 161 |
| 2 | 254 | 254 | 8 | 167 | 204 | 138 |
| Total | 764 | 764 | 104 | 556 | 614 | 401 |

The evidence arm beats its matched control by 58 rows in aggregate, but there
are 140 evidence-only wins and 82 control-only wins. Rows share construction
groups; a row-level binomial p-value would overstate independent replication.
The lesion establishes that the trained channel influences decisions, not that
its influence is uniformly correct. Family-level results expose the problem:

| Family | Population | Ordinary | Control | Evidence |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | 320 | 78 | 208 | 251 |
| Cataphoric | 48 | 26 | 32 | 27 |
| Counterfactual | 36 | 0 | 21 | 32 |
| Fork/join | 192 | 0 | 152 | 170 |
| Natural alias | 24 | 0 | 21 | 23 |
| Natural source | 48 | 0 | 39 | 44 |
| Reserved alias | 48 | 0 | 41 | 36 |
| Role binding | 48 | 0 | 42 | 31 |

The evidence arm regresses against the control in cataphoric, reserved-alias,
and role-binding families. It is not promotable from this source-only comparison.
Fold reports and weights are under
`~/.aura/rlc-evidence/semantic-recovery-full-fold{0,1,2}-{treatment,control}-20260923/`.

## Exposed validation gap

A new target-blind bank over the 23 previously exposed validation failures
reached a verified-equivalent program on 18/23. Five are *unresolved* under
incomplete chart/graph search; they are not proved unreachable. Applying the
three source-trained evidence checkpoints without fitting validation labels
selected 3/23, 9/23, and 13/23 respectively. These are diagnostic results on
an exposed slice, not fresh replication. Their spread and the five missing
candidates prevent candidate promotion and G03 closure.

The same 23 requests were replayed with 12 charts and four graphs per chart,
still without target access during proposal generation. Candidate reach rose
to 23/23; the five former misses were bounded-search misses, not an algebraic
impossibility. The *unchanged* fold-2 ranker selected only 6/23 on this wider
bank, down from 13/23 on the 4x2 bank. Its training distribution did not
contain that larger candidate set. This is evidence against treating search
coverage as answer quality, and motivates matched-width source training rather
than tuning against the exposed validation answers.

Artifacts:

- `~/.aura/rlc-evidence/semantic-recovery-exposed-gap-bank-20260923/report.json`
- `~/.aura/rlc-evidence/semantic-recovery-exposed-gap-treatment-fold{0,1,2}-20260923.json`
- `~/.aura/rlc-evidence/semantic-recovery-exposed-gap-bank-12x4-20260923/report.json`
- `~/.aura/rlc-evidence/semantic-recovery-exposed-gap-12x4-treatment-fold2-20260923.json`

Next, train/evaluate at matched search width on source-only construction folds
and measure a selection rule that does not sacrifice alias and role-binding cases.
Only a frozen candidate can proceed to broad validation, public answer checks,
fresh transfer, and the later G-ledger claims.
