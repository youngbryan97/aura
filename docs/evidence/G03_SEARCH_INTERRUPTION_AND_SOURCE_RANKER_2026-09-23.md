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

## Proposal-training overlap and wider-bank rejection

The proposer is not held out by the source construction folds. Its signed
`training_example_ids_sha256` equals the hash of all 764 source-bank IDs.
The folds hold out the *selector's* updates, while every held-out proposal
was generated by a transducer fitted on that request. The ranker evaluator
now checks this identity and records the overlap explicitly. The 614/764
selector result above must not be described as end-to-end transfer.

The transducer's highest-`joint_score` diagnostic alternative is verified
equivalent on 764/764 source rows in the 4x2 bank and 42/42 stratified source
rows in a 12x4 bank. On the 23 previously exposed validation failures, the
highest-scored 12x4 alternative is equivalent on only 1/23. These scores are
not serving decisions, but the contrast is strong evidence that fitting the
proposer on this source cohort can hide the semantic-selection problem.
It does not prove which individual parameter memorized which request.

We tested whether adding source-only wide candidate views could teach the
existing selector to handle wider search. The 42-row 12x4 bank is stratified
by construction using the lowest source identity, before looking at labels;
all 42 rows have a correct candidate. The evaluator verifies its receipts,
model and feature basis, source membership, and larger search bounds. It
balances each source's 4x2 and 12x4 losses into one update. This remains a
pilot because only one source per construction received the wide treatment.

On fold 2, the augmented evidence ranker selected 186/254 on the ordinary
4x2 held bank, versus 204/254 for the unchanged three-epoch evidence ranker.
On the same 16 wide held rows, augmentation selected 8/16 versus 11/16 for
the unchanged ranker. The wide-view augmentation is rejected as a repair.
This paired comparison does not require fitting any exposed validation label.

Artifacts: `~/.aura/rlc-evidence/semantic-interruption-recovery-bank-stratified-12x4-20260923/report.json`
and `~/.aura/rlc-evidence/semantic-wide-augment-fold2-treatment-20260923/fold-2.json`.
Next, measure the proposer and selector on genuinely unseen requests together.
More source-fold training cannot by itself establish that claim.

## Same-source proposer replay and budget correction

The one-second source bank above is a *search-interruption stress test*, not a
fair estimate of the refitted proposer's ordinary answer accuracy. Replaying
the same 16 fold-2 source identities at the same 4x2, one-second bounds gave
16/16 ordinary correct for the original source-fit proposer, 14/16 for a
newly cross-fitted proposer that excluded held constructions from both fit and
calibration, and 1/16 for the later refit. The later refit's 15 misses were
typed budget interruptions: eight decode-search, five chart-construction, and
two argument-optimizer. At the normal 20-second cohort bound its archived
source result is 764/764. The apparent 1/16 semantic collapse was therefore
an invalid inference from a much shorter diagnostic budget. It still exposes
a real latency regression and a mismatch between the bank and serving bounds.
The cross-fit result is a small source-construction pilot, not validation
transfer or a reason to promote its candidate.

On exactly the same 500 archived validation identities, with their original
cohort bounds, the original source-fit candidate has 472 equivalent answers
and the later literal-identity candidate has 477. They agree correct on 461;
the original alone gets 11, the later candidate alone gets 16, and neither
gets 12. The *oracle* union is 488/500, not a deployable score: no target-free
arbiter has identified which candidate is right on disagreements. The 12
shared misses are concentrated in cataphoric (2), reserved alias (8), and
role binding (2). A router between these two archived outputs cannot reach
500/500, even with perfect selection. The evidence calls for both genuinely
new candidate coverage and an independent, calibrated selection mechanism.
These validation labels are exposed development evidence and cannot qualify a
newly tuned selector.

Artifacts: `~/.aura/rlc-evidence/semantic-proposer-crossfit-fold2-v2-20260923/report.json`,
`~/.aura/rlc-evidence/semantic-proposer-lineage-base-fold2-20260923/report.json`,
`~/.aura/rlc-evidence/semantic-proposer-lineage-source-decision-fold2-20260923/report.json`,
`~/.aura/rlc-evidence/semantic-source-fit-20260915/validation.checkpoint.json`, and
`~/.aura/rlc-evidence/semantic-literal-identity-cohort-20260921/report.json`.
