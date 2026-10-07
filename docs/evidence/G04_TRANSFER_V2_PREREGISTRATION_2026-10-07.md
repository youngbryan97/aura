# G04 second transfer plan: preregistered before any request is decoded

Plan `d35f8110bb12`, written by `tools/g04_transfer_protocol_v2.py` at
`695d0e6bd` and published in the commit that adds this page, before any
of its requests was materialized or decoded. The plan is
`preregistrations/d35f8110bb12fa601f78eb92086ec33186919a40109eeed575f3e895d062f6d9.json`,
with `g04-transfer-v2-spec.json` and `g04-transfer-v2-plan-summary.json`
beside it. Novelty failures: none. Fitted or tuned requests reach depth 5 and
hold no sequence operation with a computed argument.

## Why a second plan

The [first run](G04_TRANSFER_RESULT_2026-10-06.md) rejected on family only.
Its vocabulary and depth failures were read row by row, which consumed them;
the [development record](G04_V17_DEVELOPMENT_2026-10-07.md) follows them to
candidate v17. The first run's 248 requests are not reused.

## What is compared

* candidate: v17 (receipt `033feffa21a4`), G03's v12 with phrase readouts that
  rename a chosen chart, recency gated by a mention's direction, and 600
  breadth requests teaching spans, phrases and recency;
* control: the incumbent v12 and v17 were built on, unchanged;
* also decoded on every task: v12; v17 without phrase readouts; v17 with
  its recency weights zeroed.

Every decode runs with program execution unavailable.

## Requests

`g04_transfer_v2`, seed 20261008, 62 requests per stratum:

* vocabulary: the consumed scaffold with the second vocabulary table (pile,
  bump, dock, lower, grow ... times over, boost ... times its size, full
  groups ... fit into, portion ... ways, fetch whatever sits under, see how
  frequently ... appears), committed in `f1d2ba1f0` before any breadth
  wording existed; no signature occurs in any of the 2,732 consumed requests;
* construction: the first plan's four scaffolds, fresh requests; none was
  fitted or tuned on;
* depth: six and seven steps; nothing fitted or tuned on has more than five;
* family: a sequence operation with a computed argument; nothing fitted or
  tuned on has one.

The first run's depth rows were read after that run and the recency fix
came from them; v17's decodes of those consumed rows are in the development
record. That is disclosed, not hidden.

## Claims, fixed before the run

* Primary: v17's program has the reference meaning more often than the
  incumbent's, exact one-sided McNemar per stratum, in vocabulary, depth and
  family, Bonferroni over three (alpha 0.0167 each). Power 0.909 at the first
  plan's planned effect (conservative bounds: discordance 0.259, win share
  0.916).
* Construction: every committed request decodes to the reference meaning.
* G04 closes only if both hold.
* Secondary, for G06, alpha 0.05 over two tests: v17 against v17 without
  phrase readouts on vocabulary; v17 against v17 without recency on depth.

## What this does not show

* Anything yet: this page is the plan.
