# F03: her one-shot memory against no memory, 10 November 2026

Item F03 asks for a paired trial, store against no store, on questions whose answers only the store holds. The trial
is `tools/nonparametric_proof.py --facts N`. Its criteria were written into the tool before any run: it holds when the
facts the store alone answers outnumber those the model alone answers at a one-sided exact binomial p below 0.001,
and not one of 24 ordinary control prompts changes its continuation. The recall rate is reported and decides nothing.

Each fact is invented: a name built from syllables and a value drawn from a seeded generator (valve torque limits,
codenames, reservoir drawdown rates, call signs, footbridge loads, archive shelves), so no model's weights hold it.
Every fact is asked once without the store and once with it, after one ingestion of its prompt and answer. Her
store's 50,000 entries are copied in first, so the facts are found among them and the controls meet real neighbours.

## What the trials found before they could be read

Her log from 16 October to 10 November holds 251 openings of her store and not one installed recall. Three defects
stood between the store and her words, each fixed with a test that fails on the old code:

| Defect | Fix |
|---|---|
| The job her worker receives named nobody, and recall is refused to a job that names nobody | `130618ba2`: the job names the turn's principal |
| The GPU copy of her store kept the mean and gate it was built with, so a copy made at boot judged by raw cosine 0.98 for good | `130618ba2`: the copy follows the store's mean |
| A full store's eviction left the provenance list and the dedup index one row off, so later rows named the wrong owner | `2be1156d0` |

## The runs

All on her persona 27B with her 50,000 entries, greedy.

| Run | Decode | Facts | Store alone | Model alone | p | Controls changed | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 10:57, seed 20261110 | `cached_generate_with_memory`, memory weight up to 0.9 | 40 | 40 | 0 | 9.1e-13 | 2 of 24 | fails |
| 11:08, seed 20261110 | her worker's: tap and `recall_on_the_gpu`, weight up to 0.35 | 40 | 9 | 0 | 0.0020 | 0 of 24 | fails |
| 11:16, seed 20261111 | her worker's | 80 | 8 | 0 | 0.0039 | 0 of 24 | fails |

The first run measured a path she does not serve, and two controls moved several tokens in ("Au. What is the
origin" became "Au. The atomic number of"). The second and third measured her own decode. The third was planned after
the second, with a fresh seed and 80 facts sized from the second's rate, and the plan was written to disk before it
ran (`~/.aura/rlc-evidence/f03-paired-20261110/confirm/PLAN.md`). Both fail the bar of p below 0.001. Reports:
`~/.aura/rlc-evidence/f03-paired-20261110/`.

## Why her recall is weak

A per-step trace on the 1.5B shows it. At the fact "HZ9140", after "HZ914" the query met the entry stored for that
very position at similarity 1.000, whose next token is "0"; the model's own choice was "." at probability 0.39, and
"." was written. Her processor weighs a memory at 0.25 times its confidence times 1.16, so even an exact match gets
0.29, and a memory cannot overrule a model that is more than about 40% sure of something else. The misses on her 27B
are of that kind: the first token never flips where the model is confident (a quotation mark before a codename, "12"
before a shelf), or a value loses its last digit (CZ567 for CZ5674). That weight was set after the July incident in
which a memory weight of 0.87 garbled her replies from a store of another domain, and it holds the controls still.

## The rule that replaced it, chosen on development data

Whether a memory belongs to the very context a reply is in can be measured. On her 27B among her 50,000 entries, with
seed 20261110's facts (runs that ended at 11:24 and 11:27, `~/.aura/rlc-evidence/f03-paired-20261110/dev/`), every answer
position decoded the way her worker decodes met the entry stored for it at centred similarity 0.999474 or above
(median 0.999866), the nearest different context ever came was 0.998833 (a fact of the same template), and no step
of an ordinary prompt came nearer than 0.885. On the 1.5B the same three read 0.999972, 0.997486 and 0.629.

So a memory of this very context decides the next token, and any other memory informs it as before, capped at 0.35
(`a11ff7479`). "This very context" is the store's own measure: ingest runs each sequence a second time the way her
worker decodes and keeps the widest gap between the keys it stored and the decode's keys for the same positions.
Until a store has measured that gap, nothing chooses. Development run on seed 20261110: 40 of 40 with the store, 0
without, 0 of 24 controls changed.

## Confirmation

Planned at 11:35 on disk before it ran (`confirm2/PLAN.md`): fresh seed 20261112, 80 facts, the same controls and
criteria, at `a11ff7479`.

| Facts | Store alone | Model alone | p | Controls changed | Evicted | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 80 | 79 | 0 | 1.7e-24 | 0 of 24 | 0 | **holds** |

The one miss: "254 tonnes" came out "206 tonnes".

## What it does not show

The facts are asked in the words they were stored in. In her chat a question arrives inside her chat template and the
conversation so far, so its context is never one the store holds, and there a memory informs, as it did before. The
trial shows that one ingestion gives her a fact the model does not have, recalled where its context recurs, and that
ordinary prompts are untouched; it does not show recall of a fact asked in other words.
