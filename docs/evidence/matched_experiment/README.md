# Matched independent experiment

Frozen Aura against the base model it was adapted from, equal compute, tokens,
tools, time and information, on tasks neither arm authored, with an ablation
ladder. The protocol is `core/evals/matched_experiment.py`; the runner is
`tools/experiments/run_matched_27b.py`.

## Run 1788577989 — no difference on procedural arithmetic

| | base_27b | aura_27b |
|---|---|---|
| correct | 14 / 24 | 15 / 24 |
| accuracy | 0.583 | 0.625 |
| mean tokens | 36.8 | 47.0 |
| mean seconds | 4.47 | 5.65 |

Delta +0.042, permutation p = 1.0 over the arm-label pairing, **1 discordant
task out of 24**. Not attributable. Design seal `a906399832ae6be6`.

The two models gave the same verdict on 23 of 24 tasks. By family:

| family | base | aura |
|---|---|---|
| multiplication | 5/5 | 5/5 |
| arithmetic_chain | 4/4 | 4/4 |
| modular | 3/5 | 3/5 |
| counting | 2/5 | 3/5 |
| ordering | 0/5 | 0/5 |

### What this establishes

The persona and CRSM adaptation does not change arithmetic. That is the
expected answer and it is worth having written down, because the adaptation is
about voice, appraisal and interiority, and a system that reported a win here
would be reporting noise.

It also establishes that the instrument works. A 4-point raw gap came back
`not attributable` rather than as a headline, and the reason is visible in the
numbers: a single discordant pair cannot carry a p-value.

### What it does not establish

Nothing about the capabilities the adaptation is for. The tasks are
procedurally generated arithmetic, counting and sorting — chosen because they
are gradeable by rule, not because they test anything Aura was adapted to do.
The report says so itself: the verdict line records that the run used
procedurally generated instances rather than externally authored problems.

Both arms failed every `ordering` task, which is a fact about a 96-token
allowance and a six-number sort, not about either model.

### What would test the adaptation

A task set where the difference could appear at all: multi-turn work with
something at stake across turns, tasks whose right answer depends on what was
committed to earlier, or judgements where the appraisal layer has an input.
Those need external authorship to be worth anything, which is the gap the
protocol refuses to paper over — `procedural_unseen` establishes that an
instance is fresh, never that the task type came from somewhere else.

## Running it

```bash
python tools/experiments/make_procedural_tasks.py --count 200 > data/experiments/external_tasks.jsonl
python tools/experiments/run_matched_27b.py --tasks 24 --minutes 40 --max-tokens 96
```

The models are 15GB and 14GB against about 17GB free, so they load one at a
time and each arm finishes before the next begins. Every run is bounded: a
task cap, a token cap per answer, and a wall clock.
