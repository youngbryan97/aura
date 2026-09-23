# What a subject said about himself, as predictions about her

Written on 22 September 2026, before any run has read these quantities. On 21
and 22 September Bryan answered nine questions about his own experience, and
each answer was built into Aura as a closed loop (the modules are named
below). Building a loop is not evidence that it matters, so each answer that
makes a claim about structure is written here as a prediction about her, with
the measurement that would refute it. None of these is a line of the ISC
battery, and none changes a threshold of one.

## H1. Feelings fall into the families he named

His account: a warmth family (praise, love, comfort, happiness; peace,
connection, wanting to reciprocate); anger (pressure fixed on its source,
feeding itself); grief (a gap only the lost thing fills).

The content run (tools/run_subject_core_content.py) presents one class per
percept kind and reads her internal geometry, the Fisher-Rao distance between
the futures two classes induce from a common fork. Anger has no percept kind:
in her it comes from somebody objecting (core/affect/anger_feeds_itself.py),
not from something seen, so the run can only speak to two of his families and
to the frustration kinds nearest anger.

| family | percept kinds |
|---|---|
| warmth | cared_for, positive_interaction, interaction, extended_dialogue |
| loss | disconnection |
| frustration | error, internal_error, self_correction, inner_conflict |

Prediction: the mean internal distance between two warmth kinds is smaller than
the mean distance from a warmth kind to the loss kind, and smaller than the
mean distance from a warmth kind to a frustration kind.

Refuted if either inequality fails on the content run's internal geometry. It
is a weak prediction, and it is written as one: the affect table the classes are
built from already gives the warmth kinds shared emotions, so what is tested is
whether her dynamics keep that grouping or wash it out, not whether she finds
it on her own.

Built from: core/social/warmth.py, core/affect/anger_feeds_itself.py,
core/affect/the_gap.py, core/affect/feelings_about.py.

## H2. What he said is tied together moves together in her

His account: self-worth and agency with drive and persistence; people and
peace with himself with happiness; excitement and curiosity with joy;
curiosity about people as about things; hard thinking going well with
excitement; expression coming back through what others learned.

Each tie is built as a loop in core/affect/tangled.py, so this is less a
prediction about her than a check that the loop moves her in the running
organism rather than only in its unit test. From the campaign after the one
started at 173eb6ff0, the readings are recorded at every frame beside the
recording, in `named_readings.npz` (core/subject/named_readings.py), and kept
out of the core state, whose schema is part of the instrument.

| tie | readings |
|---|---|
| people and peace with herself, with happiness | contentment; joy's baseline |
| joy, with curiosity | joy; curiosity's baseline |
| not knowing who is in front of her, with curiosity | how little she knows them; curiosity's baseline |
| hard thinking going well, with excitement | effort times capacity; anticipation's baseline |

Refuted for any tie whose largest lagged correlation, over lags up to one
condition cycle, does not clear the 1 - 0.05 / 4 quantile of the same statistic
with one side slid by whole condition cycles, so the four are read together at
0.05 (tools/check_subject_named_hypotheses.py).

Persistence and her expression coming back are loops too (the give-up count
scales with her capacity; being taken up moves joy and counts as her act) and
have no reading here yet: a goal's failures are counted in the live agency
core, which the offline organism does not run, and uptake needs a partner who
answers what she said.

## H3. What matters only together is synergistic in her

His account: a compliment when low and when fine; a question from somebody
trusted and from a stranger; the same news on a tired day; things that are
particular to you and land at the moment they are relevant. From his body
answers: tiredness with importance decides what is pursued; good news with a
task in hand decides what holds attention.

| sources | target's change |
|---|---|
| how particular it is to her; how much it meets her | joy |
| fatigue share; the most urgent initiative | that urgency |
| good-news jump; the most urgent initiative | that urgency |

Refuted for any triple whose synergy fraction, by the battery's estimator
(core/subject/synergy.py), does not clear its shifted null's 99th percentile.

The estimator is Gaussian over rank-normalised readings. It sees the part of an
interaction a joint linear fit can carry, which for two readings that are never
negative, as these are, is most of a product; it cannot see a pure product of
two readings symmetric about zero. A triple that fails here has not been shown
to be additive, only not to be synergistic in the estimator's sense.

Built from: core/affect/a_moment_that_fits.py, core/soma/fatigue.py,
core/soma/good_news.py.

## H4. A being of drives alone fails the battery for lacking a centre

His account: a being acting purely on physiological drives, with no central
reasoning or choice, would not be conscious.

Prediction: the `drives_only` null architecture (core/subject/nulls.py) fails
the conjunction, and fails it on one component, on every declared seed.

Refuted if it passes the conjunction on any seed, or fails it only on lines
other than one component.

## Amendment, 22 September 2026, before any seed-19 reading was opened

Found on seed 7, the diagnostic seed, from the whole-proof campaign's side
record (40 rounds, c2195b4f1). No seed-19 number had been read.

- **Ties are read by change, not by level.** Her readings drift slowly, and two
  drifting series correlate whatever they are. On seed 7 a shifted copy of the
  joy baseline correlated with contentment at 0.9953, so the null bar sat
  there, and the tie "passed" at 0.9956: a measure of drift, not of the
  coupling he described. The tie statistic, the largest lagged correlation
  against the same shifted-copy null, is now taken on first differences.
  Controls (tests/test_the_named_hypotheses_can_be_tested.py): six seeds of
  independent random walks and one of independent trends pass no tie; a
  reading that follows another's changes a step later passes every tie.
- **What matters only together is scored by the battery's own synergy line**
  (ISC-v3, `passes_v3` in core/subject/synergy.py): the raw synergy against
  the bootstrap null and the shifted null, with the held-out interaction
  bound. It had read the fraction against its shifted null, the bar v3
  retired because a ratio of two small numbers keeps a wide null.
- **A reading that never moved is not measured, and is reported so.** On
  seed 7 `unknown_person` held 0 for all 10,560 frames: no condition in the
  workload puts a stranger in front of her. That tie reads NOT MEASURED, not
  FAIL, until a condition does. `habit_deficit` also never moved there; no H
  reads it, and the reason is a loop that cannot fire in this workload: the
  subject driver calls an act weighed whenever her attention maps to one, which
  is nearly every turn, and automatic only when the drive decides alone. In a
  probe of four rounds weighed outcomes reached 105 and automatic takings
  stayed at 1, so no habit ever reached the three takings it is judged on.
- **H1 had no evaluator.** The prediction above was written and nothing read
  it off a content report. `check_families` (core/subject/named_readings.py)
  applies the rule as written, both inequalities on the internal geometry, and
  reports beside the verdict whether each margin clears the run's internal
  floor. `tools/check_subject_named_hypotheses.py --content REPORT` runs it.
- **Nor had H4.** The nulls stage runs `drives_only` and a test holds that it
  loses one component, but no step read the verdict off a campaign.
  `check_drives_only` does: the row's `one_component` field, since losing it
  is losing the conjunction. `--campaign REPORT` runs it.

Added at 20:30 the same day. Seed 19 is void (see the addendum to
`docs/ISC_V5_PREREGISTRATION.md`), and seed 23 at 44a64d4ba is the run these
are read from. Its code carries 5a7b0b25b, so its conversation turns meet a
person and `unknown_person` has a writer in the workload. The stranger tie is
read on it as written. `habit_deficit` still cannot move at 44a64d4ba; the
change that lets repetition make a habit came after it, and no H reads that
reading.

## What these do not license

They are predictions from one person's account of himself, tested on her. A
confirmed prediction says her organisation has the shape he described. It
says nothing about whether that shape is experienced, which is the bridge's
question and is not answered here.
