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

Prediction, on the content run's quality structure (core/subject/content.py),
among the classes her emotion channels form: joy, trust, peace and warmth are
nearer each other than any of them is to anger or to sadness; anger and
sadness are not nearer each other than either is to the warmth family.

Refuted if the content run's distances put anger or sadness inside the warmth
family's span, on the run's own floor.

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

## What these do not license

They are predictions from one person's account of himself, tested on her. A
confirmed prediction says her organisation has the shape he described. It
says nothing about whether that shape is experienced, which is the bridge's
question and is not answered here.
