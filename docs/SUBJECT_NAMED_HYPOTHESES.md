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

Prediction, on a campaign recording: each named pair covaries more than the
same channels shuffled against each other in time (the replay surrogate the
battery already builds), at the battery's own level:

| pair | her channels |
|---|---|
| agency, persistence | agency efficacy; goals still pending after a failure |
| contentment, happiness | warmth peace and self peace; joy's baseline |
| joy, curiosity | joy above its rest; curiosity's baseline |
| not knowing a person, curiosity | 1 - confidence in the partner; curiosity |
| effort going well, excitement | fatigue effort times capacity; anticipation |
| taken up, joy | carried times added; joy |

Refuted for any pair whose lagged cross-correlation does not clear its replay
surrogate's 95th percentile.

Built from: core/affect/tangled.py.

## H3. What matters only together is synergistic in her

His account: a compliment when low and when fine; a question from somebody
trusted and from a stranger; the same news on a tired day; things that are
particular to you and land at the moment they are relevant. From his body
answers: tiredness with importance decides what is pursued; good news with a
task in hand decides what holds attention; risk with the other's reaction
decides how it lands.

Prediction: in each named triple the target carries information about the
pair that neither source carries alone, by the synergy estimator the battery
uses (core/subject/synergy.py), above its shifted null:

| sources | target |
|---|---|
| how particular it is to her; how much it meets her | the moment's weight's effect on joy |
| fatigue share; initiative importance | which initiative is kept |
| good-news jump; task urgency | the arbiter's lightness |
| risk of what she said; their next frustration | the form ledger's weighting |

Refuted for any triple whose synergy fraction does not clear its shifted
null's 95th percentile.

Built from: core/affect/a_moment_that_fits.py, core/soma/fatigue.py,
core/soma/good_news.py, core/soma/on_the_edge.py.

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
