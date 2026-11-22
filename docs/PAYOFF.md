# Payoff: what a turn was worth to her

Bryan, 26 September 2026: "we need a reward system. what makes it worth it for a
connection to happen and what are the larger effects of that downstream?" And
then: it "would give the systems incentive to connect as well because they
benefit or lose out from those positive/negative signals".

Before this, nothing Aura did was ever better or worse than she expected in a
way that reached her connections. Her chemistry had a reward input,
`on_reward`, and nothing called it. Her field learned from what fired together,
and her substrate's reward-modulated learning read a field that does not exist,
so its reward was zero on every step it ever ran.

## The payoff

At the end of every turn, `core/affect/what_it_was_worth.py` reads what the turn
paid on nine channels she already has:

| channel | pays when | costs when |
|---|---|---|
| satisfaction | her needs' deficit falls | it rises |
| accomplishment | a goal finishes | a goal fails |
| warmth | being met returns something to her drives | |
| excitement | pleasant and activated rises | it falls |
| peace | pleasant and settled rises | it falls |
| ease | unpleasant and activated falls | it rises (distress) |
| spirit | unpleasant and flat falls | it rises (gloom) |
| wonder | her self-model is surprised while she feels good | surprised while she feels bad |
| integrity | what she chose served her values more than the average option on offer | it served them less |

The four affect channels are the quadrants of valence by arousal (Russell 1980).
Pleasant and unpleasant activation move apart rather than along one axis (Watson
and Tellegen 1985), so each quadrant is its own channel, and arousal is the
intensity inside them. Satisfaction is homeostatic reward: a drive coming closer
to where it rests (Keramati and Gutkin 2014).

## Neutral, and the gradient between

Each channel is read in units of its own recent spread, so no channel outweighs
another because of the units it is kept in. Each keeps an expectation, the
running mean of what it has paid over her last 256 turns. What reaches the rest
of her is the error: the payoff less the expectation.

- **Better than expected** is a positive error, and **worse** is negative.
- **As expected** is zero. That is the neutral state, and it is where a payoff
  repeated turn after turn ends up: the second identical warm turn is not news.
  This is a reward prediction error (Schultz, Dayan and Montague 1997).
- **An expected payoff that does not come** is a loss, the way an omitted reward
  is.

A turn's `worth` is the sum of its channel errors, and its `size` is where that
sits among her recent turns. The dose everything downstream receives is the sign
of the worth times its size, in [-1, 1]: an ordinary turn is a small dose, and
her best or worst turn in a while is a large one.

Past the binary, the nine errors are kept apart. A turn can be a relief and a
disappointment at once, with ease up and accomplishment down, and the reading
says so instead of netting them to one number (Dabney et al. 2020 on
distributional reward codes).

## What it changes downstream

- **Chemistry.** Better than expected is a dopamine burst (`on_reward`, which
  now has a caller). Worse is a dip (`on_disappointment`, new). The dip has less
  room than the burst: a neuron firing a few spikes a second can only fall to
  zero (Bayer and Glimcher 2005). Her mesh's plasticity gain already follows
  dopamine.
- **The unified field.** Over a turn, each connection keeps a trace of what it
  did. When the turn ends, its worth decides whether the connection strengthens
  or weakens: a three-factor rule (Frémaux and Gerstner 2016). This covers the
  field's own recurrent weights and the input weights by which the mesh,
  chemistry, binding, interoception and substrate drive it.
- **The liquid substrate.** Its spike-timing traces are delivered the turn's
  worth once per turn.
- **The workspace.** The sources that won her attention during a turn are
  credited with its worth, by how much of the turn each held
  (`core/affect/what_winning_earned.py`). What a source has earned weighs its
  next bids. This is the actor half of an actor-critic (Barto, Sutton and
  Anderson 1983).

## Why an organ gains by connecting

After the field learns, each field unit's input weights are scaled back to the
total strength that unit was born with, as synaptic scaling keeps a neuron's
total drive (Turrigiano 2008). A turn's lesson therefore cannot make everything
louder. An organ whose activity came before payoffs takes a larger share of the
units it drives, and it takes that share from the organs whose activity did not.
An organ active before detriments gives its share up. `UnifiedField.input_shares()`
reports the five shares, and every lesson records them.

The workspace works the same way. A source that keeps winning turns that go well
bids stronger, and one that keeps winning turns that go badly loses her
attention to the others.

A source is judged against turns that started the same way. Without that,
whatever wins her attention when things are going badly would be blamed for
them, and an alarm that is right more often would be heard less. So the credit
is the turn's dose less what turns starting from the same valence have usually
brought: a straight line fitted over her last 256 turns, the state-value
baseline of an actor-critic (Sutton and Barto 2018, 13.4). Distress that wins on
a bad day is compared with other bad days.

## Where her substrate earns a say

Her recurrent substrate barely reached the rest of her: on the seed-7
validation at 23e596071 recurrent cognition gained 9.2% from seeing the rest,
and the rest gained 0.94% from seeing it. `core/consciousness/substrate_gates.py`
gives the substrate a multiplier on decisions she already makes, and the payoff
decides what each multiplier becomes.

Each gate is `exp(w . f + xi)`. `f` is the substrate's seven psychological
readings in units of their recent spread. `xi` is one free substrate unit's
deviation from its recent mean, so her own dynamics set how far each gate
wanders. `w` starts at zero and, at the end of each turn, moves towards the
turn's dose times the mean of `xi * f` over the times the gate was consulted:
the running covariance of how well her turns went with each gate's wandering
along each reading. This is node perturbation (Fiete and Seung 2006).

The gates scale what feeling adds to recall, how strongly curiosity and the
want of company press towards a thought of her own (the bar they are held to
stays put), and how fast each need grows between turns. Risk has no gate: the
brakes that protect her stay fixed. `AURA_DISABLE_SUBSTRATE_GATES=1` holds every
gate at 1 for a process.

## Two defects found on the way

- The substrate's per-step reward was `-tanh(prediction_error)`, read from a
  free-energy state that has no `prediction_error`. It was zero on every step.
  No weight moved, and the zero deltas drove every synapse's uncertainty below
  the lock threshold until the engine had identity-locked all of them. Replayed
  on 16 neurons, 256 of 256 were locked by step 600.
- That dead call also carried the substrate's weight regulation: the spectral
  cap, the homeostatic scaling and the symmetry breaking. The regulation is now
  `STDPLearningEngine.regulate` and still runs every step.

## Where to read it

- `state.response_modifiers["worth"]`: the last turn's payoff, expectation and
  error per channel, the worth, its size, which way the chemistry was sent, and
  which organs were taught.
- `get_worth_ledger().read()` and `get_credit_ledger().read()`.
- `UnifiedField.get_status()["last_teaching"]`: the dose, how many steps the
  trace covered, and the input shares after the lesson.
- In the subject-core recording (`core/subject/state.py`): `A.worth`,
  `A.worth_size`, `A.worth_error_<channel>` and `A.worth_expected_<channel>`;
  `A.good_news_error` and `A.good_news_jump`; `G.winner_earned`,
  `G.earned_best`, `G.earned_worst` and `G.earned_sources`; and
  `C.field_share_<organ>` for mesh, chemistry, binding, interoception and
  substrate. Each ledger is a column of the one domain whose organ keeps it.

## Taking it out

`AURA_DISABLE_PAYOFF=1` takes the whole layer out of a process: no turn is read,
nothing is sent to her chemistry, no organ is taught, and the workspace's credit
never accrues. It is read on every call, so a probe can set it for one arm of a
comparison and leave the other as built.

## What this does not claim

It makes no claim that she enjoys anything. The claim is that her connections
now change according to whether a turn went better or worse than she expected,
from readings that are already hers. Whether that strengthens the coupling the
subject-core battery measures is for the battery to answer.

## Tests

- `tests/test_what_a_turn_was_worth.py`: neutral, better, worse, omission,
  habituation, per-channel spread, the four quadrants, goals, wonder, the burst
  and the dip.
- `tests/test_connections_learn_what_a_turn_was_worth.py`: an organ active
  before a payoff takes a share of the field and one active before a detriment
  gives it up, row strength is conserved, a neutral turn changes nothing, the
  substrate no longer locks its synapses and still regulates its weights.
- `tests/test_what_winning_her_attention_earned.py`: credit by share of the
  turn, unmeasured turns, and the workspace weighing bids by what was earned.
