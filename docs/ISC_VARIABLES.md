# ISC variables, with examples

Status: Guide · Updated 25 October 2026

This page translates the symbols in the ISC documents. Start with
[ISC explained](ISC_EXPLAINED.md) for the overall idea. For the measured
results, use [current status](ISC_STATUS.md).

The numbers in the worked examples below are invented to explain the
arithmetic. They are not additional Aura experiment results.

The [scientific source map](SCIENCE_AND_SOURCES.md#measurement-and-statistical-interpretation)
links methods references and explains their role. For example,
[Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)
address finite, randomly drawn permutation calculations. The actual ISC
statistics and decision rules remain those in its versioned protocol; a
methods citation does not validate a comparison with failed controls.

## State: the information available at one moment

**State** means the values and records the software carries now: what is in
memory, which goals are active, what has attention, and the variables that
influence its next computation.

Think of a saved game. The save must contain more than a picture of the
screen: it needs position, health, inventory and anything else that changes
what can happen next. The comparison is about saving relevant information;
it does not imply that Aura is a game or that all its state is already known.

| Variable | Plain meaning | Practical use |
|---|---|---|
| `t` | The moment or step being discussed | Separates what the system carries now from what happens next |
| `K_t` | The selected core's recorded state at step t | The information the test claims is enough to describe the core |
| `K_(t+1)` | That core's next recorded state | What the prediction and intervention tests try to explain |
| `E` | Declared input from the environment | Lets the test account for the situation rather than crediting the core for every outside influence |
| `X` or periphery | Other internal information outside the selected core | Used to ask whether an important internal cause was left out |
| State column | One recorded measurement | A number the analysis can compare across moments; its meaning depends on its declared reader |
| Schema | The list and meaning of the recorded columns | Prevents two differently measured systems from being treated as the same experiment |

## The ten lettered domains

The letters name groups of measurements. They are not ten little people or ten
separate language models.

| Letter | Technical name | Ordinary meaning | An example of what could change |
|---|---|---|---|
| P | Perception | What has just arrived through the senses or input | A new event becomes available to interpret |
| I | Interoception | The system's reading of its own operating condition | Resource strain changes how much work it can sustain |
| A | Affect | Variables representing a more positive, negative or activated condition | A changed condition alters attention or recall |
| G | Global workspace | The current competition for attention and broadcast | One concern wins and becomes available to other parts |
| C | Recurrent cognition | Internal computation that feeds back into later computation | A state continues to shape processing over several steps |
| S | Self-state | Records and predictions concerning the system itself | Uncertainty about its own performance changes a choice |
| M | Active memory | What has been retrieved or kept available now | A remembered event changes the interpretation of an input |
| W | World model | Its current representation of the situation | A new observation revises an expectation |
| D | Deliberation | Goals, choices and intended actions | One task gains priority over another |
| N | Ontogeny | State carrying development over time | Earlier processing leaves a lasting change |

The same symbol can mean different things in different equations. `S` names
self-state in `A,S → G`; in the carrier mathematics it can instead name a
set of components. Read its definition beside the equation.

## Valence and arousal

**Valence** is an internal positive/negative coordinate. **Arousal** describes
activation or intensity. Their names identify the role of a software variable;
they do not, by themselves, establish that the condition is felt.

Consider a mood-dependent memory search. The words in a request stay the same,
but a change in the internal condition changes which memories come back. The
practical claim is that the variable affects retrieval. Whether that is a
feeling requires an additional argument.

A thermostat offers a limited comparison: its current reading affects what it
does next. Aura's measurements cover many interacting variables and learned
representations, rather than one temperature. The comparison explains causal
use of a value; it does not settle which systems have experience.

## Joint effects: what the fork letters mean

The synergy comparison restores the same moment several times:

| Symbol | What happens in that copy |
|---|---|
| `N1`, `N2`, `N3`, `N4` | Leave the chosen sources unchanged |
| `F` | Change the first source |
| `S` | Change the second source; here S means the second fork, not the self domain |
| `B` | Change both sources |
| `delta` or Δ | The configured intervention size; its practical units depend on the domain's writer |

Suppose the target is one number. It reads 10 unchanged, 12 after the first
change, 13 after the second, and 19 after both.

The separate effects are +2 and +3. Simply adding them would predict 15.
The combined result is 19, leaving an extra effect of 4:

```text
interaction = B - F - S + N1
            = 19 - 12 - 13 + 10
            = 4
```

A recipe is a useful comparison: the effect of heat can depend on whether
yeast is present. Measuring each ingredient separately misses their joint
effect. The comparison explains interaction; it does not mean every
interaction satisfies ISC or indicates consciousness.

| Quantity | What it says | Worked example |
|---|---|---|
| Main effect | How much one source changes the target on its own | +2 or +3 |
| Interaction | What remains after subtracting both separate effects | +4 |
| Norm | The size of a change when the target has many coordinates | For one coordinate, it is the absolute size |
| Interaction/main-effect ratio | Joint change compared with the larger individual change | 4 / 3 = 1.33 |
| Matched sham | The same arithmetic over unchanged copies | Four identical readings of 10 give 10 - 10 - 10 + 10 = 0 |
| Direction consistency | Whether the joint change tends to point the same way across moments | A positive effect that repeatedly reverses may fail this bar |

Aura's actual targets are vectors, not single numbers. The experiment also
changes coordinates so differences are measured against ordinary variation.
That is **whitening**. It is like comparing changes in each instrument's own
units of variation before combining them; it does not erase the need to know
which physical or computational quantities were recorded.

## Closure: whether the chosen boundary leaves something out

Imagine describing a car's next movement from its speed and steering angle
while leaving out the driver's brake input. A predictor given the missing
input can do better. That is a boundary problem in the description.

ISC makes a related comparison: predict the next core state from the core and
declared environment, then ask whether other internal information improves
that prediction. The metaphor concerns missing causes. A software core and a
car do not share every property.

| Variable | Meaning | Practical consequence |
|---|---|---|
| Prediction loss | How much a prediction misses | Lower is better for that prediction task |
| Leak | The share of the remaining prediction error saved by outside internal information | A detectable improvement suggests the chosen core may omit a useful cause |
| Surrogate floor | A comparison after relevant relationships are disrupted | Helps detect apparent improvement caused by trends or the estimator |
| Resolution | The smallest planted contribution this method can detect on these data | A result below that size may remain unanswered |
| Closure margin | The largest remaining contribution the claim permits, set by a predeclared rule or number | Lets “closed enough for this stated claim” have a testable meaning |
| Lower / upper bound | Limits estimated around the quantity | The upper bound must be below the declared margin for a positive bounded-closure claim |

For an invented example, a test might be able to detect contributions as small
as 0.02, while a predeclared acceptable margin is 0.03. If the upper bound is
0.01 and no other required check fails, the test can support closure within
that margin. If the method can only detect 0.05, it cannot establish the
0.03 claim by failing to see a contribution.

**Current method, through `96487c6f1`:** `CLOSURE_MARGIN` is `0.02 / 1.02`
(about 0.0196078); `CLOSURE_MARGIN_RULE` remains `None`. The earlier least-domain
rule stays withdrawn. The night addendum adopts a conventional small-effect
margin; that convention is not a measured boundary for consciousness.
Closure also requires an interventional `CLOSED` result on the scored seed.
See [the current method and source limits](ISC_STATUS.md#the-night-addendum-include-what-the-transition-reads-then-test-both-ways).

The estimator now predicts core change from declared inputs, enters periphery
columns as increments, and refuses to use a failed fold to establish closure.
A failed predictor is not evidence of a closed boundary. Read the
[withdrawal record](evidence/ISC_CLOSURE_WITHDRAWAL_2026-10-25.json) and
[formula guide](FORMULAS.md#closure-margin-and-decision). The earlier numerical
margin example above is illustrative arithmetic, not a newly selected threshold.

The experiment's clock is also distinct from wall-clock waiting. A two-turn
comparison can be defined as 66 simulated processing frames even when the
loaded machine takes much longer to compute them. Do not read that horizon
as a measured conversational response time.

## The other numbers in the results

| Quantity | How to read it | What it does not say |
|---|---|---|
| `rho` | A rank correlation: near +1 means two sets of changes tend to have the same ordering; near -1 means opposite ordering | The fraction of experience or the probability of consciousness |
| `p` | How unusual the test statistic would be under the specified null comparison and its assumptions | The probability that Aura is conscious or that the entire hypothesis is false |
| `n` / anchors | How many comparison moments were sampled; check how many were actually accepted | That every recorded answer was usable |
| `alpha` in a statistical test | The test's declared decision level | Steering strength |
| `alpha` in steering | How strongly the experimental direction is applied under that implementation | Statistical confidence or a percentage of emotion |
| `tau` / horizon | How far into the future the comparison runs | The age of the whole system |
| `g` / grain | How much detail the state description keeps | A guarantee that the chosen description preserves every relevant cause |
| `Phi_FR` | The initial carrier design's Fisher–Rao distance between future distributions | The current paired-displacement cut score; the method changed |
| `F_intrinsic` | In the carrier specification, a rate of partition-sensitive change for an admissible support | A direct meter of experience or the battery's 24-check count |
| `J*` | A combined description of carrier, related contents and causal history | A scalar synergy score to maximize |

## Numbers in a report should change what we do

A readable-answer count tells us how much of a self-report experiment the
conclusion covers. A failed sham tells us to repair the comparison before
celebrating a cut effect. A leak identifies a boundary to investigate. An
unresolved result means that a required answer is still missing.

Those practical consequences are why the definitions matter. The
[reproduction guide](ISC_REPRODUCTION.md) explains how to retain them beside
the source files and measurements.

## K*, held state and the two closure tests

**K** is the original ten-domain core. **K*** is K plus the declared attributes
that its transition depends on in the discovery intervention. **Held state**
means those attributes are counted on the core's side of the boundary rather
than as outside predictors. The source stores these readings under `held.`.

For example, a display can show a decision while a separate history buffer
helps produce it. Recording the display alone omits part of the process.
K* adds identified dependencies to the candidate record; the analogy does not
establish that this record includes every possible cause.

**Observational closure** asks whether outside readings improve prediction
beyond the permitted margin. **Interventional closure** asks whether the
specified outside-state swaps actually change K or held state at the tested
anchors and two-turn horizon. Both are required by the current rule. Passing
neither is implied by having written the code that measures them.
