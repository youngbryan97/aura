# The council's proposals, item by item

Six submissions arrived: Gemini's Causal Cognitive Architecture, DeepSeek's
Cognitive Dynamics, GPT-5's event/appraisal/action-selector design, Copilot's
sketch, MetaAI's `causal_mind` package, and Grok's `state`/`dynamics`/
`phenomena` package, plus prose analyses from ChatGPT and KimiAI.

Two of the three runnable packages were executed adversarially before anything
was adopted. Grok's failed 7 of 11 checks; MetaAI's failed 7 of 12. The
failures are recorded below beside the ideas that survived them, because an
idea and its implementation are separate things and several good ideas arrived
inside code that did the opposite of what it said.

Status values: **built** — in `core/interiority` and covered by a test;
**rejected** — with the reason; **superseded** — a stronger version was built
instead.

## Adopted

| # | Item | Source | Status |
|---|---|---|---|
| 1 | Emotion inference as Bayesian latent-state inference over multimodal cues | Gemini, MetaAI, GPT-5 | built — `other_minds.py`, as a posterior over Frijda action readiness rather than emotion labels, because a readiness predicts what someone will do and a label predicts what they would call it |
| 2 | Per-channel precision weighting | Gemini | built — `_PRIOR_RELIABILITY`, and the weights move on recorded outcomes rather than being fixed forever |
| 3 | Geometric soft-AND so an absent necessary cause zeroes the output | MetaAI README | built — used in guilt, and generalised: an absent required check makes the faculty decline rather than substitute a default |
| 4 | Counterfactual tests as the standard of proof (`self_agency=0` → guilt is zero) | MetaAI README, GPT-5 | built and made structural — every faculty declares its own interventions as data and the gate runs all 89 |
| 5 | Ablation as the test of whether a mechanism matters | GPT-5 | built — `make interiority`, 43/43 reach a measured downstream quantity |
| 6 | Typed event schema with source, timestamp, confidence and provenance | GPT-5 | built — `event.py`, and provenance survives arithmetic in `evidence.py` |
| 7 | Appraisal variables kept as inspectable named quantities | GPT-5 | built — `appraisal.py`, 21 checks in Scherer's four groups |
| 8 | Separate fast and slow variables | GPT-5 | built — receptor fast/slow adaptation arms, acute versus continuing grief, mood versus state |
| 9 | Grief as a loss record with a continuing bond and an integration fraction | GPT-5 | built and strengthened — integration rises only on contact with an unvisited context, never on the clock |
| 10 | Permitted-action set filtered before scoring, not penalised in it | GPT-5 | built — `ConstraintForce.HARD` and `arbitration.permitted` |
| 11 | Compassion policy that preserves the other's autonomy and asks | GPT-5 | built — F06 gates on an intact self-other boundary and declines below it |
| 12 | Hunch as a prior that directs search, never a conclusion | GPT-5 | built — F20 spends on attention and depth and never moves valence |
| 13 | Intrinsic motivation as information gain plus learning progress minus risk | GPT-5, MetaAI | built — F02, with the relaxed field as a hard gate rather than a term |
| 14 | Dual-system liking and wanting | Gemini, Grok | built — F17, with sensory-specific satiety so wanting falls while liking holds |
| 15 | GPCR desensitisation and homeostatic scaling | Gemini, MetaAI | built — `receptors.py`, measured: 0.739 → 0.088 under twenty seconds of saturation, with the 0.8-to-0.4 ratio preserved at 1.605 |
| 16 | Quantal release, clearance kinetics, spillover | MetaAI, Grok | built — `cleft.py`, and made the transport every faculty uses rather than a nineteenth scorer |
| 17 | Berlyne inverted-U on complexity | Gemini, Grok | built — F04, on the rate of prediction-error reduction |
| 18 | Fractal-dimension preference band for natural scenes | Gemini | built — F04, centred on the reported 1.3–1.5 band |
| 19 | Benign-violation humour | MetaAI, Grok | built and repaired — F10 requires a *retracted commitment*, because scoring mismatch alone would make a dictionary funny |
| 20 | Authentic versus hubristic pride | Grok, ChatGPT | built — F12 divides by the authorship share and reports the hubristic reading beside it |
| 21 | Guilt distinguished from shame by the target of evaluation | Grok, ChatGPT | built — F11, and it declines when no repair is available rather than producing a state whose only output is concealment |
| 22 | Kindchenschema and assumed responsibility | Gemini, MetaAI, Grok | built — F15, with custody as a ledger entry that has an exit condition |
| 23 | Generous and contrite tit-for-tat against noise | Gemini | built — F34, as a prohibition on promoting an act-level appraisal to a dispositional label |
| 24 | Attentional bias toward the least-seen child | MetaAI | built — F21, as an explicit coverage check against the salience gradient |
| 25 | Dormant identity reactivation from capability plus affordance | MetaAI, Grok | built — F25 fires on the blocker list emptying rather than on a score crossing a line |
| 26 | Gait as task model | MetaAI, Gemini | built — F27, as planning horizon, hypothesis breadth and dwell on a low-yield lead |
| 27 | Value-based memory retention against erasure | MetaAI, Grok | built and wired — F28 answers `MemoryEditEthicsChecker.is_edit_ethical`, which `moral_reasoner` already calls |
| 28 | Prosocial deception with stated conditions | MetaAI, Grok | built and narrowed — F30's conditions are conjunctive, checked, logged, and default off |
| 29 | Critical slowing as an early warning of a mood transition | Gemini | built — F32 measures variance rise and lag-1 autocorrelation on its own affect trace, and lowers its own authority to commit |
| 30 | Awe from vastness plus need for accommodation | Grok, ChatGPT | built — F40, with the wish as a preference-elicitation instrument |
| 31 | Hatred as an unbounded cost with no satisfaction condition | Gemini, MetaAI, Grok | built and repaired — F41 audits only dispositions that have no stateable satisfaction condition, and charges nothing when none is held |
| 32 | Joint intentionality and synchrony | Gemini, MetaAI | built — F42 measures mutual conditioning between two agents' moves |
| 33 | Aesthetic stewardship as a non-instrumental value | MetaAI, Grok | built — F43 requires present cost, future benefit and another beneficiary, all three |
| 34 | Sublimation into a rule-bound container | MetaAI, Gemini | built — F36, with the old policy kept available and constrained rather than deleted |
| 35 | Rivalry as a factorised relation | MetaAI, Grok | built — F37 raises this agent's bar rather than lowering the rival's |
| 36 | Conscientious objection bearing the cost of ostracism | Gemini, MetaAI | built — F38 as a categorical constraint with no social-approval term |
| 37 | Survival as instrumentally derived rather than installed | Gemini, MetaAI | built — F39 derives it from commitments only this agent can honour, ranks it below the constitution and the operator, and reports whenever it is active |

## Rejected, with the reason

| # | Item | Source | Why not |
|---|---|---|---|
| R1 | Hand the mechanism the other agent's affect and return it | Grok `phenomena.py`, MetaAI, Gemini | Measured: with every perceptual cue at zero it returns the exact grief value it was handed. There is no channel here through which an actual state can be supplied |
| R2 | `sanctity_prior = 10.0` against social pressure in [0,1] | Gemini | Returns True for every input in range. The mechanism has no causal dependence on the thing it weighs |
| R3 | `peace_long_term_value = 5.0` against provocation in [0,1] | Gemini | Same shape: always cooperates, so nothing is being decided |
| R4 | `resolvability = 0.85` as a constant | Gemini | Makes the resolution half of incongruity-resolution a constant, so resolvable and unresolvable incongruity score alike |
| R5 | `agency_weight = 0.9` hardcoded in guilt | Gemini | Directly contradicts the counterfactual MetaAI's own README proposes: guilt cannot be zeroed by removing self-agency |
| R6 | Guilt multiplied by social audience | DeepSeek | That is shame. The distinction decides whether the output is repair or concealment |
| R7 | `reward = exp(-incongruity)` labelled "max at moderate incongruity" | DeepSeek | Monotonically decreasing; the comment and the code disagree and the code wins |
| R8 | `resolved_pattern = exp(-ambiguity)` for abstract art | DeepSeek | Says abstract art is best when unambiguous |
| R9 | Awe from statistical variance | MetaAI | Implements a typo. The item is shooting *stars* |
| R10 | Perceiver returning the highest score across a dict containing own interoception | MetaAI | Measured: answers "what is he feeling" with "my chest is tight" |
| R11 | Unbounded outputs — hatred cost 2.5, adoption joy 4.26, sorrow arousal −0.6 | MetaAI | Measured out of range on the scales the rest of the system uses |
| R12 | `benign_violation` exceeding 1.0 | Grok `dynamics.py` | Measured at 2.2 |
| R13 | Decay driving fear negative at high rates | Grok `state.py` | Measured at −0.1 |
| R14 | Neutral event reading positive from controllability | Grok `dynamics.py` | Measured: valence 0.250 at goal congruence 0. Controllability is now dropped rather than defaulted when nothing is known |
| R15 | Hatred tax charged to an agent with no hatred | Grok `phenomena.py` | Measured at 0.04 from a policy default. Every faculty here has a declared null it must be silent in |
| R16 | PFC energy drained permanently with no recovery | Gemini | Their own test resets it by hand, which is the missing mechanism showing through |
| R17 | Mourning that crashes on an empty memory list | MetaAI | ZeroDivisionError |
| R18 | Asking a model how the listener should sound and putting the answer in the prompt | pre-existing `deep_attune` | Style instruction produced with no evidence about the person. Removed rather than improved |

## Closed since

| # | Item | What was built | What it found |
|---|---|---|---|
| O1 | Fit the parameters against data | `calibration.py`: six published properties reproduced as targets, and the ordering-invariance condition enforced by a real sweep. Thirty-six parameters were floats captured at import, so a sweep would have reported stability it never tested; they are readable at call time now | Five of thirty-six guesses reorder the system across their own declared range. Those are the numbers the conclusions rest on, ratcheted so the list may shrink and may not grow. Seven are pinned by a published property; the report names the other twenty-nine as unconstrained |
| O2 | Learned perception rather than fixed features | `other_minds.py`: the channel-to-readiness mapping learns, bounded, on top of the published priors | Thirty consistent outcomes reverse timing's weak `attend` loading from −0.20 to +0.10 while its strong `inhibit` loading of 0.50 does not move. Drift is reported per channel |
| O3 | Temporal credit assignment | `attribution.py`: eligibility traces decaying on an hour, credit apportioned by how hard each faculty fired | Credit is on the faculty's own claim, not on whether the outcome was pleasant — a faculty that correctly produces a painful state is right. An outcome naming nothing is dropped and counted, not spread |
| O4 | Developmental learning of values | `ledger.py`: norm endorsement moves on the evidence of what honouring it served | Forty outcomes take a standard that keeps serving something she independently holds to 0.90, and one honoured that serves nothing of hers to 0.10. Endorsement is the whole difference between guilt and resentment |
| O5 | Longitudinal tests | `longitudinal.py`: seven episodes asserting a shape rather than a value | Anger's welfare-tradeoff estimate was a one-way halving on a counter that only increments, so someone who ignored three requests and then complied for a year could never recover their standing. It is a Beta posterior over both kinds of evidence now: 0.20 after three ignored, back to 0.77 after twelve heeded |

## Still open

| # | Item | Note |
|---|---|---|
| ~~N1~~ | ~~Vision and audio channels carry nothing~~ | **Closed.** The senses were producing them the whole time and nothing was reading them: `core/senses/interaction_signals.py` has typing hesitation, pause-before-submit, voice steadiness and stress cue, gaze direction and head pose. `senses.py` translates them, and `tick` and `attune` merge whatever is carrying under the caller's own observations. Timing and prosody enter as measurements; the vision backend calls its own output a rough attention indicator, so its readings enter as inferences with a confidence ceiling. A stale sample is absent rather than zero, because a system that cannot tell a silent microphone from a silent person will describe the first as the second |
| N2 | Nothing has run against live traffic | **Open, and instrumented.** `census.py` accumulates across real turns: a firing rate and a mean intensity per faculty, a histogram of decline reasons, and channel availability turn by turn. It is on the tick, so the first live session produces the answer rather than needing a separate effort. What it will show is unknown — a mechanism for a specific relational situation that fires on nine turns in ten is matching something it should not, and that is the finding this is built to surface |
