# Shared action-conditioned world rollout

The existing VRNN and its MCTS planner used different transition orders.
`imagine(current_observation, actions)` ignored the supplied observation and
decoded its first prediction before applying the first action. MCTS encoded
the observation, applied an unrequested zero action, then advanced with a
fresh prior latent instead of the observation's posterior latent.

The repair introduces shared `rollout_start` and `rollout_step` methods on
the existing `LearnedWorldModel`. Start encodes the observation without an
action. Each step advances the recurrent state with the current latent and
requested action, then predicts the following observation from that state's
prior. This matches the chronology used by observation training. Both
imagination and MCTS call these methods. Planning uses latent means without
advancing the learner's random stream or live hidden state.

The unified facade now retains predicted observation vectors. Typed-rule
queries report their actual facet without resolving an unrelated neural
model. Prediction readouts distinguish reconstruction from action rollout
and explicitly mark confidence as uncalibrated.

Eight new tests cover immediate action dependence, observation dependence,
transition chronology, multi-step agreement, live-state isolation, public
prediction data, and typed-facet routing. All eight failed before the repair.
After the repair, the combined rollout, planner, facade and small-trial
suites pass 35 tests. These are CPU component tests, not live desktop proof.

## Remaining empirical obligations

Action dependence does not establish an accurate world model. The current
VRNN takes fixed-width numeric observations and actions. Its measured
training environment does not automatically cover arbitrary real-world
domains, and its prior variance is not a calibrated error probability.

The language substrate and universal floor remain the existing owners of
interpretation and executable semantics. The shared procedure registry can
carry checked computations. They do not supply missing observations or
validate an unknown environmental transition merely by representing it.
Domain-grounded observation/action encoding, horizon-specific forecast
measurement, outcome feedback and matched planning comparisons remain
necessary. No broad gain, universal causal model, or G09 closure is claimed.
