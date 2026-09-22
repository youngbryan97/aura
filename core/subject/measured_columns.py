"""The columns the battery is measured on, and why they are pinned.

Every domain is reduced to four principal components before anything is
predicted, and `_basis` standardises each column to unit variance before the
singular value decomposition. So a channel that barely moves is scaled up to
the same variance as one that carries the domain, and four directions have to
cover however many channels the schema declares.

That makes the schema part of the instrument. Between 45a74c913 and e22e89192
it grew from 210 columns to 315 as each batch of organ readings was added to
it, and the live width of four domains roughly doubled -- D from 13 to 23, A
from 16 to 33, C from 12 to 29, S from 17 to 30. Nothing about the organs is
wrong and none of this is about them: they still gate what they gate, and the
readings are still on the state where their own tests find them.

What it did to the measurement is not small. On a synthetic source that
genuinely drives its target, with the same estimator and the same four
components, adding weakly-varying columns to the source domain takes the
variance explained from 0.7027 to:

    added    explained
        0       0.7027
        4       0.3208
        8       0.1791
       16       0.1260
       24       0.1022
       32       0.0392

D gained 31 columns over that window and A gained 22.

**What this does and does not explain.** It bears on everything read through
the reduction -- irreducibility, the cut search, the surrogate comparisons --
because those all run on four components per domain.

It does not explain the interventional edges, and the first version of this
file said it did. An edge effect is not a fit. `_paired_divergence` takes each
column's peak displacement over the lags, picks the column whose margin over
its own sham floor is largest, and reads the effect off that one column. That
is a maximum over candidates, so another column can only raise it; the module
says as much where it says an existence claim must not be a function of the
schema's granularity, and it is right. A->I falling from 8.9583 at eight
conditions of eight in run_031 to 0.4307 at two of eight in the campaign after
has some other cause, and the campaign run on this pinned set is the test of
whether the schema was any part of it.

So the instrument is pinned here, to the 210 columns of 45a74c913 -- the commit
run_029, run_030 and run_031 were read on, and the one the thresholds were
preregistered against. Changing what a measurement is taken on while chasing a
number on it is the thing this file exists to stop. Adding a column is now a
deliberate edit to this list, with the campaign that justifies it.

`build_recording` keeps these and drops the rest. `full=True` records
everything, which is for looking at a new reading rather than for scoring one.
The intervention arms are built the same way, so an effect is measured on the
same columns the baseline was.
"""

from __future__ import annotations

__all__ = ["MEASURED_COLUMNS", "PINNED_AT"]

#: The commit run_029 (18/24), run_030 (19/24) and run_031 (18/24) were
#: recorded on.
PINNED_AT: str = "45a74c9130dbf50c9dc8d2ec6c010a2892ea072d"

#: In schema order, which is the order a frame is built in.
MEASURED_COLUMNS: tuple[str, ...] = (
    "P.percept_load",
    "P.percept_claim",
    "P.percept_sources",
    "P.percept_unfelt",
    "P.percept_strength",
    "P.spatial_present",
    "P.objective_len",
    "P.objective_profile_0",
    "P.objective_profile_1",
    "P.objective_profile_2",
    "P.objective_profile_3",
    "P.percept_profile_0",
    "P.percept_profile_1",
    "P.percept_profile_2",
    "P.percept_profile_3",
    "I.cpu",
    "I.vram",
    "I.temperature",
    "I.thought_ms",
    "I.perception_lag_ms",
    "I.token_velocity",
    "I.pulse_rate",
    "I.mycelium_density",
    "I.vitality",
    "I.exertion",
    "I.recall_effort",
    "I.sensor_load",
    "A.valence",
    "A.arousal",
    "A.curiosity",
    "A.engagement",
    "A.social_hunger",
    "A.free_energy",
    "A.emotion_joy",
    "A.emotion_trust",
    "A.emotion_fear",
    "A.emotion_surprise",
    "A.emotion_sadness",
    "A.emotion_disgust",
    "A.emotion_anger",
    "A.emotion_anticipation",
    "A.emotion_curiosity",
    "A.emotion_frustration",
    "A.heart_rate",
    "A.cortisol",
    "A.adrenaline",
    "A.momentum",
    "A.action_urgency",
    "A.surprise_trend",
    "A.surprise_trend_known",
    "A.mood_baseline",
    "A.mood_baseline_spread",
    "G.attention_present",
    "G.attention_profile_0",
    "G.attention_profile_1",
    "G.attention_profile_2",
    "G.attention_profile_3",
    "G.coherence",
    "G.fragmentation",
    "G.contradictions",
    "G.conversation_energy",
    "G.discourse_depth",
    "G.branch_load",
    "G.selfhood_readings",
    "G.ignition",
    "G.ignited",
    "G.candidates",
    "G.winner_priority",
    "G.winner_source_0",
    "G.winner_source_1",
    "G.winner_source_2",
    "G.winner_source_3",
    "G.tick",
    "G.tie_impasses",
    "G.inhibited",
    "G.broadcasts",
    "G.modifier_temperature",
    "G.modifier_depth",
    "G.modifier_creativity",
    "G.modifier_focus",
    "G.modifier_vitality",
    "G.modifier_urgency",
    "C.mode_reactive",
    "C.mode_deliberate",
    "C.mode_dreaming",
    "C.mode_dormant",
    "C.loop_cycle",
    "C.phi_estimate",
    "C.substrate_valence",
    "C.substrate_arousal",
    "C.substrate_dominance",
    "C.substrate_energy",
    "C.substrate_volatility",
    "C.substrate_focus",
    "C.substrate_curiosity",
    "C.substrate_frustration",
    "C.substrate_revision",
    "S.stability",
    "S.evolution_score",
    "S.bonding_level",
    "S.narrative_version",
    "S.narrative_len",
    "S.value_load",
    "S.preference_load",
    "S.trait_openness",
    "S.trait_conscientiousness",
    "S.trait_extraversion",
    "S.trait_agreeableness",
    "S.trait_neuroticism",
    "S.belief_count",
    "S.belief_version",
    "S.snapshot_count",
    "S.pending_updates",
    "S.belief_profile_0",
    "S.belief_profile_1",
    "S.belief_profile_2",
    "S.belief_profile_3",
    "S.agency_acted",
    "S.agency_efficacy",
    "S.agency_authored_share",
    "S.agency_capabilities",
    "S.agency_actor_self",
    "S.agency_actor_user",
    "S.agency_actor_world",
    "S.agency_actor_system",
    "S.agency_actor_other",
    "S.prediction_error",
    "S.prediction_surprises",
    "S.valence_error",
    "S.drive_error",
    "S.focus_error",
    "S.least_predictable_affect_valence",
    "S.least_predictable_drive_state",
    "S.least_predictable_attentional_focus",
    "S.least_predictable_other",
    "S.prediction_confidence",
    "S.agency_score",
    "S.agency_traces",
    "S.agency_pending",
    "S.agency_attribution",
    "S.agency_attribution_known",
    "M.working_load",
    "M.working_newest_is_user",
    "M.retrieved_load",
    "M.retrieved_profile_0",
    "M.retrieved_profile_1",
    "M.retrieved_profile_2",
    "M.retrieved_profile_3",
    "M.recall_score",
    "M.working_profile_0",
    "M.working_profile_1",
    "M.working_profile_2",
    "M.working_profile_3",
    "M.summary_len",
    "M.ledger_load",
    "M.thread_present",
    "M.cold_memory_load",
    "M.evolution_log_load",
    "W.entity_load",
    "W.relationship_load",
    "W.fact_load",
    "W.preference_load",
    "W.fact_profile_0",
    "W.fact_profile_1",
    "W.fact_profile_2",
    "W.fact_profile_3",
    "W.concept_load",
    "W.user_trend",
    "W.user_trend_known",
    "W.model_surprise",
    "W.model_hidden_norm",
    "W.model_mean_surprise",
    "W.model_last_surprise",
    "W.model_steps",
    "W.causal_nodes",
    "W.causal_edges",
    "W.causal_confirmed",
    "W.model_facets",
    "W.model_train_steps",
    "D.goal_load",
    "D.initiative_load",
    "D.initiative_urgency",
    "D.goal_urgency",
    "D.goal_profile_0",
    "D.goal_profile_1",
    "D.goal_profile_2",
    "D.goal_profile_3",
    "D.origin_is_user",
    "D.action_source_0",
    "D.action_source_1",
    "D.action_source_2",
    "D.action_source_3",
    "D.drive_curiosity",
    "D.drive_energy",
    "D.drive_growth",
    "D.drive_integrity",
    "D.drive_social",
    "N.novelty",
    "N.displacement",
    "N.steps",
    "N.era",
    "N.hidden_norm",
    "N.hidden_0",
    "N.hidden_1",
    "N.hidden_2",
    "N.hidden_3",
    "N.hidden_4",
    "N.hidden_5",
    "N.hidden_6",
    "N.hidden_7",
)
