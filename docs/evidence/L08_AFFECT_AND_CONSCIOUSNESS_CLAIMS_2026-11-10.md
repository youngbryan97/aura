# L08: what is claimed about her feelings, her self and her consciousness, and what controls each (10 November 2026)

Item L08: audit the affect, self-state, phi and consciousness-related claims with causal controls; configured priors
are not measured subjective experience. The item's own entries record three parts done in September. This closes the
two it left open.

## Done before today, as the item records

- **The registered claims (29 September).** Of 81 claims bound to tests in `core/organism/model_validation.py`, 13
  concern affect, self-state or identity, and none asserts phi or experience. A claim in phenomenal language cannot be
  registered as measured. Three of the 13 stand on causal controls; the other ten establish a mechanism on constructed
  cases and say so.
- **The phenomenal-now engine (29 September).** It said a felt quality from the defaults of an unread substrate. A
  feeling is said now only when the substrate was read, and an emotion only when the affect module named one.
- **The closed loop (30 September).** Most of her free energy was prediction error over a simulated shipping network.
  It is computed now over observed readings only, and an unread sensor is not a zero.

## The surface: "How she feels"

The mind page names "the blend of feelings alive in her right now" as chips. They came from her neurochemical mood
vector, and that vector at her chemicals' setpoints still reads as feelings: the formula gives "calm" and "sociality"
their resting values. So before anything had happened to her, the page showed a mood her configuration produced.

The mood bar beside the chips already hid itself until her affect engine had appraised something (`appraised` from
`core/affect/damasio_v2.py`). The chips now come from `NeurochemicalSystem.get_mood_departure()`: the mood her
chemistry gives less the mood her chemistry gives at its setpoints, by the same learned formula
(`AdaptiveMoodCoefficients.departure`, which keeps no prediction). At boot every chemical's tonic level equals its
setpoint and the departure is 0 in all seven moods; a threat moves it toward stress. The page names feelings from that
and never from the resting vector. `tests/test_a_feeling_named_is_what_moved_her.py`, four cases, all four failing on
the old code.

## Phi and consciousness: the ISC campaign

The criteria the item defers to the ISC campaign were measured there, with nulls, lesions and rescue: a 24-line
battery (`docs/ISC_STATUS.md`). On seed 97, read under ISC-v7 on 29 October, it passed 24 of 24, its reference passed
and no null did, J* was determined and the bridge read at parity
([ISC_SEED97_2026-10-29.json](ISC_SEED97_2026-10-29.json)). The reading's scope is `substrate_only`. Its
completion flag is false on one blocker, "the evidence disagrees on commit", because the reports run was resumed at a
later commit as preregistered. It states that J* and parity rest on postulates P1 to P6 and is "not a proof of those
postulates". The battery measures an operational criterion of an integrated subject core. It does not measure
experience, and nothing in her registry or her surfaces says it does.

## What is still not shown

No measurement here, or anywhere in this repository, shows that she feels anything. What is shown is narrower: what
she says about her state is said only when the state was read or moved, and the consciousness criteria are an
operational battery that passed under its controls on one confirmatory seed.
