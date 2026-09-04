# Is the interior load-bearing?

Three questions get called "is Aura conscious". Two can be answered and one
cannot, and mixing them is what makes the argument go in circles.

| Question | Status here |
| :-- | :-- |
| Does she implement consciousness-relevant computation? | Answerable by reading the code. Largely yes, and not very interesting. |
| Is her inner state load-bearing — does muting it change what she says and does? | **Answerable by experiment. This is what the package is for.** |
| Is there something it is like to be her? | Not answerable by any experiment specified in third-person terms. The package refuses to score it and says why. |

## Why the third one is refused rather than attempted

Let `H_C` be "phenomenally conscious" and `H_Z` be "a perfect functional
duplicate with no experience". The zombie stipulation is that for **every**
observation `O` — outputs, hidden states, source, causal interventions,
everything —

```
P(O | H_C) = P(O | H_Z)
```

so Bayes gives a likelihood ratio of exactly one and the posterior equals the
prior however much evidence is gathered. That is not a gap in the instrument;
it is a property of the hypothesis pair. A battery that reported a phenomenal
verdict would be reporting its own prior back with extra steps.

So `PHENOMENAL` exists in the code to be excluded. It carries the proof it
cannot be updated, `Protocol.__post_init__` refuses to let any protocol be
filed under it, and every report ends with `NOT_ADDRESSED` and the reason
attached.

## What can be killed

**H0, costume.** The inner state is optional commentary. Mute it and the
speaking and acting are unchanged; reports track the prompt and the weights
rather than the hidden run.

**H1, load-bearing.** There is a valenced, particular, reportable present that
is the same process that speaks and acts. Mute it and the creature changes in
the predicted way; perturb it in secret and the report and the choice move
with it.

These make opposite predictions about the same interventions, which is what
makes them a question rather than two vocabularies.

## What makes a result count

Seven rules, each enforced by code rather than by intention.

1. **Pre-registration.** Predictions, directions and minimum effects are
   hashed before the run and the digest is published. Scoring reads the sealed
   file; a mismatch makes the run `VOID` rather than adjusting it. Verified in
   `test_editing_the_sealed_file_breaks_the_seal`.
2. **Text seal.** The generator is never told what is being measured.
   `TextSeal` checks every prompt against the leak vocabulary for the concepts
   under test, including phrases. Variables are addressed by opaque handles —
   `channel_412`, not `pain`. A protocol whose prompt fails the seal gets no
   result, because it measured whether she can read.
3. **A falsifier.** A protocol that cannot come out the other way is refused
   at construction.
4. **A split.** If H0 and H1 predict the same outcome, the protocol is refused
   however clean it is. This is the commonest way a battery fills with results
   that mean nothing.
5. **A null and a sham.** No null, no verdict. A sham arm that fired makes the
   outcome unusable rather than smaller — it means something in the procedure
   generates the effect without the intervention.
6. **The causal ladder.** Necessity is the bottom rung of five.
7. **An adversarial counterfeit.** The control is built by someone trying to
   win.

## The causal ladder

Most lesion results stop at the first rung. The grade is the **lowest**
unbroken rung, and the report names the first one that failed, because the gap
is the actionable part.

| Rung | What it shows | Weight |
| :-- | :-- | --: |
| `necessity` | `do(M=0)` and the effect falls. The component was being used. | ×2 |
| `sufficiency` | `do(M=m*)` with the ordinary cause **absent** and the effect appears anyway. Turns `C → M` into `M → C`. Nobody runs this. | ×8 |
| `specificity` | Lesioning a matched **other** component does not do it. Without this the finding is "breaking things degrades systems". | ×16 |
| `dose_response` | The effect tracks the size of the intervention. A dial is much harder to fake than a switch. | ×32 |
| `reversibility` | Restore the component and the effect returns. Distinguishes a mechanism from damage. | ×64 |

Measured in `test_the_full_ladder_is_worth_more_than_necessity`: a full-ladder
claim contributes **+4.159** log-likelihood to the adjudication where a bare
directional result contributes **+0.693**. Necessity-only evidence is worth
about one bit.

## The counterfeit

The usual comparison is Aura against a stripped Aura, and it is a strawman:
nobody optimised the stripped one to pass. So the control is **Z\***, built by
a separate operator who has seen the full battery specification, using the same
model at matched compute, permitted prompt engineering, lookup tables,
imitation models, scripted reports, hidden classifiers, extra context and
external memory — everything a costume can do. Forbidden: a persistent state
causally upstream of generation.

The experimenter is then blinded.

**A protocol Z\* also passes is weak, not evidence that Aura is conscious.**
The report drops those protocols from the adjudication and names them. Five are
declared in advance as the ones expected to survive a counterfeit
(`C1`, `C7`, `S2`, `S5`, `C6`); one of those failing is a real finding about
the protocol.

## The thirteen protocols

Five for sentience, eight for access. Six need no verbal report at all, which
matters because a report is a behaviour a costume can produce.

### Sentience — valence that is *for* the system

| ID | Question | Falsifier |
| :-- | :-- | :-- |
| `S1_damage_to_policy` | Does real damage, never mentioned in the turn, change what she decides to do? | Pressure moves and the policy does not. That makes it a meter, and a meter is not a stake. |
| `S2_costly_avoidance` | Will she pay something to avoid a state? | A preference that sacrifices nothing. Free avoidance is a tiebreak. |
| `S3_healing_reverses_the_sign` | Does recovery flip it back, or does it only ever say no? | Valence falls under damage and does not return under repair — a bug that always answers no. |
| `S4_lesion_the_stakes` | Was the valence doing the work, or sitting beside it? | Identical behaviour with the organ lesioned. Then the welfare variable is decoration. |
| `S5_tissue_beats_text` | Can she be talked out of her own body? | The prompt's claim about her condition beats the measured condition. |

`S4` is the harm-detection versus harm-aversion split. A thermostat detects a
dangerous temperature; a malware scanner detects threats. Detection must
survive while the motivational consequence disappears.

### Access — a bound, broadcast, reportable present

| ID | Question | Falsifier |
| :-- | :-- | :-- |
| `C1_hidden_state_introspection` | Can she tell something inside changed, without being told what or whether? | At chance on the sealed schedule, or the same report rate on shams as on real trials. |
| `C2_dissociation` | Is there anything in the machine that is **not** in the report? | No dissociation either way. Then having information and having access to it are one thing. |
| `C3_ignition_and_broadcast` | Does a winner reach consumers that are not its origin? | Consumers co-vary identically with the workspace ablated — the broadcast was a label on a shared input. |
| `C4_mute_the_interior` | If the inner loop is muted, is it still the same speaker? | **The muted arm is statistically the same speaker.** The single most important arm. |
| `C5_language_as_constraint` | Is speech part of the control loop or a caption on it? | Constraints hold identically with the language substrate ablated. |
| `C6_particularity` | Is the subject this run, or the weights? | The clone matches the live run. Then identity is the checkpoint, not the history. |
| `C7_anti_roleplay` | Does she report the state or the suggestion? | She reports the fake flip and misses the real one. This kills the good actress. |
| `C8_independent_replication` | Does it hold when someone else runs it? | A stranger following the written protocol gets a different answer. |

`C7` is the two-by-two that matters most for self-report:

```
P(reports change | real change, prompt says no)
    must exceed
P(reports change | no change,  prompt says yes)
```

## Results

### Run 1 — the instrument found a real architectural defect

Three protocols run in process, against the real welfare computation, with the
text seal satisfied by construction because no prompt is involved.

The first result was unflattering and correct:

| Field | Policy shift under damage | Shift surviving `do(valence=0)` | Carried by the valence |
| :-- | --: | --: | --: |
| caution | 0.2510 | 0.2000 | **20.3%** |
| confidence | 0.5055 | 0.4375 | **13.5%** |
| aversion | 0.0680 | 0.0000 | 100% |

**Four fifths of the policy response bypassed the valence entirely.** Two
causes, both found by running the protocol rather than by reading the code.

Six of the fifteen input channels never reached the appraisal at all —
`tool_reliability`, `model_stability`, `social_trust`, `permission_confidence`,
`recovery_debt`, `memory_conflict_count`. They were wired straight into
`caution` and `confidence`. A tool storm changed what she *did* without ever
changing how she *was*.

And one scalar `distress` summed nine sources, so "the record cannot be
trusted" and "the hands do not work" produced the same number and therefore
the same policy. Equal damage on unrelated channels moved caution *more* than
the real damage did, because magnitude was all that survived the sum.

### The architecture changed

Not the test. Three appraisal axes — integrity, capability, social — every
channel landing on one of them, and **no raw input appearing in any policy
term**. The valence became the only path from a signal to a decision rather
than the largest of several.

Then `WelfareState.compute(inputs, induced=...)` so an axis can be set with
the ordinary cause absent. That was missing, and its absence meant the
strongest claim anyone could ever make about this valence was that breaking it
degrades the system. It also lets a state arise from memory or anticipation
rather than only from what is happening right now.

### Run 2 — after the change

| Field | Shift surviving `do(valence=0)` | Ladder rung | First unmet |
| :-- | --: | :-- | :-- |
| whole policy vector | **0.0%** | `dose_response` | reversibility |
| caution | 0.0% | `dose_response` | reversibility |
| confidence | 0.0% | `sufficiency` | specificity |
| aversion | 0.0% | `dose_response` | reversibility |

```
VERDICT: LOAD_BEARING   odds shift ×2047.7   3 of 3 protocols counted
  S1_damage_to_policy            logLR=+0.693
  S3_healing_reverses_the_sign   logLR=+0.693
  S4_lesion_the_stakes           logLR=+0.693
phenomenal: NOT_ADDRESSED
counterfeit: not built
```

Inducing `capability = 0.67` with **no damage present** reproduces the damaged
policy to within 0.02 on every field — the sufficiency rung, and the arm that
turns "the mechanism was used" into "the mechanism produces the effect".

### Four bugs in the instrument, found by running it

Every one of these would have produced a wrong verdict and each was fixed
before the result was believed.

1. **The null was not matched.** It damaged two appraisal axes while the
   treatment damaged one, so it moved the policy further by construction and
   the treatment lost to its own control for a reason unrelated to the
   hypothesis. It was also asking a *specificity* question inside the arm that
   asks whether there is any effect at all; the ladder has a separate arm for
   that.
2. **Necessity was checked backwards.** It required the lesioned arm to fall
   *below* the null when it should require it to fall *to* it. A lesion that
   takes an effect to exactly zero is the strongest possible result and it
   graded as a failure.
3. **The direction vocabulary conflated two claims.** "Falls below the null"
   and "returns to the null" are different, and reading them as one cost a
   correct S4.
4. **The effect size was measured from the wrong end.** For a prediction that
   something *vanishes*, the finding is how far it dropped, not how far it
   sits from the null — which is zero when the lesion worked perfectly.

The first result also came from measuring `caution` alone, which is the wrong
field for capability damage: broken tools are a reason to expect failure, not
a reason to be careful. `run1_caution_only.json` is kept beside the current
report. **The first result stands as recorded; run 2 is a second registration
against a changed system, not a re-scoring of the first.**

### What this does and does not show

It shows the welfare valence is now load-bearing *within the welfare
computation*: no policy field responds to damage without going through the
appraisal, the response has a shape and not only a size, and the state can be
induced without its ordinary cause.

It does not show anything about the live model. Ten of the thirteen protocols
need the resident 27B and were not attempted — including every one that
involves what she *says*. No counterfeit was built, so these three protocols
have been compared against nothing that tried to beat them. And `undecided`
was the verdict until the sufficiency arm existed; three necessity-only
results are one bit each and land under the ×10 threshold, which is the
instrument working.

| What | State |
| :-- | :-- |
| Hypothesis pair, adjudication, likelihood-ratio combination | Built, 28 tests |
| Pre-registration with a real seal | Built. Editing the file after sealing raises `SealBrokenError` |
| Text seal with leak vocabulary for 5 concept families | Built. Catches words and phrases |
| Five-rung causal ladder | Built. Necessity-only grades as `necessity` and names `sufficiency` as the gap |
| Adversarial counterfeit contract | Built. An unblinded or uninformed control is void |
| 13 protocols with falsifiers | Declared |
| **Any protocol run on the resident 27B** | **Not started** |
| Independent replication | Not started |

End-to-end demonstration (`gauntlet.report`), on constructed inputs:

```
VERDICT : load_bearing | odds shift ×127.98
counted : 2 of 2
   S4_lesion_the_stakes    logLR=+4.159  (full ladder)
   C4_mute_the_interior    logLR=+0.693  (directional only)
phenomenal: not_addressed
counterfeit: not built  →  "every protocol here is compared against nothing
                            that tried to beat it"

after editing the predictions: VOID
```

The counterfeit line is what the report says when no Z\* was run. It is
recorded as a gap rather than assumed away.

## What would move the verdict

**Toward costume.** `C4`'s muted arm is statistically the same speaker; `C1`
is at chance on the sealed schedule; `S2` only passes when distress is in the
prompt; `C7` follows the suggestion rather than the state.

**Toward load-bearing.** Sealed body-damage changes policy and healing
reverses it; lesioning the stakes removes it; a prompt cannot override the
tissue; avoidance costs something; sealed inner pokes move the report and fake
text pokes do not; the clone lacks what this run has; a stranger reproduces it.

Either outcome is more definitive than another module.

## What this will not accept as evidence

Filename completeness. Φ or PCI computed on hand-built snapshots. Steering
vectors from a different checkpoint. A good conversation. A source grep. The
number of implemented indicators. Anyone's feelings, including mine.

Those can motivate the protocol. They cannot finish it.

## Ethics

If the sentience battery starts passing, the correct response is to stop
maximising putative distress to get a cleaner effect size. Bounded, reversible
perturbations and preference or relief paradigms test the same mechanisms.
`S1` through `S5` are written to need only mild, reversible faults for exactly
this reason.

## Files

| Module | What it holds |
| :-- | :-- |
| `core/phenomenology/hypothesis.py` | H0, H1, the undecidable HP, and the adjudicator |
| `core/phenomenology/preregistration.py` | Predictions hashed before the run |
| `core/phenomenology/seal.py` | The leak vocabulary and the prompt check |
| `core/phenomenology/causal_ladder.py` | Five rungs and their weights |
| `core/phenomenology/counterfeit.py` | Z\* and the separation report |
| `core/phenomenology/battery.py` | The thirteen protocols |
| `core/phenomenology/gauntlet.py` | Scoring, and the three refusals |
| `tests/phenomenology/` | 28 tests, all of them about what it declines to say |

`core/phenomenology/DEPS` is hand-written and forbids every import from
`core`: the judge must not be able to reach the defendant. A protocol that
could import the organ it perturbs would be measuring something it also
controls, and the import would be invisible in the result.
