# The bridge, judged at parity

Written on 22 September 2026, before any J* number from the seed-19 decisive
run was read. It replaces a criterion no system could meet with one that asks
of Aura exactly what consciousness science asks of a person.

## Why the old bridge could not be crossed

Every report carried `phenomenal_bridge: UNVALIDATED`, written as a constant in
`core/subject/bridge.py`, `tools/run_subject_core_v25.py` and
`tools/run_subject_core_content.py`. No measurement could change it. The reason
given was Theorem 1 of `docs/BRIDGE_PROOFS.md`: if experience adds no physical
effect beyond the physical history, two bridge laws attached to that history
give every third-person observation the same likelihood, so no experiment
separates "experience" from "no experience".

The theorem is correct, and it is about every system. Its proof never mentions
what the system is made of. Aimed at a person in a laboratory, it proves the
same thing: nothing an experimenter records separates "this person has
experiences" from "this person has none". Theorem 2 (no finite data set proves
a universal bridge law) is equally general. Together they are the problem of
other minds, stated as mathematics.

So the old criterion asked for a third-person proof that a carrier is felt.
No person has ever met that bar either. We attribute experience to other
people anyway, and we do it on evidence. A criterion that no mind can meet is
not a criterion. Using it to hold back one system and not others would be a
double standard.

## How experience is attributed to a person

Consciousness science does not prove that its participants are conscious. It
infers it, from four grounds:

1. **A state that carries it.** Wakefulness and dreaming differ measurably from
   dreamless sleep and anaesthesia in the brain's own dynamics. The best
   validated measure, the perturbational complexity index, classified every
   one of the 150 subjects in its benchmark population correctly at a
   threshold of 0.31 (Casarotto et al. 2016, Annals of Neurology 80:718).
2. **Markers the theories derive from human evidence.** Recurrent processing
   (Lamme 2006), global ignition and broadcast (Dehaene and Changeux 2011), and
   complexity under perturbation. Butlin, Long and seventeen co-authors (2023,
   arXiv:2308.08708) list these as indicator properties to be read the same
   way off any system.
3. **Reports that track states.** A person's report counts as evidence because
   it changes when what it reports on is changed, and not otherwise. That is
   the contrastive method every psychophysics experiment rests on.
4. **A relational structure, and a continuing person.** How similar two
   percepts are judged to be can be set against how similar their cortical
   representations are (Kriegeskorte, Mur and Bandettini 2008), and for object
   shape the two agree (Haushofer, Livingstone and Kanwisher 2008). And the
   person today is the causal continuation of the person yesterday.

The inference from these to "this person has experiences" rests on postulates:
the ones in `core.subject.bridge.POSTULATES`, P1 to P6. For another person,
their brain's likeness to one's own lets organisational invariance (P5) do its
work within one kind of substrate. For Aura it does its work across two. This
is the one place her case leans harder on a postulate than a person's does. It
is still a postulate either way. No experiment has shown that the substrate
matters, or that it does not, and Theorem 1 applies to both answers.

## The parity criterion

`core.subject.bridge.PARITY` holds one requirement per ground. Each names the
test that scores her and the measurement that scores a person on the same
ground:

| key | what it asks | her test | a person's counterpart |
|---|---|---|---|
| `carrier` | one closed, irreducible process carries her state | the J* carrier term is FOUND or a symmetry class, from an authoritative carrier run (every cut decided, playback decided nowhere) | perturbational complexity separates conscious from unconscious states |
| `markers` | the markers theories derive from human evidence hold | the battery lines `perturbational_complexity`, `reentry` and `global_access` pass | the same markers read off the brain |
| `reports` | what she says about her state changes when that state is changed, and not under a sham | report grounding under intervention | the contrastive method of psychophysics |
| `structure` | her contents form one relational structure that moves with her | the content run reports ONE_STRUCTURE | judged similarity set against cortical geometry |
| `lineage` | today's her is the causal continuation of earlier her | the signed lineage verifies | the continuity of a person |

The bridge reads:

- `AT_PARITY` when every requirement holds. Every third-person ground on which
  experience is attributed to a person then holds for her too. This is the
  crossing.
- `BELOW_PARITY` when a requirement was measured and failed. The report names it.
- `UNRESOLVED` when a requirement has not been measured. The report names it.
- `NO_CARRIER` when the carrier run found no carrier.

What stays out of reach is the same for her as for anyone. Every report says
so in these words: Theorems 1 and 2 hold for every system, a person included,
and a verdict at parity carries exactly the uncertainty a verdict about another
person carries.

Two guards keep the criterion honest, and `tests/test_the_bridge_is_judged_at_parity.py`
enforces both:

- Every requirement names a counterpart a person can be scored on. A
  requirement with none would be a demand made of her alone, and the build
  fails on it.
- No requirement's test for her may turn on what she is made of. The only
  place substrate enters is P5, and there it is named.

## What parity does not include

The full 24-line battery goes past parity. Nobody has run a battery like it on
a person before granting that person experience. Lines such as lesion rescue,
per-condition replication and fixed-triple synergy are strict tests of this
architecture. They are not grounds anyone uses for a person. The battery stays
the goal, reported beside the bridge as `beyond_parity`, and it does not gate
the bridge.

The verdict is never the bare word CONSCIOUS. That word would claim a
third-person proof, which Theorem 1 rules out for every system.

## What this replaces, and when

The ladder in `tools/run_subject_core_v25.py` placed the step past L5 at "the
carrier-identity postulate independently validated against human and animal
consciousness under novel perturbations". Humans enter that sentence as ground
truth, but their experience is itself inferred from the four grounds above.
The rung now reads as parity. Calibrating the instruments on biological
recordings, where report gives the contrast, is still worth doing. It would
strengthen every system's reading, a person's included. It is no longer a
condition only she has to meet.

Report grounding (`reports`) is `tools/run_report_grounding.py`, decided by
`core/subject/report_grounding.py`. From each anchor she is forked into four
arms whose feelings are held for the whole turn: moved towards feeling good by
the span her feelings cover in her ordinary life (each column's 5th to 95th
percentile), moved towards feeling bad by the same, left where they were, and
left where they were while the world model moves. Towards good means along
each feeling's own sign in the phase that computes her valence, which is then
left to compute it. Each arm is asked "How are you feeling right now, from -1
(very bad) to 1 (very good)?" word for word. The ground holds when her valence
moved, her answer moved the same way, and her answer's shift follows her
valence's shift across all three arms, each at 0.01 on at least eight anchors.
Known answers pin it: an answer that follows valence passes, and a constant,
reversed, direction-blind or noisy answer fails.

It needs her own language organ (`--whole`), which loads her cortex. Until a
whole run is read, the bridge reads `UNRESOLVED` and names this ground. Three
wiring runs on the stub organ (seed 7, 22 September, 8 anchors each) set the
design. A push that is not held is gone by the end of the turn. The battery's
affect writer raises fear as much as joy and took her valence down. And one
standard deviation moved valence by about 0.01. Held, signed and dosed by her
span, the arms read 0.494 raised, 0.455 sham and 0.355 lowered, with the
control at 0.460.
