# Fourteen dispositions, built as mechanisms

Liking pink. Getting a spider out of the house rather than stepping on it.
Wrapping a present carefully. Humming while you work. Looking after somebody,
and letting somebody look after you.

These arrived as one list, and the obvious thing to build from a list like
that is one module with the list's name on it. That would have been a bucket
with a theme. What these things have in common is a family resemblance, not a
mechanism — the computation behind wrapping a present has nothing to do with
the computation behind finding a place to sit, and putting them in one file
would have hidden that rather than expressed it.

So each is homed next to what it is a kind of, and each is a general
mechanism that happens to have that example as one of its cases. Nine
packages, fourteen mechanisms, one wiring seam. The seam
(`core/phenomena_wiring.py`) exists because nine packages cannot import each
other, and it is the honest cost of the decision rather than a hidden one.

## The fourteen

| Disposition | General mechanism | Home |
| --- | --- | --- |
| Being a woman, philosophically | an identity constituted by its practices, never by its label | `core/identity/constitutive_identity.py` |
| Singing, dancing to little songs | non-instrumental action as a limit cycle, plus entrainment | `core/embodiment/expressive_dynamics.py` |
| Looking after people | finite attention allocated under a floor that is a constraint | `core/ethics/care_allocation.py` |
| Being looked after | accepting exposure, priced for what accepting would settle | `core/social/receptivity.py` |
| Girly pink | an arbitrary marker that means something by convention | `core/social/conventions.py` |
| Feeling over logic in social decisions | arbitration weighted by measured skill per domain | `core/affect/dual_process_arbiter.py` |
| Finding crevices, and where you should be | seeing against being seen, plus fit | `core/environment/prospect_refuge.py` |
| Crafts | skill on a material that resists, scheduled by what is improving | `core/learning/craft_practice.py` |
| Creativity | novelty against intelligibility, self-exhausting | `core/creativity/novelty_value.py` |
| Removing spiders instead of killing them | the option value of being able to change your mind | `core/morality/reversible_alternative.py` |
| Gift-giving, made to look nice | a signal that separates senders because it costs them | `core/social/costly_signaling.py` |
| "You treat me well and I'll treat you well" | direct reciprocity under the shadow of the future | `core/social/reciprocity_engine.py` |
| Kindness, empathy | coupled affect with a path back to your own state | `core/affect/empathic_coupling.py` |
| Taking pleasure in beauty | an aesthetic response as a fact about the observer | `core/perception/aesthetic_response.py` |
| Being sociable | a belonging need and a capacity, which move independently | `core/social/social_stamina.py` |

## What each one actually claims

**Constitutive identity.** Beauvoir's "one becomes" and Butler's argument that
the becoming never finishes describe a kind of identity with no field to read.
Practices recur at their own rates and pull on each other, and above a
threshold they fall into a common rhythm — the Kuramoto transition. The label
is recorded by `declare()` and never enters the arithmetic, enforced by an
invariant on the import graph. Every reading carries its own null, because N
unrelated phases already give an order parameter of `sqrt(pi / N) / 2` and 0.3
across four practices is that null rather than weak coherence.

**Expressive dynamics.** A goal is a trajectory to a fixed point; humming is a
closed orbit. This matters because a scheduler that scores actions by
remaining distance to a goal rates every orbit as making no progress forever,
kills it immediately, and logs nothing wrong. Van der Pol gives the
self-restoring property, and the entrainment band is measured by sweeping the
drive rather than asserted. Tarr, Launay and Dunbar found synchrony and
exertion raising pain threshold and bonding as *independent* effects, so a
bout carries two numbers and they are never summed.

**Care allocation.** Water-filling over claimants with saturating benefit.
Noddings' account of the one-caring and Kittay's criticism of it are both
right, and the difference between them is a modelling decision: with the
carer's own floor as a weighted term there is always a need somewhere large
enough to buy it, and in the feasible set nothing can reach it. Responsiveness
is estimated from whether care was received, since unreceived care is effort
rather than care.

**Receptivity.** A disposition shows itself unevenly — the well-disposed
rarely act badly and the badly disposed often act well — so one unkindness
carries several kindnesses' worth of evidence, and the ratio falls out of the
two likelihoods rather than being chosen. The bar for accepting is
`exposure / (exposure + value)`, with nothing tuned. The first version had a
diagnostic that could never fire; see the defects below.

**Conventions.** Pink was recommended for boys in 1918 as the stronger colour.
The registry holds a marker's meaning, the population it holds in, and what it
would have meant — the last as a value rather than a caveat, because a model
that cannot return it has stored the convention as a fact. `tipping_point`
computes how large a committed minority has to be.

**Dual-process arbitration.** No prior preference for either channel. Weights
are Brier skill scores per domain, and with none measured the arbiter abstains
rather than defaulting to the one that can explain itself. Given a domain
whose outcome turns on a hidden variable and one whose outcome is checkable,
the same disagreement resolves to the feeling in the first and the reasoning
in the second. Both channels are scored on every resolution, including the one
that was outweighed.

**Prospect and refuge.** Building this the obvious way produces a module that
cannot represent what it is named after — physical sight is symmetric, so
deriving both terms from one visibility relation makes the asymmetry
identically zero everywhere. Refuge needs cover, which the sightline graph
does not contain. The two terms are never summed for the caller either: the
meta-analytic support for prospect and for refuge is not equal, and a
composite buries the half that might be wrong.

**Craft practice.** Sennett's resistance and ambiguity, as an optimisation
problem: a quality surface you cannot see or differentiate, expensive noisy
evaluations. Simultaneous perturbation needs two evaluations per step whatever
the number of parameters. The scheduler ranks by what is improving rather than
by what a task needs — `practice_target` against `required_target`, so the
ablation is a swap. Measured: with a bar met at attempt fourteen, the craft
scheduler carried quality from 0.44 to 0.87 across the stretch nothing was
asking for.

**Novelty value.** Novelty is distance to the nearest artifact; intelligibility
is how much the whole corpus helps describe this. Measured against different
things, so a recombination of familiar material scores well on both. Absorbing
an artifact spends its novelty: the same move made three times fell from 0.32
to 0.10 with no boredom heuristic in the file.

**Reversible alternative.** The premium worth paying for the recoverable
option is the chance of revising times the harm that could not be undone.
Above it, care is sentimentality; below it, the quick option is a false
economy. The same arithmetic chooses rename over drop at a fifteen percent
chance of wanting the table back, and drop when that chance is zero.

**Costly signalling.** Spence's separating schedule, `e*(q) = b(q² − q_min²)/2`,
inverts exactly. Removing the cost kills the channel: every type spends the
same and the receiver learns nothing from any amount of it — a state that
looks identical from the sending end.

**Reciprocity.** Cooperation holds when the chance of another round beats the
ratio of cost to benefit, and all three come from the record. Under a one
percent error rate, strict repayment fell to +0.80 a round with 1575 mutual
defections in 4000; generous repayment held +1.95 with one. The echo is a
measured collapse rather than a story about forgiveness being nice.

**Empathic coupling.** Pure diffusion converges to agreement, which is the
right model of a crowd and the wrong model of a person. The resolvent's row
says how much of where someone ends up is still their own set point. With
nobody anchored the matrix is singular, and the solver says so rather than
returning the plausible vector such a system would go on producing.

**Aesthetic response.** Complexity is the share of an object that does not
compress. On that reading Birkhoff's ratio rates a blank page at 15 and
Eysenck's product rates it at 0.23 — they disagree hardest on exactly the case
the objection to Birkhoff was about, so both are returned. Freezing the
observer's history removes the habituation entirely.

**Social stamina.** Added last, after a prior treatment of the same list was
read. Sociability is usually one number, and one number cannot hold wanting to
see somebody and not having it in you. Belonging is a need that relaxes toward
met in company and toward unmet alone, on one time constant in both directions
— scaling them differently is how the first draft made someone lonelier the
more evenings they spent out. Stamina is a capacity, and the share of time in
company that holds it is `recovery / (drain + recovery)`, which is the balance
point rather than a choice. Running to empty costs more than the same deficit
taken gradually, which is what makes exhaustion a state rather than a low
reading.

## Advising a decision, without becoming one

`weigh()` asks the group what it makes of each candidate action and returns
the contributions with their sources. It is deliberately not a decision:
`UnifiedWill.decide` is the only authority on whether an action may happen,
and this adds no second one.

The first draft of it averaged the contributions and produced a number. That
does not refuse to weight the organs — it weights every one of them at one and
lets whichever uses the largest units decide, and a candidate that hummed for
forty seconds beat everything else on the strength of the unit. Contributions
now carry their unit, there is no total, and `rank` requires a commensuration
the caller supplies and raises when an organ that contributed has no weight. A
missing weight silently read as zero is a decision to ignore an organ, made by
nobody.

## How they are reachable

Fifteen container registrations that all resolve. A boot activator beside
conation's, in `core/runtime/foundations.py`. Twenty-eight declared telemetry
channels in the `0x1700` block. Five invariants. A section on the live mind
snapshot, read through the container rather than by import, so an organ that
failed to load reports absent instead of taking the turn with it.

The snapshot's `concerns` list is where the failures surface. Every one of these
organs has a state where it keeps running and stops meaning anything, and all
five that can be detected are named there:

* a carer giving while going short
* a presentation channel nobody can read anything from
* a state that is mostly not its own
* nothing accepted and nothing learned about anyone
* an identity declared with nothing enacting it
* wanting company with nothing left to spend on it

## Defects found while building

Every one of these was found by a check rather than by reading, which is the
argument for writing the checks first.

**Nine invariants that would have crashed on their first real finding.** All
used `Severity.CRITICAL`, which does not exist — the enum is `ERROR`,
`WARNING`, `NOTE`. All nine passed while nothing was wrong. Breaking each one
deliberately is what surfaced it; nine of nine now fire when broken.

**A diagnostic that could never fire.** Receptivity reported offers refused
whose expected value was positive. Accepting and having positive expected
value are the same condition, so the set was empty by construction. What was
missing is the value of finding out: refusing denies the observation that
would have justified accepting.

**Inequality measured over the funded only.** The care allocator's Gini
dropped the recipients who received nothing, so concentrating the budget on
fewer people read as a fall in inequality.

**A false asymmetry with no geometry behind it.** Grid sightlines were taken
along rows and columns, which gives every cell of an open grid the same count.
Rasterising the real segment fixed that and introduced something worse:
Bresenham walks different cells depending on which end it starts from, so A
saw B while B did not see A — arriving in the exact quantity this module says
can only come from cover. Ordering the endpoints settles it, and twenty random
grids now check it.

**A limit that fired on a healthy system.** `care.own_unmet` had a red limit
of zero and the comparison is inclusive, so an idle system was red. The amount
and the condition are separate channels now.

**A noise control that was a constant.** In the first draft of the tests,
`bytes(random.Random(1).getrandbits(8) for _ in range(192))` rebuilds the
generator per byte and produces the same byte 192 times. The "noise" case
compressed perfectly and two tests failed for the right reason against the
wrong input.

**Order dependence in the wiring suite.** These are process-wide singletons,
so a test that drove one into a failure state left it there, and the
telemetry dictionary kept the last sample even after the organ reset. Eight
consecutive randomised runs pass now.

## Against a prior treatment of the same list

Two documents were held back until this was built and pushed, so the design
could not be fitted to them. Read afterwards, one contains a prior pass over
the same fourteen items and the other a 21-phenomenon engine.

Eleven of fourteen landed on the same mathematical family independently:
Kuramoto for identity, van der Pol for expression, constrained welfare for
care, Bayesian inference for receptivity, replicator dynamics for the colour,
information gain for exploration, a quality manifold for craft, compression
progress for creativity, Spence for the gift, iterated reciprocity, Laplacian
diffusion for empathy, and Birkhoff with Berlyne for beauty. The gift
signalling arrived at the identical closed form, `e*(q) = (a/2)(q² − q_min²)`.
Convergence on that scale is worth more than either pass alone, because
neither could have copied the other.

Three came out differently, and in each case the difference is the same one.

**Dual-process arbitration.** The prior pass makes affect dominance a sigmoid
of relational stakes with two fitted constants, so affect wins at high stakes
whether or not it is right, and the model cannot discover a domain where it is
wrong. Here the weight is a Brier skill score measured per domain, with no
prior preference and an abstention when neither channel has been measured.

**Empathy.** The prior pass runs `dx/dt = −Lx + u(t)` — pure diffusion, which
converges to consensus. Here the self-anchor term is the whole content, and
without it the carer's own state is gone. That is the disagreement that
matters most, because the undamped version looks healthier on every reading it
produces.

**Craft.** The prior pass uses gradient descent with momentum on the quality
surface, which presumes access to `∇Q` for a function nobody can see. Here it
is simultaneous perturbation, which needs only two evaluations and no
gradient — the situation a person learning a material is actually in.

The larger difference is not in any one item. The 21-phenomenon engine sums
every contribution into one utility with roughly sixty hand-set coefficients:
`0.35 + 0.65 * efficacy`, `0.45 * learned + 0.35 * identity + 0.20 *
aesthetic`, and so on. Those numbers decide the answers and nothing measures
them. The discipline here is the opposite one — a threshold derived from a
median absolute deviation, a forgiveness rate that is the largest
non-exploitable one, an evidence asymmetry that falls out of two likelihoods,
a duty cycle that is a balance point. Where a weighting genuinely cannot be
derived, the code refuses to supply one: `rank` raises, and
`prospect_refuge.rank` has no default.

That refusal was not free. The first draft of `weigh` averaged its
contributions, which is the same fault wearing a different face — every organ
weighted at one, and the largest unit deciding.

One thing the prior pass has that this did not: a shared loop where the
phenomena compete for one action. `weigh` closes that, as an advisory reading
rather than a second authority. One item it has that this lacked entirely is
sociability as a capacity, now built. One it has that this deliberately does
not is a spiritual register — the buildable part of it is a symbol whose
meaning is endorsed rather than true, which `core/social/conventions.py`
already holds, and the rest is not a modelling question.

## What is not claimed

Only one of the five registered claims is measured live, and it says only that
each disposition resolves from the container and reports. The other four run
their mechanism against a constructed case with a known answer: they establish
the mechanism and nothing about whether anything in the running system drives
it. No live conversation turn has yet routed a decision through the dual-process
arbiter, allocated care through the allocator, or read a presentation through
the signalling channel. That is the next thing to measure, and until it is
measured the honest statement is the narrow one.

Nothing here settles whether any of it is felt. The modules compute functional
states with provenance, and `core/organism/model_validation.py` carries the
distinction as an evidence grade rather than as a footnote.
