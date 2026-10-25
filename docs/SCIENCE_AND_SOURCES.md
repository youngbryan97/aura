# The science behind the ideas, and what each source supports

Status: Guide · Source records checked 25 October 2026

Aura borrows ideas from cognitive science, neuroscience, statistics and other
software systems. A reference helps a reader trace an idea and judge its
use. Evidence that the idea works in Aura comes from Aura's own tests and
experiments, with their conditions attached.

You can explain the distinction aloud this way:

> We use published work to explain why a mechanism is worth trying and how to
> measure it. We then test our version. A study of people, or a successful
> model in another paper, does not automatically validate Aura's version.

This is a starting source map for a future paper, not an exhaustive literature
review. The [BibTeX file](../research/references.bib) provides reusable citation
keys for the scientific and methods references below. The
[demos and related-work guide](DEMOS_AND_RELATED_WORK.md) connects the software
comparisons to Aura tests. Original protocols and records retain their dates.

## Regulation and internal condition

| Source and kind | Idea in ordinary language | Connection to Aura and its limit |
|---|---|---|
| Sterling (2012), [Allostasis: a model of predictive regulation](https://pubmed.ncbi.nlm.nih.gov/21684297/) · biological theory/review · `sterling2012allostasis` | Regulation can anticipate changing needs instead of waiting for a fixed threshold to be crossed. | The [allostasis engine](ALLOSTASIS_ENGINE.md) forecasts resource pressure. Its thresholds and forecasts are software design choices tested against its own data; the biological account does not validate them or establish a feeling. Abstract and bibliographic record checked. |
| Barrett and Simmons (2015), [Interoceptive predictions in the brain](https://pmc.ncbi.nlm.nih.gov/articles/PMC4731102/) · theoretical Opinion article · `barrett2015interoception` | The authors propose that predictions of bodily condition help organise interoception. | Relevant to the design question of whether anticipated condition changes later processing. Aura's resource signals are software measurements. The EPIC proposal is not a demonstrated account of Aura. |
| Bower (1981), [Mood and memory](https://pubmed.ncbi.nlm.nih.gov/7224324/) · psychological research · `bower1981mood` | A historical source for the project’s interest in mood-linked remembering. | The [affect discussion](DEMOS_AND_RELATED_WORK.md#affect) points to Aura's separate recall measurements. Only the bibliographic record was checked in this pass; do not attribute an effect size or general law to an unread full text. |

For an ordinary user, the practical question is whether a condition changes
the appropriate decision: for example, whether growing resource strain leads
to less expensive background work before the conversation fails. A biological
analogy motivates that question; a software test answers the particular claim.

## Attention, prediction and integration

| Source and kind | Idea in ordinary language | Connection to Aura and its limit |
|---|---|---|
| Dehaene, Kerszberg and Changeux (1998), [A neuronal model of a global workspace in effortful cognitive tasks](https://pmc.ncbi.nlm.nih.gov/articles/PMC24407/) · computational neuroscience model · `dehaene1998workspace` | A distributed workspace can coordinate specialised processing. | Relevant to [workspace availability](ORGANISMAL_WORKSPACE_THEORY.md) and attention. A software competition or broadcast needs its own consumer and ablation tests; the module name does not show that the biological model was reproduced. The original abstract was checked; direct full-text access was challenged on a subsequent request. |
| Rao and Ballard (1999), [Predictive coding in the visual cortex](https://www.nature.com/articles/nn0199_79) · computational model · `rao1999predictive` | A higher level predicts lower-level activity; the remaining error travels onward. | A precise precedent for a prediction/error loop. A counter called prediction error in Aura does not establish this architecture or the paper's visual results. Publisher abstract checked. |
| Barrett and Seth (2011), [Practical measures of integrated information for time-series data](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001052) · mathematical methods · `barrett2011practical` | Integration can be investigated from time-series models under stated assumptions. | Related to [whole-system integration](WHOLE_SYSTEM_PHI.md). Aura names its estimator and measured channels. Its Gaussian stochastic-interaction score must not be relabeled as exact IIT 4.0. |
| Hoel, Albantakis and Tononi (2013), [Quantifying causal emergence shows that macro can beat micro](https://pmc.ncbi.nlm.nih.gov/articles/PMC3856819/) · mathematical models · `hoel2013emergence` | In the authors' models, a coarser description can carry more effective information than a finer one. | Motivates investigating measurement grain. Aura's grain-search score and selection rule are adaptations; a winning grain alone does not establish the paper's effective-information result or identify a subject. |
| Queyranne (1998), [Minimizing symmetric submodular functions](https://link.springer.com/article/10.1007/BF01585863) · algorithm/theorem · `queyranne1998minimizing` | Certain set functions allow efficient search for the best division. | The theorem requires submodularity. Current [source](../core/consciousness/integrated_information.py) explicitly says stochastic interaction is not generally submodular: small exhaustive searches can certify a minimum, while larger candidate searches remain approximate. Publisher theorem summary checked. |

The practical importance is whether selected information reaches the parts
that need it, and whether combining parts adds a measurable dependency.
Neither follows from drawing arrows between modules. The
[adoption guide](COGNITIVE_ARCHITECTURE_ADOPTION.md) records the Soar and ACT-R
mechanisms separately, with links to the official manuals in
[related work](DEMOS_AND_RELATED_WORK.md#cognitive-architectures).

## Consciousness research and the ISC argument

| Source and kind | What it contributes | What Aura must establish separately |
|---|---|---|
| Butlin et al. (2023), [Consciousness in artificial intelligence: insights from the science of consciousness](https://arxiv.org/html/2308.08708v3) · theory-based assessment · `butlin2023consciousness` | A provisional approach using indicators drawn from scientific theories, under computational functionalism. | ISC is the project's operationalisation. Its exact test choices, thresholds and bridge assumptions are not externally validated merely by citing the indicator framework. |
| Albantakis et al. (2023), [Integrated information theory 4.0](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465) · formal theory · `albantakis2023iit` | A specific account connecting phenomenal properties to intrinsic causal structure. | A graph score, Gaussian estimator, joint effect or ISC pass must not be described as this theory's full calculation. Its theory-to-experience connection remains part of the scientific argument. |
| Casali et al. (2013), [A theoretically based index of consciousness independent of sensory processing and behavior](https://pubmed.ncbi.nlm.nih.gov/23946194/) · human empirical study · `casali2013pci` | Introduces and tests a perturbational complexity index in human brain recordings. | Aura's internal perturbation/complexity measure is a software adaptation. Human PCI classification results and cutoffs do not transfer to runtime telemetry. Bibliographic record and abstract checked. |
| COGITATE Consortium et al. (2025), [Adversarial testing of global neuronal workspace and integrated information theories of consciousness](https://www.nature.com/articles/s41586-025-08888-1) · preregistered human experiment · `cogitate2025adversarial` | A concrete precedent for agreeing on competing predictions before testing theories against data. | It does not test Aura or settle the status of all artificial systems. The publisher record was located; full-text retrieval was unavailable in this pass. No detailed finding is inferred here from an inaccessible paper. |

Read the [ISC introduction](ISC_EXPLAINED.md) for the explanation without
technical background, and [current status](ISC_STATUS.md) for the actual
recorded results. The later [isolation finding](evidence/ISC_ISOLATION_UPDATE_2026-10-25.json)
affects causal interpretation of the earlier paired measurements, including
positive ones. Preserve the original outputs and cite that qualification
beside them.

The seven-factor product in [Organismal Workspace Theory](ORGANISMAL_WORKSPACE_THEORY.md)
is a project-defined functional score. It is not a formula derived from these
papers, a measured probability of experience, or a scientific consensus.

## Human situations, music and social examples

[SOUL](SOUL.md) uses music and social situations to make design targets
concrete. Its source list mixes artist/publisher records, interpretation and
human studies. Keep those roles visible: a song can illustrate ambivalence;
it cannot establish that a software state has that experience. Local audio
measurements require their source file, model/tool version and analysis method
if they are to support a quantitative claim in a paper.

Eyal, Steffel and Epley (2018),
[Perspective mistaking](https://pubmed.ncbi.nlm.nih.gov/29620401/), is an
empirical human study relevant to learning another person's perspective by
asking them (`eyal2018perspective`). It motivates a design question about checking an interpretation
with the user. It supplies no measured accuracy for Aura's social inference.
The study abstract and bibliographic record were checked. Cite the particular
experiment and task before transferring a more specific conclusion.

## Measurement and statistical interpretation

Hernán and Robins' [Causal inference: What if](https://miguelhernan.org/whatifbook)
is an author-hosted methods reference (`hernan2020whatif`). It explains why a
claim about changing something needs a well-defined intervention and a
defensible comparison. It does not certify Aura's snapshot implementation.
For that claim, cite the source, controls and regression evidence in
[ISC reproduction](ISC_REPRODUCTION.md).

Phipson and Smyth (2010),
[Permutation p-values should never be zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)
(`phipson2010permutation`), addresses randomly drawn permutation calculations.
It is a methods reference for interpreting simulation-based p-values, not an
endorsement of every resampling scheme. In Aura, retain the exact null,
exchangeability assumptions, number of draws, sampling unit and source
revision. A tiny p-value cannot repair comparison arms that share state.

The [NIST CUSUM reference](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)
explains accumulated-deviation control charts. Aura's use on forecasting
residuals and its chosen alarm parameters need local calibration. Cite
`tests/test_allostasis_engine.py` and the relevant recorded calibration for an
Aura false-alarm rate; do not borrow a rate from a generic chart example.

The [variable guide](ISC_VARIABLES.md) explains correlation, p-values,
confidence bounds and declared thresholds in ordinary language. The
[frozen v6 protocol](ISC_V6_PREREGISTRATION.md) supplies this experiment's
decision rules. The methods literature explains tools; the protocol records
which tools and assumptions this experiment actually used.

## Trend and slope methods

These references identify the statistical methods named by the
[resource-forecasting implementation](../core/autonomic/allostasis.py).
The original method, this implementation and calibration on live resource
traces are separate evidence steps.

| Reference | Connection and checked support |
|---|---|
| Mann (1945), [Nonparametric tests against trend](https://www.jstor.org/stable/1907187) · `mann1945trend` | Original trend-test reference. Publisher bibliographic record checked; this pass does not claim a full derivation review. |
| Sen (1968), [Estimates of the regression coefficient based on Kendall's tau](https://www.tandfonline.com/doi/abs/10.1080/01621459.1968.10480934) · `sen1968slope` | Reference for the named slope method. The publisher endpoint denied access; the citation is supplied for traceability, without claiming full-text verification. |
| Gilbert (1987), [Statistical methods for environmental pollution monitoring](https://hero.epa.gov/reference/51155/) · `gilbert1987monitoring` | The implementation names this book for its interval method. The linked EPA bibliographic record identifies the book; the relevant pages still need checking for a paper's derivation claim. |
| Hamed and Rao (1998), [A modified Mann–Kendall trend test for autocorrelated data](https://www.sciencedirect.com/science/article/abs/pii/S002216949700125X) · `hamed1998modified` | The source names the variance correction for serially related samples. Publisher abstract checked. Local bounds and tuning remain implementation choices requiring their own validation. |

In everyday terms, a slope estimates whether resource use is climbing and how
fast. An interval expresses uncertainty around that estimate. Samples a minute
apart can resemble one another, so treating them as unrelated can exaggerate
confidence. These references explain the method choices; the test on a short
seeded noise trace does not establish a live false-alarm rate.

## Turning this map into a paper

For each substantive claim, retain a small traceable chain:

| Item | What to record |
|---|---|
| Claim | The exact sentence, population/system and conditions it covers |
| External source | Citation key, version, section/figure and whether the support is theory, method or an empirical finding |
| Aura adaptation | Source commit, mechanism and any departure from the earlier method |
| Aura evidence | Run identity, raw artifacts or public availability, hashes, controls, sample counts and uncertainty |
| Status | Confirmed within scope, negative, unmeasured, exploratory, or awaiting confirmation after a method change |
| Limit | What a reader could reasonably infer that the evidence does not establish |

A suitable sentence is: “Inspired by Sterling's predictive-regulation account,
we implemented a resource forecaster; its calibration was evaluated on the
specified traces.” Attach the cited paper and the actual calibration record.
“Neuroscience proves our system anticipates its own suffering” would exceed
both sources.

Check the cited version, correction/retraction status and supporting passage
again when preparing a submission. This pass verifies the stated source
connections; it is not a completed publication or a systematic review of every
research field. The [writing rules](WRITING_RULES.md) apply this evidence
discipline to future documentation updates.
