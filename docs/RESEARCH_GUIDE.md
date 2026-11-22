# Aura — Research Guide

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Research entry points reconciled 25 October 2026

Aura is a local AI research system. It combines a language model with memory,
attention, internal condition and action control. Research asks which of those
mechanisms actually affect outcomes, under what conditions, and whether their
organisation supports a scientific argument about subjecthood.

A newcomer can say: “We add mechanisms that should make earlier events matter
to later decisions, then test whether removing or changing them makes the
predicted difference.” The [plain-English tour](PLAIN_ENGLISH_GUIDE.md) and
[variable definitions](VARIABLES_EXPLAINED.md) explain the parts before the
technical details.

## Follow a claim to its evidence

| Question | Start here | What to keep with an answer |
|---|---|---|
| What is the project claiming? | [Claim boundaries](CLAIM_BOUNDARIES.md), [unsupported claims](../CLAIMS_NOT_SUPPORTED.md) | Exact capability and evidence scope |
| What scientific work motivates the design? | [Scientific source map](SCIENCE_AND_SOURCES.md), [related work](DEMOS_AND_RELATED_WORK.md) | Original source, relevant mechanism and limits of the adaptation |
| What does the current ISC result say? | [ISC status](ISC_STATUS.md), [source-hashed results](evidence/ISC_STATUS_2026-10-25.json) | Source revision, seed, run identity, controls and later isolation correction |
| What is the experiment meant to test? | [ISC explained](ISC_EXPLAINED.md), [worked variables](ISC_VARIABLES.md), [readers' guide](ISC_READERS_GUIDE.md) | Definition, comparison, decision rule and open questions |
| How was it measured? | [ISC reproduction](ISC_REPRODUCTION.md) and the linked frozen protocols | Commands from the actual manifest, measurement readers, artifacts and refusal reasons |
| Does the reasoning machinery help? | [RLC programme](RECURSIVE_LATENT_CORTEX.md), [reasoning engines](REASONING_ENGINES.md) | Task family, held-out results, public answers and qualification status |
| What did a demonstration show? | [Demos and evidence](DEMOS_AND_RELATED_WORK.md) | Footage, fixture, script or controlled experiment, with that scope retained |
| Where is a particular document? | [Complete document map](DOCUMENTATION_INDEX.md) | Whether it is a current guide, historical record, frozen standard or generated report |

## The current ISC qualification matters

The completed seed-23 run recorded 22/24 under ISC-v6 and positive content and
withheld-report outputs. Later source repair found that comparison arms shared
consequence history. Those results remain in the record and need rechecking
under repaired isolation before being called clean causal confirmation. The
[isolation update](evidence/ISC_ISOLATION_UPDATE_2026-10-25.json) records that
finding. No full ISC pass or settled five-ground bridge result is reported.

The later closure method also keeps an open boundary distinct from an
unresolved question. The least-domain margin proposed in the afternoon was
withdrawn at `72dd0d703`. The night addendum at `96487c6f1` names K*, a
numeric margin about 0.0196 and an additional interventional closure test.
The revised estimator predicts change and excludes failed folds from closed
readings. These method changes need fresh confirmation. Use [current status](ISC_STATUS.md) for the exact
versioned reading and the distinction between discovery and confirmation.

## Source names and measured mechanisms

Names such as “qualia,” “will” and “consciousness” occur in the code. Read the
implementation and its consumers before assigning a scientific meaning to
the name. [Terminology](TERMINOLOGY.md) translates the project vocabulary.

| Source route | What a reviewer can inspect |
|---|---|
| [Agency orchestrator](../core/agency/agency_orchestrator.py) | Drive-to-outcome control route |
| [Authority gateway](../core/executive/authority_gateway.py) | Action-authority boundary |
| [Viability state machine](../core/organism/viability.py) | Operating-condition states and transitions |
| [Latent bridge](../core/brain/latent_bridge.py) | Internal state reaching generation controls |
| [Integrated-information estimator](../core/consciousness/integrated_information.py) | Named Gaussian/discrete estimators and search limits |
| [Subject snapshot](../core/subject/snapshot.py) | What an experimental fork saves and restores |

The [whole-system integration guide](WHOLE_SYSTEM_PHI.md) describes estimated
macro-dynamics. Small exhaustive partition searches can certify a minimum;
larger candidate searches do not inherit an exactness guarantee. Neither
this estimator nor the older graph approximation computes full IIT 4.0 over
Aura. Cite [the theory itself](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465)
for the formal target and the source for the implemented approximation.

## Reproduction without disturbing the live system

Read [TESTING](../TESTING.md), [the operational guide](../CLAUDE.md) and
[ISC reproduction](ISC_REPRODUCTION.md) before choosing commands. Do not run a
full suite, longevity campaign, chaos injection or another model process as
an incidental documentation check. On the shared host, preserve the live
instance and existing experiment owners. Set a separate `AURA_LOG_DIR` for
anything test-like, and declare worker, memory and time budgets.

A unit test checks its stated invariant. An experiment checks a hypothesis
under a protocol. A longevity run checks survival over its measured duration.
An independent replication checks whether someone else obtains the result.
None supplies the others merely by passing.

Preregistration records a hypothesis, metric, threshold, baseline and analysis
before confirmatory data. An exploratory change must be labeled and followed
by fresh confirmation when the claim requires it. Keep exclusions, failed
controls and unavailable measurements visible. Registration cannot make
post-hoc reinterpretation impossible; review the record and execution together.

## Preparing a paper

Use the [source map](SCIENCE_AND_SOURCES.md#turning-this-map-into-a-paper) to
connect each scientific claim to an external source and each Aura result to
its own evidence. The [BibTeX file](../research/references.bib) supplies an
initial bibliography. Check the cited passage and version again at submission.

Keep theoretical support, a source implementation and measured benefit
separate. A matched comparison is needed to claim improvement over an earlier
system. A human study can motivate the mechanism without establishing its
accuracy, biological equivalence or experience in Aura. The
[evidence standard](BEHAVIORAL_PROOF_STANDARD.md) and
[reviewer packet](REVIEWER_PACKET.md) provide the project's own review routes.
