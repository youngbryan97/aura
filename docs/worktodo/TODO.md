# WorkToDo — every actionable item

Source: WorkToDO.pdf, 48 pages. Items are the ones that name a defect or a
buildable thing. Status is measured, not asserted.

## Concrete defects with a reproducible failure

| # | Item | Status |
| :-- | :-- | :-- |
| 1 | Global Workspace race is flaky | **DONE** — competition is one instant; 12/12 and 10/10 deterministic |
| 2 | `InteriorityService.apply()` — the general effect bus (affect, somatic, goals, curiosity, action filtering, cognitive budget) has no production caller | **DONE** — all four channels were broken independently and the ledger they appraise had no writers; `core/interiority/stakes.py` fills it, `_observe_social_turn` calls apply, 19 tests |
| 3 | Stream of Being: LLM narrative can become evidence for the state it was generated from. Self-confirming phenomenology | **DONE** — four loops cut; `core/consciousness/narrative_provenance.py` binds text to the state digest it came from and grades what it may be evidence for; 13 tests |
| 4 | Whole-agent lineage manager is NOT WIRED and must not simply be turned on. Either remove/rename or isolate with explicit resource and authority boundaries | **DONE** — isolated, not turned on: `core/self_modification/lineage_enclosure.py` enforces resource, authority and identity boundaries; the false `lineage_manager` registration removed; 30 tests |

## Architectural: ontological duplication

| # | Item | Status |
| :-- | :-- | :-- |
| 5 | Canonical state ownership. Many systems own overlapping affect / self / attention / continuity / agency / welfare / goals / uncertainty. Modules become estimators and consumers of one authoritative state | **DONE** — `core/canonical` declares 19 channels across 8 domains with precision-weighted fusion; five subsystems that each owned affect are estimators; `make state-ownership` ratchets private copies (14 → 12, may only shrink) |
| 6 | Disagreement becomes a cognitive event rather than a silent average. Three subsystems differing on uncertainty is metacognitive evidence | **DONE** — spread survives fusion, raises a `Disagreement` naming each position, and `reconcile()` turns it into evidence on epistemic.uncertainty and self.coherence, excluded from their own source set so it cannot drift |
| 7 | One authoritative self. SelfObject, AuraNow.self_state, identity engine, continuity engine, workspace ownership, substrate self-representation become views over one state | **DONE** — SelfObject reads the canonical self and affect instead of one engine, defaults are flagged as defaults, attentional coherence renamed to say what it measures; private copies 14 → 5 |
| 8 | Conation / welfare / affect / Will converge into a clean causal sequence that can distinguish dislikes-X from wants-but-refuses from judges-impossible from judges-unsafe from prefers-Y | **DONE** — `core/agency/agency_kind.py` walks the four in order and names which settled it; 25 tests built as a confusion matrix so a scenario for one kind must not return another |

## Evidence and epistemics

| # | Item | Status |
| :-- | :-- | :-- |
| 9 | Matched independent experiment: frozen Aura vs the base model, equal compute, tokens, tools, time, information, on externally authored unseen tasks, with an ablation ladder | **DONE** — `core/evals/matched_experiment.py`; ran on both real 27B models, 24 tasks: +0.042 delta, p=1.0, 1 discordant of 24, correctly reported as not attributable. See `docs/evidence/matched_experiment/` |
| 10 | Causal credit assignment across thousands of mechanisms: counterfactual replay, causal graphs, Shapley-like attribution, eligibility traces, selective interventions | **DONE** — `core/verify/coalition_credit.py` adds coalition attribution over the existing lesion registry, so redundancy stops reading as irrelevance; checked against constructed ground truth where the answer is known |
| 11 | Epistemic independence: no important adaptive mechanism defines its own success criterion after observing its result | **DONE** — `core/verify/epistemic_independence.py` seals a criterion before the run and refuses a redefinition after it has judged; `make epistemic-independence` finds bars derived from what they judge; checked against its null, all three planted forms caught |
| 12 | Model horizon: know when a simulation has left its trustworthy regime and must not drive irreversible choices | **DONE** — `core/verify/model_horizon.py` measures support and local calibration, model-agnostic; four standings keep "nothing near this was checked" separate from "the model is wrong here"; composed into the agency ceiling, which gives `turn_budget()` its first caller |
| 13 | Resource rationality: value-of-computation decisions, including whether to spend on learning at all | **DONE** — `core/cognition/value_of_computation.py`; more thinking is worth something only if it can change the decision, measured against a null arm of deliberations that spent nothing; `worth_learning` refuses a real gain that will be needed twice |

## Development

| # | Item | Status |
| :-- | :-- | :-- |
| 14 | Memory allocation across representational substrates: symbolic vs adapter vs weights, by expected lifetime usefulness, confidence and interference cost | **DONE** — `core/memory/substrate_allocation.py`; confidence gates come before value, and adapter capacity contention is what makes the weights ever the right answer (without it that branch is unreachable) |
| 15 | Self-repair under unknown failures: recognise a failure the ontology does not contain, infer the broken invariant, localise, invent a repair, verify, integrate the new failure concept | **DONE** — `core/resilience/unknown_failure.py`; signatures are learned from instances rather than read off catalogue prose, so the recogniser has a null: a repeat of a known fault must come back KNOWN |
| 16 | Architectural self-discovery: shadow architectures, forked trials, compatibility interfaces, migration proofs | OPEN |
| 17 | Primitive invention that feeds later invention, rather than composition of human-supplied primitives | **DONE** — `core/cognition/primitive_invention.py`; novelty judged against the closure rather than the base set, and depth measures whether anything compounds — a wide vocabulary of generation-one inventions is depth 1 |
| 18 | Autonomous transfer: discover the invariant without anyone tagging both domains | **DONE** — `core/cognition/transfer_search.py` retrieves by naming-invariant shape and verifies each candidate against its shuffled null; the mapper itself had to stop matching relations as strings, which was scoring genuine cross-vocabulary analogies at exactly zero |
| 19 | Identity relation under deep modification: causal continuity, not a hash | **DONE** — `core/identity/continuity_relation.py`; gradual total replacement is continuous, instantaneous is not, and the two end in the same bytes; a restored backup matches the hash and breaks the chain |
| 20 | Value development levels: separate what may change from what may not, and which learning touches which | **DONE** — `core/governance/value_levels.py`; four levels, an authority table with no setter, and nothing in it reaches constitutive — absent rather than set low, so adding one is a visible act |
| 21 | Social cognition over years: trust, attachment, norms, shared history, repair after conflict, nested beliefs | **DONE** — `core/social/long_horizon.py`; trust computed from history rather than stored, so a tested bond and an untested one at the same level are distinguishable; second-order beliefs licensed only by an act that discriminates |
| 22 | Embodied epistemic agency: choose observations by expected information gain | **DONE** — `core/perception/expected_information_gain.py`; an observation every hypothesis predicts alike scores zero however interesting it looks, and gain is separated from worth |
| 23 | Native cognitive medium: internal representations optimised for computation rather than communication | OPEN |
