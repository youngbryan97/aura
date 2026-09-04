# WorkToDo — every actionable item

Source: WorkToDO.pdf, 48 pages. Items are the ones that name a defect or a
buildable thing. Status is measured, not asserted.

## Concrete defects with a reproducible failure

| # | Item | Status |
| :-- | :-- | :-- |
| 1 | Global Workspace race is flaky | **DONE** — competition is one instant; 12/12 and 10/10 deterministic |
| 2 | `InteriorityService.apply()` — the general effect bus (affect, somatic, goals, curiosity, action filtering, cognitive budget) has no production caller | **DONE** — all four channels were broken independently and the ledger they appraise had no writers; `core/interiority/stakes.py` fills it, `_observe_social_turn` calls apply, 19 tests |
| 3 | Stream of Being: LLM narrative can become evidence for the state it was generated from. Self-confirming phenomenology | **DONE** — four loops cut; `core/consciousness/narrative_provenance.py` binds text to the state digest it came from and grades what it may be evidence for; 13 tests |
| 4 | Whole-agent lineage manager is NOT WIRED and must not simply be turned on. Either remove/rename or isolate with explicit resource and authority boundaries | OPEN |

## Architectural: ontological duplication

| # | Item | Status |
| :-- | :-- | :-- |
| 5 | Canonical state ownership. Many systems own overlapping affect / self / attention / continuity / agency / welfare / goals / uncertainty. Modules become estimators and consumers of one authoritative state | OPEN |
| 6 | Disagreement becomes a cognitive event rather than a silent average. Three subsystems differing on uncertainty is metacognitive evidence | OPEN |
| 7 | One authoritative self. SelfObject, AuraNow.self_state, identity engine, continuity engine, workspace ownership, substrate self-representation become views over one state | OPEN |
| 8 | Conation / welfare / affect / Will converge into a clean causal sequence that can distinguish dislikes-X from wants-but-refuses from judges-impossible from judges-unsafe from prefers-Y | OPEN |

## Evidence and epistemics

| # | Item | Status |
| :-- | :-- | :-- |
| 9 | Matched independent experiment: frozen Aura vs the base model, equal compute, tokens, tools, time, information, on externally authored unseen tasks, with an ablation ladder | OPEN |
| 10 | Causal credit assignment across thousands of mechanisms: counterfactual replay, causal graphs, Shapley-like attribution, eligibility traces, selective interventions | OPEN |
| 11 | Epistemic independence: no important adaptive mechanism defines its own success criterion after observing its result | OPEN |
| 12 | Model horizon: know when a simulation has left its trustworthy regime and must not drive irreversible choices | OPEN |
| 13 | Resource rationality: value-of-computation decisions, including whether to spend on learning at all | OPEN |

## Development

| # | Item | Status |
| :-- | :-- | :-- |
| 14 | Memory allocation across representational substrates: symbolic vs adapter vs weights, by expected lifetime usefulness, confidence and interference cost | OPEN |
| 15 | Self-repair under unknown failures: recognise a failure the ontology does not contain, infer the broken invariant, localise, invent a repair, verify, integrate the new failure concept | OPEN |
| 16 | Architectural self-discovery: shadow architectures, forked trials, compatibility interfaces, migration proofs | OPEN |
| 17 | Primitive invention that feeds later invention, rather than composition of human-supplied primitives | OPEN |
| 18 | Autonomous transfer: discover the invariant without anyone tagging both domains | OPEN |
| 19 | Identity relation under deep modification: causal continuity, not a hash | OPEN |
| 20 | Value development levels: separate what may change from what may not, and which learning touches which | OPEN |
| 21 | Social cognition over years: trust, attachment, norms, shared history, repair after conflict, nested beliefs | OPEN |
| 22 | Embodied epistemic agency: choose observations by expected information gain | OPEN |
| 23 | Native cognitive medium: internal representations optimised for computation rather than communication | OPEN |
