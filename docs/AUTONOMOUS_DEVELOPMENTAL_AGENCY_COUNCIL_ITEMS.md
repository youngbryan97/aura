# Every proposal in the council document, adjudicated

One hundred items, taken from all seven responses. Struck through when
something in this repository holds it. Marked *rejected* with a reason where
the proposal is wrong or already refuted, and *needs a person* where it cannot
be closed from inside.

The five architectures agree with each other and with the design in
[AUTONOMOUS_DEVELOPMENTAL_AGENCY.md](AUTONOMOUS_DEVELOPMENTAL_AGENCY.md) on the
central move — developmental actions in the same choice set as ordinary ones,
scored by a value estimated from a record — so the items below are the
differences, not the agreements.

## The developmental object

1. A developmental action carries a probe, a budget, a success test and a
   rollback, not just a name and a way of doing it.
2. A cause hypothesis is a term with an open constructor, so the taxonomy
   cannot become the next hand-written ceiling.
3. An opportunity is a record — trigger, scope, hypothesis, value, uncertainty,
   cost, risk, evidence — and they queue rather than being decided fresh.
4. A developmental objective is a term: context, desired change, budget,
   invariants.
5. Artifact status: shadow, canary, active, retired, quarantined.
6. Provenance and a utility posterior on every artifact.
7. Versioning, so an installation can be pointed back at what it replaced.

## Detectors

8. Recurrence.
9. Inefficiency: cost far above the cost of comparable solved families.
10. Compression: a candidate that shortens the joint description of the corpus.
11. Disuse and obsolescence.
12. Prediction error / residual structure with no constructor.
13. Uncertainty: high variance in the self-model, reducible by experiment.
14. Slack: unused compute beside unresolved value.
15. Transfer hint: structural similarity without a label.
16. Predicted bottleneck: a forecast that the current search misses a class.
17. Meta-limit: the improver's own yield has gone flat.
18. Search waste ratio: one minus the answer's length over nodes visited.
19. Meta-utility derivative: solutions found per unit compute, plateauing.
20. Curiosity as compression progress rather than raw surprise.
21. Detectors are themselves terms, so a new one is an admission.

## Value and calibration

22. An option-value term: what a change does to reach, not only to cost.
23. Sampling from the posterior rather than taking the argmax.
24. Calibration of the estimate against what actually happened, or the policy
    games its own proxy.
25. Sequential testing with a stated confidence, not a one-shot comparison.
26. A finite-sample bound on how many probes a claim needs.
27. Non-monotonic acceptance: a change may be taken that costs now.
28. An eviction score weighing use against the cost of re-deriving.
29. A separate development budget, allocated as a bandit against ordinary work.
30. An adaptive threshold: high when resources are tight or a live task matters.

## Evaluation and governance

31. Off-trigger-sample evaluation.
32. Promotion tiers, with heavier evidence for the parts that decide.
33. A rollback stack, and archive rather than destroy.
34. Receipts that chain, carrying the initiator and an empty external-command
    field.
35. A probe hierarchy: cheap checks before dear ones.
36. Conservative promotion: beat the frozen battery and do not regress the
    ordinary one.
37. The gate is not a destination — no developmental action may install to the
    thing that decides what is kept.

## Self-model and diagnosis

38. A counterfactual self-model predicting the effect of a change before it is
    made.
39. A posterior over bottleneck hypotheses, scored by predictive usefulness.
40. `why_suboptimal` as a searched classifier rather than a switch.
41. Interventionist credit assignment: spend a small probe budget on two rival
    changes and keep the winner.
42. A capability forecaster: predicted time to invent, held against actual.
43. The bottleneck as the value of intervening, over general interventions and
    not only lesions.

## Transfer and retrieval

44. Structural fingerprints over terms, so retrieval is by shape not by name.
45. Retrieval scored by predicted gain, with rejection when it does not help.
46. The learned cross-domain relation kept when the probe wins.
47. Frequent subtree mining over the whole corpus, not pairwise comparison.

## The library

48. Merge two artifacts into one.
49. Specialise one that is too general.
50. Recompress: re-encode the library in a shorter metalanguage.
51. Retire under a size budget, with dependencies respected.
52. A library-wide objective rather than a per-entry gate.

## Live wiring

53. Developmental primitives callable from inside the term language, so any
    program can trigger development.
54. `operator_invention` reachable from an internally generated trigger.
55. The autonomy loop offers the opportunity and the policy makes the decision.
56. `_pathway_cognitive_loop` stops returning nothing.
57. Drives become causes in the value, not scripts that pick the work.
58. Rebuilding after a retraction is the controller's problem, not the
    caller's.
59. The proposer reaches past depth two.

## Experiments

60. Self-initiation lesion: remove the external channel; development must
    still happen.
61. Opportunity lesion: destroy the evidence, keep the tasks; development must
    stop.
62. Meta-causality lesion: put the old machinery back before the second
    episode; the second change must become less likely.
63. A static check that the harness calls no developmental routine.
64. The three-generation test with an event trace naming every initiator.
65. Information versus computation: a task with a hidden bit that no search can
    reach.
66. Transfer against three controls: isomorphic, superficially similar,
    unrelated.
67. Search-strategy novelty judged behaviourally and causally, not by name.
68. Metrics: initiation rate, detector precision and recall, cost, generality,
    transfer, improver quality, false promotion, harmful retirement, rollback,
    provenance integrity.
69. An ablation table where each row names what must collapse.
70. A long run, which is time rather than code.

## Theory to record

71. The envelope reading of reach: keep the predecessor, or replacement breaks
    inclusion.
72. Blum's speedup: for some functions there is no best program, so
    development need not terminate in an optimum.
73. Löb: no proof that a change preserves what it was meant to preserve.
74. The completeness result: fair search over a universal space reaches any
    finite sequence of beneficial changes with probability one, and with no
    useful bound.
75. Levin's bound as the reason reach is the operative quantity.

## Rejected

76. *A new floor primitive for reflection.* Four responses call for adding
    `reflect`, `eval_in_sandbox` and `install` to the floor. The floor is
    already universal and carries quotation, a self-interpreter and its
    certificate; adding primitives would enlarge the trusted base for no gain
    in expressiveness. What was actually missing was a caller, and a caller is
    not a primitive.
77. *"Current Aura has no homoiconic substrate / meta-levels are separated by
    language boundaries."* Six of the seven assume this. It stopped being true
    with `the_floor_she_stands_on`, and the assessments that follow from it are
    about a different system.
78. *"The library only grows / append-only."* Not true here: retirement,
    quarantine, retraction and cascade already existed.
79. *An immutable governance core outside the language.* Already the design,
    and already proved necessary by `a_gate_inside_the_space_cannot_hold`.
80. *Simulated-annealing acceptance of changes that make things worse.* The
    non-monotonicity is real; accepting a change that measurably loses on the
    probe in the hope of a later gain is not the same thing, and there is no
    evidence in the record that would set the temperature.
