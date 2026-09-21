# Bounded reach and developmental gain

Bryan's follow-up criterion is adopted as an experimental distinction:
retained, system-generated machinery must improve performance on unseen
families at a matched resource budget. A code change by an engineer does not
itself establish developmental learning. A new library entry does not itself
establish useful reach.

## Search bound

For an explicit computable program prior q and a compatible search scheduler,
the relevant bound has the form

    search_cost <= c * min_p ((execution_cost(p) + checking_cost(p)) / q(p))
                   + implementation_overhead

over solutions recognizable by the task's checker. With a prefix code of
length ell(p), q(p) can be proportional to 2**(-ell(p)). A ten-bit reduction
then reduces that candidate's multiplicative search-bound term by 1024,
provided execution, checking, normalization and overhead remain comparable.
It is not a measured speedup or a guarantee that the whole search becomes
1024 times faster. The shortest program need not minimize length-weighted
execution cost. Kolmogorov complexity is not an executable cost meter; use
the declared encoding, measured work and retained artifact identities.

The primary reference is [Optimal Ordered Problem Solver, Machine Learning
54 (2004)](https://people.idsia.ch/~juergen/oopsweb/oopsweb.html). Its
incremental searches reuse earlier solutions under a stated program bias;
its limited-storage construction and bias-optimality assumptions matter.

## What Aura must measure

Record candidate reach separately from selected-answer correctness. A correct
program in a searched set can still lose to an incorrect interpretation. G03
currently exhibits exactly that difference. Answer accuracy is bounded by
correct-candidate reach only when both refer to the same candidate set and
the returned answer comes from that set.

The retained-machine experiment needs frozen source code and base model,
separate acquisition and evaluation data, no acquisition from held-out
answers, matched execution and checking budgets, and artifact provenance.
Compare incumbent, acquired machinery, withdrawal, and rescue. Measure
regressions by family alongside aggregate gain. Report acquisition cost and
the number of later tasks needed to amortize it. Repeatedly inspected
development rows cannot become a fresh transfer claim.

Existing implementation points are `core/cognition/sequence_reach.py`,
`core/cognition/operator_invention.py`, the retained-operator kernel, and
the shared reversible causal trials. Sequence answering and developmental
measurement already share executable retained meanings. Their bounded tests
do not establish broad gain; no second substrate or search store is needed.

The current sequence check establishes agreement with supplied transitions.
Fresh task correctness needs an independent outcome check beyond those
examples. The G09 criterion remains open until that experiment is run.
Subjecthood and phenomenal experience remain separate claims; search gain
and lesion/rescue alone do not settle them.
