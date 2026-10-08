# G09: what a broad gain would be measured on, and why it is not there yet

A plan, not a result. G09 asks for a reasoning gain from her mechanisms on
broad, independent tasks; bounded synthetic success (G03 to G06) is not it.
The [G-ledger design](../G03_SEMANTIC_CORRECTNESS_CONTRACT.md), package 4,
fixes what it means: disjoint tasks in mathematical reasoning, code
execution, planning, knowledge synthesis and transfer; her ordinary runtime
route against the ordinary model; gains, regressions, failures and
uncertainty per domain, with no pooled score hiding a failed domain.

## Where her runtime stands on 7 October

Her reasoning organ is Amplifier v2 inside the response phase
(`ResponseGenerationPhase._maybe_amplify_response` calling `amplify_turn`).
It seeds the pool with her draft, adds candidates, checks every candidate
with her verifiers (exact arithmetic, a natural-deduction prover, sandboxed
code, state traces, plan structure, citation against evidence) and replaces
the draft only with a verified answer. The design is sound; on the current
model it does nothing:

* Its budgets are constants sized for a faster model: 60% of the turn's
  remaining time capped at 30 s, 24 s for each candidate outside program
  tasks, 2,048 tokens. Her 27B decodes about 16 tokens a second and a
  settled answer runs to hundreds of tokens, so no candidate finishes.
* The batched candidate lane is switched off for live turns (5df12f9d9,
  July) because it did not carry per-candidate generation metadata.
* Her logs since 4 October show it standing down, timing out or keeping the
  draft. No turn adopted a verified amplification.
* A BIG-Bench Hard pilot through the amplifier at the deployed budget timed
  out on 3 tasks of 3; one 27B candidate took 55 to 97 s against 45 s.

Measuring broad gain now would measure that it is switched off in effect.

## Order of work

1. Budgets from the turn. The amplifier's time comes from the turn's own
   deadline and each candidate's allowance from what her draft cost (its
   recorded decode seconds and tokens), and it stands down only when one
   candidate like the draft cannot fit. No new constants.
2. Candidates in one pass. The batched lane carries each candidate's
   generation metadata, so live turns can use it again.
3. A development pilot per domain on consumed or development tasks, to
   find where her verifiers can adopt anything and to size the confirmatory
   run from measured discordance.
4. A preregistered run on fresh tasks: her ordinary answer against the same
   answer after her reasoning organ, paired, one decode each, per domain:
   * math: MATH test problems outside MATH-500, graded by its own grader;
   * code: EvalPlus problems, her visible examples as the verifier's checks,
     graded by the hidden tests;
   * planning: state-tracking and constraint tasks with exact targets;
   * knowledge synthesis: multi-hop questions over supplied passages, the
     passages as the citation verifier's evidence;
   * transfer: task families no development step touched.
5. Independent verification of the rows, as for G04 and G05.

## What would count

Fixed in the preregistration, not here. The shape: a paired gain in the
pooled suite, each domain reported with its own discordant counts and
interval, and no domain whose loss rejects. A gain confined to one domain is
reported as that and does not close G09.
