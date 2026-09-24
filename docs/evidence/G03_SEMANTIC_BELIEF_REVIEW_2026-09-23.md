# G03 semantic belief proposal review, 2026-09-23

Status: shadow implementation and focused tests. No G03 closure, serving
authority, calibrated posterior, or transfer claim.

Bryan supplied two related architecture notes on an explicit semantic belief
state. Both correctly identify a gap: a typed executable program may be
available without source evidence that selects its meaning. They propose a
larger system than the present G03 result can justify. The parts below are
mapped to current owners and to tests before any live RLC integration.

| Proposal | Current owner or decision | Required evidence |
| --- | --- | --- |
| Ordered operation, mention, and definition binding | Opt-in triadic head in the existing proposer, plus `semantic_meaning_hypothesis.py` | Gold-span held-fold binding, runtime-span replay, source-role intervention, isolated lesion |
| Source occurrence identity independent of value | Source-bound token-span identity in the shadow meaning hypothesis; IR retains register identity | Equal-value role pairs and alias transformations |
| Multiple coherent meanings | Candidate bank can project retained graphs without choosing one; scores remain uncalibrated | Joint calibration on source-disjoint constructions, with an unresolved state when banks are incomplete |
| Program ancestry from meaning | `GroundedProgramProposal` compiles a typed meaning graph and binds its source-bank receipt | Source swap changes the graph and compiled program; altered program fails ancestry check |
| Target-blind search and variable widths | Existing candidate bank and operation/argument chart search | Match training to width, measure coverage and selection together; do not train on exposed validation labels |
| Exact execution and simulated probes | Semantic floor, graph counterexamples, program portfolio, and inquiry planning | Keep feasibility and predicted consequences separate from independent observations |
| Independent observations and durable feedback | Existing source arbitration, observed program inquiries, and evidence packets | Prove source/proposal ancestry and actual reinsertion into one episode before granting confidence |
| Counterexample-guided repair and structural transformations | Existing counterexample tools and source construction folds | Source-only role swaps, paraphrases, distractors, equal-value collisions, then fresh transfer |
| Adaptive search by uncertainty | Search budgets exist; no calibrated semantic uncertainty controller yet | Compare a targeted policy to equal-compute fixed search with a complete denominator |
| Recurrence over semantic state | RLC currently owns latent episodes; no semantic belief branch is qualified | Demonstrate a non-noop update from an independent observation before changing RLC serving |
| Procedure, macro, concept, and representation invention | Existing procedure currency, operator invention, and concept handle own parts of this path | Held-out benefit and independent verification before any durable or fused materialization |
| Self-modification and fast weights | Separate later research paths, not a G03 parser repair | Frozen external evaluator, rollback, causal lesions, resource accounting |

The v0 meaning hypothesis is **projected from target-blind candidate-bank
evidence**. It preserves the operation and ordered source bindings, and it can
compile a program from that graph. This proves identity and ancestry plumbing;
it does not prove the graph was inferred from the source correctly. It gives
no posterior probability, and a missing argument mention produces no graph
instead of an invented one. The existing bank is incomplete under its bounded
search, so an empty graph set means unresolved, not impossible.

The first discriminating measurement supplies gold operation and mention
spans but never an answer. A source-only joint binding head must choose the
correct typed referent on construction-held sources. Failure there implicates
the learned binding or representation before program search. Success there,
followed by failure on runtime spans, implicates recognition or proposal.
Success on both, followed by answer failure, implicates compilation, search,
or selection. The lesion control makes every opposed role a tie when this
learned signal is zeroed. This is an upper-bound diagnostic, not serving
evidence.

The notes' later claims about general reasoning and self-improvement remain
hypotheses. A meaning graph cannot recover information absent from the source;
execution of two plausible programs cannot decide which the user intended.
Any future observation update must use the existing evidence provenance
mechanism so several model views of one source do not count as independent
witnesses.

## Frozen gold-span binding diagnostic

The first source-trained triadic feature compressed operation and mention into
one product before comparing a definition. With gold operation and mention
spans on held construction fold 0, it chose 530 of 1,096 opposed roles, with
566 wrong. Receipt:
`semantic-triadic-gold-fold0-20260923/report.json` in the RLC evidence root.

A separately versioned joint feature preserves the three vectors and their
relative token geometry. It was fixed before folds 1 and 2 ran. On the same
gold-span diagnostic, fold 0 was 952/1,096, fold 1 was 992/1,086, and fold 2
was 845/1,042. The immutable receipts are under
`semantic-triadic-gold-v2-fold{0,1,2}-20260923/report.json`. These are upper
bounds with gold spans supplied, not end-to-end or serving evidence.

The matched fold-0 component lesion is decisive for interpretation:
geometry-only retained 944/1,096 correct; representation-only retained
459/1,096. The full feature improved by only eight net correct bindings over
geometry-only. Receipt:
`semantic-triadic-gold-v2-ablation-fold0-20260923/report.json`. The strong
gold-span result is mostly a source-layout shortcut. It cannot justify a claim
of semantic transfer or G03 closure. The next source study must counterbalance
layout and roles, then measure runtime-recognized spans and complete proposer
and selector crossfit before considering any serving authority.
