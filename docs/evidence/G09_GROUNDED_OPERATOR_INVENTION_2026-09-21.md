# Grounded Operator Invention

## Findings

The screenshot review's finite-budget criterion led to inspection of the
existing operator kernel, not a second substrate. Four disconnected contracts
prevented its output from supporting that criterion:

1. The live caller discarded example outputs. Its default correctness judge
   accepted any candidate returning non-None values, and passed a constant
   compression credit of one.
2. A size-three search over unary functions cannot express a binder around
   binary arithmetic such as doubling. Returning but incorrect candidates
   could also consume the separate 64-offer limit.
3. Installation discarded the floor term, so the existing library reader could
   not offer that learned term to later searches.
4. The outer developmental transaction did not snapshot the operator kernel.
   A combined test exposed an installed operator surviving an outer rejection
   that reported the change was undone.

Search coverage had a separate measurement error: `min(cap, space_size)` was
reported as observed work even when a consumer stopped early. An offer limit
or an accepted candidate therefore could look like exhaustive enumeration.

## Repairs

The sequence-answer boundary records each attempt once, with complete shown
input/output pairs. Unanswered query inputs remain separate from examples.
The proposer checks every expected output; an optional extra judge cannot
bypass that comparison. Inputs support integers and nested sequences without
coercing floating-point inputs to integers or flattening sequence examples.

The invention action searches through size four under the existing 4,000
examined-term budget without the additional offer cap. It computes a declared
structural reuse saving instead of inventing a positive credit: for `u` distinct
demonstrated uses, term size `s`, and the existing reference cost `r`, the
saving is `u * (s-r) - (s+r)`. This charges the stored body and its definition
reference. It is an AST-symbol accounting convention, not measured runtime
speedup, byte compression, or a held-out-corpus gain.

Installed operators retain their floor terms. The existing library reader
includes them; kernel rollback removes them. The existing developmental
transaction now snapshots and restores both installed semantics and kernel
rollback lineage. Attempt observations and audit verdict history remain.

Coverage reports use observed iterator counts, including early stops, and
return null coverage/exhaustion when only capacity is known. The report schema
is `aura.operator_search.reach.v2`.

## Executable Evidence

`tests/test_the_kernel_gets_a_caller.py` exercises real floor enumeration,
installation, and execution. Labeled doubling demonstrations yield a program
that also answers unseen negative, zero, and larger inputs. With the same
size-three search and 400-examined-term limit, the learned leaf permits the
held-out numeric test; removing it loses that result and restoring it recovers
it. This is a same-function reuse experiment, not transfer to a new family.

Additional tests cover unlabeled probes, contradictory examples, exact large
integers, nested sequences, live answer-boundary data flow, outer rejection,
exception rollback, and retained observations. Search tests cover unmeasured
capacity, early consumer exit, offer limits, and inconsistent counters.

The combined focused run passed 101 tests. Five additional developmental
transaction suites passed 51 tests. The initially failing combined runs were
not dismissed as flakes: they exposed the missing transaction membership.
Mechanism tests now explicitly start without an unrelated held-out history;
separate integration tests exercise rejection by that outer judge.

Smoke passed 164 tests with one skip; lint and compile passed. Integration
gates also found two stale ownership entries for already-merged hobby
write-behind calls and one undeclared memory-organ warmup dependency. The two
ownership entries retain their debt classifications. The kernel dependency
declaration permits only `core.memory.embedding_runtime`, not the entire
memory package, for the existing shared-encoder lifecycle call.

## Remaining Obligations

G09 remains open. No live deployment, fresh held-out-family improvement,
end-to-end knowledge gain, persistent operator recovery across restart, or
frontier performance is established by these tests. Library growth may still
change the ordering and cost of bounded enumeration. Its acquisition and
retrieval costs need matched measurement. The legacy kernel's unlabeled
adversarial probes test execution survival, not independent answer correctness.
The repaired production caller's example fit must not be relabeled as such
held-out correctness evidence.

Related: [screenshot review](G_LEDGER_SCREENSHOT_REVIEW_2026-09-21.md) and
[master ledger](../AURA_1_0_MASTER_TODO.md).
