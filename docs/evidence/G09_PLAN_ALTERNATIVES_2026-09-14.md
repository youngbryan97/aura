# G09: Search alternatives before learning which plan works

The dependency planner returned the first feasible plan. Its visited-state key
merged different execution paths whenever they reached the same prerequisites.
Whole-task outcome learning could rank supplied alternatives, but the planner
could not supply those alternatives itself.

`plan_procedure_candidates` now retains distinct dependency paths. It shares
regression and execution contracts with the existing first-plan API. Callers
bound depth, expansions, and plan count; each exhausted budget is reported
without discarding plans already found or claiming the frontier was exhausted.
An already satisfied goal produces no action to learn or execute.

`search_valued_procedure_plans` passes these actual alternatives to the existing
ActionValueModel. Search does not create outcome credit. Execution remains a
pending observation until an external evaluator reports the task outcome.

The regression test enumerates reader/writer cross-products that share
prerequisites. Another test executes two type-correct alternatives, measures
which sum is correct, reopens the outcome ledger, changes registry order, and
selects the successful program for new inputs. The new answer is 26. Merely
executing that new task does not add a third measured outcome.

Verification: 100 procedure tests passed; the combined procedure and fusion
suites passed 156 tests. This is shared planning infrastructure and a controlled
integration test. Ordinary language-to-plan admission and fresh broad-task gain
are not established by this test. G09 remains open.
