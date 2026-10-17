# Guarded source effects, 16 October 2026

The source reader's path limit hid branches of the recorded case-18 movement
method. Raising the limit would still enumerate combinations of independent
branches. This checkpoint adds a separate, bounded acyclic graph reader.

## Representation and integration

`core/perception/guarded_effects_in_flash_methods.py` visits each reachable
instruction in topological order. Stack, register and memory values join
under predecessor guards. Values share a DAG with 32,768 nodes at most;
instruction, effect and stack budgets remain separate. Loops require a
separate summary and are refused. Conflicting constant pools, unbound
registers and mismatched stack heights cannot supply a completed graph.

Every load and coercion retains its prior memory token and a successor
token. Getters and object coercion can have effects. Calls and deletions
also retain memory order and unknown results. A later load cannot silently
reuse a value from before an opaque call. Each effect retains its reaching
guard and exact instruction address; `slice_effect` includes that guard.

`ProgramRead.method_effects` offers this read on demand from its complete
current action sources. An absent or multiply bound method supplies no
graph. No eager whole-program pass was added to the perception loop.

Here `complete` means coverage of supported normal bytecode control flow.
It does not mean composed callees, resolved aliases, exception coverage,
known runtime state or an executable controller. `effects_composed` remains
false. The graph is source data and is never executed by the production
reader. Visible occurrences, source contact semantics and task outcomes
still require their own evidence.

## Measurements

`tools/inspect_guarded_source_effects.py` inspected the existing local source
file whose digest is
`a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
This starts no browser or model and sends no input.

The movement method retained all 450 instructions and 19 branches in 657
nodes, with one normal exit. Its effects include 17 member assignments,
21 variable assignments, 15 method calls, 10 function calls and four
deletions. The existing reader still produces 64 paths and 12 path-budget
issues with unknown keyboard state. Both readers retain their separate
qualification meanings.

The same inspection deliberately included the outcome-display helper.
That helper remains incomplete at `NewObject`, source offset 562588.
The two-method proof step therefore returned 1. The failure was retained;
it was not replaced with a passing score. The inspection took 0.028385
seconds inside the process. Its artifact is
`artifacts/flash-campaign/2026-10-16-guarded-source-effects/source-graphs.json`.

Focused tests: 86 passed. A separate concrete byte-addressed fixture VM
agreed with the graph on 100 input assignments across 16 unknown binary
branches. Other fixtures cover stack and register joins, alias mutation
through an opaque call, getters, conditional deletion, untaken calls,
multiple returns and resource refusals. These fixtures do not execute the
original game or establish AVM1 semantics beyond their stated operations.

No G09 input, holdout, accelerator job or native runtime was changed. The
native app has not loaded this checkpoint. Source-occurrence binding,
source-defined contacts and witnessed ordinary native gameplay remain open.

## Ordered register state and final gates

Review found that cached register values could outlive opaque effects.
Registers now load and store through the same ordered VM-state graph. A
fixture helper that overwrites a register changes the later observed value;
the previous cached value is not reused. Join logic separately retains
whether every predecessor initialized the register. Duplicate parameter
names and conflicting preloads are refused rather than losing an input.

After this correction, the recorded movement method still retains all
450 instructions and 19 branches, now in 771 nodes. The final two-method
inspection took 0.032805 seconds and retains the same explicit helper
refusal. Its artifact is `ordered-source-graphs.json` in the directory above.
This supersedes the initial 657-node representation, whose artifact remains.

The final focused source suite passed 188 tests. One initial expanded
invocation named a nonexistent test path and collected no tests; its
supervisor record remains separate from the corrected passing run.
Smoke passed 165 tests with one skip. Lint, compile, layering, generated
DEPS consistency and writing passed. Governance retains 131 existing
regressions; module size retains the same unrelated four regressions and
197 excess lines. Neither baseline was changed. The full suite was not run
beside the protected exclusive model job.
