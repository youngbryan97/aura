# Source arithmetic reaches ordinary control probes

The assigned original program calls methods to update movement. The old
statement reader did not preserve enough branch information to prove which
parameter selected a signed update. Its unconditional body physics remained
empty. Replacing that uncertainty with guessed acceleration would hide the
failure rather than repair it.

The new bounded AVM1 reader retains source addresses, function bindings and
branch successors. Its companion symbolic reader retains unresolved paths,
call arguments, source writes and predicates. Malformed containers and
unsupported instructions make their coverage incomplete. Equality follows
the supported AVM1 coercion cases; unsupported coercion remains unknown.
Assumed constants remain explicit assumptions.

The ordinary `read_program` path now retains the immutable action sources
and conditional method arithmetic. The existing program cache retains the
bounded arithmetic reading and its source digest; cache version 7 prevents
an older reading from bypassing this work. The guide carries this reading
into `HowItWorks.source_trial_keys`. Pixel experiments prefer these keys
among the current body's already authorized controls.

This connection changes probe order only. A vector update's field units
are not promoted to acceleration, a signed alternative is not promoted to
steering direction, and no source name identifies a visible object. The
ordinary causal and material tests still decide ownership and response.

An integration test reads the assigned original SWF, passes its result
through the real cache encoding and guide, and checks that the conditional
vector keys reach probe ordering. Its world model still has no unconditional
drives, turns or paces. Tests also reject missing current controls, incomplete
source, zero increments and nonfinite values.

The six source/material test files passed 188 tests in 15.40 seconds.
That includes the source integration and thin-outline rejection cases.
The source reader does not execute the original program, and this checkpoint
does not prove a controlled body, a win or presentation readiness.
