# Each source effect keeps its reaching conditions, 15 October 2026

The source reader's calls and writes had an address and arithmetic but no
individual reaching predicates. Their containing `MethodPath` held the
conditions at the end of that path. A later branch could therefore appear
to guard an earlier effect when downstream code inspected the whole path.

`SourceCall` and `SourceWrite` now retain the conditions and numeric
assumptions present when they are reached. A shared immutable invocation
record keeps the method target, receiver origin, supplied parameters and
known held-key context. Unknown keys remain `None`; an explicitly neutral
input is an empty frozen set. Mutable or invalid keyboard context refuses
the read. Later branches and constant reads do not rewrite earlier receipts.

Free functions also had an incorrect default receiver: an unqualified
`callback` was read as if `this` meant `callback`. An unspecified free-function
receiver now remains symbolic `this`. A caller can supply a bounded literal
receiver for one invocation, with that origin recorded. This API accepts a
conditional invocation; it does not prove a callback's runtime binding.

The focused source-motion and reference suites passed 72 tests. They cover
an effect before a branch, a call between two branches, later assumptions,
known/unknown input contexts, free callback receivers and an unsupported tail.
The new invariant is registered against its executable tests.

An offline read of original case 18's `ship_move`, with explicitly supplied
receiver `ship` and neutral keys, retained both success-looking and
failure-looking calls with their separate guard prefixes. For example, the
animation call with argument `good` retained eleven reaching guards; the
sound-stop call before the motion branches retained none. These labels were
not promoted into an outcome oracle or a task role.

The read still produced 64 partial paths, unsupported `Delete` instructions
and exhausted-path issues. `complete` and `effects_composed` both remained
false. The new custody does not repair those missing semantics or qualify
an exact simulation. The diagnostic receipt is
`artifacts/flash-campaign/2026-10-15-source-effect-prefix.json`.

No source physics, private game state, identity threshold or live runtime
was substituted for observation. Original 9, G09 and the holdouts remain
excluded. No new gameplay outcome is claimed by this change.

Smoke passed 165 with one skipped; compile, Ruff, layering (36
grandfathered), dependency consistency (168 packages) and writing passed.
The writing baselines were not raised. The full offline suite was not run
beside the exclusive model job.
