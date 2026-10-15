# Finished probes must leave the fallback

Original case 9 ran at a63be5233 and completed normally in 174.232312
seconds without a win. It selected bodies on 539 of 732 recorded actor
decisions. The return trace includes 202 facing-trial decisions, 43
provisional steering decisions and 13 ordinary planning decisions. The
current-body budget fix reached the ordinary actor, but the play still
failed. Evidence: `/private/tmp/codex-steer/owned-budget-routing-original9-1014`.

The trace locates a later idle loop. Body 74 reached 166 fallback returns
over its 167 decisions. Its controls were all marked known, but choosing
had no moving way. The global discovery count was 11 for four keys, below
the old limit of 16. The fallback therefore called `try_controls` on each
picture. That routine found no unknown control, returned without input,
and never advanced the global count. Rechecking also depended on reaching
16, so neither branch could make progress. Body 108 later showed the same
pattern.

The fallback now uses `body_probe_is_active`, as the earlier actor gate
does. The existing recheck also uses that phase result instead of assuming
the global counter must have reached its old limit. An active physical
trial and its final rest still own their interval. A finished pass can
yield to other actions, and the existing recheck can start a fresh
independent experiment when its delay expires.

The regression drives the actual actor with a retained body, measured
zero responses, no moving direction and a global discovery count below its
limit. It must skip the exhausted routine, preserve the body before the
recheck deadline, then perform the existing recheck after that deadline.
The focused discovery, unknown-control and input-receipt suites passed
42 tests in 6.12 seconds. This proves the branch repair, not successful
movement or a completed original game.

The private branch diagnostic had imported the actor before installing its
facing callback wrapper, so that run's facing-receipts file is empty. Its
separate actor return records prove facing trials were issued; they do not
prove what each facing observation measured. The callback wrapper now also
replaces the actor's imported function reference. Future branch records
include current facing state directly. Historical evidence was preserved.

All 734 recorded vision-worker inputs replayed with exact entity snapshots
and no missing captured RGB digests. This includes the inputs processed by
the worker but not published to the actor. Replay evidence is retained in
`/private/tmp/codex-steer/owned-budget-pixel-fidelity-1014`.

Ruff, writing, changed-method size, layering, dependency graph, compile and
lint checks passed. Smoke passed 165 tests with one skip in 48.31 seconds.
The known aggregate size excess, `ResponseGenerationPhase` method excess
and 123 governance regressions remain; no baseline was relaxed.
