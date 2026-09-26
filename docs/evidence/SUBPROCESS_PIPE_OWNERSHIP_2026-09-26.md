# Exited-child pipe ownership

A bounded test process stalled in `communicate` after its direct child exited.
A descendant still held the output writer. The gateway's work-bound paths
then waited for pipe EOF without a bound, including after their budget-stop
branch. The retained OS sample is
`/tmp/Python_2026-09-26_091318_fEmZ.sample.txt`.

Both work-bound paths now distinguish direct-child exit from inherited-pipe
completion. They reap their owned child, bound output draining, retain partial
text, and close their readers if EOF remains unavailable. They do not kill
unowned descendants. Incomplete output transport is reported explicitly,
rather than returning success or waiting forever.

Three real fork-based inherited-pipe tests pass. The gateway/governance
regression set passes 92 tests. Review also registered four existing canonical
gateway calls: three private-experience file-write sites and one fixed,
read-only, accelerator-free child launch. Their tests reject path escape and
caller-supplied commands. The ownership inventory changed only these four
canonical buckets and its digest; migration-debt calls remain 1,902.
No raw-write or raw-subprocess exemption was added.
