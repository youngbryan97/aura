# Decode each accepted training round

The conditional pilot's three initially wrong source examples all decode
correctly at the start of round three, after two updates. That observation
does not establish validation accuracy. The trainer previously retained each
round's numerical state but exported a complete transducer only after the
last round. Comparing an earlier update required reconstruction or another run.

When numerical checkpoints are enabled, the joint trainer now also writes
`round-N.candidate.json` before notifying the caller that the round finished.
The envelope binds the parent, round, actual transducer, and SHA-256 of the
numerical checkpoint. Publication is immutable; resume accepts identical bytes
and rejects a different existing artifact. These snapshots have no serving
authority and do not select a model using validation results.

Fifty-seven focused tests pass. They cover interrupted training, exact
coefficient restoration, numerical-checkpoint binding, repeat publication,
conflicting saved state, and the existing conditional learning path. The
already-running pilot uses the previous frozen revision and is unchanged.

No G-ledger item closes from this checkpoint facility. Earlier and later
updates still require actual paired decoding under the declared selection rule.
