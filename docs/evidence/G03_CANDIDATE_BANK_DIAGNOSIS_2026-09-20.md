# Candidate reachability and selection diagnosis

The candidate bank calls the deployed operation and argument builders without
source annotations, expected answers or family labels. It retains the ordinary
decoder's selected program unchanged. Independent diagnosis then compares the
fixed bank with the archived source annotation using structural equivalence,
symbolic reasoning and universal-floor counterexamples.

## Completed small replay

The replay revisits errors from the already exposed 100-row policy control.
It does not use validation rows for fitting and makes no fresh-transfer claim.
Each row permits eight operation charts and four distinct argument graphs per
chart, with a two-second solver allowance per graph. A limit is incomplete
search, not proof that no correct interpretation exists.

- Eight extra joint-policy errors all contain a semantically correct program
  in their actual candidate bank, but select a witnessed incorrect program.
  These are selection failures, not missing executor operations.
- The error shared by both policies has no correct program in the small bank.
  Operation and argument search are incomplete, so its cause is unresolved.
- Neither model coefficients nor serving selection changed.
- Execution and public emission were not measured. The diagnostic cannot
  turn those missing observations into success.

Ten rows completed in 63.54 seconds, including archived-feature loading.
Receipt `4186c8b932327fecd54b32639fb36e4effd4c4e8c076a9ef5db7f599dcb218dd`
is retained at
`~/.aura/rlc-evidence/semantic-candidate-bank-diagnostic-small-20260920.json`.
Each completed row also has an immutable separate receipt. The earlier larger
attempt was interrupted inside the optimizer and has no completed result;
its printed observations are not substituted for this replay.

## Engineering checks

The expanded replay of the shared error completed in 20.10 seconds. All 16
charts in the incumbent beam were visited; the correct program appears in
chart 9 as its second argument graph. It computes
`sub(in2, count_of(in0, in1))`. The incumbent instead selects
`idiv(count_of(in0, in2), in1)`. Both operation choice and argument binding
contribute to this error. No new primitive is required to express the target.

Receipt `c47b99e999643d8bda1637a4cc0b622183b9be89a2d58a4aad1209af8db140ad`
is at
`~/.aura/rlc-evidence/semantic-candidate-bank-diagnostic-expanded-20260920.json`.
This proves reachability for that source, not completeness outside the beam.
All observed errors in the exposed 100-row slice now have witnessed correct
alternatives. That does not establish coverage for the other 400 development
rows or fresh transfer.

## Receipt and selection checks

The bank binds tokens, public values, normalized hidden observations, model
basis and transducer identity. Diagnosis checks the receipt checksum against
the actual candidate payload and selected program. Complete search requires
exhaustion evidence from both operation and argument inventories. A nominally
complete search with unknown equivalence still cannot prove unreachability.

Focused tests cover source substitution, receipt mutation, payload substitution,
truthy non-boolean outcomes, missing measurements, unknown equivalence, progress
events, unchanged ordinary selection and distinct incomplete/selection outcomes.

This closes implementation of the bounded diagnostic, not G03. Broader
reachability, a source-trained selection repair, independent validation,
free public decoding and general transfer remain separate obligations.
