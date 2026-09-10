# R09 provenance is evidence, not answer ownership

## Reproduction

At 18:35 on PID 20867, "But how can it redo the changes safely if some of them
already reached the data files?" received a fixed explanation of how Aura knew
its prior answer. The preceding answer explained database crash recovery. The
word `reached` matched an epistemic-acquisition set, and `it` supplied a supposed
referent. The route returned verified_answer_provenance without model generation.

The classifier had no subject or clause relation supporting that decision. A
better list of database exclusions would preserve the same authority error.

## Repair

The lexical matcher now has retrieval authority only. Source records remain
bound to the exact delivered answer hash, session and turn. The route returns
structured facts, records their custody, and continues through CognitiveEngine
with the original visible question unchanged. It cannot render or finalize a
provenance answer. Both compact cognition and full response generation carry
validated provenance in the existing runtime-evidence channel. Neither adds a
new instruction, canned reply, classifier model, or answer-repair generation.

Historical verified_answer_provenance receipts remain readable. Qualified
state-serialization experiments remain separate and do not receive this data.

## Verification

- 29 selected tests pass in 45.36s, covering actual API dispatch for genuine,
  multi-part and incidental-overlap questions, exact source binding, typed
  context transport, compact generation, and full-phase evidence-role custody.
- Smoke before the full-phase addition: 164 passed, one skipped in 97.51s;
  lint, compile, governance-lint and layering passed.
- Final smoke: 164 passed, one skipped in 59.00s. Lint, compile,
  governance-lint and layering pass. Live replay is pending.

This removes a mechanism that prevented semantic reasoning from running. It
does not certify the model's answer as correct or close all of R09.
