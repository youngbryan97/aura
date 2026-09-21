# Source-selected competitor probe

The current pilot candidate was examined on 16 training observations chosen by
source-identity order, round-robin across geometry groups. Selection did not
use validation failures or observed correctness. Four operation charts and
four argument graphs per chart were allowed, with three seconds per solve.

The existing runtime graph-constraint miner found 61 semantically incorrect
alternatives. Every recorded margin favored the correct training program;
the smallest was about 1.328. This bounded search does not establish that no
closer competitor exists. It does establish that repeating the existing
0.1-margin fit on these witnessed pairs supplies no unsatisfied constraint.

Receipt: `d32be57772386860c28e86d21bd77b381b990f5cdfe13706820479528f575d04`.
Artifacts: `~/.aura/rlc-evidence/semantic-source-competitor-probe-20260921/`.
Revision: `095102eff`. No parameters were fitted, no validation or test row
was admitted, and no serving or transfer authority was granted.

This result must be considered alongside the two expanded-search validation
regressions, not presented as their repair. The next learning experiment needs
evidence beyond re-optimizing already satisfied sampled constraints.
