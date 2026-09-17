# G03: recover the saved aligned fit without training again

The three-round aligned-input campaign reached its 14,400-second supervisor
bound. Its terminal receipt is `timed_out`; validation did not finish. The
candidate was nevertheless published atomically before termination and loads
with receipt `912618ac158b4d32addaf4a3b25376910a9abe69f964c63762f2169c7bd8c91f`.

Training constraints, not task accuracy:

| Round | Retained pairs | Initially wrong/tied | Finally wrong/tied | Accepted steps |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9,331 | 20 | 0 | 100 |
| 2 | 10,049 | 1 | 1 | 1 |
| 3 | 10,767 | 2 | 2 | 100 |

The final candidate is not a correctness guarantee. Fresh autonomous decode
must measure new competitors and latent choices. No serving change is made.

`refit_semantic_argument_proposals.py --evaluate-existing --compare-fit-start`
can now recover the complete comparison from this saved candidate. It requires
the reconstructed starting arm's receipt to match the parent recorded by the
fit. A different comparison parent is rejected. No fitting occurs in this mode.

Evidence directory:
`~/.aura/rlc-evidence/semantic-aligned-multiround-supervisor-20260916`.
Candidate directory:
`~/.aura/rlc-evidence/semantic-aligned-multiround-20260916`.

The run also exposes a durability defect: completed optimizer steps were only
published after the full fit. Incremental checkpoint/progress work follows;
this record does not claim that repair or G03 closure.
