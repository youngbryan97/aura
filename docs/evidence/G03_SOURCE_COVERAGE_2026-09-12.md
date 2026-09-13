# G03 exact source recovery

The narrow proposal refit is not a controlled test of candidate-construction
repair alone: it replaced heads trained on 3,296 positive rows with heads
trained on 96. Its rejection remains valid, but does not isolate calibration
from lost training coverage.

The original source report at
`artifacts/rlc/semantic_program_27b_compositional_v17_alias_local_dev/report.json`
names seven source families: arithmetic, cataphoric, fork_join,
natural_alias_source, natural_source, reserved_alias and role_binding.
All seven retained feature bundles were found and their manifests matched.
After the original representation-compatibility binding, both the counts and
example hashes matched the frozen parent's 728 training and 500 validation
examples. Replication families were not added to training.

`tools/refit_semantic_argument_proposals.py` makes this recovery repeatable:
pass the frozen transducer, its source report, each `--bundle NAME=PATH`, and
a new output path. It validates manifests, shared representation, exact split
counts and exact split hashes before invoking the existing refit API. Output
is immutable and grants no serving authority. This command is for exact-source
repair experiments, not a restriction on separately designed future curricula.

Six focused tests pass. Smoke: 164 passed, one skipped. Ruff passes on the two
new Python files. The full-source numerical fit is a separate experiment;
source recovery and passing tests do not establish a compositional gain.

Additional diagnostic: on the narrow source validation set, every proposal
scale from 0 through 1.5 in increments of 0.125 ranks the positive first on
all 48 decisions. Thus that set cannot distinguish the candidate scales by
top-1 proposal accuracy. Lower binary cross-entropy alone was not evidence of
better complete-program selection.
