# Binding-face source trial and unchanged-round termination

The completed small source-bound trial is retained at
`~/.aura/rlc-evidence/semantic-binding-faces-trial-20260918.json`, with receipt
SHA-256 `8554b6d81a4ea763121da5f591e131d58647646c7819b32e8dfc4663c3142c96`.

The fitter used 2,504 comparisons and 64 updates. Wrong or tied retained
comparisons decreased from 148 to 17, but the search budget expired before
all constraints were satisfied. Training equivalence improved from 7/8 to
8/8; validation remained 6/8. One validation error became a decode refusal.
No previously equivalent answer regressed, but this is not a transfer gain.

The receipt refuses promotion because autonomous decoding refused an answer,
operation-retention search was incomplete, and retained constraints remained
unsatisfied. No serving authority was granted.

Separately, the joint campaign now compares actual coefficient bodies before
and after each fit. An unchanged round stops further identical mining rounds,
records `coefficients_unchanged`, and retains its errors and fit status. It does
not report convergence or infeasibility. Requested and completed round counts
are explicit. A focused regression verifies one mining pass instead of three;
the existing changed-coefficient test still exercises two rounds.

G03 remains open. This result does not replace the 500-case evaluation.
