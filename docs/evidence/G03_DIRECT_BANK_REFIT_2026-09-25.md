# Direct decoder bank-contrast refit: source gain did not transfer

The source-fold direct decoder can now score a bank with one source encoding
and train against witnessed runtime-bank rivals. The offline refit retains the
original source-program loss. A bounded beam supplies additional wrong
programs without a target; only training-fold supervision labels their
counterfactual disagreements. Held construction labels never guide fitting.
No part of this experiment has serving authority.

The 32-train/24-held plumbing pilot moved held selection from 21 to 22 for
the unscaled parent. A complete frozen fold-0 source-bank run with 508 train
sources and one epoch of four-wide generated rivals moved 155/256 to 157/256
for that parent. The stronger unit-variance-scaled parent instead fell from
190/256 to 167/256 under the identical training procedure. Its pre-refit
free decode had scored 187/256 on this fold, versus 100/256 for the unscaled
parent. These are different checkpoints and different readouts; they are not
interchangeable. The source bank itself was not a sufficient positive-gain
evaluation: 203/256 held rows had a verified correct program, and every one
of those 203 already had a correct incumbent. The remaining 53 had no
verified correct program. This bank measures preservation/regression, not
recovery.

The independently frozen 23-row exposed development-miss bank was replayed
with both parent/refit pairs. It has a correct candidate on all 23 and an
incorrect incumbent on all 23. The unscaled scorer selected 1/23 before
and after its refit. The scaled scorer selected 7/23 before and 6/23 after:
zero paired gains and one paired regression. The existing ranker selected
5/23 on this bound bank.
This is a selected failure subset, not a full-development net-gain estimate.
It does falsify the claim that this refit solves the exposed selection gap.

The earlier 7/23 comparison used the scaled direct checkpoint
`cef80f7ef6005f634184c840fd6c20ffa2a9eeadd5c87f79f2ac0b54e86807e2`.
The 1/23 comparison used the unscaled checkpoint
`5397f8ab7bc235d844de822cf9b96c5e5abc1e65206d63cb3e897da053e4756f`.
The bank is the same. Both replays bind the development gap report,
candidate bank, source report, literal-identity candidate, direct parent,
refit checkpoint, and source-bank receipts. Unscaled receipts are under
`~/.aura/rlc-evidence/semantic-direct-bank-all-train-full-fold0-20260924/`
and `~/.aura/rlc-evidence/semantic-direct-bank-refit-gap-20260924/`.
Scaled receipts are under
`~/.aura/rlc-evidence/semantic-direct-bank-scaled-full-fold0-20260925/`
and `~/.aura/rlc-evidence/semantic-direct-bank-scaled-gap-20260925/`.

Both refits are rejected. G03 remains open. More epochs or broader beam
width are not justified by this evidence. The missing mechanism is selection
evidence from hard requests without relying on a bank where a reachable
correct program is always the incumbent. The existing out-of-fold proposer
banks contain wrong incumbents, but using them for selector cross-validation
without nested splits would leak through the proposer fit on other folds.
