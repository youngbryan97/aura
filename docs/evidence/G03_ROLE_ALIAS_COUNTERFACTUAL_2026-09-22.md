# G03 role-alias counterfactual

The frozen `sequence-role-binding-4` validation slice has 16 examples. The
incumbent decoder and a bidirectional-definition variant each decode 16/16,
select the same programs on all 16, and return the right public value on 14/16.
The paired receipt is
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/definition-policy-role4.json`.
Expanding definition reach did not repair these two failures.

Both wrong examples use a selector and an adjustment with the same numeric
value, 3. The text names them separately: `selector 3` and `that initial
amount`. The target programs are `mul(at(in0, in1), in2)` and
`mul(count_of(in0, in1), in2)`. The decoder selects `sub(at(in0, in2), in1)`
and `sub(count_of(in0, in2), in1)`. The candidate bank contains the target
programs. In the first example its target score is only 0.0565 below the
selected score, so this is a selection error, not an unreachable program.
The source cohort has no equal-valued selector/adjustment pair in training;
validation and test each have eight. This is a measured role-alias gap.

The new corpus kind adds equal-valued but distinct-role counterfactuals to
training. It leaves the existing validation and test examples unchanged.
Passing corpus tests establishes data validity and split integrity only. It
does not establish a learned gain. Feature extraction, refitting, a frozen
paired decode, and fresh held-out evaluation remain required before this
candidate can acquire serving authority.

The exact-definition-span probe should not be treated as a semantic-link
failure: the shorter selected span overlaps the annotated definition in
16/16 of the examined role-binding cases. Full-program comparison has higher
authority than exact span recall for this question.
