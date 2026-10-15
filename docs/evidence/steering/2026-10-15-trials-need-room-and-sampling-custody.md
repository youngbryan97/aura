# Repeated trials need room and a declared sampling policy

In the custody-and-cadence case-18 recording, thrust was usable at 3.6
seconds. One lateral direction became usable at 10.5 seconds. The body
was already going toward the left boundary; the opposite direction still
lacked a second measured pulse when the body left the picture. The finite
calibration allowance broke a tie by input order and tested the outward
direction again first.

`trials_with_room_to_observe.py` now ranks repeated experiments from the
current body's measured pulse and rest accelerations. Each fit requires
four distinct settled sample times and a measured span. Residuals and
variation among recent fits widen the projected acceleration interval.
The pulse and its following rest are projected through the same unclamped
visible-path calculation used by the existing observation guard.

All candidate controls and the rest must have a usable estimate. Missing,
stale, interrupted or non-current observations preserve the previous
order. The finite allocator still chooses the least-used eligible input;
clearance breaks its tie. It neither adds attempts nor certifies a single
pulse as a learned control. Projection ranks experiments under an observed
model and cannot guarantee future motion.

## Ownership differs from independence

Choosing an experiment to preserve visible room is a feedback policy.
Those actions cannot be counted as independent evidence about which of
the other moving things answers to the controls.

`ControlTrial` now keeps its sampling policy separately from its delivered
input, owner and immutable deadline. After body identification, calibration
inputs are recorded as nonindependent. Their visible effects still teach
the own-body motion model. They do not extend discovery's independent
trial histories. Their physical leases still release the input and keep
its finite observation rest when no further picture arrives.

Tests cover both axes and both boundaries, absent projection evidence,
the real finite trial executor's ordering, unchanged control uncertainty,
and physical release plus rest without another frame. Existing discovery,
facing and calibration regressions remain required.

Validation before fresh play: 135 focused tests passed. The smoke gate
passed 165 tests with one skipped. Fresh original-game performance is
still unmeasured at this checkpoint; no landing or presentation claim is
made here.
