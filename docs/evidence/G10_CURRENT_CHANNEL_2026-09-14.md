# G10: Current-generation channel measurement

The current signed steering generation passed the owned-layer channel protocol
at alpha 0.1 on the resident 27B checkpoint. The worker ran from frozen commit
`4be676e55` for 290.69314 seconds, returned zero, and left no process descendants.
The log and supervisor receipt are retained under
`artifacts/migration/27b/recovery/current_generation_channel_20260914/`.

- Model descriptor: `52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c`.
- Signed generation: `0d192ee439b964ba9e5792e288664835f7979743021b503434c6a3ed9dd8c51a`.
- Hook basis: `8ff000d797006b8f2e401fe2a39941c56462423da1a52fa331b1dec8c6d9b4d7`.
- Protocol: `owned_layer_matched_v1`; seed 20260908; 24 decode steps per probe.
- Hooks: 16 layers, numbered 3 through 63 in steps of four.

At alpha 0.1, 7/8 public probes changed. Mean distribution shift was
0.09596194 nats versus 0.06552591 for the control. Opposing-state separation was
2.08412754. Forced-choice accuracy was 10/10 both unsteered and steered. The
right-answer margin changed by +0.02812503, versus -0.08749999 for same-length
random directions. All four channel predicates held.

The smaller alpha 0.05 failed the noise comparison: its margin improved by
0.0406, but the random control improved by 0.0531. The diagnostic incorrectly
called both improvements costs. The sign-preserving wording is repaired; the
predicate and the measured result are unchanged.

The exact JSON printed by the worker was reopened as a FusionCertificate,
checked against its computed predicates, and written through the certificate
API. Before writing, the registered checkpoint was rehashed and its signed
generation authority validated. The certificate is in `artifacts/fusion/`,
keyed by model, actual hook basis, and protocol. Runtime lookup requires all
three. This archival step did not change the active migration authority.

This result concerns the already signed generation. The separate newly trained
generation failed its 216-decode public campaign; that negative result remains
in [the campaign record](G10_PUBLIC_STEERING_RESULT_2026-09-14.md).

Eight short public probes and ten forced-choice questions do not establish
broad reasoning improvement, long-response quality, or a fresh behavioral
replication. Desktop serving has not been validated with this certificate.
G09, G10, G11, and G12 remain open.

Focused verification passed 156 tests. Lint, compile, governance, layering,
and writing gates passed. Smoke passed 163 tests with one skip and one failure:
the existing bounded-WOW installation alarm reports `resident_manifest_drift`,
with no drifted bound source files. This certificate does not adjudicate that
separate RLC activation.
