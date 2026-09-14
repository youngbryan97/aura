# G10: isolate fusion interventions from live state

The worker's idle probe shared hooks with a 20 Hz substrate publisher. The
publisher could change alpha and the composite during a measurement. A failed
random-control evaluation could also leave that control in the hook; the
worker restored only engine alpha.

The engine now owns a reentrant publication lock. Live alpha changes,
surface clamps, and substrate publication use that owner. A probe obtains
the owner before locking its hooks in layer order. It preserves each hook's
alpha, active state, source state, mood metadata, freshness, and cached CPU
and MLX composite. Exit restores the exact incoming values on success or
failure. The caller must still own generation; this does not make overlapping
model forwards safe.

Two measurement confounds were repaired in the same path. The prior serving
clamp cannot suppress the experiment that is meant to measure a candidate
alpha. A deliberately held experimental state also must not expire halfway
through its decode. The temporary measurement owner handles both; ordinary
serving retains its existing stale-input derating and injection ceiling.
Background publishing resumes when the probe releases ownership.

Random controls now match each layer's own composite norm. Previously the
first hook supplied the magnitude and direction shape for every layer.
Certificates name the corrected measurement protocol. Older certificates
remain readable, and new protocol files do not overwrite them. Serving and
idle-probe reuse require the current protocol.

The sampled steering campaign uses the same hook-state lifetime during each
decode. The running 1,024-token calibration remains frozen on the preceding
revision and is diagnostic only; it is not upgraded in place.

Validation: 153 focused tests passed. These include real MLX cached-state
restoration, multi-hook reentry, a concurrent substrate thread, failure inside
the real random-control loop, per-layer norm matching, serving-clamp recovery,
stale-state behavior after measurement, and historical-certificate retention.
The wider suite exposed an old test that expected uncentered tanh weighting.
The production mapping was already centered by eb77493ce; the test now checks
the exact centered formula, equivalence between mood and state inputs, and
zero at the declared midpoint. Production weighting was not changed.

These are engineering checks. Current-model steering qualification, live
serving proof, and broad reasoning gain remain separate unfinished claims.

## Calibration observation

The preceding frozen public-channel calibration stopped after six baseline
and three treatment samples. All six baselines reached EOS with public text
(228, 135, 187, 679, 646, and 629 generated tokens). All three alpha-0.2
treatment samples exhausted 1,024 tokens without closing the private channel.
The remaining samples were not run. This is an early diagnostic stop, not a
complete causal verdict or a valid comparison of all planned conditions.

The authenticated supervisor recorded child PID 19305, signal 15, and return
code -15. Its run directory is
`~/.aura/experiments/caa-public-calibration-20260914`; the nine durable samples
remain in the `codex-steering-public-20260914` frozen worktree. A lower-strength
calibration on the corrected ownership path must use a new experiment identity.
