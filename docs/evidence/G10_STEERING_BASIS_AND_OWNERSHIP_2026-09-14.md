# G10: steering evidence follows the loaded basis

The worker's existing fusion tests reproduced two failures: a passing
certificate yielded alpha zero, including when the governor allowed a smaller
positive alpha. The surface module imported the worker's empty model-identity
string at import time. Later attachment replaced the worker's string, but the
surface reader retained the earlier value. The reader now resolves current
worker identity at use and checks it against the attached engine.

A model-only certificate also could not distinguish old vectors from a new
generation at the same checkpoint. Hooks now expose a basis record containing
actual float32 vector bytes, layer and block identity, and substrate mapping.
The measurement, persistence, idle remeasurement and public-surface lookup all
use that basis digest. Historical model-only certificates remain readable.
They are neither deleted nor admitted as evidence for an unmeasured basis.

Jobs and environment values can request less steering than measured, not more.
Research paths retain their explicit experimental controls. This corrects the
existing measurement boundary; it does not change the causal pass predicates.

The unattended preparation tool previously deleted the fusion certificate,
broadly killed `aura_main`, and spawned a raw runtime after installation.
Preparation now ends at independently verified authority, records that it did
not install or restart, and leaves those actions to controlled publication and
lifecycle owners. It checks replay exit status before consuming a verdict and
uses result-hash-specific outputs. Its owned status writes use the file gateway.

Ninety focused tests pass in 31.85 seconds. They cover live worker lookup,
identity/basis mismatch, remeasurement without deletion, upper and lower alpha
bounds, actual hook fingerprints, and each preparation failure stage. No live
model qualification is claimed by these tests. The trained-vector campaign
was still running while this patch was tested; G10 and G11 remain open.

Continuation check: lint, compile, governance, layering and writing passed.
Smoke completed in 48.11 seconds: 163 passed, one skipped, one failed. The
failure is the existing `resident_manifest_drift` activation alarm, with no
drifted bound source files. It is not suppressed. The campaign process later
disappeared without a result artifact; its earlier progress is not a completed
measurement. Durable campaign recovery is the next repair.
