# G10: load the checkpoint the probe certifies

The fusion CLI accepted a local --model-path and built its descriptor from
that path. Its model load still used --model, whose default named a different
checkpoint. A run could therefore measure one model and attach another
model's identity to the result.

Loading, hook attachment, descriptor construction and certificate model_name
now use the same resolved checkpoint. An explicit local path does not inherit
the unrelated default repository ID. A downloaded snapshot retains its actual
repository ID as provenance.

Two runner tests cover explicit local paths and resolved repository snapshots.
The combined identity, measurement-ownership, worker and certificate suites
passed 49 tests in 11.76 seconds. No previous certificate is changed or
reissued by this repair.

The full public-channel steering campaign separately started from frozen
commit 5a63f16d2 at alpha 0.05, with four trials per each of six development
prompts across nine conditions and 3072 tokens available to each sample.
It retains every completed sample through CampaignProgress. The supervised
record is `~/.aura/experiments/caa-public-full-20260914/`. This launch is not
a completed qualification or fresh broad-reasoning experiment.
