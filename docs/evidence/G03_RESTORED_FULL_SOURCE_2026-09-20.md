# Full restored-source replay

The second retained-latent candidate was replayed on every source-training
example and all 500 development-validation examples. No test examples were
used. The evaluator ran in frozen revision `28ebd904de`.

- Source training: 764/764 semantically equivalent programs.
- Validation parent: 488/500 exact and structurally equivalent programs.
- Validation candidate: 469/500 exact, 472/500 structurally equivalent.
- Exact paired changes: one gain, twenty regressions.
- Equivalence paired changes: two gains, eighteen regressions.
- Selection: incumbent retained.
- Total replay and validation time: 667.93 seconds.

Training replay used independent counterfactual semantic comparison. The
selection gate used its existing source-anchored structural scoring; these
scores are named separately rather than treated as interchangeable.

The candidate fixes every source-training case but fails development
generalization. No promotion or G03 closure follows. The separate operation
retention-policy trial tests the classifier update identified by component
interventions; it does not inherit success from this training score.

Artifacts under `~/.aura/rlc-evidence/semantic-restored-full-source-20260920/`:

- `training.json`, receipt
  `3e03eaa2685c7fc672560e81d388e0a734d14c97879914144d53a00d4dd85498`.
- `validation.json`, report
  `442a49ff128278748327ec4277cd0abc4e33c2eb02cde70e4c4427a595ec8e85`.
- `validation.json`, enclosing receipt
  `70948d7423add21d0825757de36ba4d184ec8eca8943e901ad031992540a4865`.

Both checkpoint files bind source identities, coefficients, and implementation.
No resident model or serving manifest changed.
