# Restored latent decode

The accumulated restoration working set permits a full-parameter fit of the
captured 25,038 comparisons. All twenty wrong or tied comparisons become
positive without lowering any retained floor. Actual decoding improves from
16/25 to 23/25 on the source-only acquisition cohort.

The remaining two outputs are the same wrong programs as before. Their
selected latent evidence changed after fitting: the old comparisons now have
positive margins, while newly mined comparisons have margins -1.632421 and
-0.171465. This distinguishes changed latent bindings from a claim that a
fixed comparison certifies the decoder's subsequent search.

A second source-only fit retains the original comparisons and both newly
witnessed comparisons. It uses the first candidate as its starting point,
with the same objective and retention checks. One update leaves no wrong or
tied retained comparisons. Actual decoding reaches 25/25.

## Separate validation

The 100-example development-validation cohort was selected by source identity
across geometries. It supplied no fitting constraints or training examples.

| Cohort | Original parent | Second candidate | Gains | Regressions |
| --- | ---: | ---: | ---: | ---: |
| Source acquisition | 16/25 | 25/25 | 9 | 0 |
| Development validation | 99/100 | 94/100 | 0 | 5 |

These scores use actual decoding and universal-floor semantic comparison,
including counterfactual inputs. The candidate is not promoted. G03 remains
open. Correcting the acquired examples does not establish source-wide
retention, unseen transfer, broad reasoning gain, or frontier performance.

The full 764-example source replay is a separate measurement. Its purpose is
to identify any missed source-training obligations before another fit.

## Artifacts

All paths below are under `/Users/bryan/.aura/rlc-evidence/`:

- `semantic-accumulated-decode-20260920/result.json`, receipt
  `744a9d02fe536a55655e8197ee6f6962e3c13441fa77b7e5c720b81fd9b3a1cf`.
- `semantic-accumulated-decode-20260920/latent-replay.json` and
  `new-latent-contrasts.npz` retain the changed latent comparisons.
- `semantic-accumulated-second-decode-20260920/result.json`, receipt
  `4da30da12efa8a959352aa96c7261bc5a1fbdb4b8f8fd76cca0c3fcbda43df9a`.
- Second candidate receipt:
  `5e26fc1f9b9b6e269fde56a72cbb77d28d2a1064488e015bcf1cefcd797964e5`.

Both decodes used frozen revision `28ebd904de`, with implementation identity
`ab2179bf3523948a941fbd1d7e9d20ff80600e988bb0994f29c3b3b871098421`.
The second fit and all 125 paired decodes completed in 212.34 seconds.
No resident model, serving manifest, or steering tissue changed.
