# Expanded search regressions

This is a development diagnosis, not a full-cohort result or transfer claim.
The frozen 1,264-row audit was still running when this intervention finished.

## Intervention

Two exposed validation observations changed from equivalent to different:

- `0b3be2024353276b847a59f49f1a27cc031e661f87709b45931c4f9e0aa7d47e`
- `5acf5bfcf00a1c02029d656a007991bc6ff877ed8a6c4ab1f092c3c8d3dedc71`

Each was decoded with the prior, the prior plus typed/all-label search, and the
trained pilot. Both regressions already occur in expanded search without any
coefficient change. All six decodes used the same 20-second search allowance.
Validation annotations were used only for post-decode comparison, not fitting.

The intervention receipt is
`3f2a54ba7f777adc4b8dbb6b34e643d555eba051033070eccdf8c33fb44cf832`,
at `~/.aura/rlc-evidence/semantic-validation-regression-intervention-20260921/report.json`.

## Localized score error

Both correct programs are `idiv(in2, count_of(in0, in1))`; expanded search
selects `sub(in2, count_of(in0, in1))`. The incorrect chart uses operation spans
`[5,7)` and `[8,11)` instead of the annotated `[5,12)` and `[24,27)`.
Its operation score is `0.194906`, versus `-6.840264` for the correct chart.
The correct argument score is better in the first observation, but that does
not overcome the operation score. In the second, both components favor the
wrong chart. This is not solely an argument-binding regression.

Score decomposition receipt:
`109b2e0925eded6379f11ec70d2361b099bb2d2c59ef7411c45ca61610f3695f`,
at `~/.aura/rlc-evidence/semantic-regression-score-components-20260921/report.json`.
Both interventions ran from frozen revision `095102eff`; implementation hash
`077b082d1d0aa8a3b7e2e25b222cf53ed9338340d171e380e46d6ab8e7ea37ee`.

## Consequence

The next source-only experiment must exercise operation-span competitors as
well as binding competitors. The existing runtime graph-constraint miner
already exposes both, including alternatives on correctly decoded training
rows. Its candidate-search allowances and unresolved rows must remain visible.
The pilot froze the operation classifier; repeating that setup alone does not
address this observed operation-score error. Do not fit these validation rows
or introduce a construction-specific selector. G03 remains open.
