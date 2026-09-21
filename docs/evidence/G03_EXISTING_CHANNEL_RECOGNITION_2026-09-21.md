# Recognition evidence already present in the resident sequence

The runtime-score-coupled transition fit completed all 1,264 development
observations. Training is 764/764. Validation is 479/500, with fourteen repairs
and eleven regressions relative to the 476/500 incumbent. All three acquired
training failures are repaired. The candidate is not promoted.

The complete comparison is
`3d6976474569da5a183f698ae5dab63e30edd85569f71b8451f5ba524c6f0fe0`
in `~/.aura/rlc-evidence/semantic-transition-runtime-fit-20260921/`.
The detached supervisor recorded 2,446.23 seconds, exit zero, no restart and
an empty process lineage. Its receipt is
`508ad89d148bd9d114be590a1c09b18b08f31d0aac65b137ce4af0c351c04888`.
Positive margins on the three acquired pairs did not protect all other
interpretations. No threshold changes follow from that result.

## Existing-channel experiment

The exact labeled-span trainer accepted only final-layer mean and final-layer
mean/transition views. The source archive already carries lexical, middle and
final channels. A controlled diagnostic fits the same linear classifier on
each existing view. Three source-training folds hold out whole constructions;
the hash-ordered construction assignment was fixed before fitting. All 1,720
training operation labels and 1,192 exposed validation labels are accounted for.
Annotated operation boundaries are supplied throughout this diagnostic.

| View | Source construction cross-validation | Full-source training | Exposed validation |
|---|---:|---:|---:|
| Final mean | 1645/1720 | 1720/1720 | 1163/1192 |
| Final mean and transition | 1628/1720 | 1720/1720 | 1175/1192 |
| Middle mean | 1700/1720 | 1720/1720 | 1174/1192 |
| Lexical mean | 1656/1720 | 1720/1720 | 1108/1192 |
| All-channel mean | 1662/1720 | 1720/1720 | 1146/1192 |

Middle mean is the source-fold winner. No validation label trained any head.
This is a representation diagnostic, not an autonomous compiler result. The
folds retain other constructions from the same corpus families, so this does
not establish held-out-family transfer. The archived script, plan, fitted
heads, errors and report are in
`~/.aura/rlc-evidence/semantic-existing-view-capacity-20260921/`.
Report: `c801e10c405c11d0c34d766710f311f0cbb074d648f8073329ab25341d3c6107`.
The detached process completed in 38.60 seconds with empty child lineage.

## Implementation

The existing exact span trainer now accepts the already-defined lexical,
middle and all-channel mean views. Channel selection follows declared names
and widths; it does not assume three equal channels or a particular order.
Projection and adjoint retain their prefix-sum implementation. Runtime feature
definitions, graph scoring, serialization and default serving are unchanged.
No new model, phrase table, routing rule or prompt is introduced.

Tests compare projections and gradients against runtime feature extraction
with unequal, reordered channel widths. Missing channels and unsupported
scorers are refused rather than silently reading the final layer. Each supported
view is fitted, serialized and restored through the existing model contract.
Full autonomous measurement of the middle-channel candidate remains required.

Verification: 72 focused tests passed in 15.13 seconds. Smoke passed with
164 tests and one skip in 69.38 seconds. Focused Ruff, compilation, layering,
governance lint and writing checks passed without baseline changes.

## Research cross-check

[Herzig and Berant (2021)](https://aclanthology.org/2021.acl-long.74/)
learn compositional span structures, including non-projective cases, and report
transfer improvements on their benchmarks. Their work supports testing explicit
composition; it does not prove Aura's representations or search are sufficient.
Aura already has typed graph composition, so introducing another parser solely
to reproduce that abstraction would duplicate existing work.

[Qiu et al. (2022)](https://aclanthology.org/2022.naacl-main.323/)
learn grammar-based recombinations from training examples. That offers a
source-only expansion path if controlled recognition measurements show a data
coverage problem. It does not permit importing exposed validation phrases as
training while retaining a held-out claim.

G03 and the fresh-transfer, public-answer, broad-gain and frontier obligations
remain open. This record neither closes them nor changes the release standard.
