# G03 representation-only binding control, 2026-09-24

The prior joint source-binding head reached 952/1,096 opposed gold-span roles
on held construction fold 0. A matched component lesion showed that geometry
alone retained 944/1,096, while the fitted representation weights alone
reached 459/1,096. That lesion could not distinguish an inadequate vector
basis from an optimizer that simply relied on geometry.

A separate, source-only head was therefore trained from scratch on the same
joint operation/mention/definition vector interactions, without any token
position features. With gold operation and mention spans supplied, it chose
555 of 1,096 opposed roles on the same held fold. The immutable receipt is
`~/.aura/rlc-evidence/semantic-triadic-gold-v3-fold0-20260923/report.json`.
No exposed validation labels were used for fitting. This is a diagnostic, not
serving authority or qualification evidence.

The result rules out the narrow explanation that the representation-only
lesion failed only because the optimizer assigned all weight to geometry.
It does not prove that the resident model lacks binding information: the
linear interaction map, middle-channel pooling, construction coverage and
binary negative objective have not been exhausted. G03 remains open.
