# GitHub Topics

These are the topics **currently set** on the repository, verified against
the GitHub API on 2026-10-25. An earlier version of this file listed a
wishlist that never matched what was actually configured.

GitHub caps topics at 20 and all 20 are in use, so adding one means dropping
one.

```
active-inference
affective-computing
apple-silicon
artificial-consciousness
autonomous-agent
cognitive-architecture
cognitive-science
consciousness
embodied-ai
free-energy-principle
global-workspace-theory
identity-persistence
iit-4
integrated-information-theory
local-llm-agent
long-term-memory
mlx
self-evolving-agent
self-repair
sovereign-ai
```

Set them under Repository Settings → About (gear icon) → Topics, or from
the CLI:

```bash
gh repo edit youngbryan97/aura --add-topic <topic>
gh repo edit youngbryan97/aura --remove-topic <topic>
```

## Candidates, if a slot frees up

Accurate to the codebase but not currently set, ordered by how well each
describes something distinctive about this repo rather than the category it
sits in:

- `activation-steering` / `contrastive-activation-addition` — the actual
  differentiator. Internal state reaches generation through the residual
  stream rather than the prompt.
- `spike-timing-dependent-plasticity` — `core/consciousness/stdp_learning.py`
- `liquid-neural-networks` — the continuous-substrate LTC ODE
- `attention-schema-theory`
- `neuromorphic`

`cognitive-science` is the weakest of the current twenty — it names a field
rather than this project — and is the obvious swap if you want
`activation-steering` in.

## Picking these

Topics are discovery, so the pull is toward the biggest available words.
Worth resisting. `consciousness` and `artificial-consciousness` are already
the two loudest entries on the list, and most of this repo's documentation
exists to explain that those are labels on mechanisms rather than claims.

Topics describing what the code *does* age better than topics describing
what it might mean.
