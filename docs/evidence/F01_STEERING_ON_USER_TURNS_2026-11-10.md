# F01: her state steers her forward pass on user turns, and her words change with it (10 November 2026)

Item F01 of the master to-do closes when her log shows α > 0 on user turns and a paired trial shows the words
change with it.

## Her log

`~/.aura/logs/desktop-launch.log`, which runs from 16 October to 10 November 09:00. Each `Surface decode` line was
paired with the prompt plan just before it, whose `origin` says whose turn it was.

| Turns from | Days | Decoded at α = 0.5 | Decoded at α = 0 |
|---|---|---:|---:|
| A person (`desktop_quick_user`) | 20–22 October, before the install | 0 | 30 |
| A person (`desktop_quick_user`) | 7–9 November | 71 | 0 |
| Her influence probes (`desktop_influence_probe`) | | 0 | 327 |

The steering generation was installed on 23 October. On every person's turn since then the worker steered at the
α its certificate earned, 0.5. The probes ask for 0 by design: they measure other channels, and a steered decode
would confound them.

## The paired trial

`tools/measure_fusion_channel.py` on her persona 27B, with the installed generation's 16 hooks attached and the
resident identity required, at α = 0.5 and without writing (`--no-write`), run at `f52951daa` from 10:49 to 10:57.
Output in `~/.aura/rlc-evidence/f01-paired-20261110/measure.log`. The basis it attached, `e267aa76`, is the one her
worker runs.

| Criterion | Reading |
|---|---|
| The words change | 6 of 8 prompts decode differently steered than unsteered, from the first token (median step 1) |
| It arrives | next-token distribution moves 0.0178 nats a step |
| It beats noise | a random direction of the same length moves it 0.0024 |
| It carries content | opposing states move it apart at 1.216 of the mean distance from silence |
| It costs nothing | forced-choice accuracy 1.000 unsteered and steered, margin +0.0625 against +0.0500 for noise |
| It keeps her voice | 32 of her stored replies are 0.111 nats a token more likely under her state's push; a random direction, −0.0025 |

All five hold. The same basis read the same arrival, separation and content on 23 October; the voice reading then
was +0.243 on a different draw of her replies.

## What it does not show

The trial shows the steering changes her words on fixed prompts and keeps them hers. It does not show that a given
change of her state changes a given reply in a direction a person would notice; that is the reports ground of the
bridge, measured separately. Background loops, the curriculum and her influence probes do not steer.
