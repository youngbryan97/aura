# 27B recovery: what is proven, what is prepared, what needs the model

Status: Handoff · Written 2026-08-24 · CPU-only preparation, no model loaded

The complete non-serving portion of the Qwen3.8-27B migration and RLC recovery.
Everything below either runs today or is frozen and waiting for one model
residency. `make rlc-27b-readiness` is the single command that says whether that
residency can start.

## Proven

Measured, with a test that fails when the measurement stops holding.

**The two checkpoints share three numbers and differ in twelve.** 64 layers,
hidden size 5120, and a 1024-wide KV projection are identical; `model_type`,
intermediate size, head counts, head dimension, vocabulary, context length,
RoPE base and the whole attention layout are not. A 32B LoRA adapter therefore
loads onto the 27B without raising. Verified independently of the tooling that
first reported it.

**688 trained tensor files address modules that do not exist.** `qwen3_5` gives
a layer `self_attn` only when `(index + 1) % 4 == 0`; every retained adapter
targets layers like 16 and 17, which carry `linear_attn`.

**The digit token ids did not move.** The first migration pass concluded from
the vocabulary growing to 248,320 that every bound id was stale. Re-deriving
under both tokenizers: digits identical at 15..24, all seven opcode markers
changed, and two of the five files listed never touch a tokenizer at all.
Portability is now measured per binding, and an unmeasured binding is reported
bound rather than fine.

**A hard crash was in the recurrent SFT path.**
`hasattr(layer.self_attn, target)` raises on a layer with no `self_attn` — the
attribute lookup happens before `hasattr` can guard it. It would have failed at
layer 16, the first linear layer inside the recurrent window.

**The window silently thinned by four.** `o_proj,v_proj` over layers 16 to 47
produced 64 adapter sites on the 32B and 16 on the 27B, with a non-empty site
list and no error. The trainer now declares its expected sites and refuses a
mismatch; adding `down_proj` brings the window back to 48.

**The native MTP head is unusable.** Zero `mtp.*` tensors in the base and in the
fuse, `mlx_lm`'s `qwen3_5` loader discards the keys on load, and no supported
API reaches an internal head. Draft-model speculation is available and needs a
vocabulary match, which rules out the 1.5B and 7B rigs and leaves
`Qwen3.5-9B-4bit`.

**The fuse is text-only.** It declares `Qwen3_5ForConditionalGeneration` with
`language_model_only` false, carries no `vision_config`, ships no preprocessor,
and its index holds 1,847 tensors all under `language_model.`.

**The bounded-WOW surface is dark for the right reason.**
`active_model_mismatch,resident_manifest_drift`, with **zero** drifted source
files. The package is intact; the checkpoint under it was replaced. It fails
identically at clean `origin/main` with none of this work applied.

## Prepared, not proven

Frozen and executable. None of it has measured anything.

- **The campaign bundle** pins the checkpoint by four digests, 20 source files
  and 3 portable tissues by hash, an 11-stage graph whose 5 model-active stages
  are contiguous, and 5 futility gates each naming a measure and a threshold.
- **The recovery package identity** exists with `verdict: null` and authorizes
  nothing. Its id is derived from the checkpoint, so no payload edit produces
  the CP568 id; its evidence namespace is separate from the one holding the 32B
  verdict; a verdict without evidence measured on this checkpoint is refused.
- **The steering regeneration plan** names 16 target layers — 4 attention, 12
  linear — and lists the four pieces of evidence a regenerated vector needs
  before it may steer anything. Called before capture it returns all four.
- **The preflight** refuses on source drift, tissue drift, a swapped config,
  index, tokenizer or provenance, an absent checkpoint, an active manifest
  pointing elsewhere, a changed attention layout, a misaligned window, stale
  evidence roots, insufficient RAM or disk, host memory pressure, and an owned
  model lane.

## Still model-active

Five contiguous stages inside one residency: **calibration → training → canary →
lesion arms → export**. Verification and adjudication run after the unload, in
another process, against the exported files.

Two experiments, in order. Recovery re-earns the bounded claim on the same
frozen four-domain cohort with the same five arms. Generalization runs only
after recovery is adjudicated positive. Training finishing authorizes neither,
and authorizes ordinary chat and global serving never.

## Measured expected workload

The only wall time in the record is CP566's: **300 decodes over 60 tasks in
4,814.53 seconds** on the 32B, with the desktop up. The recovery run is the same
shape, so that is the decode figure to plan against.

Training wall time appears in **no retained receipt**. It is carried as
`training_seconds: null` rather than estimated, and anyone who needs a number
should measure the first stage rather than inherit one from here.

## The launch

```bash
make rlc-27b-readiness
```

Prints the blockers, or the single launch command when there are none. It never
prints a command while anything blocks, because a command that is correct only
under remembered conditions is how a campaign starts against a drifted tree.

Package blockers have been **0** since the bundle was last frozen. The
environmental one moved twice while this was being written — 21.1 GiB free plus
reclaimable against the 26.1 needed, then clear — which is the distinction
earning its keep: a package blocker needs somebody to act, an environmental one
needs the host to be quiet. Read the receipt at launch time rather than this
paragraph.

## Boundaries that do not move

CP566 is 32B evidence and stays labelled that way. The CP568 package stays
inactive and must not be re-sealed against the 27B; re-hashing changed ground
against evidence measured elsewhere relabels unproven code as proven. The 27B
has earned nothing yet.
