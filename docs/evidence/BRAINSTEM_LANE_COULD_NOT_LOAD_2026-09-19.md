# The brainstem lane could not load, whichever model was bound to it

**2026-09-19.** Asked whether the new 27B was live as the brainstem. It was
configured, registered, named in the router's own tier layout, and had never
served a turn. Three separate gates stood between the lane and a load, and
only one of them was about the new model.

## What was measured

Ternary-Bonsai-2-27B at 2-bit, against the Qwen3.5-9B at 4-bit it replaced,
both loaded alone on this host:

| | on disk | active | peak | load | decode | sheep | bat and ball |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bonsai-2-27B 2-bit | 8.61 GB | 7.68 GB | 8.70 GB | 2.48 s | 8.7 tok/s | correct | correct |
| Qwen3.5-9B 4-bit | — | 5.04 GB | 5.27 GB | 1.76 s | 21.4 tok/s | correct | did not land |

The 9B is 2.5x faster and ran past the bat-and-ball answer without reaching
it, 160 tokens of working and no figure. For a lane nobody waits on, that is
the wrong side of the trade.

## The first gate: a floor no host could meet

    Deferring background local endpoint Brainstem
    (desktop_background_headroom:Brainstem:67.9%/20.6GB(need <100.0% and >=22.0GB))

Every thirty seconds, for the whole uptime. The floor was 22.0 GB, written
when the lane held a 9B and the Cortex left about 28 GB free. The resident
Cortex now leaves about 20, so the floor asked for more than the machine
ever has — and because the number is model-independent, this was equally
true of the 9B. The lane went dead when the *Cortex* grew.

The comment beside that constant already recorded the same shape happening
once before: a 48% / 34 GB gate that could never admit, so mind_tick never
completed a tick, the launcher read that as death and respawned a second
32B — a self-sustaining loop, observed 2026-07-06.

The floor is now the bound checkpoint's projected footprint and half again.
That projection is itself conservative: 11.0 GB against a measured 8.70 GB
peak. It lands at 16.5 GB for Bonsai and 10.0 GB for the 9B, and it can
never sit below the footprint of the thing it admits, which is what made a
4 GB floor kill the Cortex.

After the fix, live: `need <62.0% and >=16.5GB` at 20.6 GB free. The floor
passes.

## The second gate: cold read as not-ready

    Endpoint Brainstem failed validation: lane_not_ready:cold
    Circuit OPEN for Brainstem on transient runtime failure.

`_local_client_failure_reason` lists `cold` beside `recovering`, `spawning`,
`handshaking` and `warming` as states a lane must not be given work in. That
is right for the resident foreground lane, and the comment says so: a
foreground generation submitted to a warming Cortex blocks the caller for
105 seconds while warmup contends for the single worker.

It is not right for a lane that loads on demand. Cold is where such a lane
rests, and sending it work is how it stops being cold. So validation failed,
the circuit opened, and the circuit is what then prevented the load that
would have cleared it.

The same question is already answered correctly one module over. The gate's
tier-health sweep lists `("spawning", "handshaking", "warming",
"recovering")` — without `cold` — and calls a cold brainstem `standby`,
with the reason `cold_by_policy`.

## The third gate, which is not a defect

With the floor fixed the lane defers on pressure: `67.8% >= 62.0%`. That
percentage comes from psutil's macOS accounting, which counts file-backed
cache and compressed pages as consumed, and the code already knows it — when
the kernel itself reports no pressure, the derived percentage is not allowed
to veto the small models. The remaining deferrals are the ones where the
kernel does report pressure. That is the gate working.

## A warning that describes another checkpoint

transformers says on every load of the pack that the tokenizer has "an
incorrect regex pattern", links a Mistral-Small discussion, and advises
`fix_mistral_regex=True` or expect incorrect tokenization. On a lane that
answers questions that would be a correctness problem.

Measured: the pack's `tokenizer_class` is `Qwen2Tokenizer`, and loading it
both ways gives byte-identical ids on everything that regex governs — digit
grouping, contractions, runs of whitespace, accented text. 0 differences of
7, and round-trip exact on code, unicode and the chat special tokens. The
flag is not passed, and the reason is written at the load site because the
warning fires on every boot and reads like a defect.

## What happened once the floor came down

    [PROMPT CACHE] miss — prefilling all 774 tokens;
      key=('Ternary-Bonsai-2-27B-mlx-2bit', 'default')
    [PROMPT CACHE] retained 824 tokens scope=default
      key=('Ternary-Bonsai-2-27B-mlx-2bit', 'default')
    [MLX] Soft-cancel requested for job seq=9 (generation_caller_cancelled)
    Brainstem returned no text. Trying local fallback.

It loaded, it read the prompt, it decoded, and the caller stopped waiting
after about twenty-nine seconds. So the lane works and does not yet finish
inside the budget it is given, which is the next thing rather than the same
thing.

That budget is now priced on the lane's own readings. `thinking_reserve`
keeps decode rates per model and the two sides of the process boundary were
not changed with it: the client carried the worker's measured rate across
under no model at all, and the gate's background budget cut asked for an
estimate with no model, so every lane wrote to one window and the background
lane's budget was cut on the blend. Both name the model now, and the store
shows the split taking:

| checkpoint | readings | median |
| --- | --- | --- |
| Aura-Qwen3.8-27B (cortex) | 128 | 3.0 tok/s |
| Qwen3.5-9B (the old brainstem) | 128 | 9.6 tok/s |
| Ternary-Bonsai-2-27B (this lane) | 13 | 2.3 tok/s |
| unnamed, written before the split | 128 | 6.4 tok/s |

Bonsai measures 8.7 tok/s alone and 2.3 beside a resident Cortex. The
unnamed window says 6.4. A budget cut on 6.4 hands this lane nearly three
times the tokens it can deliver, and the cut exists to stop a generation
being promised more than the clock can pay for.
