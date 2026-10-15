# Static material can be recognized without a model, 15 October 2026

The ordinary CPU browser run `2026-10-15-case18-offered-controls` at
`6d9f707cd` completed in 81.98 seconds. Its complete recording has 229 motor
frames, three normal screen snapshots, 78 input events and five normal
play choices. There were no dropped events. It identified a body after
4.98 seconds, then made five choices that retained the requested relation
and explicitly reported no bound destination. It did not replace the goal
with scenery. The result remained lost: zero landings, one crash, won=false.

In this run the normal pursuit offered and clicked both OCR-damaged Start
and Play labels. No filter discarded them. This measurement cannot explain
why the preceding run tried keys before Play. It does establish that the
new goal guard reaches ordinary browser planning, and that a missing visual
referent remains the next unresolved connection.

## Source references

JPEXS FFDec's documented command line exports static sprite artwork. Its
unmodified 26.3.0 release and an isolated Temurin 17 JRE were downloaded from
their official GitHub releases and checked against published SHA-256 values.
They are separate local tools, not code copied into Aura. The renderer used
one CPU, a 512-MB heap, headless Java2D and no OpenGL. It loaded only original
case 18's already audited source file. No ActionScript was executed and no
live game state was read. Original case 9, G09 and the holdouts were excluded.

The first `-select all:1` export failed with an upstream null-pointer error;
its failed supervisor record remains. A fresh `-select 1` export succeeded
in 1.17 seconds, but exported all sprite frames. Its directory name is not
evidence that only first frames were exported. Only the first PNG for each
of two exported references was used in the correspondence measurement.

Command documentation: [FFDec command-line arguments](https://github.com/jindrapetrik/jpexs-decompiler/wiki/Commandline-arguments).
The [SWF specification](https://open-flash.github.io/mirrors/swf-spec-19.pdf)
defines exported symbol names and sprite definitions independently of
their current execution state.

## Current pixel correspondence

`a_visual_reference_in_the_picture.py` adds a general reference matcher for
manual illustrations, icons and exported artwork. A masked FFT search
proposes bounded places and scales. Existing original patch-fit code then
checks local correspondence in both directions. At least six supported
points and a redundant rigid or affine fit are required. Scale-search
cost supplies no authority by itself. Duplicate visible copies remain
separate referents; flat and unrelated material is refused.

The source and picture byte identities remain attached to each result.
These are reference correspondences, not temporal point identities. The
component grants no controlled-body status, task role, collision boundary,
safe arrival speed or completion. It has not been added to the live actor.

On the actual captured gameplay frame `frame-0000304.png` from the complete
run above, it found one platform appearance with 27 supported points in
0.358 seconds and one ship appearance with seven in 0.217 seconds. On the
recorded instruction-screen overview, it found no platform and one ship
preview with six points in 0.189 seconds. Thus the reference method itself
demonstrates why appearance cannot establish control. The no-match platform
query on that screen took 0.379 seconds. These are CPU component timings,
not frame rates or gameplay results.

Raw measurements, point pairs and hashes are in
`artifacts/flash-campaign/2026-10-15-static-reference-first-frames/matched-material.json`.
`tools/measure_supplied_visual_references.py` makes the same component probe
repeatable for arbitrary supplied pictures and references, with fresh output
paths, search limits and explicit evidence scope.

Sixteen tests passed for scaled independent image compositions, duplicate
copies, unrelated scenes, missing support and search bounds. The remaining
integration must relate source names or instructions to the referenced
material, preserve observation ownership and retain causal body checks.
No original-game win or presentation qualification has been demonstrated.
