# Subject core work log

What was done towards 24/24, J* and the bridge, newest first. Each entry says
what broke, what fixed it, and where the evidence is. Runs live in
`~/subject-core-runs/`; the order runs are read in is
`~/.aura/subject_core/scratch/AFTER_THE_DECISIVE_RUN.md`.

## 23 September, night

### 03:13: the recovery, second attempt

The first version of the recovery (934564f60) asked the gate to warm the lane
every ten seconds. Each request extended the gate's startup quiet window, and
inside that window the warmup it had just started was refused as background
work, so a 1.5B dry run spent every turn in that loop. 39656c6fe acts only on a
stopped worker: it calls the gate's own restart hook once
(`_respawn_cortex_if_needed`, which the router was meant to call and never did
once its circuit opened) and watches the lane. The chain restarted at 03:13 on
that commit.

### 02:50: the first whole run with her feelings steering her cortex

At 02:27 the reports ground started on her 27B cortex with affective steering
attached for the first time in a whole run: 80 qualified vectors, byte for byte
the desktop's, 16 hooks, engine online. It measured nothing. Its first baseline
turn offered a tool, ran past the router's 105 s endpoint budget at about five
tokens a second, and the router aborted it by stopping the model worker.
Nothing started the worker again: with the circuit open the router never
called the gate, which is what starts a stopped worker, and recovery counts as
background work, held until a first visible conversation that a dead worker
cannot give. Every later turn ended in the failure sentence. It was stopped at
02:45 (`reports-s23-void-cortex-0245`). Since 934564f60 each turn of a whole
run first brings a stopped language worker back, and the chain restarted at
02:50 on that commit.

Meanwhile: seed 7 at the decisive design (six trials, two-turn arms) is
running in `~/subject-core-runs/design-s7`; the recurrent reference decided all
511 cuts on all three power seeds, and the star nulls are still running.

### Where things stood at 02:15

- **Decisive run (seed 23): not yet started.** The first seed-23 run at
  68922b866 was stopped at 01:22 and is `v5-s23-void-fd-0122`; no number was
  read from it. Its fork carried the standing authority's audit chain, which
  holds open descriptors, so every dropped snapshot copy closed descriptors the
  run still used.
- **Queued behind it: the reports ground.** `reports_then_decisive.sh` waits for
  the charger and 30 GB free, runs a whole-mode dry run on a 1.5B model (one
  hour at most), stops if that run crashes or a fork copy closes a descriptor,
  then runs the whole reports ground on her cortex (eight hours at most), then
  launches the decisive campaign, the five-worker sweep and the content run,
  all on one commit.
- **Power study:** paused by the load governor while the Mac is on battery.

### Why every whole run so far measured nothing

A whole run is her with her own 27B cortex, which the reports ground needs.
Every attempt died or ran degraded. The causes, in the order they were found:

1. **The fork copied objects that own OS descriptors** (39bcc5c78). A copy's
   finalizer closed the original's descriptors; Metal later reused a number and
   guarded it, and the next close was a kernel kill (EXC_GUARD).
2. **Under a run's own state root the registry could not confirm her cortex**
   (d3b45f643), so affective steering never attached. The whole environment now
   pins the key file the desktop reads.
3. **Her feelings never reached the forward pass, on the desktop either.** The
   array the model worker steers from was never written by the parent
   (1448f2385), the worker could not read an array at all, fell back to zeros
   that the hooks read as "low" on every dimension, and its governor never ran
   (c42e8d4e9). The desktop's fusion certificate had been measured by handing
   states to the hooks directly, so it certified a path live traffic did not
   take. The live desktop keeps the old behaviour until it is restarted.
4. **Only her cortex's answer is her report** (6a22d931e, ea6fa8c63, and this
   batch). Each arm records which endpoint answered and the steering applied;
   the brainstem's answer and the fixed failure sentence are excluded; the run
   refuses if steering did not attach.
5. **Found by a 1.5B dry run, three minutes in instead of eight hours** (this
   batch): the fork copied the pipes to the model worker, and a copy's
   finalizer closed the live pipe (the same EXC_GUARD, and the 98 s "no
   progress" stall in the first report run); a restore reset the model client's
   request-lock owner record so the lane stayed held forever; the worker's
   environment scrub removed the key pin because "AUTHORITY" contains "auth".
6. **What she learned about a world was never kept in a fresh root** (this
   batch): `remember` made its directory outside its governed scope, so in every
   campaign the write was refused and `recall` came back empty.

### J* structure at 24 anchors (seed 7, read 02:20)

The seed-7 content run at 24 anchors finished after 10.5 hours. It ran at
8447bc297, before the displacement was redesigned (valence plus a
reference-point shift, held for the turn) and before the noise rule became a
rank test, and it reads NOT_MEASURED under the old rule: the closest pair of
classes (0.044) was no further apart than a class from itself (0.058). Its
other numbers:

- agreement between her internal geometry and her recall geometry: rho 0.386,
  p 0.001, bar 0.3, passes;
- design recovery: rho 0.654, p 0.001, passes;
- moves together under displacement: rho 0.228, p 0.003, bar 0.3, fails.

Under the current rule the same data is out of its noise: between-class
distances exceed the within-class floors with a one-sided Mann-Whitney p of
1.2e-5, and 48% of class pairs sit above the largest floor. So at 24 anchors
the structure question comes down to "moves together". The redesigned
displacement moves the internal geometry about thirty times as far (shift
0.178 against 0.006). In two six-class quick runs the recall geometry did not
move with it (rho 0.0 and -0.30, 15 pairs each), which is weak evidence
either way. The seed-23 content run under the new design decides it.

### What seed 7 says about the lines still failing

The seed-7 whole-system proof (3 trials, one-turn arms) failed kappa,
partition irreducibility, partition beating the nulls, perturbational spread
(0.178 against 0.6), synergy, natural-runtime replication, lesion deficit,
rescue, and "beats every null". Spread is the widest gap. A displacement of A
reaches D from frame 11 but reaches C, S and N only in frames 31 and 32 of a
33-frame arm, at the end of the turn when the phases that read her state run,
so a one-turn arm cuts responses off as they land. I and N reach nothing. The
decisive design runs two-turn arms with six trials; a seed-7 run at that design,
nulls and lesion skipped, is next once the Mac is on the charger.
