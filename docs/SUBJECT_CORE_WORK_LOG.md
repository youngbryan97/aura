# Subject core work log

What was done towards 24/24, J* and the bridge, newest first. Each entry says
what broke, what fixed it, and where the evidence is. Runs live in
`~/subject-core-runs/`; the order runs are read in is
`~/.aura/subject_core/scratch/AFTER_THE_DECISIVE_RUN.md`.

## 23 September, night

### Where things stand at 02:15

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
