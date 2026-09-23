# Subject core work log

What was done towards 24/24, J* and the bridge, newest first. Each entry says
what broke, what fixed it, and where the evidence is. Runs live in
`~/subject-core-runs/`; the order runs are read in is
`~/.aura/subject_core/scratch/AFTER_THE_DECISIVE_RUN.md`.

## 23 September, morning

### 07:20: the reports ground lost her cortex again, and the two reasons why

The reports ground that started at 06:34 on 23650c6c4, with the machine to
itself, measured nothing and is `reports-s23-void-lane-0707`. Its first tool
turn ("Write today's plan into notes.txt") was decoding at about ten tokens a
second, thinking before its call, when the router stopped the model worker at
105 s. The worker came back and failed its first warmup probe, the retry stood
down because "somebody is still being answered", and the lane never became
ready: the rest of the baseline ran without her cortex and every anchor turn
ended in the failure sentence.

- **The retry waited on a copy of its own flag** (e8da9a89e). It imported the
  client's "a person is being answered" flag by name and polled that name, so a
  retry that began during a reply waited the full 30 s and stood down however
  soon the reply ended. The flag is now read from the module on each check.
- **A person's turn in a whole run was never an open turn** (this batch). The
  desktop binds a turn for every message, and the router's thread watchdog
  stands down for a bound turn with a person waiting, leaving the endpoint's
  own liveness to judge a slow answer. The subject driver runs the phases
  itself and bound nothing, so every generation somebody waited for was killed
  at the flat budget for a short reply. `person_turn` opens the turn the
  desktop opens, marks what was served and finalizes it, for a person's turn
  in a whole run only. The stub organ never reaches the router, and binding
  there would change the stub organism's error records for bookkeeping.

The reports ground is relaunched on the fix after a 1.5B dry run, with nothing
else on the machine.

### 07:15: at 320 rows the synergy line cannot see a strong product

Before building anything for the two triples that fail on her, the seed-7
recording was given synthetic product terms of known size
(`synergy_dose*.py` in the session scratchpad). A product of the self-model's
and the drives' own first principal components, integrated into all eight of
the mesh's population columns at one standard deviation per unit, lifted the
synergy over its shifted null (0.048 against 0.040) and left the interaction
gain's lower bound at 0.000. At that size the bar is out of reach. The
decisive campaign records 300 rounds, 2,400 turn rows, so a seed-7 recording
of that length is running (`~/subject-core-runs/record-s7-300`). All four
triples are read from it, clocks out, before any mechanism is designed.

What the design would have to move, from the seed-7 recordings: C's change is
led by the mesh's population statistics (a quarter of its variance), then
substrate arousal and frustration, then the reactive-deliberate switch; S's
first component is how well her self-model predicts her; D's first is her
drives' energy against growth and social need. If S,D->C still fails at scale,
the mechanism with a published basis is control allocation as incentive times
efficacy (Shenhav, Botvinick and Cohen, 2013), carried by gain in the mesh's
executive tier.

## 23 September, night

### 06:40: synergy, read on the line that counts

The 24 lines are scored with ISC-v3's synergy (`passes_v3`), which already
replaced the v2 fraction's nulls with a null that simulates her sources' own
dynamics (a VAR(1) fit), keeps both interaction bars, and leaves the fraction
only an absolute floor. That is the revision the 15 September session found
necessary, when her sources' persistence (lag-1 of 0.98 to 0.998) made the v2
fraction unable to see a coupling of two spreads.

On seed 7 at the decisive design, v3 passes two of the four preregistered
triples: P,M->W (synergy 0.374 against a simulated null of 0.010 +- 0.005) and
W,A->D (0.296 against 0.055 +- 0.014). The other two fail on her, not on the
instrument:

- A,S->G, affect and self-model on the workspace: 0.127 clears the simulated
  null but not the shifted one (0.156), and the interaction gain's lower bound
  is negative (-0.069);
- S,D->C, self-model and drives on recurrent cognition: synergy 0.017, and no
  interaction at all.

The workspace sums its terms: base salience, the urgency affect lends (its
weight times 0.3), and a free-energy boost for bids aligned with her dominant
action. Nothing makes a feeling's pull on attention depend on how much the
content concerns her, which is the joint dependence A,S->G asks about. So this
line will fail on seed 23 as she is now. Passing it needs mechanisms, such as
appraisal-style gating where a feeling's pull scales with its relevance to
her, designed and checked on seed 7 before a decisive run. The decisive run
goes ahead as planned, because partition, the largest unknown, needs it.

### 06:30: seed 7 reads 19 of 24; the decisive run restarts with three-turn arms

**Seed 7, two-turn arms, lesion measured: 19 of 24.** The lesion of the cheapest
cut (S against the other nine) and its rescue pass, the first time either line
has been measured on any run. Still failing: kappa, the two partition lines
(decided by the v5 sweep in the decisive run), synergy (scored on 320 rows; the
decisive campaign has 2,400), and "beats every null", which needs the rest.

**Kappa.** Every route into W, her world model, goes through one column, the
surprise of her learned world model. S moves it by 1.9 standard deviations;
D, M and P by 0.22 to 0.27, under the 0.3 bar, peaking late in the arm. With
three-turn arms on seed 7 kappa is 2, spread rises (A, C, S and W reach every
other domain) and all eight conditions stay strongly connected. That run failed
causal closure, which is computed from the recording before any arm runs; the
same seed does not reproduce the recording (the two seed-7 recordings differ
in all 10,560 rows), and closure read -0.004, -0.012, -0.014 and 0.068 across
four seed-7 runs. So the decisive campaign now runs three-turn arms.

**The reports ground at 03:21 measured nothing, and that was my scheduling.** It
ran beside the seed-7 runs and six power jobs at a load pressure of 1.92 per
core. A cortex recovery's warmup timed out behind other work, the lane was
marked failed, the router's circuit stayed open, and all 96 arms came back
empty. Six power jobs at once had also taken the host from load 58 to 186. The
power jobs are paused, and the chain restarted at 06:27 on 23650c6c4: a 1.5B
dry run, then the reports ground alone on the machine, then the decisive
campaign with three-turn arms, the sweep and the content run. The two-turn
decisive run that had started at 03:56 was stopped before any of its numbers was
read.

### 03:35: the reports ground is running on her cortex, steered

The reports run that started at 03:21 has her 27B cortex with steering applied
to her replies at the certified alpha of 0.2 on every user-facing generation.
Its first tool turn was aborted by the router at 105 s, which stopped the
worker again, and this time the next turn found it stopped, asked the gate, and
logged "her language organ's worker was back in 20s".

Open, not on the path to 24/24 tonight: after each generation the production
CAA's adaptive alpha raises the engine's alpha (3.8, then 4.8, up to 6.2 so far)
and writes it to the hooks. Those numbers are in the old absolute units, where
15 was "standard"; the stream-fraction default is 0.2. Since c42e8d4e9 the
governor resets the hooks every 50 ms, so the high value lasts one tick, and
user-facing decodes are clamped to the certificate either way. Before that fix
the governor never ran in the worker, so on the desktop the hooks sat at the
adaptive value between surface decodes: background generations were steered at
several times the stream, in the constant "low" direction.

### 03:25: seed 7 at the decisive design reads 17 of 24

Seed 7 with six trials and two-turn arms (`~/subject-core-runs/design-s7`, nulls
and lesion skipped, 61 minutes) against the one-turn proof's 15:

- **perturbational spread passes, 0.667 against 0.6** (was 0.178). Every source
  now reaches something: I reaches 0.556 of the other domains, N 0.333, S all
  of them.
- **natural-runtime replication passes:** all eight conditions form one
  strongly connected graph (was two of eight).
- **kappa is still 1.** W, her world model, has one kept in-edge (S->W, 1.88 in
  all eight conditions), so removing S cuts it off. The next candidates into W
  miss the preregistered bars: D->W 0.269 pooled (0.35, 0.40 and 0.42 in
  autonomy, idle and stress; the bar is 0.3 pooled), M->W 0.241, P->W 0.223.
  They peak at frame 60 or later of 66, still rising when the arm ends, the
  shape the edges into I had under one-turn arms. A seed-7 run with three-turn
  arms (`design-s7-t3`) is measuring whether they clear the bar with the time
  to arrive.
- **synergy fails** on its nulls: all four triples must pass and each misses
  differently on 320 rows (P,M->W clears its shifted null, 0.482 against 0.444,
  but not the held-out interaction bound; W,A->D has an interaction gain with a
  lower bound of 0.18 but sits under its null). The decisive campaign records
  2,400.
- **lesion and rescue** have never been measured on any run on disk; the
  seed-7 lesion is running now from the design run's own cheapest cut
  (`design-s7-lesion.log`).
- The two partition lines are scored by the v5 sweep in the decisive run, and
  "beats every null" needs the rest.

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
