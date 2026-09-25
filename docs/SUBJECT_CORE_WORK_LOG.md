# Subject core work log

What was done towards 24/24, J* and the bridge, newest first. Each entry says
what broke, what fixed it, and where the evidence is. Runs live in
`~/subject-core-runs/`; the order runs are read in is
`~/.aura/subject_core/scratch/AFTER_THE_DECISIVE_RUN.md`.

## 24 September

### 21:25: the attention fix at 2,400 rows, a drive that ran again, and a lease on the wrong clock

Seed 7 at 2,400 turn rows on the attention fix (252079897,
`record-s7-252079897`): her arousal moves (mean 0.688, sd 0.070; it was pinned
at 0.911) and the workspace's winner and ignition vary. Synergy with the
counters out:

| triple | synergy | shifted null | interaction lower bound |
|---|---|---|---|
| P,M->W | 0.392 | 0.259 | +0.263, passes |
| W,A->D | 0.103 | 0.130 | +0.050 |
| A,S->G | 0.026 | 0.021 | 0.000 (the product terms took no weight in any fold) |
| S,D->C | 0.033 | 0.074 | -0.123 (the last fold read -0.214) |

Closure still fails. The last fold broke on a drive again: with deliberation
winning the workspace often, and deliberation serving growth, each win
credited growth its full priority, and growth rose from 50 to 80 and passed
curiosity near turn 2,370. A need nearly met is satisfied less (Keramati and
Gutkin): the credit is now scaled by the unmet share (ace4ae4b2), which brings
2,400 wins to about 63.

The reports ground's first attempt tonight stopped in its dry run: the model
lane controller stamped leases with `time.time`, which a subject run replaces
with its rewindable clock, so the worker pruned the in-process embedding
engine's lease and the heartbeat left behind asked the runtime to shut down.
Lane leases are on the machine's clock now (0ff3143b8). The reports ground
restarted at 21:20 on 0ff3143b8, with the seed-7 recording on the same commit
queued behind it.

Power study: the independent null at seed 7 decided 0 of 511 cuts, as a null
built without irreducibility should; 7 of 12 jobs are done.

### 20:05: her capacity was a constant, because nothing registered what it read

The obstruction her substrate's frustration is pushed towards (Berkowitz: an
obstructed goal frustrates the one pursuing it) is the urgency of her most
pressing intention times one less her capacity, which already has the shape
S,D->C asks about. Her capacity read 0.5000 for all 2,400 turns of the seed-7
validation run while she acted throughout. It, her usefulness in her standing
and her sense of control in acting-in-decline looked the agency ledger up as a
runtime service named "agency_ledger", which nothing registers anywhere; the
subject instrument reads the same ledger through its accessor and saw it move.
The readings now use the accessor (64c4fc443). Two tests had passed by
patching the unregistered name.

A probe that builds a stub organism found 309 of the 480 service names the
code reads resolving to nothing. Most are desktop services the offline
organism does not build; sixteen have an accessor, no registration found, and
a reader on her per-turn path. That audit is its own task.

### 19:50: every report arm now reaches her cortex

The whole-mode dry runs of 23 September all exited cleanly and none measured
anything: only the first arm of each anchor got an answer. Three causes,
found in order:

- **The worker's deadline was stamped on the experiment clock** (3b8c0b9e8).
  A subject run replaces `time.time` with a clock a restore rewinds; the
  worker is another process on the machine's clock, so after the first
  restore every request arrived past its deadline
  (`deadline_exceeded_before_decode`). Deadlines now cross on
  `core/runtime/wall_clock.wall_time`, which the experiment clock leaves alone.
- **A person's turn was finalized as fail-closed** (0f2bacdca). The harness
  named it `cognitive_engine`; one turn that ended with an answer available
  but never served was escalated to a critical failure and raised, ending the
  dry run two anchors in. It is now finalized as the desktop's chat route
  finalizes a person's turn.
- **The dry-run gate read only the process's exit.** It now reads the arms:
  at most half the anchors may have an arm the cortex lane did not serve.

The last dry run (0f2bacdca): all 32 arms answered by the cortex lane, no
failure sentence, no deadline refusal. One anchor of eight is readable
because the 1.5B stand-in often answers in words or ranges rather than with a
number; her own cortex is the one the ground is scored on. The reports ground
is queued on 0f2bacdca behind the seed-7 recording, with the machine to
itself.

### 18:50: seed 7 on the candidate reads 17 of 24, and her attention had one winner

The seed-7 run at the decisive design on 164d2a560 (300 rounds, six trials,
three-turn arms, nulls and lesion skipped; `validate-s7-164d2a560`) died at its
wall-clock bound while the 2048 demo had the machine, and was resumed from its
checkpoint. It reads 17 of 24: kappa 3, perturbational spread 0.911, 81 of 90
edges kept in one component, replication, complexity, ownership and global
access all pass. The partition lines wait for the sweep and the lesion was
skipped. Two lines fail on her:

- **Synergy**, read with the counters out at 2,400 rows: P,M->W passes
  (0.372 against a shifted null of 0.196, every fold positive); W,A->D (0.037
  against 0.084), S,D->C (0.036 against 0.061, with an interaction in every
  fold since control allocation) and A,S->G (0.013) do not. The drive fix held:
  growth stayed between 50.0 and 50.9 for the whole run.
- **Causal closure**: a leak of 0.018 spread thin over the periphery, the
  largest from how many turns since the person spoke (0.0056).

Her confidence in herself, now scored against predicting no change, sits at
its floor of 0.1 for most of the run: her self-model predicts her no better
than persistence does, which is honest, and makes the efficacy half of control
allocation nearly constant.

The attention report explains A,S->G. One bid, the heartbeat's affect bid,
won 2,341 of 2,412 competitions at priority 1.0. Its priority was her arousal
plus the size of her valence, clipped at one; every win ignited the workspace
at 1.0, and the affect phase then blended her arousal towards the winner's
priority at the ignition's weight. Her arousal was told its own value back and
sat at 0.91 for the whole run while the substrate's read 0.54, and the winner
columns of the workspace held one value. The bid now joins the affect channel
(one bidder, typed as affect, no lent urgency on top) with its priority the
larger of the two readings, and a feeling that wins attention no longer sets
her arousal. A seed-7 recording at 2,400 rows on that change reads synergy and
closure next.

## 23 September, evening

### 23:30: synergy at the decisive length, and a drive nobody was feeding

Seed 7 recorded at the decisive campaign's length (300 rounds, 2,400 turn
rows, f38d4923b; `~/subject-core-runs/record-s7-300`), read with the counters
out:

| triple | synergy | shifted null | interaction lower bound | at 320 rows |
|---|---|---|---|---|
| A,S->G | 0.080 | 0.064 | -0.115 | fail |
| P,M->W | 0.307 | 0.162 | -0.009 | pass |
| W,A->D | 0.028 | 0.107 | -0.069 | pass |
| S,D->C | 0.060 | 0.048 | -0.001 | fail |

All four fail, the two that passed at 320 rows among them. Causal closure holds
at this length (closed=True).

The interaction bound is read over five forward-chaining folds, and one fold
broke three of the triples. The fold testing turns 1,919 to 2,159 read -0.21,
-0.13 and -0.16 where the folds before it read up to +0.25. In that block her
dominant drive changed. Growth's budget climbed in a straight line from 50 to
84 across the run and passed curiosity near turn 2,030; her self-prediction's
drive error, flat for 1,900 turns, began to move, and ambivalence pressure fell
21 standard deviations.

Nothing in her fed growth. A probe on seed 7 printed the drive the motivation
phase credited each turn: the same `{'drive': 'growth', 'priority':
0.9105007597813606}` on every turn, while the winner of every broadcast was
`affect_engine`, which serves no drive. The broadcast consumer replaces its
record only when a winner serves a drive, and the motivation phase credited the
record on every tick without taking it, so one early win for growth paid out for
the whole run. The reading is now taken when it is credited (75149ff26).

S,D->C read no interaction in any fold (the largest gain 0.003). The dose test
at this length says a product can register here: goal urgency times her
self-prediction confidence, integrated into the mesh's population columns,
passes at a quarter of a spread per unit and above, where urgency times her
agency's efficacy does not register at any size. Control allocation, the
executive tier's gain set from urgency times self-confidence, is built for it
(cherry-picked onto main after the drive fix).

Next: seed 7 at the decisive design on the commit that carries the drive fix,
the confidence fix and control allocation, read at 2,400 rows before any
seed-23 run.

### 20:55: the day's runs, lost to a pause and rebuilt

The 2048 demo session stopped every subject job at 07:38 so the demo had the
machine. The seed-7 recording at the decisive length and the 1.5B dry run of
the turn fix died while stopped: a wall-clock bound keeps counting through a
pause, and both came due (10:16 and 08:32). Nothing from either was read. The
recording was relaunched at 20:38 on f38d4923b, which carries the day's agency
and narration work.

The live instance, booted at 18:39 and idle since, was closed at 20:43. Beside
it, five stub organisms took the load from 30 to 154 in three minutes as its
model workers respawned; with it closed the same five ran at 17. It is
relaunched after the reports ground.

### 21:30: what the recall change does to the content run, on six seeds

Quick content runs (six classes, fifteen pairs, six anchors), with the recall
change (f38d4923b) and without it (e8da9a89e; seed 7's second reference run):

| seed | agreement with / without | moves together with / without | recall shift with / without |
|---|---|---|---|
| 7 | +0.679 / +0.109 | +0.322 / -0.300 | 0.020 / 0.011 |
| 3 | +0.451 / +0.618 | +0.289 / +0.140 | 0.009 / 0.005 |
| 11 | -0.526 / +0.407 | +0.002 / 0.000 | 0.004 / 0.000 |
| 5 | -0.119 / +0.287 | -0.160 / +0.361 | 0.033 / 0.025 |
| 13 | +0.529 / +0.266 | +0.246 / -0.109 | 0.013 / 0.005 |
| 17 | +0.410 / +0.100 | +0.630 / -0.020 | 0.020 / 0.019 |
| mean | +0.238 / +0.298 | +0.222 / +0.013 | 0.017 / 0.011 |

Recall moves further on every seed. "Moves together" rises on five of six, from
a mean of 0.013 to 0.222; agreement, the static half, falls by 0.06 on the mean
and on three seeds of six. Structure needs both at 0.3 or more: without the
change no seed reaches both, with it seeds 7 and 17 do. The change stays.

What agreement loses has a likely cause. Recall runs before the turn's percept
reaches affect, so the mood a memory is cued by is the same for every class at
an anchor and pulls every class towards the same memories. A variant that cued
only on the emotions the percept moves, the affective tone of the material in
front of her, was run on the same six seeds and is falsified: agreement +0.122,
moves together -0.174, recall shift 0.0095 (the change on main: +0.238, +0.222,
0.0165), no seed reaching both bars. Restricting the cue took away the movement
"moves together" needs. It was never committed
(`~/subject-core-runs/content-quick-appraised-s*`).

Those runs also showed where the load comes from. Three organisms at once took
the load average to 124 with a third of the processors idle and about 3 MB/s
of disk: each runs the neural mesh on the GPU through MLX, and their threads
wait on it. With them finished the load fell to 9. Organisms now run one or
two at a time.

### 21:30: her confidence in herself could not fall

Her self-prediction's confidence sat at 1.000 at the median of seed 7 while her
valence error was 1.03 times what predicting no change would miss by: the
composite error weighed valence on the scale of -1 to 1, where her valence moves
0.02 a turn. It is now scored against predicting no change on each channel
(f31096c44). The S domain carries a reading that can move, which the two
synergy triples through S need.

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
