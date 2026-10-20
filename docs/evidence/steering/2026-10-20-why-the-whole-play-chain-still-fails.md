# Why the whole play chain still fails

This is the measured state after `56502de79`, not a readiness verdict.
The browser diagnostic uses ordinary screen pursuit, a real small vision
worker and physical browser inputs. Its language-decision provider is absent;
the log says `no model to ask`. It is not the complete native 27B application.
G09 and holdouts 31, 32, 33, 34, 35, 36, 39, 42, 47 and 49 were excluded.

## The measured chain

The fresh supervised run finished in 134.88 seconds. The report says six
results (`unread`, `unread`, `lost`, `lost`, `lost`, `unread`) and no win.
An independently viewed loss frame, `frame-0000417.png`, shows zero landings
and one crash. The final motor frame, `frame-0001713.png`, is active flight
after another start; it does not prove an ending or a landing.

| Stage | What the run retained | What that establishes |
| --- | --- | --- |
| Capture and execution recording | Complete; 781 picture payloads, 250 input events, zero dropped events | The retained event sequence can be inspected. Input-port completion still does not prove game acceptance. |
| Perception and control attribution | 582 control observations; 308 with a selected body | Body selection now happens during ordinary play. It is intermittent. |
| Visual naming | 14 instance names; 24 visual looks | The measured capture clock reaches the real vision worker. Names still describe parts such as a base, window, flag and ladder. |
| Goal binding | 48 planning choices; zero bound destinations | Every recorded task-directed choice lacks its requested referent. |
| Planning | All 48 choices have `why=wait` | Some choices press up to stabilize the current position. None of these records contains a bound landing approach. |
| Motion authority | Acceleration and direct-movement classifications alternate | Known accelerations do not remain the controller's selected motion model. |
| Phase handoff | Current end screens return; zero assignments from the visual body fallback | The new phase guard carried over in this run. Causal attribution and naming still need separate scrutiny. |
| Outcome | No win; zero observed landings in the viewed result | Component gains have not closed gameplay. |

Artifacts: `artifacts/flash-campaign/2026-10-20-ordinary-terminal-phase/`.
The owned vision worker was PID 90144. Its cleanup receipt says
`process_alive=false`, and a process check found no remaining PID. No native
runtime, cortex or foreign worker was stopped.

## 1. The task does not acquire an observed destination

`a_requested_relation_in_play.py` correctly parses the requested relation
as `on` and its destination as `the landing platform`. `bind_destination`
requires one current named object containing every meaningful requested
word. `base`, `window`, `ladder` and `flag` do not meet that condition.
Its provisional containment geometry can also refuse a target whose detected
fragment is narrower than the body. The recorded choices do not distinguish
these refusal branches, so their individual contribution is still unmeasured.

On failure, `requested_target` explicitly returns the body's own location,
`wait`, and no target object. The downstream planner therefore optimizes
staying near the current position. It has no represented destination to
approach. More accurate point tracking alone cannot change this branch.

The deeper representation gap is between visual fragments, a whole material
object, its task role and the surface/contact region needed by control.
The source-reference path can publish a visible exported object and its
origin. That observation is not yet a composed task-role or contact contract.
Calling the nearest grey object the destination would hide this gap.

## 2. Measured physical behavior can lose authority without disappearing

The run learned useful acceleration estimates in working pixels per second
squared: rest near `(0, 24)`, up near `(0, -36)`, left near `(-36, 22)`,
and right near `(45, 37)`. These are noisy measurements, not exact game
constants. The same recorded body later switches from `carried=true` with
up, left and right known to `carried=false` with only down known.
Its stored up, left, right and rest acceleration estimates remain present.

`_Speeds.carried` requires its acceleration contrast statistic to exceed
both a threshold and the separate pace statistic, and also rejects histories
that appear to stop after release. These are different summaries of the
same history. A larger pace statistic is not a physical counterexample to
an acceleration law. A collision can also stop an accelerated body.
The retained recording does not include those two intermediate predicate
values, so the exact disjunct causing each live switch is still unmeasured.

This switch changes `keys_that_move_her` and `has_a_measured_response`.
Acceleration controls then cease to be the active control interface, while
direct speed correlations can become controls instead. This is a model
selection and authority problem, downstream of successful measurement.
It needs competing models scored on the same future observations, with
collision/phase evidence kept separate from an ordinary key release.

## 3. Discovery spends the same time and physical state needed for control

`_act` returns to control trials before planning while the body experiment
is active. Trials need delivered holds, settling time, repeated independent
observations and neutral intervals. Those requirements serve real purposes;
weakening them would count coincidences as controls.

The game continues during all of this. A known thrust can be useful before
the remaining lateral controls are learned, but the experiment needs room
and time to retain its track. The code has a headroom controller, which itself
requires a current body, carried classification and measured rest/control
responses. Losing that classification can disable the mechanism protecting
the next trial. A complete repair must coordinate safe control and informative
experiments under one measured deadline, including their input ownership.

## 4. Some useful actions are rejected before they reach the hands

The ordinary log twice rejects up immediately after the play control because
`where_am_i` sees no occupied text/grid places and no pointer target. A wordless
graphical world supplies neither. This causes an initial delay before the
fast gameplay path takes over. It is not the cause of all later failures;
the same run subsequently delivers many controls.

The missing contract is positive visual custody of the current task surface
through a delivered transition. Application names and page URLs alone cannot
supply that proof. Removing the guard would trade this failure for typing
into unrelated windows. A visual transition witness needs to reach this gate.

## 5. Reading the source is not yet executing its physical and success laws

The source reader can retain actions, branch conditions, effect graphs and
visible asset references. A complete effect graph is not an executable
transition model: heap effects and unresolved calls remain uncomposed.
`source_motion_checked` only admits a complete direct-control specification
checked against the current body; it explicitly declines accelerated bodies.
It does not provide a source-derived accelerated landing controller.

The task requires control over a state such as position and velocity, plus
the target's contact region and a safe-arrival predicate. The live connection
must map those quantities to current visible material and the actual input
clock, predict an action's effect, observe its delivered result and revise
the same model. Having the source file, or a controller supplied exact state
in a test, does not create those mappings.

## What must close before another readiness claim

Use one explicit state/action/task contract across perception, identification,
model selection, planning and delivery. Each quantity needs its identity,
coordinate basis, observation time, uncertainty and evidence owner. Keep a
task's currently bound referent and acceptance conditions in that contract.
Missing evidence remains unknown and must identify the next useful
observation or experiment, rather than indefinitely supplying a wait target.

First expose the exact model-selection predicates and destination refusal
branches in the existing recording. Falsify the motion classifier with
accelerated, direct-control and collision worlds; score alternative models
against the same held-out physical observations. Then connect whole-object
role and contact evidence to goal binding. Test each repair through current
pixels and delivered input, before interpreting original-game outcomes.

There is no evidence here that the remaining work is one click fix or one
tracker installation. The present failure is a chain of missing or unstable
contracts. All of these mechanisms apply to interactive graphical systems
beyond a particular game. Native ordinary play and the 56-game campaign
remain unproven.

## The missing branch records are now connected

Destination selection now retains its actual result and refusal branch:
unavailable body/request, absent name, ambiguous name, incomplete name,
stale referent, unsupported relation geometry or bound destination. The
existing binding API consumes that same result. The choice recording also
retains bounded current names, candidate counts and a bound target's stance.
Truncating diagnostics cannot remove candidates from selection.

The motion-family classifier now retains its actual ordered predicates and
the reason it selected a family. The observer records the evaluation time
and body number. Predicates skipped by short-circuit evaluation remain
unavailable. The classifier's thresholds and Boolean decision are unchanged;
this is evidence for the next repair, not a claimed motion-model correction.

The focused recording/binding/control group passed 138 checks in 11.13
seconds (12.18 supervised), followed by 62 controller and attribution checks
in 5.22 seconds (5.97 supervised). The first integration run exposed two
legacy schema assertions; diagnostic fields are now added only when the
current selector supplied a reading. Legacy observations keep their original
requested-target shape. Both pure canaries are registered with their test.

Artifacts for these checks are in
`artifacts/flash-campaign/2026-10-20-whole-chain-audit/`.
Complete gates, push, then measure the actual branches in ordinary play.

Checkpoint gates passed in 78.65 seconds: smoke (164 passed, one skipped),
lint, compile, layering, generated DEPS and writing. The two new organism
imports have narrow named-module rules; baselines were unchanged.
