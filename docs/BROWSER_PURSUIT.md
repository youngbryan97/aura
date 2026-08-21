# Pursuing a goal in a browser

*Written against the tree on 2026-08-21. See
[DOC_STATUS.md](DOC_STATUS.md) for how to read this file.*

A scripted action list presumes every selector is known before the first
click. That holds for a page you have already seen and fails for most of the
web, because the next screen depends on the answer given to the last one.
Sixty-item questionnaires, installers, wizards, checkout flows: the shape of
the failure was always the same, one step and a stop.

`pursue` is the closed loop — observe, decide, act, observe again — and it is
one mode of `core/skills/sovereign_browser.py` rather than a separate skill.

## The four modes

`BrowserInput.mode` accepts `search`, `browse`, `interact`, and `pursue`.

| Mode | Needs | What it does |
| --- | --- | --- |
| `search` | `query` | Search, optionally deep-diving the first non-ad result |
| `browse` | `url` | Navigate and read one absolute HTTP(S) page |
| `interact` | `url`, `actions` | Run a known action sequence |
| `pursue` | `goal` | Decide each action from what the page currently shows |

`pursue` takes no authority of its own. Execution is delegated to the same
`_handle_interact` path a scripted interaction uses, so the lease, the
`ActionExecutor` receipt, the origin check, and the effect verification apply
identically. The loop adds perception and choice and nothing else.

## Who may drive it

`core/capabilities/browser_authority.py` is the boundary, and two rules give
it its shape.

**Reading is not acting.** `NAVIGATE`, `READ`, `SCREENSHOT`, and `SCROLL`
need a named principal and a destination that passes
`core/runtime/url_policy.py` — scheme restriction, credential rejection,
private-address exclusion, DNS-rebinding defence, port policy. `CLICK` and
`TYPE` change someone else's system, so `BrowserAction.is_effectful` is true
for those two and they need a lease naming the action and the origin.

**A lease is spent.** `DEFAULT_LEASE_TTL_S` is 300 seconds and
`MAX_LEASE_INTERACTIONS` is 50. An approval that outlives the task it was
granted for is standing consent nobody gave, and an unbounded lease is the
same defect in slower motion.

A standing directive covering the action or the URL refuses it, and that check
fails closed: an unreadable directive file stops the action rather than
allowing it. See [../TOOL_USE_POLICY.md](../TOOL_USE_POLICY.md).

## What the loop carries between rounds

A step-picker asks "which control advances the goal" every round, from
nothing, forever. Nobody uses a website that way. A person arrives with an
aim, works out what the place is — a sixty-item survey, six to a screen, a
seven-point scale, a Next button at the bottom — and acts fluently from that.

The loop carries a standing understanding with five fields:

| Field | The question it answers |
| --- | --- |
| `here` | What this page is |
| `to_progress` | What it needs from me to move on |
| `relevant` | Which controls matter |
| `present_but_not_needed` | Which controls are merely here |
| `done_when` | How I know I am finished |

It is revised rather than regenerated. The prior understanding is given back
and the question is what changed, because a revision that discards what was
already worked out is a rebuild wearing another name.

Without it, the loop has no answer to "why this control and not that one", no
way to tell an irrelevant control from a relevant one, and no way to know it
has finished except by exhausting a step budget.

## What ends a pursuit

Progress is the bound. Not a clock: how long a questionnaire takes depends on
how many items it holds, how fast the site renders, and how often it
re-navigates, and a fixed deadline is a category error against that. A working
pursuit was once cancelled at 181 seconds mid-form and the person was told the
page had not responded.

- `PURSUE_STALL_LIMIT` is 2. A round that lands no action on a page that has
  not changed counts as stalled; two in a row ends the run. The first repeat
  may be a re-render, the second is a loop.
- `max_steps` defaults to 40 and is bounded to 1–200. It is a ceiling against
  spinning forever, not the operating limit. A run that keeps making progress
  keeps going.

## Why decisions are batched, and on two lanes

A page showing six independent questions is six decisions. Asking the model
once per control turns a sixty-item form into sixty model calls, so
`_decide_next_actions` returns several actions at once when they are genuinely
independent — the difference between minutes and most of an hour.

Answered controls stop being offered. Offering them anyway is still offering
them, and they were chosen: measured live, questions 8 and 10 were each
answered four separate times while questions further down the same screen were
never reached. They also crowd the list — one screen of six questions renders
42 radios against `PURSUE_CONTROL_BUDGET`, which is 40. Their answers stay
readable in the page text, which is where reading what she has said belongs.

The rich judgement — building and revising the understanding — runs on the
full path with her identity, affect, and workspace attached, because a bare
`router.generate(prompt)` call would answer a question like "you regularly
make new friends" from a language model's priors about what an AI is. The
small per-control choices go to the fast lane through
`router.generate(prefer_tier=...)`. That is the entry that honours the tier;
`think()` on a resolved client is endpoint-level and drops it, which is why
three earlier attempts to use the fast lane changed nothing and every round
still routed to Cortex. `_decide_on_the_fast_lane` returns `None` rather than
raising when the fast lane is unavailable, so the caller falls back — a
decision that vanishes stalls the loop, a slower decision only costs time.

Nothing in the loop knows what kind of page it is looking at. A loop that
recognises page types is a collection of special cases wearing a general name.

## Watching it

A pursuit opens a window and narrates every round as it happens, so the work
is visible without a chat turn. Narration cannot break the pursuit, and a
failed round does not destroy the finished ones. Each round tells the watchdog
it is alive.

## The same loop off the web

`core/skills/screen_pursuit.py` runs the identical shape against the screen
rather than a browser: read the screen, decide, press or click, read again. It
takes a goal, a way to recognise success in what is read back, and a bound,
and it delegates the judgement to a policy callable so the loop owns no
opinion about what to press. A build log, a wizard, an installer, and a game
are the same problem.

`core/agency/goal_pursuit.py` is the layer above both: it takes a goal that
`InitiativeArbiter` has already selected, gates it on timing, executes through
`FluidExecutor` or `ParallelExecutor`, re-plans once if it stalls, and records
a `PursuitOutcome` receipt.

## Tests

| File | What it holds |
| --- | --- |
| `tests/test_browser_pursue_is_a_closed_loop.py` | Delegation with resolved selectors, an index off the page refused, a wall stopping the loop, malformed model output as an error rather than a crash |
| `tests/test_browser_effects_are_vetted_and_authorized.py` | URL canonicalisation, scheme restriction, an unauthorized open never reaching AppleScript |
| `tests/test_browser_observation_sees_real_controls.py` | Transparent controls still offered, `display:none` still hidden, every element carrying a resolvable selector, the question travelling with its controls |
| `tests/test_pursuit_is_watchable.py` | A window for a pursuit and not for a search, per-round narration, narration never breaking the run, the watchdog heartbeat |
| `tests/test_a_stalled_pursuit_tries_something_different.py` | A failed approach not chosen again, a replan matching the failed plan refused |
| `tests/test_goal_pursuit.py` | The agency-layer orchestrator |
