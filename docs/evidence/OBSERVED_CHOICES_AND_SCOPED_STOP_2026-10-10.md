# Observed choices and scoped stopping, 10 October 2026

Base revision: `e0e82a43f`.
Presentation qualification remains **unproved**.

The installed app loaded this revision in one runtime. An ordinary chat request
opened the archived game, entered the kitchen and watched its instructions.
Aura then pressed Test Trap without placing a part. The control had no matching
procedure step, so the earlier matched-step prerequisite did not cover it.
Method prose was also treated as a literal control name, and unnamed choices
could not bind to the options exposed by a selector.

The Stop button sealed the chat, but browser work continued. The model gate saw
`chat_delivery_cancelled_by_user`; the screen loop did not consult that token.
Normal application quit completed graceful shutdown with exit code zero.
The full failure segment is retained locally in
`/Users/bryan/.aura/flash-proof-2026-10-10/live-e0e82a43f-complete.log`.

Method words now identify input delivery separately from the named target.
A selector can reach an unnamed choice prerequisite. Its owned, fresh observed
effect binds newly revealed options to that choice. The binding requires a
current capture, the same surface and viewport, and a currently visible option.
Execution and evaluation controls in construction procedures require current
measured placement even when the control is absent from the procedure. Old drag
records and descriptions of built work cannot supply that prerequisite.

Interruptible scopes connect the existing stop token to the task executing an
effect, including work behind a shielded waiter. The execution loop rechecks
ownership between perception, decisions, approval, action and retry. Real-time
play and page input check the same owner. Listeners detach when a scope ends;
nested scopes avoid repeated cancellation during input release. Cleanup can
release held controls after Stop. Cancellation remains cancellation rather than
becoming another fallback attempt. A synchronous worker already executing outside
the event loop still requires its own cooperative stop handling.

[AutoGen's cancellation implementation](https://microsoft.github.io/autogen/stable/_modules/autogen_core/_cancellation_token.html)
links pending futures to a token. [BehaviorTree.CPP's asynchronous action guide](https://www.behaviortree.dev/docs/guides/asynchronous_nodes/)
requires running actions to support interruption. These informed the task-owner
binding and checks between actions. Neither library was installed; Aura's own
token, execution gateway and receipt path remain the implementation.

Validation used the shared virtual environment and isolated test logs:

- Execution, stopping and procedure regressions: **134 passed**, 14.25 seconds.
- Updated invariants and action-executor admission/effect contracts:
  **75 passed**, 8.54 seconds. These runs overlap and are not additive counts.
- Smoke: **165 passed, 1 skipped**, 64.73 seconds.
- Repository lint, compile and layering passed; layering retains 36 grandfathered
  edges. One narrow module permission admits the pure readiness canary.
- Governance lint still reports 101 pre-existing regressions. Comparing changed
  production modules against the base found **zero effect bucket changes**.
- Registered claims remain synthetic: current choice binding, rejection of stale
  construction credit, and detachable cancellation listener lifetimes.

Full-suite validation, live Stop verification, visible construction and completed
ordinary app gameplay remain outstanding. The earlier live failure is evidence
for these repairs, not evidence that they are now presentation ready.
