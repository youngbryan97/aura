# R09 Continuation History

## Mechanical Repair

`b293c085d` preserves exact trailing whitespace in native tokenizer
continuations and removes the worker assessor's independent 12-message
history cutoff. The real resident tokenizer passed four whitespace variants;
135 focused tests and smoke (164 passed, one skipped) passed before deployment.

The next repair removes an explicit history reset in the CognitiveEngine
continuation/obligation builder. Both paths retain the already admitted,
runtime-attested transcript. Newly assembled telemetry and repair instructions
remain excluded. Tests cover 40 complete exchanges, exact partial bytes,
obligation parent/segment ordering, and rejection of unattested exchanges.
The route subset passed 47 tests in 99.80 seconds. Smoke passed 164 tests with
one skipped in 69.13 seconds. Compile, governance lint, and layering passed.
Lint exposed an upstream import-order defect in screen_pursuit; sorting that
import restored the gate without changing behavior.

## Live Evidence

The supported launcher started PID 57659 on `b293c085d`. Bootstrap reported
matching expected/actual commit, workspace and shell hashes, source_current
true, and no revision issues. The 27B resident model reached conversation
readiness. The previous evening's conversation survived shutdown and launch.

At 00:55:54 PDT the desktop question was: "Which novel did you recommend for
our reading group after the correction?" Neither title appeared in this
question. Aura correctly named Gone Girl by Gillian Flynn and explained that
The Tell-Tale Heart had been replaced because it was a short story. Delivery
`aura-chat-8fafb80f-7f63-4ef4-9a6d-8cb6b2bc0d78` completed in 39.574 seconds.
Worker measurements: 2582-token prefill in 19.80 seconds; 42-token decode in
3.65 seconds. Browser reload retained the complete answer.

At 00:57:47 the paraphrase "What did you get wrong before that, and what was
my reason for correcting you?" failed. The worker's initial draft correctly
identified the short-story error in the retained excerpt but was rejected as
fabricated_shared_history. The excerpt is incomplete, so it does not establish
that every sentence of that draft was grounded. After another generation the
turn fell back to the smaller model, which denied having the conversation.
Delivery `aura-chat-a2845c4f-f424-40be-a545-3df5bd64a071` was terminal failed;
it is not a successful R09 proof. The delivery boundary then rewrote that
denial into capability-registry prose, an additional failure.

The actual neural feed reported repeated conversation_history_read timeouts
at the 1500 ms limit and deferred episodic writes. A supported SIGUSR1 stack
sample found the durable history thread inside get_recent_sessions at its
SQL query. These observations remain under investigation.

## Scope

R09 remains open. One post-restart recall and one browser reload passed;
paraphrased follow-up reliability, episodic capture/over-window recall, and
the complete streaming/cancellation/reconnection matrix are not yet closed.
This checkpoint does not claim broad RLC transfer or Aura 1.0 readiness.
