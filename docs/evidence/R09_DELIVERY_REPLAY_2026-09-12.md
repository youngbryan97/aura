# R09 delivery replay, September 12

## Follow-up repair

Delivery `aura-chat-8b628020-fbcd-4348-afad-d6c0a2db4728`, turn
`268660c680fb4550b79025f87014e5bb`, completed at 00:28:58 PDT on runtime
11182. It resolved "those processes" to evaporation and boiling after restart.
Forty exchanges reached generation. The delivery receipt records one model
generation, zero retries, and one deterministic text mutation.

That mutation prevents a clean acceptance result. The capability ledger
replaced a sentence in the scientific example with Aura's interoceptive
readings. The stage was `chat.capability_claim_reconciliation`, the method
`measured_sentence_replacement`, and the reason
`contradicted_capability:interoception`. Before/after hashes were
`da46bccb8d5e8efbcf236aa60dd9cb3d0fbf845b4fe203a5f1eed6baafe81251` and
`b92009efe65d1bc17ddb4263134bc3d2279578fc4f6e4f282b050bd1e4076b36`.
The retained journal does not contain the replaced sentence's original bytes.

The checker treated retention-loss predicates, including "evaporates", as
universal denials. A sentence containing that verb and the capability's topic
word "energy" or "body" could therefore be replaced by a vitals summary.
The repair makes failure predicates an explicit capability declaration.
Memory and deferred intentions retain their loss checks; other capabilities
cannot use those predicates to claim a contradiction. Ordinary explicit
denials remain checked. Tests cover physical evaporation, discarded energy,
external signals, retained reminders, and the production reply reconciler.

## Numbered-list display

The journal's numbered paragraphs rendered as three lists starting at 1.
Blank lines separate list blocks; the renderer discarded their start value.
Each ordered list now preserves its declared first ordinal. A Node test runs
the production renderer for loose lists, continued numbering, a new list
starting at 1, ordinary prose, and escaped markup. The regression failed
before the change and passed after it.

## Gates and deployment

The final focused run passed 119 checks in 16.57 seconds. Earlier attempts
exposed two test-fixture defects: a missing Availability.summary and a patch
against the moved function instead of its call-time import. Both were corrected.
Smoke passed 164 tests with one skipped in 35.83 seconds. Compile and layering
passed; 37 grandfathered layering edges remain. Ruff passes on the three
touched Python files. The aggregate lint gate reports 69 errors in newly
fetched, unrelated files; it is not recorded as green.

The stopped desktop was launched through the installed signed application
after verifying no resident model/training owner and 39.1 GiB available RAM.
Only the two production repair blocks were applied to the dirty canonical
checkout; unrelated edits were preserved. The launcher was built and installed
with its existing signing identity. Runtime 7761 reached ready with verified
source snapshot
`a81ad0fedd37274043d0144e8220ade32e03b31274497270025f3931c7e10bf7`.

The restored live pane now renders the retained answer's ordinal markers as
1, 2, 3. The unchanged multipart request was submitted at 16:08:01 PDT.
Its completion and a clean follow-up are still pending at this entry.

## Completed multipart and follow-up replay

The multipart delivery `aura-chat-24419db2-9302-495b-b6f4-310909c584c8`,
turn `223086a5c5904a679c0d58118e267884`, sealed in 76.24 seconds. The
initiating browser was reloaded during prefill. It restored the pending
question and received one complete answer without another manual reload.
The answer contained all three numbered points and both everyday examples;
the rendered ordinal markers were 1, 2, 3. There was no PARTIAL badge or
private reasoning in the public answer. The journal records one attempt,
one foreground generation, zero repair/completion retries, zero text
mutations, and answer_delivery_proven=true.

The next question, "Which of those processes cools the remaining liquid, and
why?", was submitted through the same desktop input at 16:09:44 PDT.
Delivery `aura-chat-24c34582-e132-4fdd-97f1-b3f11f8959d1`, turn
`046513ede6594db39c3a7e8c06441db2`, sealed in 62.73 seconds. It identified
evaporation, explained the loss of higher-energy molecules, and retained its
sweating example without replacing it with Aura's vitals. The browser showed
the complete answer at 16:10:47. The receipt again records one attempt, one
generation, 40 admitted exchanges, no retries or text mutations, and an
answer-delivery proof. These are contextual-delivery results; the discussion
of boiling assumes continued heating and is not a proof about every boiling
regime.

The worker measured 2,400 prefill tokens in 13.44 seconds and 503 generated
tokens in 46.63 seconds for the follow-up. First-token latency was 13.66
seconds; peak worker memory was 16.64 GB. Stabilization took 285.94 ms and
runtime reconciliation 9.88 ms. The repaired reader also recovered all 100
display exchanges after the initial RAM-only bootstrap, without replacing
the already visible current answers or raising model context admission.

After rebasing onto the concurrent integration work, the delivery-journal,
cancellation, stream-identity, UI-history, rendering, and capability suites
passed 153 tests in 37.14 seconds.

## Separate runtime observations

The actual neural feed and terminal remained under observation throughout.
At 16:15:35 the feed described a phi value as "2.02 out of 1". The terminal
identified 2.02100 as the maximal six-node complex and 0.33943 as full-system
phi, with state-summary rather than activation-level grounding. The displayed
denominator is therefore not supported by that measurement; R08 owns the
unit/meaning correction. Neither value is evidence of consciousness.

A 60 ms hold of core.canonical.state.singleton on the event loop raised a
lockdep splat during the next turn. R06 retains that blocking-work obligation.
Background model admission and semantic-cache work yielded to the foreground;
those deferrals did not cancel these two answers. R11 still owns measured
latency improvement. The earlier worker death and launch races remain in
their dated records; successful chat delivery does not erase them or certify
the wider runtime.

## Final retention and reload result

At 16:15:47 the desktop submitted: "What novel did we settle on for the
reading group, and why did we replace your first suggestion?" It supplied
neither title nor the correction. Delivery
`aura-chat-5f8633e4-060a-45a8-b050-6842947ced9d`, turn
`8786c8d414a5485d8468e30924dee0ff`, completed in 99.56 seconds. The answer
named Gone Girl by Gillian Flynn, identified the first suggestion as The
Tell-Tale Heart, and explained that a short story did not meet the request
for a novel with unreliable narration. Forty admitted exchanges crossed the
turn boundary. The journal records one attempt, one generation, zero repair
or completion retries, zero text mutations, and answer_delivery_proven=true.

The initiating browser showed the complete answer at 16:17:26. Reloading it
restored all three current-run questions and their answers, one bubble per
turn, including the unchanged sweating example and the older-topic correction.
The ordered list still displayed 1, 2, 3. Boot health at 16:18:44 reported
PID 7761, ready=true, source_current=true, no revision issues, and the same
workspace snapshot recorded above. No runtime replacement was needed between
these turns. The restored answer timestamps use their exchange timestamps;
they are not measurements of completion latency.

This completes the remaining R09 replay. The final row-by-row disposition is
in [the acceptance matrix](R09_ACCEPTANCE_MATRIX_2026-09-10.md#r09-closeout-september-12).
It does not close R06, R08, R11, a long-duration soak, or any RLC reasoning-gain
claim. The current ordinary answers used one recurrent loop and zero surface
steering alpha; their success must not be attributed to untested tissue.

Final cancellation verification after integration passed 37 selected checks
in 16.41 seconds, including the tool fallback boundary, latent caller Stop,
and MLX cooperative cancellation. Writing and governance-lint passed their
existing baselines. Doc-drift reported one broken-reference regression each
in README.md and docs/README.md; that aggregate gate remains red. No production
code changed during these final checks.
