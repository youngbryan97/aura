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
