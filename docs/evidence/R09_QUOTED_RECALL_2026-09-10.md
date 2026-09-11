# R09 Quoted Recall

## Reproduction

The failed desktop turn at 00:57:47 PDT is recorded in
[continuation history](R09_CONTINUATION_HISTORY_2026-09-10.md). Later inspection
of the parent log recovered the rejected candidate's 400-character head and
200-character tail (601 characters total). The second rejection was caused
by this span:

> Your reason for correcting me: you said it right there in that message — "That is a short story.

The original quotation continued with "I asked for a novel." Both quoted
sentences appear in the user's durable correction. An offline reproduction
using the preceding twelve durable turns flags this span. It also flags the
minimal exact-quote example without any speculative tail. The failure is
therefore reproducible without assuming the missing character in the log or
judging the unseen portion of the first draft.

The sentence splitter cut inside quoted speech. Attribution selection then
measured the words introducing the quote instead of the quoted correction.
The same quote without its reporting frame passed.

## Repair

The existing language typography module now supplies balanced quotation
offsets. The shared-history checker preserves multi-sentence quotations and
resolves a direct-speech complement across grammatical reporting frames.
Quoted terms inside substantive claims remain part of those claims.
Contractions, escaped quotes and nested quotations have separate tests.

The focused suite passed 119 tests. Negative controls still reject invented
quotations, fabricated second sentences, a separate unsupported claim after
a grounded quotation, and a substantive false claim preceding a true quote.
The prior invented-directory regression also remains enforced.

## Remaining Evidence

Live replay on the repaired revision is pending. R09 remains open. The
fallback model's missing transcript and repeated durable-history read
timeouts are separate observed defects, not evidence that this repair passed.
