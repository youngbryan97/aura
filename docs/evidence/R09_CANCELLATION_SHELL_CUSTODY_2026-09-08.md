# R09 cancellation, shell identity, and answer custody

## Live finding

The 19:49:38 replay on PID 79188, commit `2d6223122`, completed with zero
repair retries. The preceding run had retried four times. Delivery key:
`aura-chat-95b8ed4c-22e7-4a66-bc4c-39783189ef19`.

The recurrent answer contained 685 characters. A later wrapper delivered 666.
The removed 19 characters were the closing marker and its separators. The log
reported `Trimmed a reply that stopped mid-clause`. The structural quality
judge accepted the footer; the independent trimmer removed it. The final
contract also retained the original answer's quality hash after changing its
bytes. This run does not close complete delivery.

The actual neural feed showed repeated shell connections during backend
workspace changes. A browser reload was tied to the whole workspace hash,
although its signed shell assets had not changed. Warnings also included
event-loop lag and a tick-duration SLO fault; those remain R06/R08/R11 work.

## Repairs

- Use the same truncation predicate for admission and terminal trimming.
- Invalidate byte-bound proofs when the terminal wrapper replaces an answer;
  record both hashes and the reason using the existing custody helper.
- Add explicit cancellation bound to the authenticated journal identity and
  its current turn/generation owner. Disconnect remains a transport event.
  Repeated cancellation cannot interrupt cleanup or execute the turn again.
- Add the desktop stop control. Its acknowledgement does not render a second
  final answer or advance the queue; the delivery observer retains ownership.
- Address immutable shell bytes by source-root identity and asset digest.
  Backend source drift remains reported separately and cannot reload the UI
  when the served bytes are unchanged.

## Verification

138 focused tests passed, including cancellation, journal replay, answer-byte
custody, footer preservation, immutable snapshots, and health revision capture.
Smoke: 164 passed, one skipped. These are local tests, not live closure.

R09 remains open for the updated live cancellation, reconnect, final-answer,
and follow-up replay. R05 remains separately open.
