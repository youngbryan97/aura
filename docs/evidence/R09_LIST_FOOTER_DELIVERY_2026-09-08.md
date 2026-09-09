# R09: List footer delivery repair

Status: classifier repaired; end-to-end replay pending. R09 and R05 remain open.

## Live observation

The desktop turn `aura-chat-1d2902b4-8f7f-490f-8939-c42a8421664b`
requested five numbered points about durable chat behavior followed by a
standalone completion marker. The durable receipt is in
`~/.aura/data/chat_delivery.sqlite3`.

RLC produced 846 characters, five numbered points, and 112 words. Its output
quality check passed and generation ended at EOS. Identity alignment removed
one character. The downstream reliability check then classified the closing
line as `truncated_tail`, initiating repeated model repairs. The final delivery
was a fallback, with status
`desktop_required_final_quality_failed:served_repairable_draft`.

The source revision also drifted during this turn. That is a separate live
qualification failure, not evidence that the truncation classification was
correct. No semantic correctness claim is made about the generated points.

## Cause and repair

`_has_truncated_tail` applied the requirements for a final list item to any
last line of an answer containing a numbered list. A standalone footer therefore
failed because it had no list marker. The repair scopes list-item requirements
to list items and evaluates trailing prose separately. Existing dangling-word,
unclosed-quotation, and bare-heading checks remain active. A long unpunctuated
paragraph after a list still fails when generation exhausted its budget.

## Verification

- Focused completion and salvage suites: 64 passed.
- Smoke: 164 passed, one skipped.
- No prompts, token limits, or model weights changed.
- Live replay, cancellation, follow-up semantics, and context retention still
  require acceptance evidence before R09 can be checked off.
