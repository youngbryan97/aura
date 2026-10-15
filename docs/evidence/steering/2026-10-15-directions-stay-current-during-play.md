# Directions stay current during play, 15 October 2026

The explicit spatial request added earlier was installed only when a play
stretch began. Later screen instructions could update the general rules
without updating the destination. A direction heard earlier could also be
discarded as globally familiar when it became current again.

The normal running actor now passes its existing meeting state into the
screen-rule update. New explicit motion directions reach that state in
capture order before wording deduplication. A counter supplies no goal
revision. Finite valid captures advance the reading watermark; a late older
reading, an invalid timestamp or a familiar line cannot replace a newer
direction. Conflicting explicit directions abstain from a unique override.

The parser also refuses infinitive purposes such as going "to find cargo"
as named destinations, and refuses pronouns without a named referent. This
does not add object-placement planning or a general language model.

Validation: 82 targeted tests passed, including actual rule updates on the
normal run object, changed directions, wording that becomes current again,
reordered delivery, invalid clocks and existing reading/reference/recording
tests. Smoke passed 165 with one skip. Ruff, layering and generated DEPS
checks passed. The realtime actor remains at 1998 lines.

This is a state-propagation repair. No new original gameplay measurement or
native prompt proof is claimed. Original case 9, G09 and the holdouts remain
outside this work. Presentation readiness is still unproved.
