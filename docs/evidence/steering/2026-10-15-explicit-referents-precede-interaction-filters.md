# Explicit referents precede interaction filters, 15 October 2026

The explicit spatial-goal resolver searched `choosing.others()`. That list is
for ordinary contact decisions: it removes still objects sharing the body's
visual kind, known writing and attached objects. Those heuristics could remove
a requested destination before name resolution. They could also remove a
competing named object and create a false unique match.

Explicit destination lookup now receives all published observed instances.
Its existing checks still exclude the controlled object, require a current
finite geometry, preserve competing named candidates and refuse ambiguity.
The interaction filters and subsequent hazard-stance check are unchanged.
Missing observations supply no candidate or fallback destination.

The normal chooser test now resolves a named still destination sharing the
body's kind even though ordinary contact filtering removes it. A second test
keeps a named rival in ambiguity when the interaction list removes that rival
as writing. The focused resolver and recording checks passed 84 tests in
11.50 seconds. Smoke passed 165, one skipped, in 54.54 seconds. Lint and compile
passed. The registered destination canary also checks the unfiltered observed
instance handoff.

This does not fix instance naming or supply a source-contact contract. Source
reference acquisition, continuous body registration and ordinary gameplay
proof remain open. No original-game win or native-app qualification follows.
G09, original game 9, holdouts and resident runtime ownership remain untouched.
