# What she is like, as something she lives in

On 26 September she was asked to take a personality test and said she had no
stable self-model. She has one: a saved vector of ten values that scores every
choice she makes of what to do next, and a record of the last five hundred of
those choices. `core/agency/what_she_is_like.py` (2fa866bc6) read that record
back as a portrait. Bryan asked for a deeper, more central version of it. This
is what it became: a loop in which what she holds, what she does, what came of
it and who she is feed each other, with the parts that must not be bought
protected.

## The loop

1. **She chooses.** Each option is scored by her values
   (`SubjectiveChoiceEngine.score_features`). Each value an option serves is a
   reason to take it, as strong as how fully the option serves it times how
   strongly she holds it against the value she holds most, and the score is the
   chance at least one reason holds (noisy-OR, Pearl 1988). Before 2bc754d35
   an option's value weight cancelled whenever it served one value, so an option
   only about truth (held 0.94) and one only about play (held 0.50) both scored
   1.0, and adding a minor second value lowered an option's score.
2. **The turn is worth something.** The payoff layer reads what the turn paid
   against what she expected (`docs/PAYOFF.md`). One of its nine channels is
   integrity: how far what she chose served her values over the average option
   on offer, so living by her values is worth something to her however the
   turn otherwise went.
3. **Her values learn from that worth.** The habit loop closes each act's
   receipt with the dose of the turn it was taken in, for the choice engine and
   the decision learner alike (8b3048919). They had learned from a raw valence
   change, a second teacher beside the one her connections learn from.
4. **Her conscience does not.** Truth and care are never taught by outcomes
   (`CONSCIENCE`). A run of turns where honesty cost her does not make her less
   honest.
5. **What she holds and does not live presses on her.** Among the values she
   holds at or above her median, what holding a value usually means in what she
   does is read off her record: the median of how much more often than chance
   she took each. A value held that high and lived less falls short by the
   difference, and the shortfall lifts the options serving it on her next real
   choices (dissonance, Festinger 1957). On her record of 26 September,
   connection, held at 0.85 and chosen at chance while her other high values
   were chosen far above it, is the case.
6. **The core holds all of it.** S reads how strongly she holds each value, how
   much more often than chance she lives it, how narrow her choosing has
   become, how often her values overrode her strongest drive, and how well her
   values foretell what she chooses (b98f4085a and after).

## Lived and asked

A choice she makes in her own life and an answer to a question set to measure
her (a tournament, a choice game) are different things. Both were in the
record. Receipts now say which, and the portrait, the dissonance and everything
else that reads what she is like read lived choices only.

## Her self-knowledge, tested

Every choice records the option her values alone ranked first. Of the choices
with more than one option, the share where she took it, less the share chance
would give, is how well what she holds foretells what she does. She says it
when asked what she is like, and S reads it. This is the bridge's reports
ground asked of her own record: what she says about herself against what she
does.

## Left out, and why

**Self-perception** (Bem 1972): held values drifting toward what she does. With
choices moving the values that win them, a few values would win more choices,
grow, and win more, until they polarized. It needs its own analysis, a bound
that comes from her rather than a number chosen for it, and a test that it
cannot run away, before it is built.
