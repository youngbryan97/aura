# Resolved inertial axes and the made goal, 15 October 2026

The `2026-10-15-case18-made-choice` CPU-only browser run at `543ef27db`
completed in 86.8 seconds. It reported zero landings and one crash. Its
nine recorded normal planning choices aimed first at a stationary fragment
near the left edge, then at material from the collision. The first aim was
object 10, kind 6, centre `(20.5, 140)`, extent `(17, 8)` in working pixels.
The instruction had been read as bringing the controlled body safely onto
the landing platform. Target selection did not ground that requested role.
These are actual choices retained by the normal actor before input delivery.
They establish neither a win nor presentation readiness.

The same record exposed a separate axis defect. An established inertial
body's pulse-minus-neutral acceleration was converted to velocity change
over 0.5 seconds, then compared with the 20-pixel-per-second discovery
threshold. A resolved acceleration of 34 pixels per second squared therefore
lost its horizontal axis. The chooser could receive its predicted motion
while treating horizontal motion as absent.

Established acceleration now uses the displacement resolution of the
tracker: `a*T*T/2 > 4` working pixels over the existing `T=0.5` second
horizon. Both pulse and neutral measurements are required. Direct pace and
body-discovery thresholds remain unchanged. This does not identify a body,
certify a target, or add source-derived game controls.

Validation: 161 focused tests passed, including both axes and both signs,
subresolution effects, missing neutral evidence, and registered claims.
Smoke passed with 165 tests and one skip. Ruff and diff whitespace checks
passed. The initial smoke command used the system interpreter and could not
import pytest; the rerun explicitly used the shared project virtualenv.

Next work is grounding the requested relation and target in current
observations before scoring contact. No post-change original play has yet
been measured. G09, original case 9 and the recorded holdouts remain outside
this work. Native Aura prompt proof and the 56-case campaign remain open.
