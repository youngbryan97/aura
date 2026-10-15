# Visual declarations keep their program, 15 October 2026

The supplied-reference experiment at `52c03c1d2` matched source artwork
against captured pixels. Its external export had no repeatable link back to
the source reader's dictionary identities. `read_flash` kept names but
discarded their character IDs and declaration addresses.

`visual_assets_in_a_flash_file.py` now reads bounded root dictionary and
export declarations through the existing strict container decoder. A visual
declaration retains the original file SHA-256, dictionary character ID,
definition tag/address and every consistent export address. Addresses are
relative to the uncompressed root tag stream, as in the action-source
reader. Different compressed files have separate source namespaces.

Missing or duplicate definitions, conflicting names, external imports,
malformed export strings and incomplete containers remain explicit issues.
The `complete` property concerns the declaration mapping only. It does not
validate nested artwork or assert that it can be rendered. A name lookup
requires a complete mapping and a unique matching declaration. Document
classes, sounds and fonts do not become visual references.

`ProgramRead`, its merge operation and the current `Guide` retain these
tables. Incomplete visual metadata cannot discard existing controls.
Guide memory does not serialize the tables; they must be read again on a
later visit. Nothing assigns observed names, task roles or control identity
from these declarations.

Only original case 18 was read for this check. Its existing bundle file
has source SHA-256
`a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
The normal program reader retained 33 visual declarations with no mapping
issues and still read its three arrow controls. The raw metadata receipt is
`artifacts/flash-campaign/2026-10-15-source-visual-declarations.json`.

The first focused run passed 112 tests and failed one test assertion that
treated the `as_seen` method as a collection. That assertion was corrected
to inspect `by_kind`. The repeat passed all 113 tests, including the existing
action-source and guide suites. Smoke passed 165 with one skipped. Ruff
passed on the changed modules. No new gameplay run or native-app proof was
performed for this metadata change. No landing or presentation qualification
is claimed.

The first continuation heartbeat fired while the chat was already working
and was retired through the app tool. The later one remains scheduled.
G09 induction PID 4559 remained active; no model, runtime or holdout was
touched.

## Repeatable static export

`80853022e` was committed locally while another agent advanced main.
Those three upstream commits were merged without conflicts in `0a831a553`,
then pushed. Their runtime and model jobs were left alone.

`tools/export_program_visual_reference.py` now accepts an exact exported
name, resolves its dictionary ID from the strict source metadata, stages
immutable source bytes and invokes an explicitly supplied external FFDec/JRE.
The asset name is data; it never enters the command or an output path.
The child uses one CPU, a 512-MB heap and headless Java2D with OpenGL disabled.
The existing proof-step supervisor owns its finite deadline and process group.
Local declarations with unresolved imports or conflicts refuse export.

The receipt retains the source digest, declaration addresses, renderer/JRE
digests, staged command and each output PNG's digest and dimensions. Image
inventory refuses symlinks, another dictionary character, unexpected files
and exhausted byte/image bounds. External renderer output remains external
tool evidence; no renderer code was copied or represented as clean-room work.

Local CLI help revealed the earlier frame-selection error: bare `-select 1`
addresses the main timeline. The new command derives both `-selectid ID` and
`-select ID:1` from the selected declaration. Case 18's exported `pad`
therefore produced exactly one frame with ID 57, in 0.94 seconds including
the supervisor. Its PNG digest is
`b2357b7794887c7e23a9cb6825b5812f942e48804ce68bd1e89e391cbf9eabc4`.

The supplied-reference probe matched that new PNG against the recorded
gameplay frame `frame-0000304.png`, with one appearance, 27 points and
0.417 seconds of correspondence work. The receipt and match are under
`artifacts/flash-campaign/2026-10-15-custodied-reference/`. This links static
source identity to current pictured material in an offline experiment.
It does not establish a task role, temporal identity, source registration
point or native-app play. The program reader still reports no `hers` or
ending structure for this procedural source; classes cannot be assumed.

The export and declaration tests passed 35 checks. These tests exercise
actual PNG decoding and byte hashes, wrong-character and corrupt output,
path custody and bounded inventory. The real renderer was also exercised
separately above. A landing and demo readiness remain unproved.
