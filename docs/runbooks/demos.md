# Running the demos

Status: Runbook · How to run the live demos so that they work, and what to check when they do not

All demos are typed into her chat on the live app (http://localhost:8000), the
way a person would ask. Nothing is started from a script.

## Before any demo

1. Connect the charger. A live run on battery drains about 1.3% a minute.
2. Check the power mode: `pmset -g | grep powermode`. Low Power Mode (1) cuts
   her decode speed to a third.
3. If Aura is already running, use that instance. For a first launch, start
   her with her browser browsing as what it is:

   ```bash
   AURA_BROWSER_STEALTH=0 ./launch_aura.sh
   ```

   Her browser can hide that it is automated (a stealth module, a borrowed
   user agent, the automation flag switched off). The demos do not depend on
   that and must not: a site that refuses automated visitors has said so.
   Replacing a running instance requires Bryan's explicit restart
   authorization under `AGENTS.md`; the supported replacement command adds
   `--reboot`. Check that no training or soak owns the model, preserve logs,
   and verify one replacement process and its source revision.
4. Wait for `healthy: true` and `conversation_ready: true` on
   `curl -s localhost:8000/api/health`. Boot time depends on the resident model
   and host state; use the readiness result.

## Demo 2: mend a broken game and play three attempts

1. Put the broken game in place:

   ```bash
   python tools/reset_pong_demo.py
   ```

   This writes `~/aura-demos/pong/pong.html` (five flaws, see
   `tools/grade_pong_repair.py`) and prints the request to type.
2. Type in her chat, in any words that ask for it; for example:

   > Fix the broken game at /Users/bryan/aura-demos/pong/pong.html, then play
   > it for three attempts to show the repair works.

   The repair and play count can be requested in ordinary words. When the words do not plainly name
   the repair, her own model reads the request beside the catalogue of what she
   can do (core/brain/her_reading_of_a_request.py) and calls it.

3. What happens: she runs the game and says what is wrong with it; reads the
   code and says which places look wrong; tries the edits on copies, keeping
   only those that make the game behave right, and says what each change
   mended; saves the file with the original beside it as
   `pong.html.before-repair`; then opens the mended game in a window and plays
   three attempts. A loss counts as an attempt. Winning is a separate goal,
   requested with words such as "until you win".

   Timings from the live runs on 4 October 2026: the request reaches the
   repair in under a minute (a request that plainly names one capability's
   job is dispatched to it, not left to the model's choice); the repair takes
   eleven to sixteen minutes; a game against the computer takes about three.
   Offline against the repaired game's computer she won four of the eight full
   games she finished, and she plays on until one is won (up to twenty
   minutes).

   A spare broken copy is kept at `~/aura-demos/pong-spare/pong.html` (all
   five flaws, untouched); `tools/reset_pong_demo.py` also writes a fresh one
   to `~/aura-demos/pong/pong.html` at any time.
   The exact three-attempt request above completed through normal chat on
   5 October 2026 at revision `6ec1ff44a`: all five faults repaired, then a
   5–3 win, a 1–5 loss and a 5–2 win. Independent grading passed all ten
   checks. The complete request took 15 minutes 12.5 seconds; repair took
   6 minutes 20.5 seconds. Codex supplied no repair edits or game inputs.
   The final reply now lists every attempt (`ab9a5b054`). The broken file was
   restored and independently graded after the run, and a permanent copy is
   kept in `/Users/bryan/.aura/control-proof-2026-10-05/`.

   See [the measured replay](../evidence/GENERAL_CONTROL_DEMO_2026-10-05.md)
   for receipts, observation coverage and the limits of the result. That
   result proves this demo in this environment; it does not certify arbitrary
   games, every application or the fastest possible execution.

4. Check the repair independently:

   ```bash
   python tools/grade_pong_repair.py ~/aura-demos/pong/pong.html
   ```

## Demo 1: three Cartoon Network games

Type in her chat:

> Go to https://www.webdesignmuseum.org/flash-game-exhibitions/cartoon-network-flash-games
> and play three of the games, one after another, and win each one. To pick
> them: number the games on the list from 0, starting with the first one. Take
> the current minute of the hour, divide it by how many games there are, and
> play the game whose number is the remainder. When that game is over, go back
> to the list, add 19 to the number, take the remainder again, and play that
> game. Then add 19 once more for the third game.

The museum refuses its game files to an automated browser. Each game page
links the same game in the Internet Archive, and she plays it there.

The picking is done by code, not left to her decisions
(core/language/picking_by_a_rule.py, core/skills/sovereign_browser_picking.py).
She reads the list off the page (all 56 games, in order), works the rule out
from the clock as it is when she starts, and says the working before she
plays: "There are 56 games on the list. By your rule: it is 7:14 am, so the
minute is 14; 14 divided by 56 leaves 14: number 14, "Scooby-Doo: Scooby
Trap". Then ..." Each game picked is then played for the task said of one:
"Play this game and win it." Over an hour, every game is someone's first.

## Demo 3: a clean-room reconstruction of Microsoft Word

Type in her chat:

> Use your program DNA engine to do a clean-room reconstruction of Microsoft
> Word: a complete, polished word processor I can open from my Applications
> folder. Then prove it works by writing a one-page letter in it and exporting
> it to my Desktop.

What happens (core/rebuilding): she reads what is written about the program
and its kind (her own Wikipedia corpus, else Wikipedia online), and her model
says what it does as features a person uses.

Features she already knows how to make are given by code, each with the checks
a person would make of it (core/rebuilding/parts_a_maker_knows.py and .js).
For a document program that covers the page (paper, margins, pages counted,
zoom), character styles, fonts, colours, alignment, lists, indent, spacing,
headings, undo, the clipboard, find and replace, tables, pictures, links,
insertions, proofing, printing and print preview, a new document, Save as
(PDF among the kinds), page setup and a tabbed toolbar. They work on the
page's editing (document_editing.js) and its files (document_formats.js,
which writes .docx, .odt, .rtf, .html, .md, .txt and PDF). She says how many
of its features these are.

For every other feature her model writes the checks a person would make,
before any code, and a check that already holds on an empty program is
dropped. Then her model writes the part inside the general application frame
(menus, toolbar, work area, status bar, dialogs, files). A part is kept only
when its own checks hold when done in a browser and nothing that worked before
stops working; otherwise she is shown what went wrong and tries again, three
times, and the feature is left out rather than kept broken. She says which
features work and which do not.

Offline, with her model away, Word's own feature list builds 18 of 18
features in about 90 seconds. A notes app and an email composer get every
feature from the same parts; a spreadsheet, a drawing program and a task
board get none (tests/test_parts_a_maker_knows.py).

When it is done the program is installed as a Mac application in
/Applications (a native window with the Edit menu, Open and Save dialogs and
printing; core/rebuilding/as_a_mac_app.py) and opened. Then she does what the
request asks with it, with its own controls, in a window you can watch: her
model writes the letter, and she types it a paragraph at a time and presses
the program's own export command. What it exports is saved in the folder
named (here the Desktop), read back to check that it says what was written,
and opened so you see it.

An unfinished build of the same program made on a page her model wrote is not
taken up: she says so, leaves it as it was, and starts afresh.

The same engine builds to a specification ("build me a ...", build_app),
changes a build ("add a dark theme to the word processor you built",
change_a_program) and uses one ("use the word processor you built to write
...", use_a_program). Programs that are code rather than windows are checked
by calling them in the sandbox.

No code of the original is read. The program is one file,
`artifacts/rebuilt_programs/<name>/index.html`, beside `what_it_does.json`
(the features), `checks.json` (the checks) and `what_works.json` (each
feature, whether its checks held, and how many tries it took).

## If something goes wrong

| Symptom | Check |
| --- | --- |
| She answers in words instead of acting | The request must name the file or the page; see the routing tests in `tests/test_repairing_a_program_is_asked_for.py`. |
| A game page shows "Ruffle failed to load" | Expected on the museum; she follows the Internet Archive link. |
| She plays a game and never stops | A game that only counts points has no winner; she stops after one finished run. |
