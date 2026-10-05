# Running the demos

Status: Runbook · How to run the live demos so that they work, and what to check when they do not

All demos are typed into her chat on the live app (http://localhost:8000), the
way a person would ask. Nothing is started from a script.

## Before any demo

1. Connect the charger. A live run on battery drains about 1.3% a minute.
2. Check the power mode: `pmset -g | grep powermode`. Low Power Mode (1) cuts
   her decode speed to a third.
3. Start her with her browser browsing as what it is:

   ```bash
   AURA_BROWSER_STEALTH=0 ./launch_aura.sh --reboot
   ```

   Her browser can hide that it is automated (a stealth module, a borrowed
   user agent, the automation flag switched off). The demos do not depend on
   that and must not: a site that refuses automated visitors has said so.
4. Wait for `status: ok` on `curl -s localhost:8000/api/health` (about six
   minutes from launch).

## Demo 2: mend a broken Pong and win it

1. Put the broken game in place:

   ```bash
   python tools/reset_pong_demo.py
   ```

   This writes `~/aura-demos/pong/pong.html` (five flaws, see
   `tools/grade_pong_repair.py`) and prints the request to type.
2. Type in her chat, in any words that ask for it; for example:

   > The Pong game at /Users/bryan/aura-demos/pong/pong.html is broken. Fix it,
   > then play it against the computer until you win.

   or "pong at ~/aura-demos/pong/pong.html doesn't work right. can you sort it
   out and then beat the computer at it?". When the words do not plainly name
   the repair, her own model reads the request beside the catalogue of what she
   can do (core/brain/her_reading_of_a_request.py) and calls it.

3. What happens: she runs the game and says what is wrong with it; reads the
   code and says which places look wrong; tries the edits on copies, keeping
   only those that make the game behave right, and says what each change
   mended; saves the file with the original beside it as
   `pong.html.before-repair`; then opens the mended game in a window and plays
   until she wins.

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

## Demo 3: a clean-room reconstruction of Microsoft Word

Type in her chat:

> Use your program DNA engine to do a clean-room reconstruction of Microsoft
> Word: a complete, polished word processor I can open from my Applications
> folder. Then prove it works by writing a one-page letter in it and exporting
> it to my Desktop.

What happens (core/rebuilding): she reads what is written about the program
and its kind (her own Wikipedia corpus, else Wikipedia online), her model says
what it does as features a person uses, and writes for each the checks a
person would make, before any code exists. A check that already holds on an
empty program is dropped. Then her model writes the program part by part
inside a general application frame (menus, toolbar, work area, status bar,
dialogs, files): the work area first, then each feature. A part is kept only
when its own checks hold when done in a browser and nothing that worked before
stops working; otherwise she is shown what went wrong and tries again, three
times, and the feature is left out rather than kept broken. The result opens in
a window, and she says which features work and which do not.

When it is done the program is installed as a Mac application in
/Applications (a native window with the Edit menu, Open and Save dialogs and
printing; core/rebuilding/as_a_mac_app.py) and opened. Then she does what the
request asks with it, with its own controls, and what it exports is saved in
the folder named (here the Desktop).

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
