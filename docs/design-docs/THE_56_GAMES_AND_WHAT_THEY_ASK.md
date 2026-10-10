# The 56 games and what they ask

Status: Design map · What each Cartoon Network game in the Web Design Museum's collection asks a player to do, said as general ways of playing, and where in Aura each way is done

The games' own files were read offline, once, for this map: the words each
game says to its player and the signs of what its code does (key handlers,
drags, hit tests, timers). What they ask is said, mechanic by mechanic, for
any place in `core/agency/mechanics_she_knows.py`, with what each means and
how it is played: games, sites, forms, documents, programs.

Wherever she is, she keeps a guide to the place
(`core/cognition/a_guide_to_a_place.py`): from what it shows and tells her,
its page, the program it runs where that can be had
(`core/perception/reading_a_program.py`, informational only: keys, pointer,
controls and instructions, never what play would show her later), and what
she looked up. It is kept up to date as the place changes, and it tells her
how; she decides what.

A way of playing is named for what it asks of anyone, not for a game: a body
steered to meet some things and keep clear of others is the same way in a
shooter, a catching game and a car park. `core/agency/ways_of_playing.py`
lists every way she has, the function in her live play that does it, and the
test that shows it working; `tests/test_every_way_of_playing_is_live.py` fails
if any of them is not called from her live play or has no test.

## The ways

| Way | What it asks | Where she does it |
|---|---|---|
| steer | move a body to meet some things and keep clear of others, by keys or the pointer | `core/agency/playing_as_it_happens.py` |
| shoot | fire or throw at things, aimed | `playing_as_it_happens._trigger` |
| strike | act on what is close by a key that sends nothing out (a punch, a swing): stand her ground against what comes on slowly enough, and press when it is within the reach that has paid | `core/agency/how_far_her_blow_reaches.py` |
| charge | hold a key the words say to hold and let go, while playing on with the others, and let it go after the length of hold that has paid | `core/agency/holding_to_charge.py` |
| click things | click things as they show or cross | `playing_as_it_happens._click_things` |
| send | press, pull or hold, let go; set how hard and which way | `core/agency/playing_by_shots.py` |
| time a press | press when something moving is at the right place, learned from what each press paid by where it was | `core/agency/when_a_press_pays.py` |
| jump | clear what comes at her, or a gap, with a press whose effect plays out over the next moment: learned as a curve by watching, made when its whole path is clear | `core/agency/what_a_press_does.py` |
| a view going by | a world that scrolls past | `core/perception/how_the_scenery_goes_by.py` |
| keys shown | press the keys a screen draws mid-play while it shows them, in turn and fast where several are lit by turns | `core/agency/pressing_what_is_shown.py` |
| copy a sequence | do again, in order, what was shown | `core/agency/doing_again_what_was_shown.py` |
| remember what was shown | remember what each place showed when turned over, and turn together two that showed alike | `core/agency/things_that_go_together.py` |
| a board in turns | a board, moves in turns, someone on the other side | `core/skills/screen_pursuit.py` with `core/agency/looking_ahead.py` |
| a grid's rule | a grid whose rule is found by moving in it | `core/skills/screen_pursuit.py` (her rule of the world) |
| type | type what a screen asks for: an answer, a name, a question | `core/agency/typing_what_is_asked.py` |
| make | make something and say what | `core/skills/sovereign_browser_drawing.py` (MADE) |
| use things | take a thing (it goes elsewhere or is chosen where it is), then use it on another: two clicks | `core/agency/taking_and_using.py` |
| carry | press on a thing and carry it, the button held, to a place, and let go there; a place marked as the one ("here", the one that stands out) first | `core/agency/putting_things_in_place.py` |
| a chain | place pieces so each leads to the next, toward an end: each part put on from where the last one that worked went | `core/agency/putting_things_in_place.py` (in_chain_order) |
| serve | give each what it asks for: a thing taken given first to what looks most like it | `core/agency/putting_things_in_place.py` (served_first) |
| unseen | keep out of what reaches her without touching her: the distance and side at which each kind has cost her | `core/agency/how_far_a_thing_reaches.py` |
| stack | drop or place pieces to build up: square over the top of what is built | `core/agency/building_up.py` |
| switch | change which of several she controls: a key after which her keys move another of hers, the first kept hers | `core/agency/which_one_answers_to_her.py` |
| counters | read score, lives and time | `core/agency/what_meeting_things_does.py` |
| play as told | take up the way the place says it is played and play it, stretch after stretch, while it is learned or pays | `core/agency/the_way_it_is_played.py` |
| a legend | know a thing drawn beside words about it (get it, keep clear of it, shoot it) when it turns up in play | `core/perception/what_a_legend_shows.py` |
| bars | read the bars that fill and empty (health, energy, paint, a boss, time); a fall of what she has left is a loss, and low, she keeps wider of what costs her | `core/perception/how_full_a_bar_is.py` |
| carried | a body carried on by its own going and pushed by her keys (a lander, a ship): known by what each key adds to its going, steered by where that takes it a moment ahead, brought onto a thing slowly | `core/agency/which_one_answers_to_her.py` (pushes) |
| pause | a screen that says it is paused goes on by what paused it, which is not taken again in play | `core/skills/screen_pursuit_decision.py` |
| a guide | know how the place she is in is worked, from what it shows and says, its page, its program and what she looked up; kept up to date as it changes | `core/cognition/a_guide_to_a_place.py` |
| checked against the guide | each act she weighs scaled by what the guide says of it, learned from what acts did | `core/cognition/checking_the_debate.py` |
| read the program | read the program a page runs, where it can be had, for how it is worked; nothing that would spoil it | `core/perception/reading_a_program.py` |
| take stock | read the rules; ask what she knows when stuck | `core/cognition/taking_stock.py` |

## The games

Numbered as the museum's list numbers them, from 0. What each asks is from its
own words where it has any, else from what its code does.

| # | Game | Ways |
|---|---|---|
| 0 | Scooby-Doo: Scooby Snapshot | click things (snap ghosts), make (album) |
| 1 | Batman: The Riddler's Secret Identity Inventor | type (answer the riddle), make |
| 2 | Cartoon Network: Food Bash | shoot (aim with the mouse, click to throw), steer (dodge), counters |
| 3 | Dexter's Laboratory: Runaway Robot | use things (glasses, battery), steer, counters |
| 4 | Ed, Edd n Eddy: Spin Stadium | send (aim, click to release the top), counters |
| 5 | Scooby-Doo: Ask Swami Shaggy | type (a yes or no question, then Ask) |
| 6 | Cartoon Network: Cartoon Cove Mini Golf | send (putt), counters |
| 7 | Cartoon Network: Operation S.T.A.T. | use things (find items, drag to assemble), counters |
| 8 | Courage the Cowardly Dog: Nightmare Vacation | steer, a grid's rule |
| 9 | Cow and Chicken: Ballet Parking | steer (park each car), counters |
| 10 | Dexter's Laboratory: Clone-A Doodle Doo | send (place a post, drag, let go), counters |
| 11 | Ed, Edd n Eddy's Candy Machine Deluxe | carry (add a part, drag it to where it attaches), a chain (tubes from where the jawbreaker drops to the bucket), counters (tries) |
| 12 | Samurai Jack: Code of the Samurai | steer, jump, shoot, counters |
| 13 | Scooby-Doo and the Creepy Castle | use things (pick up objects, use them on ghosts), click things (doors), counters |
| 14 | Scooby-Doo: Scooby Trap | steer, jump, counters |
| 15 | Billy & Mandy: Zap to It! | copy a sequence (the arrows in the book), counters |
| 16 | Toonami: Tunnel Rush | steer, shoot, counters |
| 17 | KND: Numbuh Generator | make |
| 18 | KND: Operation Tommy | carried (a lander: thrust against a pull, onto a pad slowly), steer |
| 19 | Foster's: A Friend in Need | send (aim and toss), counters |
| 20 | Foster's: Coco's Egg Scramble | steer (mouse), shoot (click to throw), counters |
| 21 | Foster's: Door to Door | click things (the doors), counters |
| 22 | Foster's: Mid-Flight Snack | steer (mouse), jump (click), counters |
| 23 | Foster's: Simply Smashing | steer (catch), counters |
| 24 | Foster's: Wilt's Wash N' Swoosh | send (click and hold to aim, let go), steer |
| 25 | Camp Lazlo: Paintcan Panic | steer (cover the ground), unseen |
| 26 | KND: Tummy Trouble | shoot, steer, counters |
| 27 | The Batman: The Cobblepot Caper | steer, jump, strike (S to punch, D to kick), shoot (A for the batarang), counters |
| 28 | Tom's Trap-O-Matic | a chain (devices to the cage) |
| 29 | Ben 10: Hero Matrix | make |
| 30 | Code Lyoko: Monster Swarm | steer, jump, strike (Z at close enemies), charge (hold X, let go), shoot |
| 31 | Foster's: Big Shot Checkers | a board in turns |
| 32 | Ben 10: Blockade Blitz | steer (paddle by mouse), shoot (click), counters |
| 33 | Ben 10: Krakken Attack | shoot (aim for the chest), counters |
| 34 | Cartoon Network: Ready, Im, Fire! | steer (dodge), shoot (throw) |
| 35 | Operation Z.E.R.O. Out-Mandy'd | steer, shoot, unseen, keys shown (caught: ← and → lit by turns) |
| 36 | KND: Flight of the Hamsters | send, time a press |
| 37 | KND: Rainbow Monkey Rundown | steer (mouse), shoot (click), counters |
| 38 | Foster's: Team Work | steer (catch), counters |
| 39 | Powerpuff Girls: Attack of the Puppybots | steer, shoot (named keys), a view going by, switch, counters |
| 40 | Pokemon: Towering Legends | stack (same colour), steer (mouse) |
| 41 | Ben 10: Cavern Run | steer, jump, counters |
| 42 | Total Drama: H-Bomb's Killah Beatz | make |
| 43 | Adventure Time: Jumping Finn | jump, steer |
| 44 | Regular Show: All-Nighter | time a press, counters |
| 45 | Gumball: Blind Fooled | steer (WASD), use things (pick up and drop boxes), stack (make steps) |
| 46 | Annoying Orange: Escape from Dr. Fruitenstein | click things (balloons, rope), a chain |
| 47 | Cartoon Network: SnowBrawl Fight! | steer, jump, shoot, counters |
| 48 | Teen Titans Go! Grab that Grub | steer, counters |
| 49 | Regular Show: Killer Z's | steer, jump, shoot, a view going by |
| 50 | Regular Show: Ride 'Em Rigby | jump (click, double click), steer, counters |
| 51 | Regular Show: Dance of Doom | copy a sequence, steer |
| 52 | Gumball: Water Sons | shoot, counters |
| 53 | Regular Show: High Flying Halloween | steer, jump |
| 54 | Sonic Boom: Link 'N Smash | a grid's rule, time a press |
| 55 | Gumball: Battle Bowlers | send (bowl), steer, switch (number keys) |

Games 51 and 53 had no file the archive served when this map was made; what
they ask is from their names and the museum's categories, and is the least sure
of the map. On 9 October 2026 the museum's pages for both linked a game file,
and a browser fetched it.
