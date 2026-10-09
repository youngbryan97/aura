# The 56 games and what they ask

Status: Design map · What each Cartoon Network game in the Web Design Museum's collection asks a player to do, said as general ways of playing, and where in Aura each way is done

The games' own files were read offline, once, for this map: the words each
game says to its player and the signs of what its code does (key handlers,
drags, hit tests, timers). They are the design map and an answer key for
checking her perception. She never reads a game's code while she plays; she
plays from what she sees and what the game tells her on screen.

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
| click things | click things as they show or cross | `playing_as_it_happens._click_things` |
| send | press, pull or hold, let go; set how hard and which way | `core/agency/playing_by_shots.py` |
| time a press | press when something moving is at the right place, learned from what each press paid by where it was | `core/agency/when_a_press_pays.py` |
| jump | clear what comes at her, or a gap, with a press whose effect plays out over the next moment: learned as a curve by watching, made when its whole path is clear | `core/agency/what_a_press_does.py` |
| a view going by | a world that scrolls past | `core/perception/how_the_scenery_goes_by.py` |
| keys shown | press the keys a screen draws mid-play while it shows them, in turn and fast where several are lit by turns | `core/agency/pressing_what_is_shown.py` |
| copy a sequence | do again, in order, what was shown | `core/agency/doing_again_what_was_shown.py` |
| remember what was shown | turn things over and match what was seen where | not yet |
| a board in turns | a board, moves in turns, someone on the other side | `core/skills/screen_pursuit.py` with `core/agency/looking_ahead.py` |
| a grid's rule | a grid whose rule is found by moving in it | `core/skills/screen_pursuit.py` (her rule of the world) |
| type | type what a screen asks for: an answer, a name, a question | `core/agency/typing_what_is_asked.py` |
| make | make something and say what | `core/skills/sovereign_browser_drawing.py` (MADE) |
| use things | take a thing (it goes elsewhere or is chosen where it is), then use it on another: two clicks | `core/agency/taking_and_using.py` |
| a chain | place pieces so each leads to the next, toward an end | not yet |
| serve | give each what it asks for | not yet |
| unseen | keep out of what can see her | not yet (keeping clear of things, not of what they can see) |
| stack | drop or place pieces to build up | not yet |
| switch | change which of several she controls | not yet (keys named on screen are tried, not known as a switch) |
| counters | read score, lives and time | `core/agency/what_meeting_things_does.py` |
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
| 11 | Ed, Edd n Eddy's Candy Machine Deluxe | a grid's rule (drop candy left or right), time a press |
| 12 | Samurai Jack: Code of the Samurai | steer, jump, shoot, counters |
| 13 | Scooby-Doo and the Creepy Castle | use things (pick up objects, use them on ghosts), click things (doors), counters |
| 14 | Scooby-Doo: Scooby Trap | steer, jump, counters |
| 15 | Billy & Mandy: Zap to It! | copy a sequence (the arrows in the book), counters |
| 16 | Toonami: Tunnel Rush | steer, shoot, counters |
| 17 | KND: Numbuh Generator | make |
| 18 | KND: Operation Tommy | send, steer |
| 19 | Foster's: A Friend in Need | send (aim and toss), counters |
| 20 | Foster's: Coco's Egg Scramble | steer (mouse), shoot (click to throw), counters |
| 21 | Foster's: Door to Door | click things (the doors), counters |
| 22 | Foster's: Mid-Flight Snack | steer (mouse), jump (click), counters |
| 23 | Foster's: Simply Smashing | steer (catch), counters |
| 24 | Foster's: Wilt's Wash N' Swoosh | send (click and hold to aim, let go), steer |
| 25 | Camp Lazlo: Paintcan Panic | steer (cover the ground), unseen |
| 26 | KND: Tummy Trouble | shoot, steer, counters |
| 27 | The Batman: The Cobblepot Caper | steer, jump, shoot (named keys), counters |
| 28 | Tom's Trap-O-Matic | a chain (devices to the cage) |
| 29 | Ben 10: Hero Matrix | make |
| 30 | Code Lyoko: Monster Swarm | steer, jump, shoot |
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

Games 51 and 53 have no file the archive serves; what they ask is from their
names and the museum's categories, and is the least sure of the map.
