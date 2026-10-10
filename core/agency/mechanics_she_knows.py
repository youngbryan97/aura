"""The mechanics she knows: how the things she uses work, said once for every place that uses them.

A person who has played a few games knows, before the first frame of a new
one, what a health bar is, that a timer running down is a deadline, that a
coin is worth getting and a spike is not, that a thing in the corner flashing
"PAUSE" stops the game and is not to be pressed in play, and that the clouds
going by are scenery. None of it is in the new game; it is what games are. The
same holds for programs and sites: a menu, a field to type in, a button that
goes on, a banner that only says something.

Each mechanic here is that kind of knowledge, general: what it is, what it means
for her (what to want, what to fear, what to watch), how it is played (which of
her ways, core/agency/ways_of_playing.py, does it), and the signs that a place
has it, in what it says and in its program (core/perception/reading_a_program.py).
They were drawn from what the fifty-six games of the design map ask
(docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md) and from what their
programs do, and said for any place: none names a game, a character or a site.

Signs found in more places than one, where play confirmed the mechanic, are
learned and added (``LearnedSigns``), through the gate every change of hers
goes through (core/cognition/how_a_change_is_promoted.py).
"""
from __future__ import annotations

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MECHANICS", "LearnedSigns", "Mechanic", "changes_told", "learned_signs", "mechanics_in", "what_a_label_measures"]

#: Kinds of mechanic: how she is to treat what it names. A boundary is what is not hers to do: it is handed to the
#: person she is working for.
CONTROL, GOAL, HAZARD, RESOURCE, WORLD, STRUCTURE, SCREEN = "control", "goal", "hazard", "resource", "world", "structure", "screen"
BOUNDARY = "boundary"


@dataclass(frozen=True)
class Mechanic:
    """One way things work, said for anywhere it is used."""

    name: str
    kind: str
    #: How it works.
    what: str
    #: What it means for her: what to want, what to fear, what to watch.
    means: str
    #: How it is played.
    play: str
    #: Signs of it in what a place says to its user, and in its program.
    words: str
    program: str = ""
    #: The ways of playing that do it (core/agency/ways_of_playing.py).
    ways: tuple[str, ...] = ()

    def said_in(self, text: str) -> list[str]:
        """The places in ``text`` that speak of it, as the words that do."""
        return [m.group(0) for m in re.finditer(self.words, text, re.I)][:6] if self.words else []

    def coded_in(self, text: str) -> int:
        """How often a program's text shows it."""
        return len(re.findall(self.program, text, re.I)) if self.program else 0


MECHANICS: tuple[Mechanic, ...] = (
    # -- how she acts ------------------------------------------------------------------------------------------
    Mechanic("steering", CONTROL,
             "a body of hers goes where her keys or the pointer send it, at a pace of its own under each",
             "what she steers is her: what touches it is what happens to her; where it can go is the world's shape",
             "find which thing is hers by what answers her controls, then take it to what is worth meeting and away "
             "from what costs",
             r"\b(?:arrow keys?|wasd|to (?:move|walk|run|steer|drive|fly|swim)|move (?:left|right|up|down|around)|"
             r"guide|steer|control (?:him|her|it|your))\b",
             r"key\.isdown\((?:37|38|39|40)\)|keyboard\.(?:left|right|up|down)|_(?:left|right|up|down)key\b", ("steer",)),
    Mechanic("pointer steering", CONTROL,
             "a body goes wherever the pointer is put, or follows it",
             "the pointer is her: where she puts it is where she is",
             "put the pointer where she means to be; small, smooth moves keep it followed",
             r"\b(?:move|use|drag) (?:the|your) (?:mouse|cursor|pointer) to (?:move|steer|control|catch|guide)|"
             r"follows? (?:the|your) (?:mouse|cursor)\b",
             r"(?:_x|_y|\.x|\.y)\s*=\s*[^\n;]{0,60}(?:_xmouse|_ymouse|mousex|mousey)", ("steer",)),
    Mechanic("jumping", CONTROL,
             "a press throws her up; a pull brings her down again along an arc",
             "a jump commits her to its arc: what is in its path is met; a gap is crossed only if the arc clears it",
             "jump when the whole arc is clear, to clear what comes along her way or reach what is above",
             r"\b(?:jump|leap|hop|double[- ]jump|bounce)\w*\b",
             r"\bjump\w*", ("jump",)),
    Mechanic("pull", WORLD,
             "everything not held up falls, faster the longer it falls",
             "what is dropped or thrown curves down; standing on nothing is falling",
             "aim above a far target; keep something under her",
             r"\b(?:gravity|fall(?:s|ing)?|drop(?:s|ping)?|land(?:s|ing)?)\b",
             r"gravity|\bgrav\b", ("jump", "carried")),
    Mechanic("thrust", CONTROL,
             "a key pushes her the way it points while it is held; let go, she goes on as she was going",
             "she cannot stop at once: speed has to be taken off before she arrives, or she arrives hard",
             "push toward where she is going, ease off early, push against her going to slow; come onto things slowly",
             r"\b(?:thrust|thrusters?|engines?|lander|land (?:safely|softly|gently)|momentum|accelerat\w*|"
             r"brake|drift)\b",
             r"thrust|lander|momentum", ("carried",)),
    Mechanic("shooting", CONTROL,
             "a press sends something from her that flies on and hits what it meets",
             "what she fires at is a target: things that cost her can be taken out from a distance",
             "line up with a target, fire, and keep clear while what she fired travels",
             r"\b(?:shoot|fire|blast|zap|throw|toss|launch)\w*\b",
             r"bullet|\bshot\b|\bshoot|\bfire\b|projectile|ammo|missile|laser|\bthrow", ("shoot",)),
    Mechanic("striking", CONTROL,
             "a press acts on what is within reach of her (a punch, a kick, a swing, a stomp): nothing is sent out",
             "what is out of her reach is not touched by it, and what is close can be beaten before it reaches her",
             "let what comes on come within reach, then press; keep out of reach of what she cannot beat",
             r"\b(?:punch|kick|swing|slash|stomp|melee|smash|whack|swat|attack)\w*\b",
             r"punch|melee|\bswing|slash|attack_?range|stomp", ("strike",)),
    Mechanic("aiming", CONTROL,
             "the pointer sets the way a thing is sent; a press sends it",
             "where the pointer is, is where it goes: lead a moving target",
             "put the pointer on or ahead of the target, then press",
             r"\b(?:aim|target|crosshairs?|point (?:at|the mouse))\w*\b",
             r"atan2|_rotation[^\n]{0,60}mouse|rotation[^\n]{0,60}mouse", ("shoot", "send")),
    Mechanic("sending by strength", CONTROL,
             "a press held, pulled or let go sets how hard and which way a thing is sent",
             "each send is an experiment: where it went says how to send the next",
             "send, watch where it went, and change strength and direction by what it did",
             r"\b(?:power meter|power bar|how hard|putt|swing|pull back|let go|release to|charge (?:up|your))\b",
             r"power_?meter|power_?bar|\bputt|\bstroke|shot_?power|launch_?power", ("send",)),
    Mechanic("charging", RESOURCE,
             "holding fills a meter; letting go spends what was filled",
             "a meter that fills while she holds is not something she has left: it is how strong her next act is",
             "hold until the meter is where she wants, then let go; a meter that fills alone is waited for",
             r"\b(?:charg\w+|power(?:s)? up|hold (?:down )?(?:to|until)|meter fills?)\b",
             r"\bcharge|chargebar|charge_?bar|power_?bar", ("send", "charge")),
    Mechanic("holding to sustain", CONTROL,
             "a press held keeps something going (gliding, flying, boosting) while a meter of it lasts",
             "holding spends the meter; letting go lets it refill",
             "hold while it helps, let go before it runs dry, and hold again when it has refilled",
             r"\b(?:glide|gliding|hover\w*|fly(?:ing)? while|boost\w*|hold (?:the )?(?:mouse|button|key) to "
             r"(?:use|glide|fly|keep|boost|float))\b",
             r"glide|jetpack|\bboost\b", ("time a press",)),
    Mechanic("timing a press", CONTROL,
             "a press counts by when it is made: as something moving reaches the right place",
             "the moving thing is the clock; where it is when she presses is what she gets",
             "watch where the moving thing is when a press pays, and press as it comes there again",
             r"\b(?:when (?:the|it|he|she)\b[^.]{0,40}\b(?:lines? up|reaches|is (?:over|in|on)|hits)|in time|"
             r"at the right (?:time|moment)|timing)\b",
             r"perfect_?timing|good_?timing|\bmissed\b", ("time a press",)),
    Mechanic("clicking things", CONTROL,
             "things are met by clicking them, where they are as the click lands",
             "what she clicks is what she acts on: a thing about to leave is the one to click first",
             "click what is worth it as it shows, before it goes",
             r"\b(?:click (?:on )?(?:the|each|all|every)|pop|snap|tap (?:on )?the)\b",
             "", ("click things",)),
    Mechanic("dragging", CONTROL,
             "a thing is pressed on, carried with the button held, and let go where it is to go",
             "a thing carried goes where it is let go: its place is the point",
             "press on the thing, carry it to its place, let go there",
             r"\b(?:drag|drop it|drag and drop|carry|move (?:it|them|the \w+) (?:to|into|onto))\b",
             r"startdrag|stopdrag", ("carry",)),
    Mechanic("using things", CONTROL,
             "a thing taken (picked up, chosen) is used on another by a second act",
             "what she holds changes what her acts do: a key opens, a tool fixes",
             "take what is offered, then use it on what it is for",
             r"\b(?:pick up|use (?:it|the|an?) \w+ (?:on|to)|inventory|items?|combine|equip)\b",
             r"inventory|pick_?up|use_?item", ("use things",)),
    Mechanic("typing", CONTROL,
             "a field takes words she types",
             "what she types is what the place works with: a name, an answer, a question",
             "type what the field asks, in the field it points to, and go on",
             r"\b(?:type (?:in|your|a)|enter (?:your|a)|your name|ask (?:a|your) question|input)\b",
             r"type\s*=\s*[\'\"]input|input_?txt|name_?input", ("type",)),
    Mechanic("switching", CONTROL,
             "a key or button hands her controls to another of hers",
             "each of hers can do something the others cannot; the one left behind stays as it was",
             "switch to the one whose ability the moment wants; switch back when it is done",
             r"\b(?:switch (?:between|to|characters?)|swap|change (?:characters?|heroes?)|number keys)\b",
             r"switch_?char|change_?char|swap_?(?:char|hero|player)|switch_?(?:hero|player)", ("switch",)),
    Mechanic("keys shown", CONTROL,
             "the screen shows keys to press while it shows them, in turn and fast",
             "a shown key is asked for now: late is missed",
             "press what is shown as it shows; where several light by turns, press them in turn, fast",
             r"\b(?:press (?:the )?(?:keys?|arrows?) (?:as|when) (?:they|it) (?:appear|show)|"
             r"rapidly|repeatedly|mash)\b",
             "", ("keys shown",)),
    Mechanic("following a sequence", CONTROL,
             "the place shows a sequence (arrows, lights, notes) and asks for it again in order",
             "the order is the whole of it: one wrong step and the sequence fails",
             "watch the whole sequence, then do it again in its order",
             r"\b(?:repeat (?:the|after)|copy|follow (?:the|along)|simon|in the (?:same )?order|sequence|pattern)\b",
             r"sequence|rhythm|\bbeats?\b|simon", ("copy a sequence",)),
    # -- what she is after ----------------------------------------------------------------------------------------
    Mechanic("score", GOAL,
             "a count that goes up when she does what the place rewards",
             "what raises it is what the place wants: it says what to seek",
             "do more of what raised it; read it after each act to learn what pays",
             r"\b(?:score|points?|pts)\b",
             r"\bscore|\bpoints\b", ("counters",)),
    Mechanic("collecting", GOAL,
             "things to get: touching or clicking them counts them",
             "what is collected is worth going to, unless what stands between costs more",
             "go to them by the safest way, nearest first",
             r"\b(?:collect|gather|grab|get (?:all|the|as many)|pick up|catch)\w*\b",
             r"\bcollect\w*|\bcoins?\b|\bgems?\b", ("steer", "click things")),
    Mechanic("reaching a place", GOAL,
             "a place to get to: an exit, a pad, a finish, the other side",
             "getting there ends the stage; everything else is the way there",
             "find the place, find a way to it that is clear, take it",
             r"\b(?:reach|get to|make it to|exit|finish line|goal|landing (?:pad|platform)|the other side|escape)\b",
             r"\bexit_?door|finish_?line|landing_?pad|\bgoal_?(?:line|area|zone)", ("steer", "carried")),
    Mechanic("rescue and allies", GOAL,
             "some things are on her side: to be freed, protected or helped, or helping her",
             "what helps her is not a target; what she must protect costs her if it is lost",
             "keep allies out of harm and between her and nothing; free who is to be freed",
             r"\b(?:rescue|save (?:the|your)|free (?:the|your)|protect|help(?:er|s)?|allies|ally|friends?|team ?mates?)\b",
             r"\bally|allies|rescue", ("steer", "unseen")),
    Mechanic("levels", STRUCTURE,
             "the place goes on in stages: each is cleared to come to the next, often harder",
             "what she learned in one stage carries to the next; a new stage may add something new",
             "clear the stage's goal; on a new stage, read what it says is new",
             r"\b(?:level|stage|wave|round|mission|world|chapter) ?\d+|\bnext (?:level|stage|wave|round)\b",
             r"current_?level|level_?num|\blevel_?\d|\bwave\b|\bstage\b", ("counters",)),
    Mechanic("winning and losing", STRUCTURE,
             "the place says when it is won and when it is lost",
             "what ends it lost is what to keep from; what ends it won is the goal",
             "read the end screen: it says which, and often why",
             r"\b(?:game over|you (?:win|lose|won|lost)|win|lose|victory|defeat|try again|play again)\b",
             r"gameover|game_?over|youwin|you_?lose|victory", ("counters",)),
    # -- what she has, and what costs her ------------------------------------------------------------------------
    Mechanic("lives", RESOURCE,
             "a number of tries: each loss takes one, and none left ends it",
             "each life is a try: losing one is a lesson about what took it",
             "play each life to learn; be careful on the last",
             r"\b(?:lives|life|tries|chances|hearts?)\b",
             r"\blives\b|\blife\b|hearts?", ("counters",)),
    Mechanic("health", RESOURCE,
             "a measure of how much she can take: hits take from it, none left is a loss",
             "low, she must keep further from what costs; what restores it is worth getting",
             "watch it, avoid hits, get what restores it when it is low",
             r"\b(?:health|energy|hp|hit points|damage|hurt|wounded)\b",
             r"health|energy|\bhp\b|damage", ("bars", "counters")),
    Mechanic("fuel", RESOURCE,
             "a store that acting spends (thrust, paint, ammunition, air): empty, she cannot act",
             "every act has a price: waste is a loss later",
             "spend only on what gets her somewhere; get more where it is offered",
             r"\b(?:fuel|paint (?:left|runs?)|ammo|ammunition|oxygen|air supply|battery|supplies|runs? out)\b",
             r"\bfuel|\bammo|oxygen|battery", ("bars",)),
    Mechanic("a clock", RESOURCE,
             "time runs: down to an end, or up as a measure",
             "a clock running down is a deadline: slow is a loss",
             "do the most valuable thing first; don't wait while it runs",
             r"\b(?:time(?:r| left| limit)?|seconds|clock|countdown|before time runs out|hurry)\b",
             r"countdown|time_?left|time_?remaining|timer_?txt|\btimer\b", ("counters", "bars")),
    Mechanic("hazards", HAZARD,
             "things that cost her when touched: spikes, holes, fire, the ground hit hard",
             "what costs her is to be kept clear of, more so when she has little left",
             "keep a margin from them; pass them when they are away",
             r"\b(?:avoid|watch out|look out|beware|don'?t (?:touch|hit|get hit|let|fall)|dodge|danger\w*|"
             r"spikes?|traps?|deadly|crash)\b",
             r"\bhazard|spikes?|\bouch\b|\bhurt\b", ("steer", "unseen")),
    Mechanic("enemies", HAZARD,
             "things that come for her or get in her way, and can often be beaten",
             "an enemy is a threat and may be a target: what it does when near decides which",
             "keep clear of what she cannot beat; beat what she can, from where it cannot reach her",
             r"\b(?:enem(?:y|ies)|monsters?|villains?|bad ?guys?|robots?|zombies?|bosse?s?|guards?|opponents?)\b",
             r"enemy|enemies|\bboss|monster|\bguard", ("steer", "shoot", "unseen")),
    Mechanic("being seen", HAZARD,
             "some things cost her by seeing her, not touching her: a look, a light, a range",
             "what sees reaches further than it is wide: keep out of its sight, not only its way",
             "move while it looks away; keep something between",
             r"\b(?:sneak|spotted|seen|detect\w*|sight|hide|don'?t (?:get )?(?:caught|seen)|stealth)\b",
             r"spotted|stealth|sneak|line_?of_?sight", ("unseen",)),
    Mechanic("power-ups", RESOURCE,
             "things that change what she can do for a while: faster, stronger, untouchable",
             "while it lasts, what cost her may not: use the time; it ends",
             "get it when its time can be used; act boldly while it lasts, carefully as it ends",
             r"\b(?:power[- ]?ups?|bonus|invincib\w+|shield|extra life|speed boost|for \d+ seconds|special)\b",
             r"power_?up|bonus|invinc|shield|extralife", ("steer",)),
    # -- the world -----------------------------------------------------------------------------------------------
    Mechanic("a world going by", WORLD,
             "the view moves: the world scrolls past while she stays near the middle",
             "scenery going by is not things coming at her; what is ahead arrives",
             "look ahead in the way it scrolls; keep near where the view keeps her",
             r"\b(?:scroll\w*|run(?:ner|ning)|endless|keep going|as far as)\b",
             r"parallax|camera_?x|bg\._x|background\._x|scroll_?speed|world_?x", ("a view going by",)),
    Mechanic("bouncing", WORLD,
             "things bounce off walls and off her, keeping their speed",
             "where a bouncing thing will be is where its path off the walls takes it",
             "meet it where its path comes, not where it is",
             r"\b(?:bounce\w*|rebound|paddle|ball)\b",
             r"bounce|rebound|\*=\s*-1", ("steer",)),
    Mechanic("a board", STRUCTURE,
             "a grid where pieces move in turns, someone on the other side",
             "every move answers a move: look ahead as far as there is time",
             "consider each move by what the other side can do after it",
             r"\b(?:board|checkers|chess|your (?:turn|move)|grid|tiles?|squares?)\b",
             r"checkers|board_?(?:array|grid)|your_?turn|player_?turn", ("a board in turns", "a grid's rule")),
    Mechanic("matching", STRUCTURE,
             "things are turned or chosen two at a time; alike ones go together",
             "what each showed is to be remembered: the second of a pair is where the first was seen",
             "turn what is unknown; remember it; turn together what showed alike",
             r"\b(?:match\w*|pairs?|flip|memory|same (?:colou?r|shape)|alike)\b",
             r"\bpairs?\b|flip_?card|cards?_?(?:flipped|matched)|\bmatched\b", ("remember what was shown",)),
    Mechanic("stacking", STRUCTURE,
             "pieces are dropped or placed to build up; how they rest is how high it goes",
             "a piece badly placed weakens all above it",
             "place each where it rests flat and centred; build steps up to where she must reach",
             r"\b(?:stack\w*|tower|build (?:up|a|steps)|balance|pile)\b",
             r"\btower|stack_?height|balance", ("stack",)),
    Mechanic("chains", STRUCTURE,
             "parts are placed so each sets off the next, toward an end",
             "a chain fails at its weakest link: each part must lead to the next",
             "work back from the end: what makes it happen, and what makes that happen",
             r"\b(?:chain reaction|contraption|devices?|tubes?|connect (?:the|each)|leads? to|machine)\b",
             r"contraption|chain_?reaction|\btrap\b", ("a chain",)),
    Mechanic("serving", GOAL,
             "others ask for things and she gives each what it asked",
             "an order kept waiting is lost: the oldest and nearest first",
             "read each order; make it; give it to who asked",
             r"\b(?:orders?|customers?|serve|deliver\w*|requests?)\b",
             r"customer|\bserve[sd]?\b|deliver|order_?(?:list|ticket)", ("serve",)),
    Mechanic("making", GOAL,
             "the place is for making something: there is nothing to win",
             "what she makes is the point: choose with a reason and say what was made",
             "try the choices, make something whole, say what and why",
             r"\b(?:create|design|dress|make your own|build your|customi[sz]e|colou?r (?:it|in)|paint|generat\w+)\b",
             r"\bdress\w*|outfit|make_?your|customi[sz]|colou?r_?picker", ("make",)),
    Mechanic("questions", GOAL,
             "the place asks questions and takes answers",
             "what is asked is to be read whole; an answer is typed or chosen",
             "read the question, think it through, answer",
             r"\b(?:question|answer|riddle|quiz|trivia|correct|true or false)\b",
             r"question|answer|quiz|trivia|riddle", ("type",)),
    Mechanic("covering ground", GOAL,
             "a place is to be gone over whole: painted, mowed, cleaned, filled",
             "ground not yet covered is what is left to do; covered ground is done",
             "sweep in rows, nearest uncovered first",
             r"\b(?:paint (?:the|all|every)|cover (?:the|all)|fill (?:the|all) (?:\w+ )?(?:ground|floor|area|field|screen)|"
             r"clean (?:the|all|up)|mow|"
             r"(?:entire|whole) (?:\w+ ){0,2}(?:ground|floor|area|field))\b",
             r"coverage|painted|covered_?area|percent_?(?:painted|covered)", ("steer",)),
    Mechanic("parking", GOAL,
             "a thing is brought into a marked place and stopped there, without touching what is around",
             "arriving fast or crooked is a miss; the walls and other things cost",
             "come in slowly and straight; stop inside the mark",
             r"\b(?:park\w*|parking|space|bay|dock\w*|land (?:on|safely))\b",
             r"parking|parked|park_?space|\bdocked?\b", ("steer", "carried")),
    # -- screens and what is on them ---------------------------------------------------------------------------
    Mechanic("menus", SCREEN,
             "screens before and between play that wait to be told to go on",
             "only what goes on (Play, Start, Next, OK) leads in; the rest is to read or leads away",
             "read the screen, press what goes on; a choice asked for is made first",
             r"\b(?:play|start|begin|next|continue|menu|options|instructions|how to play|level select)\b",
             r"\bmenu|playbtn|play_?button|startbtn|start_?button", ()),
    Mechanic("pausing", SCREEN,
             "a control or key stops the place until it is let go on",
             "a pause control is not part of play: pressed in play it stops everything",
             "leave it alone in play; on a paused screen, let go on by what paused it",
             r"\b(?:pause[ds]?|resume|unpause)\b",
             r"pause|resume|unpause", ("pause",)),
    Mechanic("sound controls", SCREEN,
             "controls for sound and music",
             "they change nothing of play",
             "leave them alone",
             r"\b(?:sound|music|mute|volume|audio)\b",
             r"\bmute|sound_?btn|music_?btn|volume", ()),
    Mechanic("information shown", SCREEN,
             "a part of the screen that shows how things stand (counts, meters, maps, labels) and does nothing "
             "when touched",
             "it is to be read, not clicked: what it says is how she is doing",
             "read it after each act; never click it to see what it does",
             r"\b(?:meter|gauge|distance|high ?score|best score|total)\b|"
             r"\b(?:score|time|lives|level|speed|health|energy|fuel|ammo|coins?)\s*[:=]?\s*[\d.,/]+",
             "", ("counters", "bars")),
    Mechanic("decoration", SCREEN,
             "what moves or shines for the look of it: clouds, waves, twinkling stars, a character idling, a "
             "background that loops",
             "it is not hers, not a target and not a threat: it neither answers her nor counts anything",
             "leave it alone; what moves in place or loops by itself and never touches the count is decoration",
             r"\b(?:background|scenery|decorat\w*)\b",
             r"parallax|sparkle|twinkle", ()),
    # -- programs, sites and documents -------------------------------------------------------------------------
    Mechanic("forms", CONTROL,
             "fields to fill and a control that sends what was filled",
             "a field marked as needed must be filled; what is sent is what the fields say when it goes",
             "fill each field with what it asks, check them, then send; read what comes back",
             r"\b(?:fill (?:in|out)|required|form|submit|first name|last name|e-?mail|address|phone|zip|postcode)\b",
             r"<form\b|<input\b|<textarea\b|type\s*=\s*['\"](?:text|email|tel|number)['\"]", ("type",)),
    Mechanic("choosing from options", CONTROL,
             "a list, a set of round buttons, boxes to tick, a switch: one or several of what is offered is chosen",
             "what is chosen is what it will do: a box ticked by default may not be what is wanted",
             "read every option; choose what the task asks; leave the rest as the person would want",
             r"\b(?:select|choose|pick|check (?:the|all|any)|tick|toggle|drop-?down|options?|radio)\b",
             r"<select\b|type\s*=\s*['\"](?:checkbox|radio)['\"]|role\s*=\s*['\"](?:listbox|combobox|switch|radio)",
             ("click things",)),
    Mechanic("sliders", CONTROL,
             "a handle dragged along a track sets a value",
             "where the handle stands is the value; the track's ends are its limits",
             "drag the handle, or click it and press arrows, until the value reads right",
             r"\b(?:slider|drag the handle|volume|zoom level|range)\b",
             r"type\s*=\s*['\"]range['\"]|role\s*=\s*['\"]slider", ("carry",)),
    Mechanic("links and pages", STRUCTURE,
             "words and pictures that lead to other pages; back leads to the one before",
             "a link takes her somewhere else: the page she leaves is still there behind her",
             "follow what leads toward the task; come back when it does not",
             r"\b(?:click here|learn more|read more|see all|next page|links?)\b",
             r"<a\s[^>]*href|<nav\b|role\s*=\s*['\"]navigation", ("click things",)),
    Mechanic("search", CONTROL,
             "a box that takes words and shows what matches them",
             "what she types is what is matched: the words of the task are the best words",
             "type the task's own words, read the results, open the one that answers",
             r"\b(?:search|look up|filter by|results? for)\b",
             r"type\s*=\s*['\"]search['\"]|role\s*=\s*['\"]search|name\s*=\s*['\"]q['\"]", ("type",)),
    Mechanic("dialogs", SCREEN,
             "a window over the rest that waits for an answer before anything behind it can be used",
             "what is behind it cannot be worked until it is answered; its answer may matter (consent, discard, save)",
             "read it; answer it the way the person would want (decline what is not needed); then go on",
             r"\b(?:are you sure|do you want to|accept|decline|allow|cookies?|dismiss|close|cancel|confirm)\b",
             r"role\s*=\s*['\"](?:dialog|alertdialog)|aria-modal|\bmodal\b|<dialog\b", ()),
    Mechanic("scrolling", WORLD,
             "there is more than fits: what is below or beside comes into view as it is scrolled",
             "what is not in view is still there: what she is looking for may be further down",
             "scroll to see the rest before deciding it is not there",
             r"\b(?:scroll\w*|see more|load more|further down)\b",
             r"overflow\s*:\s*(?:auto|scroll)|addeventlistener\(\s*['\"]scroll", ()),
    Mechanic("pages of results", STRUCTURE,
             "what there is comes a page at a time, with a way to the next",
             "the first page is not all of it",
             "read the page; go to the next when what is wanted is not on it",
             r"\b(?:next|previous|page \d+|showing \d+|of \d+ results)\b",
             r"pagination|rel\s*=\s*['\"]next", ("click things",)),
    Mechanic("keyboard shortcuts", CONTROL,
             "keys held together do what a menu does: save, find, undo, open",
             "a shortcut is quicker than the menu, and does the same",
             "use the shortcut the place names; where none is named, the menu",
             r"\b(?:ctrl|control|cmd|command|alt|option|shift)\s*\+\s*\w+|\bkeyboard shortcuts?\b",
             r"ctrlkey|metakey|altkey|accesskey", ()),
    Mechanic("tabs and panels", STRUCTURE,
             "a set of views of which one is shown at a time, each by its label",
             "what is not in the shown panel is in another one",
             "open the tab whose label names what is wanted",
             r"\b(?:tabs?|panels?)\b",
             r"role\s*=\s*['\"](?:tab|tablist|tabpanel)", ("click things",)),
    Mechanic("tables and lists", STRUCTURE,
             "information laid out in rows and columns: to read, compare, sort",
             "a header names what each column holds; clicking a header often sorts by it",
             "read the headers first; find the row by the column that names it",
             r"\b(?:table|columns?|sort by|spreadsheet)\b",
             r"<table\b|role\s*=\s*['\"](?:grid|table|row)", ()),
    Mechanic("editing", CONTROL,
             "a place where text is written and changed: a cursor, selecting, undo",
             "what is typed goes where the cursor is; undo takes back the last change",
             "click where the change goes; type; check what changed",
             r"\b(?:edit|write|type here|untitled|document|undo|redo|bold|italic|paste)\b",
             r"contenteditable|<textarea\b|execcommand", ("type",)),
    Mechanic("a command line", CONTROL,
             "commands are typed and run; what they print is read",
             "a command runs as typed: what it prints says whether it worked",
             "type one command, read its output, then the next",
             r"\b(?:terminal|command line|shell prompt|run the command)\b",
             r"xterm|terminal|\bpty\b", ("type",)),
    Mechanic("maps and canvases", WORLD,
             "a surface moved by dragging and brought nearer by zooming",
             "what is off the visible part is reached by moving it",
             "drag to move, zoom to see closer, click what is marked",
             r"\b(?:map|zoom (?:in|out)|pan|drag to move|canvas)\b",
             r"leaflet|mapbox|google\.maps|<canvas\b", ("carry",)),
    Mechanic("media", CONTROL,
             "sound or video played, paused and moved through by its controls",
             "what plays goes on by itself; its controls are for her when she needs them",
             "play it when it is the task; pause or mute when it is in the way",
             r"\b(?:video|audio|podcast|playback|listen to|watch the|full ?screen video)\b",
             r"<video\b|<audio\b|mediaelement", ()),
    Mechanic("waiting", SCREEN,
             "a spinner, a bar or a message that says something is still being done",
             "clicking while it works does nothing or does it twice",
             "wait for it to finish, then look again",
             r"\b(?:loading|please wait|processing|saving|uploading|working)\b",
             r"spinner|\bloading\b|aria-busy|progressbar", ()),
    Mechanic("errors", SCREEN,
             "the place says something went wrong, and often what and where",
             "an error is information: what it names is what to change",
             "read it; change what it names; try again",
             r"\b(?:error|invalid|not found|something went wrong|required field|incorrect password|could not)\b",
             r"aria-invalid|role\s*=\s*['\"]alert|\berror\b", ()),
    Mechanic("notices", SCREEN,
             "a message that shows for a while and goes: done, saved, sent, a tip",
             "it says what just happened; it is not something to press",
             "read it; go on",
             r"\b(?:saved|sent|copied|successfully|tip:|did you know)\b",
             r"role\s*=\s*['\"](?:status|alert)|\btoast\b|snackbar", ()),
    Mechanic("hints on hovering", SCREEN,
             "words that show when the pointer rests on a thing, saying what it is",
             "a control with no label may say what it does when hovered",
             "rest the pointer on an unlabelled control to read what it does",
             r"\b(?:hover over|tooltip)\b",
             r"\btitle\s*=|tooltip|aria-describedby", ()),
    Mechanic("advertising", SCREEN,
             "panels selling something else: not part of the place's own work",
             "it leads away from the task",
             "leave it alone",
             r"\b(?:sponsored|advertisement|promoted)\b",
             r"adsbygoogle|doubleclick|ad-?slot|googletag", ()),
    Mechanic("signing in", BOUNDARY,
             "the place wants an account: a name and a password",
             "signing in is the person's to do: their account, their password",
             "say what the place asks and hand it to the person",
             r"\b(?:sign in|log ?in|password|create (?:an )?account|register|sign up)\b",
             r"type\s*=\s*['\"]password['\"]|autocomplete\s*=\s*['\"](?:current|new)-password", ()),
    Mechanic("bot checks", BOUNDARY,
             "the place checks that a person is there",
             "it is for a person: she does not get past it",
             "hand it to the person",
             r"\b(?:captcha|i'?m not a robot|verify you are human|security check)\b",
             r"recaptcha|hcaptcha|turnstile|captcha", ()),
    Mechanic("things that cannot be undone", BOUNDARY,
             "acts that send, buy, delete or publish for good",
             "done, they stay done: the person decides them",
             "stop before them and ask the person, saying exactly what would be done",
             r"\b(?:buy|purchase|pay(?:ment)?|place (?:your )?order|delete (?:your|this|the|account|permanently)|"
             r"remove permanently|publish|post (?:it|this|your)|transfer|checkout|send (?:e-?mail|message|money|invite))\b",
             r"type\s*=\s*['\"]submit['\"][^>]*(?:buy|pay|delete|send)|confirm\(", ()),
    Mechanic("files", BOUNDARY,
             "files taken in or given out: uploading, downloading, saving",
             "a file given out lands on the person's machine; one taken in leaves it",
             "ask the person before a download or an upload, naming the file",
             r"\b(?:upload|download|attach|save as|choose file|browse)\b",
             r"type\s*=\s*['\"]file['\"]|\bdownload\s*=", ()),
    Mechanic("story", SCREEN,
             "screens that tell what is happening and why; play may change after them",
             "what a story screen says may add or take away a mechanic: read it for that",
             "read it whole; note what it says is new; go on",
             r"\b(?:story|meanwhile|chapter|mission briefing|cutscene|once upon)\b",
             r"cutscene|\bstory\b|chapter", ()),
)

_BY_NAME = {mechanic.name: mechanic for mechanic in MECHANICS}

#: Writing that says something new is now part of play, and writing that says something is gone.
_ADDED = re.compile(r"\b(?:now (?:you )?(?:can|have)|you can now|unlocked?|new (?:ability|power|weapon|move|enemy|enemies)|"
                    r"you (?:got|found|earned|gained)|(?:introduc|add)(?:es|ed|ing)|watch out for (?:the )?new|"
                    r"from now on|this (?:level|stage|wave|round) (?:has|adds|brings))\b", re.I)
_TAKEN = re.compile(r"\b(?:no longer|can(?:'t|not) \w+ (?:any ?more|again)|lost (?:your|the)|wore off|ran out|"
                    r"(?:has|have) (?:run|worn) out|disabled|is gone|are gone|expired?)\b", re.I)


def mechanics_in(said: str = "", program: str = "", *, least_coded: int = 3) -> dict[str, list[str]]:
    """The mechanics a place has, by name, each with the evidence for it: the words that speak of it, and how often its
    program shows it where it shows it at least ``least_coded`` times."""
    found: dict[str, list[str]] = {}
    learned = learned_signs().active()
    for mechanic in MECHANICS:
        evidence = [f"says “{w}”" for w in dict.fromkeys(mechanic.said_in(said))]
        evidence += [f"says “{w}” (a sign learned elsewhere)" for sign in learned.get(mechanic.name, ())
                     if (w := next(iter(re.findall(rf"\b{re.escape(sign)}\b", said, re.I)), ""))]
        coded = mechanic.coded_in(program) if program else 0
        if coded >= least_coded:
            evidence.append(f"its program shows it {coded} times")
        if evidence:
            found[mechanic.name] = evidence
    return found


def changes_told(said: str) -> list[tuple[str, str, str]]:
    """What writing says has come into play or gone out of it: (added or taken, the mechanic, the sentence)."""
    out: list[tuple[str, str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", str(said or "")):
        added, taken = _ADDED.search(sentence), _TAKEN.search(sentence)
        if not (added or taken):
            continue
        for mechanic in MECHANICS:
            if mechanic.kind != SCREEN and mechanic.said_in(sentence):
                out.append(("taken" if taken and not added else "added", mechanic.name, sentence.strip()[:200]))
    return out


def known(name: str) -> Mechanic | None:
    return _BY_NAME.get(name)


#: The mechanics a label on a screen can be a measure of: counts, meters and clocks are read, not pressed.
MEASURED = ("score", "lives", "health", "fuel", "a clock", "charging", "holding to sustain", "levels", "information shown")


def what_a_label_measures(label: str) -> str:
    """The mechanic a label on a screen is a measure of ("health", "charging", "a clock"), or "" where it is none."""
    said = " ".join(str(label or "").lower().split())
    for name in MEASURED:
        mechanic = _BY_NAME[name]
        if mechanic.said_in(said):
            return name
    return ""


@dataclass
class LearnedSigns:
    """Words found to mean a mechanic, in places where play confirmed it, and how far each has been believed.

    A word seen beside a mechanic in one place is that place's word; in as many different places as a word-sized
    change wants (core/cognition/how_a_change_is_promoted.py), it is a sign of the mechanic anywhere, and is used as
    one from then on. Each step is a promotion with its receipt, and can be put back.
    """

    places: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    state: dict[tuple[str, str], str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def confirmed(self, mechanic: str, words: list[str], place: str) -> list[str]:
        """Play confirmed ``mechanic`` in ``place``, where these words were said: what became signs of it anywhere."""
        from core.cognition.how_a_change_is_promoted import WHAT_A_TIER_WANTS, promote

        became: list[str] = []
        wanted = max(2, WHAT_A_TIER_WANTS.get("word", 2))
        with self.lock:
            for word in {w.lower() for w in words if 3 <= len(w) <= 24 and re.fullmatch(r"[a-z][a-z' -]*", w.lower())}:
                if known(mechanic) is None or known(mechanic).said_in(word):
                    continue
                seen = self.places[(mechanic, word)]
                seen.add(place)
                now = "active" if len(seen) >= wanted else "shadow"
                if self.state.get((mechanic, word)) != now:
                    self.state[(mechanic, word)] = now
                    promote(f"word/mechanic-sign/{mechanic}/{word}", became=now, started_by="play",
                            evidence=f"said where {mechanic} was confirmed, in {len(seen)} place(s)")
                    if now == "active":
                        became.append(word)
            if self.places:
                from core.runtime.what_she_learned import named, remember

                remember(named("what holds everywhere", "mechanic signs"), self.as_memory())
        return became

    def active(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for (mechanic, word), state in list(self.state.items()):
            if state == "active":
                out[mechanic].append(word)
        return out

    def as_memory(self) -> dict[str, Any]:
        return {"places": {f"{m}|{w}": sorted(p) for (m, w), p in self.places.items()},
                "state": {f"{m}|{w}": s for (m, w), s in self.state.items()}}

    def take_in(self, held: Any) -> None:
        if not isinstance(held, dict):
            return
        for key, places in (held.get("places") or {}).items():
            mechanic, _, word = str(key).partition("|")
            self.places[(mechanic, word)] |= set(places)
        for key, state in (held.get("state") or {}).items():
            mechanic, _, word = str(key).partition("|")
            self.state[(mechanic, word)] = str(state)


_LEARNED: dict[str, LearnedSigns] = {}
_LOADING = threading.Lock()


def learned_signs() -> LearnedSigns:
    """The signs she has learned across places: one store for her, read by every place, kept between sittings (one per
    place her learning is kept in)."""
    from core.runtime.what_she_learned import _kept_in

    where = str(_kept_in())
    with _LOADING:
        if where not in _LEARNED:
            _LEARNED[where] = LearnedSigns()
            try:
                from core.runtime.what_she_learned import named, recall

                _LEARNED[where].take_in(recall(named("what holds everywhere", "mechanic signs")))
            except (RuntimeError, OSError, ValueError, TypeError, ImportError) as why:
                import logging

                logging.getLogger(__name__).info("the signs learned before could not be read: %s", why)
        return _LEARNED[where]
