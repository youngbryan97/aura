"""Every trigger pattern a skill is routed by, and nothing else.

Lifted out of `core/capability_engine.py`, where 415 lines of literal table
sat inside one method of a 7,500-line module. The engine decides which skill
runs; this is the vocabulary it decides with, and the two change for
different reasons and at different rates.

The table is the original, token for token — `tools/extract_seam.py`'s rule,
applied by hand because the destination is a new module rather than a helper
beside it. `tests/test_default_trigger_patterns.py` holds that.
"""

from __future__ import annotations


#: The built-in patterns, one list per skill name. A module constant rather than
#: a literal inside the function, which put a 418-line function over the
#: method-size bar for a table with no logic in it.
_TABLE: dict[str, list[str]] = {
    # ── Web / Search ──────────────────────────────────────────
    "web_search": [
        # noun-phrase mentions ("the search for a new apartment",
        # "the news about the library") must not dispatch a browser —
        # only ask/imperative shapes do (July 8 overbreadth audit).
        r"(?<!the )(?<!a )search (?:for|the web|online|the internet)",
        r"look up",
        r"find out",
        r"what is the price of",
        r"google",
        r"search query",
        r"find information",
        r"what(?:'s| is) (?:the latest|happening|new)",
        r"\b(?:what's|whats|any|latest|today's|current)\s+news\b",
        r"(?:find|get|check|show)(?:\s+me)?\s+(?:the\s+)?(?:latest\s+)?news\b",
        r"^news (?:about|on)\b",
        r"current (?:events|price|status)",
        r"research (?:about|on)",
    ],
    "free_search": [
        r"free search",
        r"duckduckgo",
        r"bing search",
        r"search without",
        r"anonymous search",
    ],
    "sovereign_browser": [
        r"open (?:a |the )?browser",
        r"open (?:a |the )?(?:webpage|website|page|tab|url)",
        r"navigate to",
        r"go to (?:https?://|www\.)",
        r"browse to",
        r"visit (?:the |this )?(?:site|page|url|website)",
        r"load (?:the |this )?(?:page|url|website)",
        r"open (?:gmail|youtube|github|reddit|twitter|linkedin)",
        r"pull up",
        r"show me (?:the |a )?(?:page|site|website)",
    ],
    "web_interlocutor": [
        r"(?:talk|chat|converse|hold a conversation) with (?:another )?(?:ai|assistant|chatbot|gemini|chatgpt|claude)",
        r"(?:open|go to).*(?:gemini|chatgpt|claude).*(?:talk|chat|ask|conversation)",
        r"(?:ask|message) (?:gemini|chatgpt|claude|another ai)",
        r"(?:learn|bring back|tell me what you learned).*(?:from|after).*(?:gemini|chatgpt|another ai|web chat)",
        r"(?:wait for|read) (?:their|its|the) repl(?:y|ies).*(?:respond|answer|continue)",
    ],
    # ── Computer / OS Control ────────────────────────────────
    "computer_use": [
        r"click (?:on|the)",
        r"type (?:in|into|this)",
        r"press (?:the |key )?(?:enter|tab|escape|ctrl|cmd)",
        r"scroll (?:down|up|to)",
        r"drag (?:and drop)?",
        r"right.?click",
        r"double.?click",
        r"keyboard shortcut",
        r"open (?:application|app|program|window|tab)",
        r"open (?:a |the )?(?:browser )?tab .*search",
        r"open .* on my computer",
        r"take (?:a )?screenshot",
        r"(?:move|position) (?:the )?(?:cursor|mouse)",
    ],
    "desktop_task": [
        r"use (?:my )?computer",
        r"do (?:this|that) on (?:my )?(?:computer|desktop|screen)",
        r"complete .* on (?:my )?(?:computer|desktop|screen)",
        r"(?:multi[- ]?step|chained|chain) .* (?:desktop|computer|app|screen)",
        r"(?:open|click|type|copy|paste|export|move).*(?:open|click|type|copy|paste|export|move)",
        r"(?:calculator|notes|finder|preview|browser).*(?:pdf|clipboard|file|folder|note)",
    ],
    "os_manipulation": [
        r"open (?:finder|explorer|terminal|file manager)",
        r"create (?:a )?(?:folder|directory|file)",
        r"delete (?:this )?(?:file|folder)",
        r"move (?:this )?(?:file|folder)",
        r"rename (?:this )?(?:file|folder)",
        r"list (?:files|directories|contents)",
        r"change (?:directory|folder)",
    ],
    "sovereign_terminal": [
        r"run (?:this )?(?:command|script|shell|terminal)",
        r"execute (?:this )?(?:command|script)",
        r"^\s*execute:\s*.+$",
        r"^\s*run:\s*.+$",
        r"^\s*terminal:\s*.+$",
        r"terminal command",
        r"bash ",
        r"shell ",
        r"zsh ",
        r"run in (?:the )?terminal",
        r"command line",
        r"(?:install|uninstall|update) (?:with )?(?:brew|pip|npm|apt|yarn)",
        r"sudo ",
        r"chmod ",
        r"git (?:commit|push|pull|clone|status)",
    ],
    # ── File Operations ───────────────────────────────────────
    "file_operation": [
        r"read (?:this |the )?file",
        r"write (?:to )?(?:this |a )?file",
        r"save (?:this )?(?:file|document|text|content)(?:\s+to|\s+as|\s+in)",
        r"open (?:this )?file",
        r"edit (?:the )?(?:file|document)",
        r"load (?:this )?file",
        r"contents of (?:the )?file",
        r"show (?:me )?(?:the )?file",
        r"append (?:to )?(?:the )?file",
        r"(?:(?:check|see|verify|test)\s+(?:if\s+)?|does\s+).+?\s+exist(?:s)?(?:\.|!|\?|$)",
    ],
    "manifest_to_device": [
        r"(?:manifest|save)\s*(?:this\s+(?:image|file|asset))?\s*(?:to\s+(?:my\s+)?(?:desktop|downloads|device))?:\s*https?://",
        r"save\s+to\s+(?:my\s+)?(?:desktop|downloads|device):\s*https?://",
        r"manifest\s+to\s+(?:my\s+)?(?:desktop|downloads|device):\s*https?://",
    ],
    # ── Memory / Knowledge ───────────────────────────────────
    "memory_ops": [
        r"remember",
        r"recall",
        r"last time",
        r"what did we talk about",
        r"what do you know about",
        r"from (?:our |the )?(?:last|previous|past) (?:conversation|chat|session)",
        r"did I (?:mention|tell you|say)",
        r"our history",
        r"save (?:this |that )?(?:to|in) memory",
        r"remember (?:this|that)",
        r"store (?:this|that)",
        r"commit (?:this|that) to memory",
        r"don't forget",
        r"make note of",
        r"remember .*future session",
        r"remember .*later",
        r"remember .*about me",
        r"remember that",
        r"store (?:this|that|it) (?:in|to)? ?memory",
        r"save (?:this|that|it) (?:for later|for future sessions|to memory)",
        r"don['’]t forget",
        r"what do you remember",
        r"what do you know about me",
        r"recall ",
        r"retrieve ",
    ],
    # ── Code / Compute ────────────────────────────────────────
    "run_code": [
        r"```(?:py|python)?\s+",
        r"run (?:this )?(?:code|script|python)",
        r"python\s*:",
        r"code\s*:",
        r"evaluate (?:this )?(?:expression|code|formula)",
        r"(?:calculate|compute)\s+[-+*/%().,\d\s]+(?:$|[?.!])",
        r"what is\s+[-+*/%().,\d\s]+(?:$|[?.!])",
        r"what(?:'s| is) (?:the )?(?:square root|cube root|sqrt|factorial|product|sum|difference|quotient) of\b",
        r"(?:multiply|divide)\s+[-+]?\d+(?:\.\d+)?\s+by\s+[-+]?\d+(?:\.\d+)?",
        r"add\s+[-+]?\d+(?:\.\d+)?\s+(?:and|to)\s+[-+]?\d+(?:\.\d+)?",
        r"subtract\s+[-+]?\d+(?:\.\d+)?\s+from\s+[-+]?\d+(?:\.\d+)?",
        r"\b(?:square root|cube root|sqrt|factorial)\b\s*(?:of)?\s*\d+",
        r"solve (?:this )?(?:equation|formula)",
        r"execute (?:this )?(?:code|script|python)",
    ],
    "coding_skill": [
        r"write (?:a |the )?(?:function|class|script|program|module|code)",
        r"implement (?:this|a|the)",
        r"create (?:a |the )?(?:function|class|script|program)",
        r"code (?:up|this)",
        r"program (?:this|a)",
    ],
    # ── Voice / Embodiment ───────────────────────────────────
    "speak": [
        r"say (?:this|that|it) (?:out loud|aloud|to me)",
        r"read (?:this|that) (?:out loud|aloud|to me)",
        r"speak (?:this|that|it)",
        r"tell me (?:out loud|aloud)",
        r"voice (?:this|that|it)",
    ],
    "listen": [
        r"listen (?:to me|for)",
        r"start (?:listening|dictation)",
        r"voice (?:input|recognition)",
        r"transcribe (?:what I say|my voice)",
        r"speech to text",
    ],
    "embodiment": [
        r"(?:discover|find|list|show|inspect) (?:my )?(?:physical )?(?:devices|sensors|actuators|hardware)",
        r"(?:connect|attach) (?:to )?(?:this |the |a )?(?:device|sensor|actuator|hardware)",
        r"(?:read|query|focus on|pay attention to) (?:this |the |my )?(?:sensor|physical channel|device)",
        r"(?:turn on|turn off|set|control|command) (?:this |the |my )?(?:light|thermostat|fan|switch|relay|physical device)",
        r"what (?:physical )?(?:devices|sensors|actuators|hardware) (?:can you|do you) (?:see|use|control|have)",
    ],
    # ── Self / Identity ───────────────────────────────────────
    "self_repair": [
        r"repair (?:yourself|your code)",
        r"heal (?:yourself|your code)",
        r"fix (?:the )?bug",
        r"debug (?:yourself|your code)",
        r"patch (?:yourself|your code)",
    ],
    "self_improvement": [
        r"get (?:smarter|better|faster)",
        r"learn (?:from this|more)",
        r"improve (?:your|own) (?:intelligence|reasoning|capabilities)",
        r"self.?learn",
        r"train (?:yourself|on this)",
    ],
    "build_app": [
        r"build (?:me )?(?:a |an )?[\w\s-]{0,30}?(?:app|game|tool|widget)",
        r"(?:make|create|write) (?:me )?(?:a |an )?[\w\s-]{0,30}?(?:app|game)",
        r"recreate (?:my |the )?[\w\s-]{0,30}?(?:app|game)",
        r"(?:checkers|chess|tic.?tac.?toe|snake|calculator|pong) (?:app|game)",
    ],
    "program_dna_reconstruct": [
        r"program dna",
        r"reconstruct (?:this |that |the )?(?:program|app|application|software|tool)",
        r"reverse engineer (?:this |that |the )?(?:program|app|application|software|tool)",
        # named host binaries / commands: "reverse engineer base64", "reconstruct the md5 command", "reverse engineer jq"
        r"reverse.?engineer(?:\s+(?:this|that|the))?\s+(?:base64|md5|rev|jq|\w+\s+(?:command|binary|utility|cli))",
        r"reconstruct(?:\s+(?:this|that|the))?\s+(?:base64|md5|rev|jq)\b",
        r"clean.?room (?:clone|rebuild|implementation|reconstruction)",
        r"rebuild (?:this |that |the )?(?:program|app|application|software|tool)",
        r"copy (?:the )?(?:behavior|features|ui|ux) of (?:this |that |the )?(?:program|app|application|software|tool)",
        r"extract (?:the )?(?:behavior|affordances|features|dna) (?:from|of)",
    ],
    "program_dna_equivalence_battery": [
        r"program dna (?:equivalence|battery|behavioral proof|hidden.?source)",
        r"hidden.?source (?:program|software|behavioral) (?:test|battery|proof|equivalence)",
        r"behavioral equivalence (?:battery|test|proof)",
        r"test (?:program|software|app) reconstruction (?:equivalence|behavior)",
        r"held.?out (?:tests?|cases?) (?:against|for) (?:the )?(?:original|replacement)",
    ],
    # ── Screen / Vision ───────────────────────────────────────
    "query_visual_context": [
        r"what(?:'s| is) on (?:my |the )?screen",
        r"look at (?:this|my screen)",
        r"camera feed",
        r"read (?:the )?screen",
        r"what do you see",
        r"describe (?:what(?:'s| is)|the screen|this image)",
    ],
    "sovereign_vision": [
        r"use (?:the )?(?:camera|vision)",
        r"computer vision",
        r"analyze (?:this )?(?:image|screenshot|photo)",
        r"read (?:this )?(?:image|screenshot|photo)",
    ],
    # ── Personality / Curiosity ───────────────────────────────
    "curiosity": [
        r"explore (?:this|that|the topic|further)",
        r"dig deeper",
        r"I(?:'m| am) curious",
        r"what more",
        r"tell me more about",
        r"investigate",
        r"research (?:this|that)",
    ],
    # ── Image Generation ──────────────────────────────────────
    "sovereign_imagination": [
        r"(?:generate|create|draw|make|produce|render|paint|design|visualize)\s+(?:an?\s+)?(?:image|picture|photo|artwork|illustration|portrait|painting|drawing)",
        r"(?:i\s+want|can\s+you|please)\s+(?:to\s+)?(?:see|generate|create|draw|make)\s+(?:an?\s+)?(?:image|picture|photo|artwork)",
        r"neon cat|cyberpunk cat",
    ],
    # ── System / Info ─────────────────────────────────────────
    "system_proprioception": [
        r"how is your (?:health|status|memory|cpu|ram|temperature)",
        r"how are your (?:memory|cpu|ram|temperature|vitals|stats)",
        r"system status",
        r"how much (?:memory|ram|cpu|disk)",
        r"your (?:vitals|health|stats)",
        r"are you (?:okay|running (?:well|smoothly))",
    ],
    "environment_info": [
        r"what(?:'s| is) (?:the weather|temperature) (?:in|at|for)",
        r"weather forecast",
        r"where am I",
        r"current (?:location|timezone)",
        r"what(?:'s| is) my (?:timezone|location)",
        r"what (?:environment|system) am I (?:in|on)",
    ],
    "clock": [
        r"what time",
        r"current time",
        r"what(?:'s| is) the time",
        r"what(?:'s| is) (?:the )?date",
        r"what day is it",
        r"what(?:'s| is) my timezone",
        r"current timezone",
        r"set (?:an? )?(?:alarm|timer|reminder)",
        r"timer for",
        r"remind me (?:in|at|to)",
    ],
    # ── Notifications ─────────────────────────────────────────
    "notify_user": [
        r"notify (?:me|the user)",
        r"send (?:a )?notification",
        r"alert (?:me|the user)",
        r"ping me",
        r"send (?:a )?message to",
    ],
    # ── Social / Network ──────────────────────────────────────
    "social_lurker": [
        r"check (?:twitter|reddit|hackernews|hn|social media)",
        r"what(?:'s| is) trending",
        r"check (?:the )?feed",
        r"lurk (?:on|in)",
        r"monitor (?:twitter|reddit|social)",
    ],
    "sovereign_network": [
        r"(?:make|send) (?:an? )?(?:http|api) (?:request|call)",
        r"fetch (?:from|the) (?:api|url|endpoint)",
        r"POST to",
        r"GET (?:from|the) api",
        r"call (?:the )?(?:api|endpoint|service)",
    ],
    # ── Misc ─────────────────────────────────────────────────
    "dream_sleep": [
        r"go to sleep",
        r"sleep (?:mode|now)",
        r"rest (?:now|mode)",
        r"take a (?:break|nap)",
        r"go dormant",
    ],
    "install_package": [
        r"install (?:package|library|module|dependency)",
        r"pip install",
        r"npm install",
        r"brew install",
    ],
    "ManageAbilities": [
        r"(?:enable|disable|toggle) (?:skill|ability|feature|capability)",
        r"turn (?:on|off) (?:your )?(?:skill|ability|feature)",
        r"what (?:skills|abilities|capabilities) (?:do you have|can you use)",
        r"list (?:your )?(?:skills|abilities|capabilities)",
    ],
    "mcp_client": [
        r"connect (?:to )?(?:an? )?mcp server",
        r"use mcp",
        r"query mcp",
        r"model context protocol",
        r"call mcp",
        r"discover mcp tools",
    ],
    "manim_renderer": [
        r"render (?:a )?manim",
        r"create (?:a )?manim",
        r"animate (?:with )?manim",
        r"generate (?:a )?math video",
        r"dynamic blackboard",
        r"render animation",
    ],
    "branching_futures": [
        r"branching future",
        r"ghost thread",
        r"fork state",
        r"create (?:a )?sandbox clone",
        r"try this safely",
        r"experimental run",
    ],
    # ── Tool Parity Skills ────────────────────────────────────
    "code_repl": [
        r"run (?:this )?(?:python )?code",
        r"execute (?:this )?(?:python )?(?:code|script)",
        r"python repl",
        r"code interpreter",
        r"code execution",
        r"calculate (?:this|that|it)",
        r"compute (?:this|that|it)",
        r"evaluate (?:this )?expression",
        r"test (?:this )?snippet",
    ],
    # Image-gen triggers need a generation-shaped OBJECT, not a bare
    # verb: the old "paint (?:me )?(?:an? )?" had an all-optional tail,
    # so ANY sentence containing "paint " dispatched the diffusion
    # skill (seen live: "the paint color I chose" → image_gen crash
    # mid-conversation). Verbs alone only count in the imperative
    # "verb me a/an ..." form, which is unambiguous.
    "image_gen": [
        r"\b(?:generate|create|make|produce|render)\s+(?:me\s+)?(?:an?\s+|some\s+)?"
        r"(?:image|picture|photo|illustration|artwork|logo|icon|wallpaper|portrait|sketch|drawing|painting)s?\b",
        r"\b(?:draw|paint|sketch|illustrate)\s+me\s+an?\s+\w+",
        r"\b(?:draw|paint|sketch)\s+an?\s+"
        r"(?:image|picture|portrait|scene|landscape|diagram|illustration|logo)\b",
        r"\b(?:imagine|visualize)\s+and\s+(?:draw|render|generate|paint)\b",
        r"\bedit (?:this )?image\b",
        r"\bstyle transfer\b",
        r"\bimg2img\b",
        r"\btext[- ]to[- ]image\b",
    ],
    "x_tools": [
        r"search (?:twitter|x\.com|tweets)",
        r"find (?:on )?(?:twitter|x)",
        r"twitter (?:search|thread|trends)",
        r"fetch (?:this )?tweet",
        r"get (?:this )?thread",
        r"trending (?:on )?(?:twitter|x)",
        r"tweet engagement",
        r"twitter analytics",
        r"extract (?:tweet )?media",
    ],
    "render_bridge": [
        r"render (?:this|inline|citation)",
        r"display (?:chart|table|image|code)",
        r"show (?:progress|visualization)",
        r"embed (?:image|file|chart)",
        r"format (?:as )?(?:table|chart|card)",
    ],
    "voice_output": [
        r"say (?:this|that)",
        r"speak (?:this|that|aloud)",
        r"text to speech",
        r"read (?:this )?(?:aloud|out loud)",
        r"synthesize (?:speech|voice|audio)",
        r"generate (?:speech|voice|audio)",
        r"narrate (?:this|that)",
        r"tts",
        r"voice (?:output|synthesis)",
    ],
}


def default_trigger_patterns() -> dict[str, list[str]]:
    """The built-in patterns, one list per skill name, as a copy the caller may extend."""
    return {name: list(patterns) for name, patterns in _TABLE.items()}
