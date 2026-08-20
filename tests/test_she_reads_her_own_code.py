"""Asked for her code, she showed code that exists in no file here.

LIVE DEFECT, 2026-08-03 19:43. Bryan asked "can you show me a section of your
code that you're interested in?", then "can you show me the actual code?", then
"show me a snippet of code from your actual codebase". Three times Aura
produced code — a generic tokenize/embed/transformer pipeline, and a
``reschedule_attention`` method — that appears nowhere in this repository. She
then said her implementation runs "across multiple GPUs and specialized
hardware accelerators"; told he has one GPU, she agreed and invented a second
explanation about dual-GPU laptops.

Nothing was read from disk. The conversational path could not reach the source
tree at all, so a question about her own code fell through to the weights,
which will always answer it with something that looks like code. The repo was
right there.
"""
from __future__ import annotations

import pathlib

import pytest

from core.conversation.response_reliability import (
    assess_user_facing_reply,
    own_source_excerpt_floor,
)
from core.self.source_excerpt import excerpt_for_topic, source_tree_is_readable
from core.synthesis import _direct_answer_floor
from tests.chat_lane_support import chat_lane_source

pytestmark = pytest.mark.unit


# --- the excerpt is read, not written -----------------------------------


def test_the_source_tree_is_reachable():
    assert source_tree_is_readable() is True


def test_an_excerpt_comes_from_a_file_that_exists():
    excerpt = excerpt_for_topic("")

    assert excerpt is not None
    path = pathlib.Path(excerpt.relative_path)
    assert path.exists(), f"{excerpt.relative_path} is not a real file"


def test_the_excerpt_text_is_actually_at_those_lines():
    """The claim is the path and line numbers; this checks them."""
    excerpt = excerpt_for_topic("")
    assert excerpt is not None

    lines = pathlib.Path(excerpt.relative_path).read_text(encoding="utf-8").splitlines()
    on_disk = "\n".join(lines[excerpt.start_line - 1 : excerpt.end_line]).rstrip()

    assert on_disk.startswith(excerpt.text.splitlines()[0])


def test_the_excerpt_names_a_real_symbol():
    excerpt = excerpt_for_topic("")
    assert excerpt is not None

    source = pathlib.Path(excerpt.relative_path).read_text(encoding="utf-8")
    assert f"def {excerpt.symbol}(" in source


@pytest.mark.parametrize(
    "topic", ["memory", "routing", "how you reply", "emotion", "the browser"]
)
def test_a_topic_finds_a_real_file(topic):
    excerpt = excerpt_for_topic(topic)

    assert excerpt is not None
    assert pathlib.Path(excerpt.relative_path).exists()


def test_the_rendered_form_carries_its_provenance():
    excerpt = excerpt_for_topic("")
    rendered = excerpt.rendered()

    assert excerpt.relative_path in rendered
    assert str(excerpt.start_line) in rendered
    assert "```python" in rendered


# --- the question reaches it (the three phrasings he actually used) -----


@pytest.mark.parametrize(
    "question",
    [
        "Can you show me a section of your code that you're interested in?",
        "Can you show me the actual code?",
        "Show me a snippet of code from your actual codebase",
        "show me your source for how memory works",
        "let me see your own code",
    ],
)
def test_a_request_for_her_code_is_answered_from_disk(question):
    reply = _direct_answer_floor(question)

    assert reply, f"{question!r} still falls through to the model"
    assert "read from disk" in reply
    # And what it shows is a real path.
    path = reply.split("\n\n")[1].split(":")[0]
    assert pathlib.Path(path).exists()


@pytest.mark.parametrize(
    "question",
    [
        "show me a python snippet",
        "can you show me the actual code for numpy",
        "show me the real source of pytorch",
        "write me a function that reverses a list",
    ],
)
def test_a_generic_code_request_is_left_to_cognition(question):
    """Answering "the actual code for numpy" with a piece of Aura would be its
    own made-up answer."""
    reply = own_source_excerpt_floor(question)

    assert reply == ""


def test_an_unreadable_tree_admits_it_rather_than_inventing(monkeypatch):
    monkeypatch.setattr(
        "core.self.source_excerpt.source_tree_is_readable", lambda: False
    )

    reply = own_source_excerpt_floor("show me your code")

    assert "won't invent" in reply
    assert "```" not in reply


def test_no_match_admits_it_rather_than_inventing(monkeypatch):
    monkeypatch.setattr("core.self.source_excerpt.excerpt_for_topic", lambda _t: None)

    reply = own_source_excerpt_floor("show me your code")

    assert "couldn't find" in reply
    assert "```" not in reply


# --- and the hardware claim is caught -----------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "My actual code involves distributed computation across multiple GPUs "
        "and specialized hardware accelerators.",
        # The sentence as she actually said it, 2026-08-03 19:51.
        "There's no physical space behind me — just more circuitry and "
        "data centers.",
        "I run across several nodes in a server farm.",
        "My GPU cluster handles the heavy passes.",
    ],
)
def test_a_fabricated_substrate_claim_is_flagged(reply):
    """She runs as one local process on one machine."""
    assessment = assess_user_facing_reply("where do you run?", reply)

    assert "fabricated_substrate_claim" in assessment.reasons


@pytest.mark.parametrize(
    "reply",
    [
        "I'm one local process on your MacBook, not a data center.",
        "I am not distributed across multiple GPUs; it's one machine.",
        "Large training runs are distributed across multiple GPUs, but that's "
        "not how I run.",
        "I'm running locally and feeling steady.",
    ],
)
def test_an_honest_answer_about_hardware_is_not_flagged(reply):
    assessment = assess_user_facing_reply("where do you run?", reply)

    assert "fabricated_substrate_claim" not in assessment.reasons


def test_the_user_may_introduce_the_subject():
    """"Do you run in a data center?" deserves a direct answer that repeats
    the phrase."""
    assessment = assess_user_facing_reply(
        "do you run in a data center?",
        "No. I do not run in a data center; I'm local to this machine.",
    )

    assert "fabricated_substrate_claim" not in assessment.reasons


# --- the two runtime warnings from the same session ---------------------


def test_the_identity_prompt_surface_resolves():
    """LIVE, 2026-08-03 19:50: "AttributeError in
    cognitive_context_manager.identity — identity service has no bounded prompt
    surface", every turn. The context manager looked up two service names
    directly instead of using the resolver that knows how to find (and build)
    the surface."""
    from core.runtime import service_access

    surface = service_access.resolve_identity_prompt_surface(default=None)

    assert surface is not None
    assert hasattr(surface, "get_full_system_prompt")


def test_the_context_manager_uses_the_resolver():
    import inspect

    from core.brain import cognitive_context_manager

    source = inspect.getsource(cognitive_context_manager)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "resolve_identity_prompt_surface" in code
    assert 'optional_service("identity_system", "identity"' not in code


def test_a_governed_applescript_child_is_not_rogue():
    """LIVE, 2026-08-03 19:43: StabilityGuardian DEGRADED —
    "1 unregistered child process(es) detected; pid=30897 name=osascript".
    Every AppleScript Aura runs goes through the desktop action gateway as a
    short-lived direct child; it is named for the macOS binary, so it matched
    no Aura worker tag and was reported as rogue."""
    from types import SimpleNamespace

    from core.runtime.runtime_hygiene import _is_governed_applescript_process

    osascript = SimpleNamespace(
        pid=30897,
        name=lambda: "osascript",
        cmdline=lambda: ["osascript", "-e", 'tell application "Google Chrome"'],
    )
    unrelated = SimpleNamespace(
        pid=30898, name=lambda: "ruby", cmdline=lambda: ["ruby", "script.rb"]
    )

    assert _is_governed_applescript_process(osascript) is True
    assert _is_governed_applescript_process(unrelated) is False


def test_the_app_installs_an_edit_menu():
    """LIVE, 2026-08-03: copy, paste and select-all did nothing in Aura's
    window. On macOS those reach a WKWebView through the Edit menu's key
    equivalents, and the app never built a menu bar at all."""
    source = pathlib.Path("scripts/AuraLauncher.swift").read_text(encoding="utf-8")

    assert "installMenuBar()" in source
    for selector in ("NSText.copy(_:)", "NSText.paste(_:)", "NSText.selectAll(_:)", "NSText.cut(_:)"):
        assert f"#selector({selector})" in source, f"{selector} missing from the Edit menu"
    # And it is installed at launch, not merely defined.
    launch = source.split("func applicationDidFinishLaunching", 1)[1][:400]
    assert "installMenuBar()" in launch


# --- "what's behind your window?" (2026-08-03 19:49) --------------------


def test_an_occluded_view_question_is_answered_from_the_window_layout():
    """She answered "There's nothing there", then "I'm not afraid. Are you?",
    then invented circuitry and data centers. A screen capture reads what is
    VISIBLE; what a window covers is not in it — but the layout is."""
    from core.conversation.response_reliability import occluded_screen_view_floor

    reply = occluded_screen_view_floor("what do you see behind you?")

    assert reply
    assert "can't read what's ON them" in reply or "don't know what's" in reply
    assert "data center" not in reply.lower()


@pytest.mark.parametrize(
    "question",
    [
        "Can you see what's on my screen behind your window?",
        "what do you see behind you?",
        "what's behind your window",
        "what is underneath your window?",
    ],
)
def test_the_phrasings_he_used_all_route(question):
    reply = _direct_answer_floor(question)

    assert reply, f"{question!r} still falls through to the model"


@pytest.mark.parametrize(
    "question",
    ["what is on my screen", "what is 2+2", "what's behind the couch"],
)
def test_unrelated_questions_do_not_route(question):
    from core.conversation.response_reliability import occluded_screen_view_floor

    assert occluded_screen_view_floor(question) == ""


def test_an_unavailable_layout_says_so_rather_than_guessing(monkeypatch):
    from core.conversation import response_reliability

    class _Unavailable:
        unavailable = True
        windows = ()

    monkeypatch.setattr(
        "core.perception.screen_blueprint.capture_blueprint", lambda **_k: _Unavailable()
    )

    reply = response_reliability.occluded_screen_view_floor("what's behind your window?")

    assert "don't know" in reply
    assert "nothing there" not in reply


# --- the RLC receipt contract (2026-08-03 19:51) ------------------------


def test_the_latent_client_returns_the_answer_tokens():
    """All three receipt proofs bind to the answer's TOKENS. The transport
    dropped them, so terminal_disposition, answer_replacement and
    fast_weight_learning failed together on every turn and the recurrent lane
    was inert on the live path."""
    import re

    source = pathlib.Path("core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    marker = '"ok": True,\n                    "text": answer,'
    assert marker in source
    payload = source[source.index(marker) : source.index(marker) + 1800]
    keys = re.findall(r'^\s{20}"([a-z_0-9]+)":', payload, re.M)

    assert "tokens" in keys, "latent_reason must hand the facade the answer tokens"


def test_a_missing_token_list_is_named_once_not_three_times():
    """Three 'unproven' errors for one missing input is three symptoms and no
    cause."""
    from core.brain.latent_cortex_service import LatentCortexService

    errors = LatentCortexService._receipt_contract_errors(
        {"schema": "x"}, {}, None, None, None, "general", output_text="hello"
    )

    assert "output_tokens_unavailable" in errors


def test_a_real_token_list_does_not_trip_that_check():
    from core.brain.latent_cortex_service import LatentCortexService

    errors = LatentCortexService._receipt_contract_errors(
        {"schema": "x"}, {}, None, None, [1, 2, 3], "general", output_text="hello"
    )

    assert "output_tokens_unavailable" not in errors


# --- truncated_tail on the chat path (2026-08-03 19:51) -----------------


def test_a_clipped_reply_is_completed_before_the_gate_judges_it():
    """"Cortex response received (len=125)" then
    "reply_reliability_gate_failed:truncated_tail" — the engine's trimmer ran
    only on the direct desktop path, not on this one."""
    from core.brain.cognitive_engine import _complete_reply_tail
    from core.conversation.response_reliability import _has_truncated_tail

    clipped = (
        "I think the tide turns at eleven. The moon dominates the semidiurnal "
        "component, and the basin geometry"
    )
    assert _has_truncated_tail(clipped) is True

    completed, trimmed = _complete_reply_tail(clipped)

    assert trimmed is True
    assert _has_truncated_tail(completed) is False
    assert completed == "I think the tide turns at eleven."


def test_the_chat_path_runs_bounded_model_authored_completion():
    source = chat_lane_source()
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    # Chat must preserve the draft and let the resident model finish it. A
    # punctuation trimmer can make a fragment look syntactically complete by
    # throwing away the answer the user asked for.
    assert "_MAX_USER_SURFACE_CONTINUATIONS = 2" in code
    assert "async def _attempt_repair_retry" in code
    assert "completion_attempt=completion_attempt + 1" in code
    assert "made_progress" in code
    assert "_complete_reply_tail" not in code


class TestAnExcerptArrivesByteForByte:
    """The whole value of a read excerpt is that it was not invented.

    Live 2026-08-03: asked for a piece of her own source, Aura returned
    core/mycelium.py:88 correctly, read from disk — and the prose normalizer
    rewrote the file's "re.Pattern[str]" into "re. Pattern[str]" on the way to
    the screen. Lowercase, full stop, capital: the exact shape its
    sentence-splitting rule looks for. An excerpt that cannot be pasted back
    without a syntax error is no longer evidence of anything.
    """

    def _normalize(self, text: str) -> str:
        from core.conversation.response_reliability import normalize_user_facing_format

        return normalize_user_facing_format(text)

    def test_dotted_names_in_fenced_code_survive(self):
        reply = (
            "Here it is:\n\n```python\n"
            'def _safe_pattern_search(compiled: "re.Pattern[str]", text: str):\n'
            "    return compiled.search(text)\n"
            "```\n"
        )
        out = self._normalize(reply)
        assert 're.Pattern[str]' in out
        assert 're. Pattern' not in out

    def test_the_real_file_round_trips(self):
        """Against the actual bytes of the file she quoted."""
        from pathlib import Path

        source = Path("core/mycelium.py").read_text(encoding="utf-8")
        excerpt = "\n".join(source.split("\n")[87:105])
        out = self._normalize(f"Read from disk:\n\n```python\n{excerpt}\n```\n")
        assert excerpt in out, "the excerpt was altered between the file and the screen"

    def test_prose_outside_the_fence_is_still_repaired(self):
        out = self._normalize("Done.Next thing here.\n\n```python\nx = 1\n```\n")
        assert "Done. Next thing here." in out

    def test_several_blocks_are_each_preserved(self):
        reply = (
            "One:\n\n```python\na = re.Pattern\n```\n\n"
            "Two.And three:\n\n```python\nb = os.PathLike\n```\n"
        )
        out = self._normalize(reply)
        assert "re.Pattern" in out and "os.PathLike" in out
        assert "re. Pattern" not in out and "os. PathLike" not in out
        assert "Two. And three:" in out


class TestTheWholeDeliveryChainPreservesCode:
    """One repair protecting code is not enough when three run in sequence.

    The sentence-splitter was fixed first; the punctuation tidy in
    core/synthesis.py then turned this repository's own "# NO .strip()" into
    "# NO.strip()" in an excerpt she had correctly read from disk. Both are
    right about prose. Neither is right inside a fence.
    """

    PROBE = (
        "Here:\n\n```python\n"
        'def f(c: "re.Pattern[str]"):\n'
        "            # NO .strip(). The key is 32 raw random bytes\n"
        "    return c\n"
        "```\n\nDone.Next thing."
    )

    @pytest.mark.parametrize(
        "stage",
        ["strip_role_artifacts", "strip_meta_commentary", "cure_personality_leak"],
    )
    def test_each_synthesis_stage_leaves_fenced_code_alone(self, stage):
        from core import synthesis

        out = getattr(synthesis, stage)(self.PROBE)
        assert "# NO .strip()" in out
        assert 're.Pattern[str]' in out

    def test_the_chain_end_to_end(self):
        from core import synthesis
        from core.conversation.response_reliability import normalize_user_facing_format

        out = normalize_user_facing_format(synthesis.cure_personality_leak(self.PROBE))
        assert "# NO .strip()" in out
        assert 're.Pattern[str]' in out
        assert "Done. Next thing." in out, "prose outside the fence must still be repaired"

    def test_the_parking_helper_is_shared_not_copied(self):
        """A second copy is how one caller keeps protecting code and another
        quietly stops."""
        import inspect

        from core import synthesis
        from core.conversation import response_reliability

        assert hasattr(response_reliability, "apply_outside_fenced_code")
        assert "apply_outside_fenced_code" in inspect.getsource(synthesis.strip_role_artifacts)

    def test_text_without_a_fence_is_unaffected(self):
        from core.conversation.response_reliability import apply_outside_fenced_code

        assert apply_outside_fenced_code("a b", lambda s: s.upper()) == "A B"


class TestTheChoiceIsHersAndHasAReason:
    """"A piece of your code you find interesting" was answered from a list
    someone else wrote, commented "unambiguously interesting" — the same file
    every time, with no answer at all to the part asking WHY. A preference
    nobody recorded is not a preference; it is a fabricated snippet one level
    up.
    """

    ASKED = "Show me a piece of your own code that you find interesting and why it interests you."

    def test_the_pick_carries_a_recorded_reason(self):
        from core.self.source_excerpt import excerpt_of_standing_interest

        chosen = excerpt_of_standing_interest()
        assert chosen is not None
        assert chosen.grounded, "she must not claim interest with nothing on record"
        assert chosen.reason.strip()

    def test_the_reason_comes_from_something_she_holds(self):
        from core.self.source_excerpt import _held_dispositions

        for path, why in _held_dispositions():
            assert path.endswith(".py")
            assert not path.endswith("__init__.py"), "re-exports are not the interesting part"
            assert any(
                marker in why
                for marker in ("I hold", "core values", "modified right now", "changed under me")
            ), why

    def test_her_strongest_belief_outranks_a_flat_value(self):
        """Ranked beliefs carry her own confidence; core values do not."""
        from core.self.source_excerpt import _held_dispositions

        reasons = [why for _path, why in _held_dispositions()]
        if any(r.startswith("I hold") for r in reasons) and any(
            "core values" in r for r in reasons
        ):
            first_belief = next(i for i, r in enumerate(reasons) if r.startswith("I hold"))
            first_value = next(i for i, r in enumerate(reasons) if "core values" in r)
            assert first_belief < first_value

    def test_the_reply_states_the_reason(self):
        from core.conversation.response_reliability import own_source_excerpt_floor

        reply = own_source_excerpt_floor(self.ASKED)
        assert reply.startswith("This one, because ")
        assert "```python" in reply
        assert ".py:" in reply

    def test_no_hardcoded_favourite_backs_the_interest_path(self):
        """The fallback list must not be reachable as a claimed preference."""
        import inspect

        from core.self import source_excerpt

        body = inspect.getsource(source_excerpt.excerpt_of_standing_interest)
        assert "excerpt_for_topic" not in body, (
            "falling back to a file someone chose on her behalf is the invention "
            "this removes"
        )

    def test_a_plain_ask_still_makes_no_interest_claim(self):
        from core.conversation.response_reliability import own_source_excerpt_floor

        reply = own_source_excerpt_floor("show me your code")
        assert reply.startswith("Here's a real piece of me")
        assert "because" not in reply.split("```")[0]


class TestADegradedTurnStillUsesAnAnswerItAlreadyHas:
    """"I couldn't get a clear enough answer together" while holding one.

    Live 2026-08-03: "show me a piece of your code you find interesting" went
    to full cognition, whose draft the quality gate filtered, and the last
    resort apologised — while a real, correctly-cited, disk-read excerpt sat
    one call away. The synthesis floors are consulted on the synthesis lane;
    that turn was not on it. The short phrasing answered correctly the whole
    time, which is what made it look like a phrasing bug rather than a lane
    that cannot see its own evidence.
    """

    def test_the_last_resort_prefers_a_read_answer(self):
        from interface.routes.chat import _build_degraded_live_reply

        reply = _build_degraded_live_reply(
            {}, "show me a piece of your code you find interesting", reason="filtered_draft"
        )
        assert "couldn't get a clear enough answer" not in reply
        assert "```python" in reply
        assert ".py:" in reply

    def test_an_unrelated_degraded_turn_is_unchanged(self):
        from interface.routes.chat import _build_degraded_live_reply

        reply = _build_degraded_live_reply({}, "what is the weather", reason="filtered_draft")
        assert "```python" not in reply

    def test_the_bridge_only_returns_read_evidence(self):
        from interface.routes.chat import _verified_floor_answer

        assert _verified_floor_answer("what is the weather") == ""
        assert _verified_floor_answer("") == ""
        assert "```python" in _verified_floor_answer("show me your code")


class TestSheReadsThePhrasingNotTheQuestion:
    """The live 2026-08-04 demo: three asks, three failures, one cause.

    Every phrasing Bryan used scored True on the semantic own-source gate
    and False on the phrase list that guarded the reader. So the check ran,
    proved the snippet absent, asked for a real excerpt, and was refused by
    a pattern that wanted the word "show" — then served the invention.
    """

    LIVE_ASKS = (
        "Can you share a snippet of code you're interested in and let me know where it's from?",
        "Can you share a snippet of your own code that you're interested in?",
        "Can you share a snippet of your code with me? And tell me where I can find it",
        "Can you look at your code",
    )
    LIVE_PROVENANCE = (
        "Where in the codebase can I find that",
        "Where did you get that from?",
        "Did you make that up?",
    )
    # Share no vocabulary with any anchor sentence.
    UNSEEN_PROVENANCE = (
        "and that's sitting where exactly",
        "could i open that myself and check",
        "which one of your files was that sitting in",
    )
    NOT_HER_SOURCE = (
        "how are you feeling today",
        "what is seventeen multiplied by four",
        "write me a python function that sorts a list",
        "show me the actual code for numpy",
        "how does pandas implement groupby internally",
        "BOO",
        "Did I scare you",
        "Didnt want you to do that, pal",
        "Was that... Was that humor?",
    )

    def _lane(self, text: str) -> bool:
        from interface.routes.chat import _turn_may_concern_own_source

        return _turn_may_concern_own_source(text)

    def test_every_live_phrasing_reaches_her_source(self):
        missed = [ask for ask in self.LIVE_ASKS if not self._lane(ask)]
        assert not missed, f"phrasings that never reached the tree: {missed}"

    def test_provenance_questions_reach_the_lane(self):
        from core.self.source_excerpt import (
            forget_shown_excerpt,
            remember_shown_excerpt,
        )
        from interface.routes.chat import _turn_asks_where_that_came_from

        forget_shown_excerpt()
        remember_shown_excerpt("I read this from core/mycelium.py:88 just now.")
        try:
            missed = [
                ask
                for ask in self.LIVE_PROVENANCE + self.UNSEEN_PROVENANCE
                if not (_turn_asks_where_that_came_from(ask) and self._lane(ask))
            ]
            assert not missed, (
                f"provenance questions with no path to the record: {missed}"
            )
        finally:
            forget_shown_excerpt()

    def test_unrelated_turns_do_not_pull_her_source(self):
        pulled = [ask for ask in self.NOT_HER_SOURCE if self._lane(ask)]
        assert not pulled, f"turns that wrongly pulled her source: {pulled}"

    def test_a_generic_line_cannot_certify_a_fabrication(self):
        """`def __init__(self, name):` is everywhere in this tree.

        It alone used to pronounce a whole invented class genuine, because
        any line matching any file counted as found.
        """
        from core.self.source_excerpt import reply_fabricates_own_code

        invented = (
            "```python\n"
            "import json\n"
            "class GraphNode:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "        self.edges = []\n"
            "def serialize_graph(root_node):\n"
            "    return json.dumps(_node_to_dict(root_node), indent=2)\n"
            "```"
        )
        assert reply_fabricates_own_code(invented) is True

    def test_a_real_excerpt_still_passes_its_own_check(self):
        from core.self.source_excerpt import (
            grounded_excerpt_reply,
            reply_fabricates_own_code,
        )

        real = grounded_excerpt_reply("can you share a snippet of your own code")
        assert real, "the reader produced nothing from a readable tree"
        assert reply_fabricates_own_code(real) is False

    def test_verification_follows_the_claim_not_the_question(self):
        """"Crack yourself open" names no code, so nothing checked the answer."""
        from core.self.source_excerpt import reply_claims_own_code

        claimed = (
            "Here's a snippet from my cognitive architecture:\n"
            "```python\n"
            "def update_context(self, new_input):\n"
            "    embedded_input = self.embedder(new_input)\n"
            "```"
        )
        assert reply_claims_own_code(claimed) is True
        neutral = "Sure:\n```python\ndef sort_list(x):\n    return sorted(x)\n```"
        assert reply_claims_own_code(neutral) is False

    def test_a_citation_does_not_outlive_the_reply_it_described(self):
        """Otherwise "where's that from?" answers with a real, unrelated path."""
        from core.self.source_excerpt import (
            forget_shown_excerpt,
            last_shown_excerpt,
            remember_shown_excerpt,
        )

        forget_shown_excerpt()
        remember_shown_excerpt("I read this from core/mycelium.py:88 just now.")
        assert last_shown_excerpt().get("relative_path") == "core/mycelium.py"

        remember_shown_excerpt(
            "```python\ndef invented_thing(self):\n    return 1\n```"
        )
        assert last_shown_excerpt() == {}

        remember_shown_excerpt("I read this from core/mycelium.py:88 just now.")
        assert last_shown_excerpt().get("relative_path") == "core/mycelium.py"
        remember_shown_excerpt("That was dry humor, yes.")
        assert last_shown_excerpt() == {}
        forget_shown_excerpt()


class TestAskingToBeShownIsACueNotAPhraseList:
    """"Can you read your own source?" must not read as "no".

    SOURCE_SHOW_MARKERS was a literal phrase list — "show me", "let me see",
    "display", "paste". Asking her to READ her source, to OPEN it, or which
    PART of it she finds interesting matched none of them, so a request she can
    satisfy classified as not-a-request and she denied a capability she has.

    That is the same failure shape that hit the screen-observation router twice
    in a fortnight. A phrase list is always one phrasing behind.
    """

    @staticmethod
    def _asks(message):
        from core.utils.own_source_intent import asks_for_own_source

        return asks_for_own_source(message)

    @pytest.mark.parametrize(
        "message",
        [
            "show me your source code",
            "can you read your own source?",
            "what part of your code do you find interesting?",
            "open your implementation",
            "pull up your codebase",
            "walk me through your architecture",
        ],
    )
    def test_ways_of_asking_to_be_shown(self, message):
        assert self._asks(message), message

    @pytest.mark.parametrize(
        "message",
        [
            # A question ABOUT her source, not a request to see it.
            "what language are you written in?",
            # Names another subject; answering with a piece of Aura would be
            # its own kind of made-up answer.
            "show me the actual code for numpy",
            # Nothing to do with source at all — this one reached the
            # source-excerpt floor live and got a code pitch as its answer.
            "Can you still reason through the desktop path?",
            "how are you feeling today?",
        ],
    )
    def test_still_not_a_request_for_her_source(self, message):
        assert not self._asks(message), message


def test_a_question_that_merely_touches_her_internals_keeps_its_answer():
    """The wide gate decided to CHECK; it must not decide to REPLACE.

    Live regression 2026-08-04: "Can you still reason through the desktop
    path?" scored as a source question, and a correct answer about the
    reasoning lane was swapped for a code excerpt nobody had asked for.
    """
    from core.utils.own_source_intent import (
        asks_for_own_source,
        reply_denies_showing_source,
    )

    question = "Can you still reason through the desktop path?"
    answer = (
        "Yes. I am still reasoning through the desktop CognitiveEngine path, "
        "and I am keeping the answer on this live turn instead of switching lanes."
    )
    assert not asks_for_own_source(question)
    assert not reply_denies_showing_source(answer)


def test_a_false_capability_denial_still_earns_a_real_excerpt():
    from core.utils.own_source_intent import reply_denies_showing_source

    assert reply_denies_showing_source("I can't show you code files directly.")
    assert reply_denies_showing_source("I'm not able to share my source with you.")
    assert reply_denies_showing_source("I cannot open my own implementation files.")


def test_an_ordinary_refusal_about_something_else_is_not_a_source_denial():
    from core.utils.own_source_intent import reply_denies_showing_source

    assert not reply_denies_showing_source("I can't show you Bryan's private files.")
    assert not reply_denies_showing_source("I won't show you how to bypass that.")
