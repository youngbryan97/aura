from __future__ import annotations


def test_shared_desktop_objective_detector_covers_general_document_tasks() -> None:
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert looks_like_desktop_objective(
        "Open a tab for Google Docs and start typing a coherent essay about climate adaptation."
    )
    assert looks_like_desktop_objective(
        "Could you open my notes app, write a timestamped summary, and save it as a PDF?"
    )
    assert looks_like_desktop_objective(
        "Open a browser window, search for climate news, and show me the articles."
    )
    assert looks_like_desktop_objective(
        "Create a local file with the draft and save it on my desktop."
    )
    assert looks_like_desktop_objective(
        "Use my computer to resize the current browser window and arrange it on the left side of the screen."
    )


def test_present_desktop_requests_use_the_shared_language_substrate() -> None:
    """Imperative and polite syntax must not require different action routes."""
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective
    from interface.routes.chat import _desktop_objective_self_sufficient_without_cognitive_text

    requests = (
        "Find a suitable image of a blue whale online and make it my desktop background.",
        "Please find a nebula image and use it as my wallpaper.",
        "Could you find a redwood photo and set it as the desktop background?",
    )
    for request in requests:
        assert looks_like_desktop_objective(request), request
        assert _desktop_objective_self_sufficient_without_cognitive_text(request), request

    assert not looks_like_desktop_objective(
        "Why would someone find a nebula image and use it as a desktop background?"
    )


def test_capability_questions_do_not_intercept_concrete_external_requests() -> None:
    from core.runtime.skill_task_bridge import (
        looks_like_capability_inventory_dialogue_request,
    )

    for request in (
        "Can you change my desktop background to a blue whale?",
        "Could you find a redwood photo and set it as the desktop background?",
        "Would you open Notes and copy this text into it?",
    ):
        assert not looks_like_capability_inventory_dialogue_request(request), request

    for question in (
        "Do you have tools for changing the desktop background?",
        "Could you tell me whether you can change my desktop background?",
        "What desktop capabilities do you have?",
    ):
        assert looks_like_capability_inventory_dialogue_request(question), question


def test_shared_desktop_objective_detector_rejects_explanation_only_requests() -> None:
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert not looks_like_desktop_objective("Can you explain Docker Compose documentation?")
    assert not looks_like_desktop_objective("What is a browser tab?")
    assert not looks_like_desktop_objective("How would you open Notes if you had to?")


def test_chat_and_voice_use_the_shared_desktop_objective_detector() -> None:
    from core.voice.voice_bridge import VoiceConversationBridge
    from interface.routes.chat import _looks_like_desktop_objective

    prompts = [
        "Open a tab for Google Docs and start typing a coherent essay about climate adaptation.",
        "Could you open my notes app and save a PDF?",
        "Can you explain Docker Compose documentation?",
    ]

    for prompt in prompts:
        assert _looks_like_desktop_objective(prompt) is VoiceConversationBridge._looks_like_desktop_objective(prompt)


def test_screen_observation_requests_route_to_desktop_body() -> None:
    """Asking Aura to read the screen needs the desktop body
    (read_screen_text) even though it carries no action+surface verb pair —
    'what's on my screen' used to silently do nothing."""
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert looks_like_desktop_objective("Read my screen and tell me what text you see.")
    assert looks_like_desktop_objective("What's on my screen right now?")
    assert looks_like_desktop_objective("What do you see on my screen?")
    assert looks_like_desktop_objective("Look at the screen and describe it.")
    assert looks_like_desktop_objective("Take a screenshot.")
    assert looks_like_desktop_objective("Read the text on my screen, word for word.")

    # Must not over-trigger on unrelated 'screen' / 'read' mentions.
    assert not looks_like_desktop_objective("Read me a poem.")
    assert not looks_like_desktop_objective("I watched a movie on the big screen last night.")


def test_everyday_english_app_names_do_not_route_a_turn_to_desktop_control():
    """"Remember the word LANTERN" is not Microsoft Word.

    The surface terms matched on bare word boundaries, so an action phrase from
    one sentence plus an everyday noun from another classified the whole turn as
    desktop control. Measured live: "Remember the word LANTERN ... show me the
    real output. Run a Python snippet that prints the PID and CPU cores" was
    dispatched to desktop_task -> os_automation, which correctly refused for
    lack of an observable acceptance contract — so a request the sandbox could
    have answered came back as a failure.
    """
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    not_desktop = (
        # The exact live failure.
        "Aura, it's Bryan. Remember the word LANTERN — I'll ask for it later. Now "
        "something concrete: actually execute something on this machine right now and "
        "show me the real output. Run a Python snippet that prints the current "
        "process's PID and how many CPU cores this host has.",
        "Show me your reasoning — in other words, walk me through it.",
        "Can you repeat that word for word?",
        "Show me three pages of thinking on this.",
        "Show me how you'd drive the point home.",
    )
    for message in not_desktop:
        assert looks_like_desktop_objective(message) is False, (
            f"everyday English must not become a desktop objective: {message[:60]!r}"
        )


def test_real_app_references_still_route_to_desktop_control():
    """The fix must not cost any genuine desktop objective."""
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    desktop = (
        "Open Microsoft Word and write a paragraph about latency.",
        "Create a Word document with two bullet points and save it.",
        "Write this up in Word and export it as a PDF.",
        "Open Apple Pages and start a new file.",
        "Save that file to Google Drive.",
        "Open Chrome and search for climate news.",
        "Open Notes and write down the anchor number.",
        "Read my screen and tell me what's there.",
    )
    for message in desktop:
        assert looks_like_desktop_objective(message) is True, (
            f"a real desktop objective was lost: {message[:60]!r}"
        )


def test_screen_questions_survive_missing_apostrophes_and_plain_nouns():
    """LIVE 2026-08-10: the cue class was one phrasing short, for the third time.

    "whats on my screen right now" missed because \\bwhat\\b cannot match inside
    "whats" — one absent punctuation mark was the whole difference between
    looking and not. "which window is in front" missed because a window was not
    a screen surface.
    """
    from core.runtime.desktop_objective_intent import _SCREEN_OBSERVATION_RE

    for asked in (
        "take a look at my screen — which application is frontmost?",
        "what is on my screen",
        "whats on my screen right now",
        "which window is in front",
        "whats up on the display",
        "read my screen",
        "tell me what you see on my screen currently?",
        "what windows do i have open",
    ):
        assert _SCREEN_OBSERVATION_RE.search(asked), asked


def test_talking_about_a_screen_is_not_asking_to_read_one():
    """A capture that fires on a remark is worse than one that misses a request."""
    from core.runtime.desktop_objective_intent import _SCREEN_OBSERVATION_RE

    for remark in (
        "close the window",
        "open a new window",
        "my screen protector is cracked",
        "i bought a new monitor yesterday",
        "what is consciousness",
    ):
        assert _SCREEN_OBSERVATION_RE.search(remark) is None, remark
