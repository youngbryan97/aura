"""A private browser window on screen refuses a screen read, whatever the window server calls it.

LIVE 14 November 2026: a Chrome incognito window on example.com was "Example Domain" to CoreGraphics and "Example
Domain - Google Chrome (Incognito)" to accessibility. Her resident bridge and her window inventory both admitted a
full-screen read with it in front; asked "What is on my screen right now?", she named it and read the screen around it.
Her ambient sense, which reads accessibility titles, had refused the same window.
"""
from __future__ import annotations

from core.security import screen_capture_policy as policy

_INCOGNITO = "Example Domain - Google Chrome (Incognito)"
_FRAME = {"X": 0, "Y": 70, "Width": 1708, "Height": 1013}


def _window(owner, title, pid, bounds=_FRAME):
    return {"kCGWindowLayer": 0, "kCGWindowOwnerName": owner, "kCGWindowName": title, "kCGWindowOwnerPID": pid,
            "kCGWindowBounds": bounds}


def _titles(*given):
    return lambda _app: list(given)


def test_an_incognito_window_in_front_is_private_though_the_window_server_names_only_its_page():
    windows = [_window("Google Chrome", "Example Domain", 11), _window("Notes", "Notes", 12)]
    refused = policy._a_visible_browser_window_is_private(
        windows, foreground_pid=11, titles_of=_titles((_INCOGNITO, (0, 70, 1708, 1013))))
    assert refused is not None and refused.reason is policy.ScreenCaptureDenial.PRIVATE_FOREGROUND


def test_one_behind_another_app_is_private_and_visible():
    windows = [_window("Notes", "Notes", 12), _window("Google Chrome", "Example Domain", 11)]
    refused = policy._a_visible_browser_window_is_private(
        windows, foreground_pid=12, titles_of=_titles((_INCOGNITO, (0, 70, 1708, 1013))))
    assert refused is not None and refused.reason is policy.ScreenCaptureDenial.PRIVATE_VISIBLE


def test_a_private_window_off_screen_does_not_refuse_a_read_of_what_is_on_it():
    windows = [_window("Google Chrome", "Inbox", 11, {"X": 25, "Y": 33, "Width": 1708, "Height": 1013})]
    titled = _titles(("Inbox - Google Chrome - Bryan", (25, 33, 1708, 1013)), (_INCOGNITO, (0, 70, 1708, 1013)))
    assert policy._a_visible_browser_window_is_private(windows, foreground_pid=11, titles_of=titled) is None


def test_a_browser_whose_titles_cannot_be_read_refuses_the_read():
    windows = [_window("Google Chrome", "Example Domain", 11)]
    refused = policy._a_visible_browser_window_is_private(windows, foreground_pid=11, titles_of=lambda _app: None)
    assert refused is not None and refused.reason is policy.ScreenCaptureDenial.BROWSER_TITLE_UNKNOWN


def test_no_browser_on_screen_asks_nothing():
    asked = []
    assert policy._a_visible_browser_window_is_private(
        [_window("Notes", "Notes", 12)], foreground_pid=12, titles_of=lambda app: asked.append(app)) is None
    assert asked == []


def test_one_window_of_a_browser_is_judged_by_the_title_its_browser_gives_it():
    titled = _titles((_INCOGNITO, (0, 70, 1708, 1013)), ("(429) Home / X - Google Chrome - Bryan", (0, 48, 1708, 1013)))
    refused = policy._this_browser_window_is_private("Google Chrome", "Example Domain", titles_of=titled)
    assert refused is not None and refused.reason is policy.ScreenCaptureDenial.PRIVATE_FOREGROUND
    assert policy._this_browser_window_is_private("Google Chrome", "(429) Home / X", titles_of=titled) is None
    assert policy._this_browser_window_is_private("Notes", "Notes", titles_of=titled) is None


def test_an_allowed_whole_screen_read_is_refused_when_a_visible_browser_window_is_private(monkeypatch):
    monkeypatch.setattr(policy, "screen_allowed", lambda: True)
    monkeypatch.setattr(policy, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    monkeypatch.setattr(policy, "_resident_bridge_capture_admission",
                        lambda: policy.ScreenCaptureAdmission(allowed=True, context_known=True, authority="resident_bridge"))
    monkeypatch.setattr(policy, "_visible_windows_and_foreground",
                        lambda: ([_window("Google Chrome", "Example Domain", 11)], 11))
    monkeypatch.setattr(policy, "_accessibility_window_titles", _titles((_INCOGNITO, (0, 70, 1708, 1013))))
    admission = policy.evaluate_screen_capture_admission()
    assert not admission.allowed and admission.reason is policy.ScreenCaptureDenial.PRIVATE_FOREGROUND
