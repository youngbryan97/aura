"""Launcher contracts: the neuron mark, and a blocked boot that says so.

Two things the launcher must keep doing:

1. The boot mark is a NEURON (soma / dendrites / myelinated axon / travelling
   spikes) in a retro-arcade idiom — square "pixel" nodes and CRT scanlines —
   not the old orbital atom. It must be the neuron on EVERY surface that shows
   a mark: the native launcher, the web splash, and the app icon. A single
   surface left on the atom is the whole point of these tests.
2. When a start is positively refused because another runtime holds the
   instance lock, the window shows THAT, instead of spinning on
   "Aura is waking up… waiting for boot health".
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWIFT = (PROJECT_ROOT / "scripts" / "AuraLauncher.swift").read_text(encoding="utf-8")
SPLASH = (PROJECT_ROOT / "interface" / "static" / "index.html").read_text(encoding="utf-8")
SPLASH_CSS = (PROJECT_ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")
ICON_GEN = (PROJECT_ROOT / "scripts" / "build_launcher_icon.py").read_text(encoding="utf-8")


# ── the neuron mark ────────────────────────────────────────────────────────


def test_mark_is_a_neuron_not_an_atom():
    assert "Aura's neuron mark" in SWIFT
    for part in ("soma", "dendrite", "axon", "myelin", "bouton"):
        assert part.lower() in SWIFT.lower(), f"neuron anatomy missing: {part}"


def test_orbital_atom_is_gone():
    """The old mark drove "electrons" around ellipses; none of that remains.

    The check used to ban the substring "electron" anywhere in the file. It is
    also the name of a UI framework, and a comment explaining why WKWebView
    does not honour Electron's `-webkit-app-region: drag` failed a test about
    a drawing. Bans the mark's own vocabulary instead.
    """
    assert "Aura's orbital mark" not in SWIFT
    assert 'forKey: "orbit"' not in SWIFT
    lowered = SWIFT.lower()
    for token in ("electronlayer", "electrons", "electronpath", "orbitlayer"):
        assert token not in lowered, token
    # And no bare "electron" outside the one framework mention.
    framework_mentions = lowered.count("electron property") + lowered.count(
        "electron's"
    )
    assert lowered.count("electron") == framework_mentions, (
        "an electron survived that is not the UI framework"
    )


def test_retro_arcade_idiom_is_present():
    # Square pixel nodes, stepped (discrete) animation, and CRT scanlines are
    # what make it read as a sprite rather than a diagram.
    assert "func pixel(" in SWIFT, "square pixel nodes are the arcade vocabulary"
    assert "calculationMode = .discrete" in SWIFT, "stepped motion, not smooth glide"
    assert "scanline" in SWIFT.lower()
    assert "lineDashPattern" in SWIFT, "myelin segments"


def test_spikes_ride_real_paths():
    # A spike must be bound to its fibre's path, not approximated with offsets.
    assert "CAKeyframeAnimation(keyPath: \"position\")" in SWIFT
    assert 'forKey: "spike"' in SWIFT


def test_mark_keeps_its_public_shape():
    # Call sites construct it by diameter; renaming/reshaping would break them.
    assert "private final class AuraSigilView: NSView" in SWIFT
    assert "init(diameter: CGFloat)" in SWIFT


# ── the web splash carries the same mark ───────────────────────────────────


def test_web_splash_is_a_neuron_not_an_atom():
    # This is the screen the user actually sees while Aura boots in the app.
    assert "Aura's neuron mark" in SPLASH
    for part in ("soma", "dendrite", "axon", "myelin", "bouton", "nucleus"):
        assert part.lower() in SPLASH.lower(), f"neuron anatomy missing: {part}"


def test_web_splash_has_no_orbital_leftovers():
    for ghost in ("sigil-orbit", "sigil-electron", "sigil-core", "sigil-pulse"):
        assert ghost not in SPLASH, f"atom leftover in the splash markup: {ghost}"
        assert ghost not in SPLASH_CSS, f"atom leftover in the splash CSS: {ghost}"
    # The old cage/pulse keyframes went with the electrons they drove.
    assert "sigilCage" not in SPLASH_CSS
    assert "sigilPulse" not in SPLASH_CSS


def test_web_splash_spikes_ride_real_fibre_paths():
    # A spike must be bound to its fibre via mpath, not positioned by hand — the
    # same discipline the electrons used, kept through the redesign.
    for fibre in ("#dend-a", "#dend-b", "#dend-c", "#dend-d", "#axon"):
        assert f'<mpath href="{fibre}"/>' in SPLASH, f"no spike bound to {fibre}"
    assert SPLASH.count("<animateMotion") == 5


def test_web_splash_keeps_the_arcade_idiom():
    # Square sprites and stepped motion are what make it read as a game object.
    assert 'calcMode="discrete"' in SPLASH, "stepped motion, not a smooth glide"
    assert 'class="neuron-soma"' in SPLASH and "polygon" in SPLASH, "a hexagon soma, not a sphere"
    assert 'stroke-dasharray' in SPLASH, "myelin segments"
    assert "<circle" not in SPLASH[SPLASH.index("Aura's neuron mark"):SPLASH.index("splash-logo\">")].replace(
        '<circle cx="100" cy="100" r="96" fill="url(#sigil-halo-fill)"/>', ""
    ), "no round dots in the mark — pixels only (the halo is the one exception)"


def test_web_splash_mark_has_no_full_bleed_overlay():
    """The container carries a drop-shadow, so a full-bleed rect glows as a BOX.

    Observed live: a 200x200 scanline rect inside .splash-sigil rendered as a
    lit rectangle around the neuron because the parent's filter picked it up.
    """
    mark = SPLASH[SPLASH.index("Aura's neuron mark"):SPLASH.index('<h1 class="splash-logo">')]
    assert 'width="200" height="200"' not in mark, "a full-bleed rect will glow as a box"
    assert "url(#scanlines)" not in mark


def test_reduced_motion_still_stills_the_mark():
    block = SPLASH_CSS[SPLASH_CSS.index("@media (prefers-reduced-motion: reduce)"):]
    block = block[:block.index("}\n}") + 3]
    for animated in (".neuron-soma", ".neuron-nucleus", ".neuron-nodes"):
        assert animated in block, f"{animated} keeps animating under reduced motion"
    assert "animateMotion" in block, "travelling spikes must stop too"


# ── the app icon carries the same mark ─────────────────────────────────────


def test_app_icon_is_the_neuron():
    assert "_draw_neuron" in ICON_GEN
    for part in ("soma", "dendrite", "axon", "myelin", "bouton"):
        assert part.lower() in ICON_GEN.lower(), f"neuron anatomy missing: {part}"


def test_app_icon_has_no_orbital_ring_or_orb():
    for ghost in ("Orbital ring", "orb_radius", "orb_draw", "Orb body"):
        assert ghost not in ICON_GEN, f"atom/orb leftover in the icon generator: {ghost}"


def test_only_one_generator_writes_the_icon():
    """A second generator writing the same path silently repaints the mark.

    scripts/generate_icon.py drew concentric rings + radiating spokes into the
    SAME aura_icon.icns; running it would have put the atom back.
    """
    generators = []
    for path in (PROJECT_ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "aura_icon.icns" not in text:
            continue
        # Consumers (bundle/install/export) merely reference the path; a
        # GENERATOR draws pixels or invokes iconutil to produce it.
        if "ImageDraw" in text or "iconutil" in text:
            generators.append(path.name)
    assert sorted(generators) == ["build_launcher_icon.py"], (
        f"competing icon generators: {sorted(generators)}"
    )


def test_icon_renders_and_is_not_blank():
    pytest = __import__("pytest")
    pytest.importorskip("PIL")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_icon_gen", PROJECT_ROOT / "scripts" / "build_launcher_icon.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 128px is the Dock size: the anatomy has to survive the downscale, so a
    # render that collapses to a smudge is a real regression.
    icon = module.build_icon(256).convert("RGBA")
    assert icon.size == (256, 256)
    px = icon.load()
    cyan = sum(
        1
        for y in range(icon.height)
        for x in range(icon.width)
        if (lambda c: c[3] > 200 and c[1] > 180 and c[2] > 160 and c[0] < 120)(px[x, y])
    )
    assert cyan > 200, f"the cyan dendrites did not render (only {cyan} px)"


# ── the blocked-boot notice ────────────────────────────────────────────────


def test_launcher_reads_the_boot_blocked_notice():
    assert "boot_blocked.json" in SWIFT
    assert "readBootBlockedNotice" in SWIFT


def test_blocked_boot_short_circuits_the_waking_up_screen():
    # The check must run BEFORE the "waiting for boot health" copy is rendered.
    pending = SWIFT.index("private func renderPendingLaunch")
    body = SWIFT[pending:pending + 900]
    assert "readBootBlockedNotice()" in body
    assert body.index("readBootBlockedNotice()") < body.index("Waiting for Aura to publish boot health")


def test_blocked_screen_shows_reason_and_remedy():
    start = SWIFT.index("private func renderBootBlocked")
    body = SWIFT[start:start + 700]
    assert "Another Aura is already running" in body
    assert "notice.reason" in body and "notice.remedy" in body
    assert ".rose" in body, "an instance conflict is a problem state, not progress"


def test_dead_holder_is_not_treated_as_a_live_blocker():
    start = SWIFT.index("private func readBootBlockedNotice")
    body = SWIFT[start:start + 1200]
    # kill(pid, 0) liveness probe: a notice about an exited process must clear.
    assert "kill(pid_t(pid), 0)" in body
