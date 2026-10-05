"""A program she built, as a Mac application: in the Applications folder, opened like any other.

A page in a browser tab is not what a person means by a program they can use.
This wraps any page she built in a native application (a_mac_app_shell.swift):
its own window and Dock icon, a menu bar with the Edit menu text editing needs,
real Open and Save dialogs, printing, and storage that lasts between launches.
The shell is compiled once, from source kept here, and every application is
that binary with its own name, icon and program beside it, signed for this
machine and installed where the person's applications are.

An application already in the folder under the same name is replaced only if
she made it; anyone else's is left alone and hers is given another name.
"""
from __future__ import annotations

import hashlib
import logging
import plistlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("Rebuilding.MacApp")

__all__ = ["as_a_mac_app", "where_applications_go"]

_SHELL_SOURCE = Path(__file__).with_name("a_mac_app_shell.swift")

#: Every application she makes says so in its identifier.
_HERS = "com.aura.built."


def _the_shell() -> Path:
    """The shell's binary, compiled from its source the first time and whenever the source changes."""
    from core.runtime.state_ownership import state_root

    source = _SHELL_SOURCE.read_bytes()
    built = state_root() / "app_shell" / hashlib.sha256(source).hexdigest()[:16] / "AppShell"
    if built.exists():
        return built
    built.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["swiftc", "-O", "-o", str(built), str(_SHELL_SOURCE), "-framework", "AppKit", "-framework", "WebKit"],
        check=True, capture_output=True, timeout=600,
    )
    return built


def where_applications_go() -> Path:
    """The Applications folder when it can be written to, else the person's own."""
    shared = Path("/Applications")
    try:
        probe = shared / f".aura-probe-{abs(hash(shared))}"
        probe.write_text("")
        probe.unlink()
        return shared
    except OSError:
        own = Path.home() / "Applications"
        own.mkdir(exist_ok=True)
        return own


def _an_icon(name: str, accent: str, out: Path) -> Path | None:
    """An icon: the program's first letter on its own colour, rounded as Mac icons are."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    colour = accent if re.fullmatch(r"#[0-9a-fA-F]{6}", accent or "") else "#2b579a"
    size = 1024
    picture = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(picture)
    inset = 100
    draw.rounded_rectangle((inset, inset, size - inset, size - inset), radius=185, fill=colour)
    letter = (re.sub(r"[^A-Za-z0-9]", "", name) or "A")[0].upper()
    font = None
    for candidate in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, 560)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    box = draw.textbbox((0, 0), letter, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text(((size - width) / 2 - box[0], (size - height) / 2 - box[1] - 20), letter, font=font, fill="white")
    iconset = out.with_suffix(".iconset")
    iconset.mkdir(parents=True, exist_ok=True)
    for points in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = points * scale
            suffix = "" if scale == 1 else "@2x"
            picture.resize((pixels, pixels), Image.Resampling.LANCZOS).save(iconset / f"icon_{points}x{points}{suffix}.png")
    try:
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as why:
        logger.info("no icon could be made: %s", why)
        return None
    finally:
        shutil.rmtree(iconset, ignore_errors=True)
    return out


def _its_place(folder: Path, name: str, identifier: str) -> Path:
    """Where the application goes: its own name, unless someone else's application has it."""
    wanted = folder / f"{name}.app"
    if not wanted.exists():
        return wanted
    try:
        theirs = plistlib.loads((wanted / "Contents" / "Info.plist").read_bytes()).get("CFBundleIdentifier", "")
    except (OSError, plistlib.InvalidFileException):
        theirs = ""
    if theirs == identifier or str(theirs).startswith(_HERS):
        return wanted
    return folder / f"{name} (Aura).app"


def as_a_mac_app(page: Path, name: str, *, accent: str = "", version: str = "1.0", where: Path | None = None) -> Path:
    """``page`` (a program she built) as ``name``.app, installed and signed; its path."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "program"
    identifier = _HERS + slug
    folder = where or where_applications_go()
    target = _its_place(folder, name, identifier)
    with tempfile.TemporaryDirectory() as scratch:
        app = Path(scratch) / f"{name}.app"
        (app / "Contents" / "MacOS").mkdir(parents=True)
        (app / "Contents" / "Resources").mkdir(parents=True)
        executable = re.sub(r"[^A-Za-z0-9]", "", name) or "Program"
        shutil.copy2(_the_shell(), app / "Contents" / "MacOS" / executable)
        shutil.copy2(page, app / "Contents" / "Resources" / "index.html")
        icon = _an_icon(name, accent, app / "Contents" / "Resources" / "AppIcon.icns")
        info = {
            "CFBundleName": name,
            "CFBundleDisplayName": name,
            "CFBundleIdentifier": identifier,
            "CFBundleExecutable": executable,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": version,
            "CFBundleVersion": version,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSHumanReadableCopyright": "Built clean-room by Aura",
            **({"CFBundleIconFile": "AppIcon"} if icon else {}),
        }
        (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True, capture_output=True, timeout=120)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(app, target, symlinks=True)
    logger.info("made %s", target)
    return target
