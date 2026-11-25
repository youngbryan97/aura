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
import os
import plistlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

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
    _a_build_step(["swiftc", "-O", "-o", str(built), str(_SHELL_SOURCE), "-framework", "AppKit", "-framework", "WebKit"],
                  timeout=600, what="swiftc")
    return built


def _a_build_step(argv: list[str], *, timeout: float, what: str) -> subprocess.CompletedProcess:
    """One step of building her app, through the subprocess gateway in a governed scope: it writes what it builds
    (Master Todo Q06). Raises CalledProcessError when the step fails, as before."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with local_internal_governed_scope(f"as_a_mac_app.{what}", domain="tool_execution", receipt_prefix="mac-app",
                                       constraints={"artifact": "aura.mac_app.v1", "operation": what}):
        return get_subprocess_gateway().run(argv, timeout=timeout, check=True, source=f"as_a_mac_app.{what}",
                                            accelerator_capability="none")


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


def _an_icon(name: str, accent: str, out: Path, picture: Path | None = None) -> Path | None:
    """An icon, rounded as Mac icons are: the picture there is of the program, or its first letter on its own colour."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        logger.info("The app keeps the default icon: Pillow could not be imported (%s).", exc)
        return None
    colour = accent if re.fullmatch(r"#[0-9a-fA-F]{6}", accent or "") else "#2b579a"
    size = 1024
    drawn = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(drawn)
    inset, side = 100, 824
    draw.rounded_rectangle((inset, inset, size - inset, size - inset), radius=185, fill=colour)
    if picture is not None and _laid_over(drawn, picture, inset, side):
        return _as_icns(drawn, out)
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
    return _as_icns(drawn, out)


def _laid_over(drawn: Any, picture: Path, inset: int, side: int) -> bool:
    """Lay a picture over an icon's ground, cut to its rounding, with the light along its top that Mac icons have."""
    from PIL import Image, ImageDraw

    try:
        with Image.open(picture) as opened:
            image = opened.convert("RGB")
    except OSError as why:
        logger.info("the picture could not be read: %s", why)
        return False
    short = min(image.size)
    left, top = (image.width - short) // 2, (image.height - short) // 2
    image = image.crop((left, top, left + short, top + short)).resize((side, side), Image.Resampling.LANCZOS)
    mask = Image.new("L", (side * 2, side * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, side * 2 - 1, side * 2 - 1), radius=370, fill=255)
    mask = mask.resize((side, side), Image.Resampling.LANCZOS)
    light = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    ImageDraw.Draw(light).rounded_rectangle((1, 1, side - 2, side - 2), radius=184, outline=(255, 255, 255, 60), width=4)
    image = Image.alpha_composite(image.convert("RGBA"), light)
    drawn.paste(image, (inset, inset), mask)
    return True


def _as_icns(drawn: Any, out: Path) -> Path | None:
    """A picture of 1024 by 1024, as the file a Mac keeps an application's icon in."""
    from PIL import Image

    iconset = out.with_suffix(".iconset")
    iconset.mkdir(parents=True, exist_ok=True)
    for points in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = points * scale
            suffix = "" if scale == 1 else "@2x"
            drawn.resize((pixels, pixels), Image.Resampling.LANCZOS).save(iconset / f"icon_{points}x{points}{suffix}.png")
    try:
        _a_build_step(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], timeout=60, what="iconutil")
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
    # not a failure: an app with no readable identifier is not taken as hers
    except (OSError, plistlib.InvalidFileException):
        theirs = ""
    if theirs == identifier or str(theirs).startswith(_HERS):
        return wanted
    return folder / f"{name} (Aura).app"


def as_a_mac_app(page: Path, name: str, *, accent: str = "", version: str = "1.0", where: Path | None = None, picture: Path | None = None) -> Path:
    """``page`` (a program she built) as ``name``.app, installed and signed, with ``picture`` for its icon where there is one; its path."""
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
        icon = _an_icon(name, accent, app / "Contents" / "Resources" / "AppIcon.icns", picture)
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
        _a_build_step(["codesign", "--force", "--deep", "--sign", "-", str(app)], timeout=120, what="codesign")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(app, target, symlinks=True)
        os.utime(target)  # so that its icon is read again
    logger.info("made %s", target)
    return target
