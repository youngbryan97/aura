"""The pictures of a program, painted by her: its icon, what stands behind a game, and the things seen in it.

What each shows is hers to decide, in the design. It is painted by her own image model (core/skills/image_gen.py), a few of
each, and which is used is measured and not judged: an icon has to be told from the rest of hers and read at the size of a
button; a thing has to come away clean from what it was painted on. Where none does, or there is no memory to paint in, the
program is drawn as it was before there were pictures, and its icon is its first letter.
"""
from __future__ import annotations

import asyncio
import base64
import colorsys
import io
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from core.making.design import Design, Picture
from core.understanding.workbench import Work

logger = logging.getLogger("Making.Pictures")

__all__ = ["as_data", "by_her", "colour_word", "icon_at", "paint", "taken_from_its_ground"]

#: Paint: (what is seen, width, height, seed) -> the file it is in, or None.
Paints = Callable[[str, int, int, int], Awaitable[Path | None]]

#: How many are painted of each kind before the best of them is used, and how big each is.
OF_EACH = {"icon": 3, "backdrop": 1, "thing": 3}
SIZE = {"icon": (512, 512), "backdrop": (896, 560), "thing": (512, 512)}
#: How big each is kept, in pixels of its longer side.
KEPT = {"backdrop": 960, "thing": 256}
#: How long one picture is waited for, in seconds: the first has the model to load.
A_PICTURE_S = 240.0

_COLOURS = {"white": (0.97, 0.97, 0.97), "black": (0.06, 0.06, 0.06), "grey": (0.5, 0.5, 0.5), "red": (0.85, 0.16, 0.16), "orange": (0.95, 0.5, 0.12),
            "amber": (0.98, 0.72, 0.2), "yellow": (0.98, 0.9, 0.25), "lime": (0.6, 0.85, 0.2), "green": (0.2, 0.65, 0.3), "mint": (0.6, 0.92, 0.75),
            "teal": (0.1, 0.55, 0.55), "cyan": (0.2, 0.8, 0.9), "sky blue": (0.5, 0.75, 0.95), "blue": (0.2, 0.4, 0.9), "navy": (0.08, 0.12, 0.35),
            "indigo": (0.3, 0.25, 0.7), "violet": (0.55, 0.35, 0.85), "purple": (0.45, 0.15, 0.55), "magenta": (0.85, 0.2, 0.65), "pink": (0.97, 0.6, 0.75),
            "brown": (0.45, 0.28, 0.16), "cream": (0.98, 0.94, 0.84), "gold": (0.85, 0.65, 0.2), "plum": (0.35, 0.12, 0.3)}


def colour_word(colour: str) -> str:
    """What a colour is called."""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(colour).strip())
    if not m:
        return ""
    rgb = tuple(int(m.group(1)[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return min(_COLOURS, key=lambda name: sum((a - b) ** 2 for a, b in zip(rgb, _COLOURS[name], strict=True)))


def _seen(picture: Picture, design: Design) -> str:
    """What is asked of her painter for a picture: what it shows, and the manner of its kind, in the design's own colours."""
    colours = list(dict.fromkeys(w for w in map(colour_word, [design.look.accent, *design.look.palette[:3]]) if w))
    in_them = f", {' and '.join(colours[:2])}" if colours else ""
    if picture.kind == "icon":
        return f"app icon of {picture.shows}, one bold simple subject filling the middle, clean rounded shapes, soft studio light, plain {' and '.join(colours[:2]) + ' ' if colours else ''}background, no text"
    if picture.kind == "backdrop":
        return f"{picture.shows}, wide game background, painted, layered depth, nothing in the foreground{in_them}, no text, no people"
    return f"{picture.shows}, single game sprite, whole, centered, facing right, clean outline, isolated on a plain flat white background, no shadow, no text"


async def by_her(seen: str, width: int, height: int, seed: int) -> Path | None:
    """Painted by her own image model, through the door every picture of hers goes through."""
    from core.capability_engine import execute_tool

    result = await execute_tool(
        "image_gen",
        {"prompt": seen, "enhance": False, "width": width, "height": height, "steps": 4, "guidance_scale": 0.0, "seed": seed},
        context={"origin": "work_asked_for", "source": "making.pictures", "route": "making.pictures", "objective": "Paint a picture of the program they asked her to make",
                 "desktop_execution_contract": True, "foreground_request": True, "user_requested_action": True, "user_explicit_action_request": True,
                 "user_explicitly_authorized": True, "local_desktop_action": True, "verification_required": True,
                 "predicted_outcome": "A picture is in her folder of pictures, to be put in the program."})
    path = Path(str(result.get("path") or "")) if isinstance(result, dict) and result.get("ok") else None
    if path is None:
        logger.info("No picture came: %s", (result or {}).get("error") if isinstance(result, dict) else result)
    return path


async def paint(design: Design, folder: Path, work: Work, *, paints: Paints = by_her, others: Iterable[Path] = ()) -> list[str]:
    """Paint what the design has pictures of, into its folder; the names of those there are now."""
    done: list[str] = []
    for n, picture in enumerate(design.pictures):
        if await asyncio.to_thread(_file_of(folder, picture).is_file):
            done.append(picture.name)
            continue
        best: tuple[float, Any] | None = None
        for k in range(OF_EACH[picture.kind]):
            width, height = SIZE[picture.kind]
            try:
                path = await asyncio.wait_for(paints(_seen(picture, design), width, height, 1 + 97 * n + 13 * k), timeout=A_PICTURE_S)
            except TimeoutError:
                path = None
            if path is None:
                break
            try:
                got = await asyncio.to_thread(_judged, path, picture.kind, list(others))
            except (OSError, ValueError) as exc:
                logger.info("A picture could not be read: %s", exc)
                continue
            if got is not None and (best is None or got[0] > best[0]):
                best = got
        if best is None:
            continue
        await asyncio.to_thread(_kept, best[1], picture, folder)
        done.append(picture.name)
        work.painted("icon" if picture.kind == "icon" else picture.name, picture.shows)
    return done


def _file_of(folder: Path, picture: Picture) -> Path:
    return folder / "icon.png" if picture.kind == "icon" else folder / "pictures" / f"{_safe(picture.name)}.png"


def _safe(name: str) -> str:
    return re.sub(r"[^\w-]+", "-", name)[:60]


def icon_at(folder: Path) -> Path | None:
    """The icon she painted for a program, where she did."""
    path = Path(folder) / "icon.png"
    return path if path.is_file() else None


# ── which of them ─────────────────────────────────────────────────────


def _judged(path: Path, kind: str, others: list[Path]) -> tuple[float, Any] | None:
    """How good a picture is for what it is for, and the picture as it would be kept; None where it will not do."""
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if kind == "icon":
        return _reads_small(image) * _stands_apart(image, others), image
    if kind == "backdrop":
        return 1.0, image
    return taken_from_its_ground(image)


def _reads_small(image: Any) -> float:
    """How well a picture can be told at the size of a button: how much of what is light and dark in it is left, and how much of
    that is in the middle."""
    import numpy as np

    small = np.asarray(image.resize((32, 32)).convert("L"), dtype=np.float32) / 255.0
    middle = small[6:26, 6:26]
    edge = np.concatenate([small[:3].ravel(), small[-3:].ravel(), small[:, :3].ravel(), small[:, -3:].ravel()])
    apart = float(abs(middle.mean() - edge.mean()) + middle.std())
    return apart / (0.08 + float(edge.std()))


def _stands_apart(image: Any, others: list[Path]) -> float:
    """How unlike the icons she has made already a picture is, by its colours: one, where there are none."""
    import numpy as np
    from PIL import Image

    def colours(one: Any) -> Any:
        hist, _edges = np.histogramdd(np.asarray(one.resize((48, 48)).convert("RGB"), dtype=np.float32).reshape(-1, 3) / 256.0, bins=(4, 4, 4), range=((0, 1),) * 3)
        return hist.ravel() / hist.sum()

    mine, least = colours(image), 1.0
    for path in others:
        try:
            with Image.open(path) as opened:
                least = min(least, 1.0 - float(np.minimum(mine, colours(opened)).sum()))
        except (OSError, ValueError):
            continue  # not a failure: an icon that is not there is not one to be told from
    return 0.35 + least


def taken_from_its_ground(image: Any) -> tuple[float, Any] | None:
    """A thing, cut from the plain ground it was painted on: what is like the edge of the picture and joined to it is ground.
    None where the ground is not plain, or where nothing, or nearly all of it, is left."""
    import numpy as np
    from PIL import Image, ImageFilter
    from scipy import ndimage

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([pixels[:6].reshape(-1, 3), pixels[-6:].reshape(-1, 3), pixels[:, :6].reshape(-1, 3), pixels[:, -6:].reshape(-1, 3)])
    ground = np.median(border, axis=0)
    off = np.abs(border - ground).max(axis=1)
    if float((off > 42).mean()) > 0.12:
        return None
    like = np.abs(pixels - ground).max(axis=2) < 30
    marked, _n = ndimage.label(like)
    at_edge = set(np.unique(np.concatenate([marked[0], marked[-1], marked[:, 0], marked[:, -1]]))) - {0}
    is_ground = np.isin(marked, list(at_edge))
    thing = ndimage.binary_opening(~is_ground, iterations=1)
    parts, n = ndimage.label(thing)
    if not n:
        return None
    sizes = ndimage.sum(thing, parts, range(1, n + 1))
    # The thing, and what is near as big as it: a speck of the ground is neither.
    thing = np.isin(parts, [i + 1 for i, size in enumerate(sizes) if size >= max(sizes) * 0.06])
    share = float(thing.mean())
    if not 0.04 < share < 0.86:
        return None
    alpha = Image.fromarray((thing * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.1))
    out = image.convert("RGBA")
    out.putalpha(alpha)
    box = alpha.getbbox()
    whole = min(box[0], box[1], image.width - box[2], image.height - box[3]) > 2 if box else False
    cut = out.crop(box) if box else out
    side = max(cut.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(cut, ((side - cut.width) // 2, (side - cut.height) // 2))
    return (1.0 if whole else 0.4) + 1.0 / len(sizes), square


def _kept(image: Any, picture: Picture, folder: Path) -> None:
    from core.rebuilding.her_build_writes import governed

    if picture.kind in KEPT:
        image = image.copy()
        image.thumbnail((KEPT[picture.kind], KEPT[picture.kind]))
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    path = _file_of(folder, picture)
    governed("ensure_directory", path.parent)
    governed("write_bytes", path, out.getvalue())


# ── in the program ────────────────────────────────────────────────────


def as_data(folder: Path, design: Design) -> dict[str, str]:
    """The pictures of a program, as they are put in its page: each by its name."""
    from PIL import Image

    out: dict[str, str] = {}
    for picture in design.pictures:
        path = _file_of(folder, picture)
        if not path.is_file():
            continue
        try:
            with Image.open(path) as opened:
                image, held, how = opened.copy(), io.BytesIO(), "png"
                if picture.kind == "icon":
                    image.thumbnail((256, 256))
                if picture.kind == "thing":
                    image.save(held, format="PNG", optimize=True)
                else:
                    how = "jpeg"
                    image.convert("RGB").save(held, format="JPEG", quality=86)
        except (OSError, ValueError) as exc:
            logger.info("A picture of %s is not put in: %s", design.name, exc)
            continue
        out["icon" if picture.kind == "icon" else picture.name] = f"data:image/{how};base64,{base64.b64encode(held.getvalue()).decode()}"
    return out


def a_ground(accent: str) -> tuple[str, str]:
    """Two colours for a plain icon's ground, from a program's accent: lighter at the top, and a little round the wheel."""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(accent).strip())
    r, g, b = (int((m.group(1) if m else "4c6ef5")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    said = lambda hue, lit: "#" + "".join(f"{round(255 * c):02x}" for c in colorsys.hls_to_rgb(hue % 1.0, max(0.0, min(1.0, lit)), s))  # noqa: E731
    return said(h - 0.03, light + 0.12), said(h + 0.03, light - 0.12)
