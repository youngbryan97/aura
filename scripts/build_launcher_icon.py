from __future__ import annotations
#!/usr/bin/env python3
"""Generate Aura's launcher icon assets."""


from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "interface" / "static"


def _vertical_gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    mask = Image.linear_gradient("L").resize((size, size))
    return Image.composite(
        Image.new("RGBA", (size, size), bottom + (255,)),
        Image.new("RGBA", (size, size), top + (255,)),
        mask,
    )


def _radial_mask(size: int, blur: int = 0) -> Image.Image:
    mask = Image.radial_gradient("L").resize((size, size))
    mask = ImageChops.invert(mask)
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def _ellipse_bounds(cx: int, cy: int, radius: int) -> tuple[int, int, int, int]:
    return (cx - radius, cy - radius, cx + radius, cy + radius)


def build_icon(size: int = 1024) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    inset = int(size * 0.07)
    corner = int(size * 0.23)

    # Background plate
    bg = _vertical_gradient(size, (5, 3, 14), (18, 10, 30))
    bg_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(bg_mask).rounded_rectangle(
        (inset, inset, size - inset, size - inset),
        radius=corner,
        fill=255,
    )
    canvas = Image.composite(bg, canvas, bg_mask)

    # Vignette
    vignette = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    vignette_mask = _radial_mask(size, blur=int(size * 0.015))
    vignette.putalpha(Image.eval(vignette_mask, lambda px: int(px * 0.42)))
    canvas = Image.alpha_composite(canvas, vignette)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (inset, inset, size - inset, size - inset),
        radius=corner,
        outline=(105, 84, 164, 118),
        width=max(4, size // 170),
    )

    cx = size // 2
    cy = int(size * 0.515)
    orb_radius = int(size * 0.29)

    # Ambient glow
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(
        _ellipse_bounds(cx, cy, int(orb_radius * 1.28)),
        fill=(133, 72, 255, 108),
    )
    glow_draw.ellipse(
        _ellipse_bounds(int(cx + orb_radius * 0.18), int(cy - orb_radius * 0.26), int(orb_radius * 0.78)),
        fill=(103, 245, 255, 80),
    )
    glow_draw.ellipse(
        _ellipse_bounds(int(cx - orb_radius * 0.28), int(cy + orb_radius * 0.18), int(orb_radius * 0.72)),
        fill=(255, 91, 222, 74),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(int(size * 0.05)))
    canvas = Image.alpha_composite(canvas, glow)

    # Orb body
    orb = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    orb_draw = ImageDraw.Draw(orb)
    orb_draw.ellipse(_ellipse_bounds(cx, cy, orb_radius), fill=(82, 34, 158, 255))
    orb_draw.ellipse(
        _ellipse_bounds(cx, cy, orb_radius),
        outline=(190, 133, 255, 210),
        width=max(6, size // 128),
    )
    canvas = Image.alpha_composite(canvas, orb)

    # Inner bloom
    inner = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    inner_draw = ImageDraw.Draw(inner)
    inner_draw.ellipse(
        _ellipse_bounds(int(cx - orb_radius * 0.07), int(cy - orb_radius * 0.12), int(orb_radius * 0.76)),
        fill=(246, 215, 255, 132),
    )
    inner_draw.ellipse(
        _ellipse_bounds(int(cx + orb_radius * 0.12), int(cy - orb_radius * 0.18), int(orb_radius * 0.58)),
        fill=(118, 244, 255, 108),
    )
    inner_draw.ellipse(
        _ellipse_bounds(int(cx - orb_radius * 0.1), int(cy + orb_radius * 0.22), int(orb_radius * 0.7)),
        fill=(109, 58, 210, 122),
    )
    inner = inner.filter(ImageFilter.GaussianBlur(int(size * 0.035)))
    canvas = Image.alpha_composite(canvas, inner)

    # Highlight and shadow for depth
    accent = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    accent_draw = ImageDraw.Draw(accent)
    accent_draw.ellipse(
        (
            int(cx - orb_radius * 0.52),
            int(cy - orb_radius * 0.63),
            int(cx + orb_radius * 0.02),
            int(cy - orb_radius * 0.12),
        ),
        fill=(255, 255, 255, 118),
    )
    accent_draw.ellipse(
        (
            int(cx - orb_radius * 0.7),
            int(cy + orb_radius * 0.12),
            int(cx + orb_radius * 0.45),
            int(cy + orb_radius * 0.85),
        ),
        fill=(18, 7, 37, 138),
    )
    accent = accent.filter(ImageFilter.GaussianBlur(int(size * 0.03)))
    canvas = Image.alpha_composite(canvas, accent)

    # Orbital ring
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring)
    ring_bounds = (
        int(cx - orb_radius * 1.08),
        int(cy - orb_radius * 0.92),
        int(cx + orb_radius * 1.08),
        int(cy + orb_radius * 1.12),
    )
    ring_draw.ellipse(
        ring_bounds,
        outline=(124, 232, 255, 124),
        width=max(5, size // 180),
    )
    ring = ring.filter(ImageFilter.GaussianBlur(int(size * 0.012)))
    canvas = Image.alpha_composite(canvas, ring)

    # Small spark points
    sparks = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sparks_draw = ImageDraw.Draw(sparks)
    for px, py, radius, color in (
        (int(cx + orb_radius * 0.55), int(cy - orb_radius * 0.62), int(size * 0.014), (122, 244, 255, 255)),
        (int(cx - orb_radius * 0.66), int(cy + orb_radius * 0.18), int(size * 0.011), (210, 162, 255, 235)),
        (int(cx + orb_radius * 0.12), int(cy - orb_radius * 0.86), int(size * 0.009), (255, 220, 255, 215)),
    ):
        sparks_draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    sparks = sparks.filter(ImageFilter.GaussianBlur(int(size * 0.004)))
    canvas = Image.alpha_composite(canvas, sparks)

    return canvas


def save_png_variants(image: Image.Image) -> None:
    image.save(ROOT / "aura_icon.png", format="PNG")
    image.save(STATIC_DIR / "icon.png", format="PNG")
    image.resize((512, 512), Image.LANCZOS).save(STATIC_DIR / "icon-512.png", format="PNG")
    image.resize((192, 192), Image.LANCZOS).save(STATIC_DIR / "icon-192.png", format="PNG")


def save_icns(image: Image.Image) -> None:
    image.save(ROOT / "aura_icon.icns")


def main() -> None:
    image = build_icon()
    save_png_variants(image)
    save_icns(image)
    print(f"Updated {ROOT / 'aura_icon.png'}")
    print(f"Updated {ROOT / 'aura_icon.icns'}")
    print(f"Updated {STATIC_DIR / 'icon.png'}")
    print(f"Updated {STATIC_DIR / 'icon-512.png'}")
    print(f"Updated {STATIC_DIR / 'icon-192.png'}")


if __name__ == "__main__":
    main()
