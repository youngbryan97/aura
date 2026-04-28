"""Generate Aura app icon - dark geometric brain/circuit design."""
from PIL import Image, ImageDraw, ImageFont
import math
import os

SIZE = 1024
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background: rounded dark rectangle
margin = 40
bg_color = (10, 10, 14, 255)
draw.rounded_rectangle(
    [margin, margin, SIZE - margin, SIZE - margin],
    radius=180, fill=bg_color,
)

# Outer glow ring
cx, cy = SIZE // 2, SIZE // 2
for i in range(60, 0, -1):
    alpha = int(2.5 * i)
    radius = 340 + i
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=(0, 180, 255, alpha), width=1,
    )

# Inner core rings
for ring_r, a, w in [(280, 120, 2), (220, 80, 2), (160, 50, 1)]:
    draw.ellipse(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        outline=(0, 200, 255, a), width=w,
    )

# Neural circuit lines radiating from center
num_lines = 12
for i in range(num_lines):
    angle = (2 * math.pi / num_lines) * i
    inner_r = 90
    outer_r = 300
    x1 = cx + inner_r * math.cos(angle)
    y1 = cy + inner_r * math.sin(angle)
    x2 = cx + outer_r * math.cos(angle)
    y2 = cy + outer_r * math.sin(angle)
    draw.line([(x1, y1), (x2, y2)], fill=(0, 180, 255, 60), width=2)
    for frac in [0.4, 0.7]:
        nx = cx + (inner_r + (outer_r - inner_r) * frac) * math.cos(angle)
        ny = cy + (inner_r + (outer_r - inner_r) * frac) * math.sin(angle)
        node_r = 6
        draw.ellipse(
            [nx - node_r, ny - node_r, nx + node_r, ny + node_r],
            fill=(0, 220, 255, 180),
        )

# Central bright core
for cr in range(100, 0, -1):
    frac = cr / 100.0
    alpha = int(255 * (1 - frac) ** 2)
    color = (int(100 * (1 - frac)), int(180 + 75 * (1 - frac)), 255, alpha)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=color)

# "A" letter in the center
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 200)
except Exception:
    font = ImageFont.load_default()
bbox = draw.textbbox((0, 0), "A", font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
tx = cx - tw // 2 - bbox[0]
ty = cy - th // 2 - bbox[1] - 5
draw.text((tx + 3, ty + 3), "A", fill=(0, 0, 0, 180), font=font)
draw.text((tx, ty), "A", fill=(220, 250, 255, 255), font=font)

# Save PNG
png_path = os.path.join(os.getcwd(), "aura_icon.png")
img.save(png_path, "PNG")
print(f"Saved PNG: {png_path}")

# Generate .icns for macOS
icns_path = os.path.join(os.getcwd(), "aura_icon.icns")
iconset_dir = "/tmp/aura_icon.iconset"
os.makedirs(iconset_dir, exist_ok=True)

for s in [16, 32, 64, 128, 256, 512, 1024]:
    resized = img.resize((s, s), Image.LANCZOS)
    resized.save(os.path.join(iconset_dir, f"icon_{s}x{s}.png"))
    if s <= 512:
        retina = img.resize((s * 2, s * 2), Image.LANCZOS)
        retina.save(os.path.join(iconset_dir, f"icon_{s}x{s}@2x.png"))

import subprocess
subprocess.run(["iconutil", "-c", "icns", "-o", icns_path, iconset_dir],
               capture_output=True, timeout=30)
if os.path.exists(icns_path):
    print(f"Saved ICNS: {icns_path}")
else:
    print("iconutil failed - will use PNG fallback")

print("Icon generation complete.")
