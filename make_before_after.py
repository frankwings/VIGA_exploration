"""Create before/after comparison: broken (left) vs fixed (right), with target at top."""
import sys
from PIL import Image, ImageDraw, ImageFont

target_path = sys.argv[1]
broken_path = sys.argv[2]
fixed_path = sys.argv[3]
output_path = sys.argv[4]
label = sys.argv[5] if len(sys.argv) > 5 else ""

img_target = Image.open(target_path).convert("RGB")
img_broken = Image.open(broken_path).convert("RGB")
img_fixed = Image.open(fixed_path).convert("RGB")

# Resize all to same height
h = 512
w_target = int(img_target.width * h / img_target.height)
w_broken = int(img_broken.width * h / img_broken.height)
w_fixed = int(img_fixed.width * h / img_fixed.height)
img_target = img_target.resize((w_target, h), Image.LANCZOS)
img_broken = img_broken.resize((w_broken, h), Image.LANCZOS)
img_fixed = img_fixed.resize((w_fixed, h), Image.LANCZOS)

# Layout: three columns
label_h = 36
gap = 6
canvas_w = w_target + gap + w_broken + gap + w_fixed
canvas_h = h + label_h
canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

# Paste
x = 0
canvas.paste(img_target, (x, label_h))
x += w_target + gap
canvas.paste(img_broken, (x, label_h))
broken_x = x
x += w_broken + gap
canvas.paste(img_fixed, (x, label_h))
fixed_x = x

# Labels
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except Exception:
    font = ImageFont.load_default()

draw.text((4, 6), f"Target: {label}", fill=(0, 0, 0), font=font)
draw.text((broken_x + 4, 6), "BEFORE (broken transforms)", fill=(200, 0, 0), font=font)
draw.text((fixed_x + 4, 6), "AFTER (fixed transforms)", fill=(0, 150, 0), font=font)

# Separators
for sep_x in [w_target + gap // 2, w_target + gap + w_broken + gap + gap // 2]:
    draw.line([(sep_x, 0), (sep_x, canvas_h)], fill=(180, 180, 180), width=2)

canvas.save(output_path)
print(f"Saved: {output_path}")
