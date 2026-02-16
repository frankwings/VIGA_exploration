"""Create side-by-side comparison: 2D input (left) | 3D render (right)."""
import sys
from PIL import Image, ImageDraw, ImageFont

input_2d = sys.argv[1]
render_3d = sys.argv[2]
output_path = sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else ""

img_2d = Image.open(input_2d).convert("RGB")
img_3d = Image.open(render_3d).convert("RGB")

# Resize to same height
h = max(img_2d.height, img_3d.height)
w2d = int(img_2d.width * h / img_2d.height)
w3d = int(img_3d.width * h / img_3d.height)
img_2d = img_2d.resize((w2d, h), Image.LANCZOS)
img_3d = img_3d.resize((w3d, h), Image.LANCZOS)

# Create canvas with labels
label_h = 40
gap = 10
canvas_w = w2d + gap + w3d
canvas_h = h + label_h
canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

# Paste images
canvas.paste(img_2d, (0, label_h))
canvas.paste(img_3d, (w2d + gap, label_h))

# Draw labels
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

draw.text((10, 8), f"2D Input: {label}", fill=(0, 0, 0), font=font)
draw.text((w2d + gap + 10, 8), f"3D Render (MoGe cam, fixed)", fill=(0, 0, 0), font=font)

# Draw separator line
draw.line([(w2d + gap//2, 0), (w2d + gap//2, canvas_h)], fill=(200, 200, 200), width=2)

canvas.save(output_path)
print(f"Saved: {output_path}")
