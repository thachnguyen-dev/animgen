"""Contact sheet on a checkerboard: python sheet.py out.png cols height img1 img2 ..."""
import sys

from PIL import Image

out, cols, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ims = []
for p in sys.argv[4:]:
    im = Image.open(p).convert("RGBA")
    ims.append(im.resize((max(1, im.width * h // im.height), h)))
cw = max(i.width for i in ims)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGBA", (cw * cols, h * rows))
for y in range(0, sheet.height, 16):
    for x in range(0, sheet.width, 16):
        c = 200 if (x // 16 + y // 16) % 2 else 245
        sheet.paste((c, c, c, 255), (x, y, x + 16, y + 16))
for i, im in enumerate(ims):
    sheet.alpha_composite(im, ((i % cols) * cw, (i // cols) * h))
sheet.convert("RGB").save(out)
