"""Draw the MDATK application icon (PNG for Linux, ICO for Windows) with Pillow.

The source tree has no bitmap icon (the desktop file uses the theme icon
``applications-science``); the stand-alone builds need one, so it is drawn
here at build time: an RNA double helix on a dark rounded square.

    python packaging/make_icon.py OUT_DIR      # writes mdatk.png (256 px) and mdatk.ico
"""

from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw

SIZE = 1024  # drawn large, scaled down (anti-aliasing)
BG = (28, 45, 72, 255)
STRAND_A = (255, 170, 60, 255)
STRAND_B = (90, 200, 240, 255)
RUNG = (220, 225, 235, 255)


def draw() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = SIZE // 32
    d.rounded_rectangle((pad, pad, SIZE - pad, SIZE - pad), radius=SIZE // 6, fill=BG)
    top, bottom = SIZE * 0.14, SIZE * 0.86
    cx, amp, turns = SIZE / 2, SIZE * 0.25, 1.5
    n = 400

    def x_at(t: float, phase: float) -> float:
        return cx + amp * math.sin(2 * math.pi * turns * t + phase)

    ys = [top + (bottom - top) * k / n for k in range(n + 1)]
    for k in range(20, n, 40):  # base-pair rungs
        t = k / n
        d.line((x_at(t, 0), ys[k], x_at(t, math.pi), ys[k]), fill=RUNG, width=SIZE // 36)
    r = SIZE / 36  # strands: overlapping discs give a smooth, round-capped tube
    for phase, color in ((math.pi, STRAND_B), (0.0, STRAND_A)):
        for k in range(n + 1):
            x, y = x_at(k / n, phase), ys[k]
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return img


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "."
    os.makedirs(out, exist_ok=True)
    img = draw()
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(out, "mdatk.png"))
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(out, "mdatk.ico"),
                                                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                                       (128, 128), (256, 256)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
