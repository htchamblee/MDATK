"""Program icons of the stand-alone builds (PNG for Linux, ICO for Windows) with Pillow.

MDATK: an RNA double helix on a dark rounded square, drawn here: at build time
for the builds' PNG and ICO, and with ``--assets`` into ``mdatk/assets``
(``mdatk-icon-<size>.png``, committed), the icons that ``mdatk desktop install``
puts into the user's icon theme (the desktop-entry template itself names the
theme icon ``applications-science``).

RNAdo: the icon is committed as PNGs in ``mdatk/assets`` (drawn by
``packaging/make_rnado_art.py``); its ICO is put together from those PNGs, one
image per size, so Windows shows the simplified small sizes as drawn.

    python packaging/make_icon.py OUT_DIR
        # writes mdatk.png (256 px), mdatk.ico, rnado.png (256 px) and rnado.ico
    python packaging/make_icon.py --assets
        # redraws mdatk/assets/mdatk-icon-<size>.png (after changing the drawing)
"""

from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mdatk", "assets")
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)  # 256 px is the largest size an ICO holds
ASSET_SIZES = ICO_SIZES  # mdatk/assets/mdatk-icon-<size>.png: the hicolor sizes of the menu entry

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


def rnado(out: str, assets: str = ASSETS) -> None:
    """rnado.png (256 px) and rnado.ico from the committed PNGs (the largest one is the ICO's base image,
    the others are stored as they are; sizes without a PNG are scaled from it)."""
    images = {n: Image.open(os.path.join(assets, f"rnado-icon-{n}.png")).convert("RGBA")
              for n in ICO_SIZES if os.path.isfile(os.path.join(assets, f"rnado-icon-{n}.png"))}
    if 256 not in images:
        raise SystemExit(f"{assets}/rnado-icon-256.png is missing: run packaging/make_rnado_art.py")
    base = images.pop(256)
    base.save(os.path.join(out, "rnado.png"))
    base.save(os.path.join(out, "rnado.ico"), sizes=[(n, n) for n in ICO_SIZES],
              append_images=[images[n] for n in sorted(images)])


def mdatk_assets(assets: str = ASSETS) -> None:
    """mdatk/assets/mdatk-icon-<size>.png for every size of :data:`ASSET_SIZES` (Pillow writes the same
    bytes for the same drawing, so redrawing an unchanged icon changes no file)."""
    img = draw()
    for n in ASSET_SIZES:
        img.resize((n, n), Image.LANCZOS).save(os.path.join(assets, f"mdatk-icon-{n}.png"), optimize=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--assets"]:
        mdatk_assets()
        return 0
    out = argv[0] if argv else "."
    os.makedirs(out, exist_ok=True)
    img = draw()
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(out, "mdatk.png"))
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(out, "mdatk.ico"),
                                                sizes=[(n, n) for n in ICO_SIZES])
    rnado(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
