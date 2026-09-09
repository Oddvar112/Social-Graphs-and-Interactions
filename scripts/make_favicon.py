"""
The site icon: a tiny network with one gold hub, in the site's palette.

Writes assets/favicon.svg (crisp at any size, used by modern browsers) and two
PNG fallbacks rendered from the same geometry with Pillow: assets/favicon-32.png
for older browsers and assets/apple-touch-icon.png (180 px) for iOS home screens.

Run:  python scripts/make_favicon.py
"""

import pathlib

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"

BG, HUB, WEB, RED = "#0c1122", "#ffc93f", "#5fe3ff", "#ff4b50"

# Geometry in a 64 x 64 box: a hub off centre, five satellites, one of them red.
HUB_XY, HUB_R = (27, 34), 10.5
SATELLITES = [((52, 14), 5.5, WEB), ((55, 40), 5, WEB), ((44, 56), 5, RED), ((12, 54), 5, WEB), ((10, 16), 5.5, WEB)]
EXTRA_EDGES = [((52, 14), (55, 40)), ((10, 16), (12, 54))]


def svg():
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">',
             f'<rect width="64" height="64" rx="14" fill="{BG}"/>']
    for (x, y), _, _ in SATELLITES:
        lines.append(f'<line x1="{HUB_XY[0]}" y1="{HUB_XY[1]}" x2="{x}" y2="{y}" stroke="{WEB}" stroke-width="2.6" stroke-opacity="0.75"/>')
    for (x1, y1), (x2, y2) in EXTRA_EDGES:
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{WEB}" stroke-width="2" stroke-opacity="0.45"/>')
    for (x, y), r, colour in SATELLITES:
        lines.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{colour}"/>')
    lines.append(f'<circle cx="{HUB_XY[0]}" cy="{HUB_XY[1]}" r="{HUB_R}" fill="{HUB}"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def png(size):
    S = 8                                                    # supersample, then shrink for smooth edges
    im = Image.new("RGBA", (64 * S, 64 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, 64 * S - 1, 64 * S - 1), radius=14 * S, fill=BG)

    def line(p, q, width, colour, alpha):
        c = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(c).line((p[0] * S, p[1] * S, q[0] * S, q[1] * S), fill=colour, width=int(width * S))
        c.putalpha(c.getchannel("A").point(lambda a: int(a * alpha)))
        im.alpha_composite(c)

    for (x, y), _, _ in SATELLITES:
        line(HUB_XY, (x, y), 2.6, WEB, 0.75)
    for p, q in EXTRA_EDGES:
        line(p, q, 2.0, WEB, 0.45)
    d = ImageDraw.Draw(im)
    for (x, y), r, colour in SATELLITES:
        d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=colour)
    x, y, r = HUB_XY[0], HUB_XY[1], HUB_R
    d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=HUB)
    return im.resize((size, size), Image.LANCZOS)


def main():
    (OUT / "favicon.svg").write_text(svg(), encoding="utf-8")
    png(32).save(OUT / "favicon-32.png")
    png(180).save(OUT / "apple-touch-icon.png")
    print("wrote assets/favicon.svg, assets/favicon-32.png, assets/apple-touch-icon.png")


if __name__ == "__main__":
    main()
