"""
Pre-render maroon checkbox / radio indicator icons for the Qt stylesheet.

Qt stops drawing its native check/dot as soon as an ::indicator is styled via
QSS, so to keep a real check mark (and radio dot) in the app's maroon accent we
draw the four states here and reference them from views/styles.py with
`image: url(...)`. Icons have a transparent background so they read on both
light and dark themes.

Run from the project root:  python tools/render_indicators.py
Outputs:  assets/checkbox_unchecked.png, checkbox_checked.png,
          radio_unchecked.png, radio_checked.png
"""

import os

from PIL import Image, ImageDraw

ACCENT = (164, 52, 58)  # #A4343A maroon
WHITE = (255, 255, 255)

SS = 128  # supersample canvas (downscaled to FINAL)
FINAL = 32  # stored icon size in px (displayed smaller via QSS width/height)


def _save(img, out_dir, name):
    img = img.resize((FINAL, FINAL), Image.LANCZOS)
    img.save(os.path.join(out_dir, name))


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "assets")
    os.makedirs(out_dir, exist_ok=True)

    border = 10  # stroke width on the SS canvas (~2.5px final)
    box = (12, 12, SS - 12, SS - 12)

    # Checkbox unchecked: rounded-square outline.
    im = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle(box, radius=24, outline=ACCENT, width=border)
    _save(im, out_dir, "checkbox_unchecked.png")

    # Checkbox checked: filled rounded square + white check mark.
    im = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box, radius=24, fill=ACCENT)
    d.line([(34, 66), (54, 88), (96, 34)], fill=WHITE, width=14, joint="curve")
    _save(im, out_dir, "checkbox_checked.png")

    # Radio unchecked: circle outline.
    im = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse(box, outline=ACCENT, width=border)
    _save(im, out_dir, "radio_unchecked.png")

    # Radio checked: circle outline + filled maroon dot (transparent gap between).
    im = Image.new("RGBA", (SS, SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse(box, outline=ACCENT, width=border)
    d.ellipse((42, 42, SS - 42, SS - 42), fill=ACCENT)
    _save(im, out_dir, "radio_checked.png")

    print(f"Wrote 4 indicator icons ({FINAL}x{FINAL}) to {out_dir}")


if __name__ == "__main__":
    main()
