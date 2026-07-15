"""
Pre-render the equations shown in the info/*.html help dialogs to PNG images.

The help dialogs are displayed with QMessageBox, which only supports Qt's
limited rich-text subset (no MathJax/MathML). To show properly typeset math we
render each equation once with matplotlib's mathtext (no LaTeX install needed)
and embed the resulting images via <img> tags.

Run from the project root:  python tools/render_equations.py
Outputs:  info/equations/*.png  (transparent, high-DPI)
It also prints the display width/height (device-independent px) for each image,
which the HTML <img> tags use.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Classic LaTeX (Computer Modern) look for the math.
plt.rcParams["mathtext.fontset"] = "cm"

# Render at 2x and display at half size so the images stay crisp on HiDPI screens.
# FONT_SIZE is tuned so the displayed equations sit just above the 16px body text.
FONT_SIZE = 13
DPI = 200
DISPLAY_SCALE = 0.5

# name -> mathtext string. Names are shared across tabs where the equation is identical.
EQUATIONS = {
    "wavelength": r"$\lambda = h\, /\, \sqrt{2\,m_e\,eV\,(1 + eV / (2\,m_e c^2))}$",
    "gamma": r"$\gamma(f) = -\frac{\pi}{2}\,C_s\,\lambda^3 f^4 + \pi\,d_f\,\lambda f^2 + p$",
    "fs": r"$f_s = C_c\,\sqrt{(\Delta V / V)^2 + (2\,\Delta I / I)^2 + (\Delta E / eV)^2}$",
    "env_temporal": r"$E_T(f) = \exp\!\left(-\frac{\pi^2 \lambda^2 f_s^2 f^4}{2}\right)$",
    "env_spatial": (
        r"$E_S(f) = \exp\!\left(-\left(\frac{\pi\,e_a}{\lambda}\right)^{\!2}"
        r"\left(C_s \lambda^3 f^3 + d_f \lambda f\right)^{2}\right)$"
    ),
    "env_detector": r"$E_D(f) = \mathrm{DQE}(f / \mathrm{Nyquist})\, /\, \max(\mathrm{DQE})$",
    "env_total": r"$E_{total}(f) = E_T(f)\cdot E_S(f)\cdot E_D(f)$",
    "ctf": r"$\mathrm{CTF}(f) = -\sin\!\left(\gamma(f) + \arcsin(A_c)\right)\cdot E_{total}(f)$",
    "azimuth": r"$\varphi = \arctan(f_y / f_x)$",
    "defocus_astig": (
        r"$d_f = \frac{1}{2}\left[\,d_u + d_v + (d_u - d_v)\cos\!\left(2(\varphi - "
        r"\varphi_a)\right)\right]$"
    ),
    "ctf_thickness": (
        r"$\mathrm{CTF}(f) = -\mathrm{sinc}\!\left(\pi \lambda f^2\, t / 2\right)\cdot "
        r"\sin\!\left(\gamma(f) + \arcsin(A_c)\right)$"
    ),
    "convolution": (
        r"$\mathrm{Convolved\ image} = \mathrm{FFT}^{-1}\!\left("
        r"\mathrm{FFT}(\mathrm{Image}) \cdot \mathrm{CTF}\right)$"
    ),
}


def main():
    import io

    from PIL import Image

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "info", "equations")
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1: render each equation tight-cropped, opaque white, downscaled 1:1.
    rendered = {}
    for name, tex in EQUATIONS.items():
        # Render on an OPAQUE WHITE background (matching the info dialog's page)
        # so thin strokes stay solid black rather than partial-alpha grey.
        fig = plt.figure(figsize=(0.1, 0.1), facecolor="white")
        fig.text(0, 0, tex, fontsize=FONT_SIZE, color="black")
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=DPI,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)

        # Downscale the 2x supersample to the final display size here with
        # high-quality LANCZOS. Saving at the exact display size means Qt draws
        # the image 1:1 (no runtime bilinear scaling that greys/drops hairlines).
        buf.seek(0)
        with Image.open(buf) as im:
            im = im.convert("RGB")
            w, h = im.size
            disp_w = max(1, round(w * DISPLAY_SCALE))
            disp_h = max(1, round(h * DISPLAY_SCALE))
            rendered[name] = im.resize((disp_w, disp_h), Image.LANCZOS)

    # Pass 2: paste every equation onto a canvas of uniform height (vertically
    # centered) so the info-dialog table rows are all equal and the spacing
    # between equations is even regardless of each equation's own ink height.
    row_h = max(im.height for im in rendered.values()) + 2  # small even margin
    dims = {}
    for name, im in rendered.items():
        canvas = Image.new("RGB", (im.width, row_h), "white")
        canvas.paste(im, (0, (row_h - im.height) // 2))
        canvas.save(os.path.join(out_dir, f"{name}.png"))
        dims[name] = (im.width, row_h)
        print(f"{name}.png  final={im.width}x{row_h}")

    print(f"\nUniform row height = {row_h}px")
    print("HTML img tags (relative to {IMG_DIR}):")
    for name, (w, h) in dims.items():
        print(f'<img src="{{IMG_DIR}}/{name}.png" width="{w}" height="{h}">')


if __name__ == "__main__":
    main()
