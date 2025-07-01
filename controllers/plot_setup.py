import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from utils.image_processing import load_and_prepare_image
from utils.frequency_helpers import compute_resampled_freqs


def setup_1d_plot(ctrl):
    """
    Call this once at app startup to create the 1D CTF figure:
      - set titles/labels
      - plot the five initial lines (Et, Es, Ed, Etotal, CTF)
      - store references to the Line2D objects on `ctrl`
    """
    fig = ctrl.ui.canvas_1d.fig
    fig.subplots_adjust(hspace=0.3, top=0.875, bottom=0.125, left=0.125, right=0.9)

    ax = ctrl.ui.canvas_1d.axes[1]
    ax.set_title(
        "1-D Contrast Transfer Function",
        fontsize=ctrl.ui.font_sizes["large"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["large"],
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Spatial Frequency (Å⁻¹)", fontsize=ctrl.ui.font_sizes["medium"])
    ax.set_ylabel("Contrast Transfer Function", fontsize=ctrl.ui.font_sizes["medium"])
    ax.tick_params(axis="both", which="major", labelsize=ctrl.ui.font_sizes["small"])

    # horizontal zero‐line
    ctrl.line_y0 = ax.axhline(y=0, color="grey", linestyle="--", alpha=0.8)

    # five curves: Et, Es, Ed, Etotal, CTF
    freqs = ctrl.freqs_1d
    ctf1d = ctrl.ctf_1d  # shorthand
    wrap_fn = lambda x: x  # initial wrap; controller will override if needed

    (line_et,) = ax.plot(
        freqs,
        wrap_fn(ctf1d.envelope.temporal(freqs)),
        label="Temporal Envelope",
        linestyle="dashed",
        linewidth=ctrl.ui.linewidth,
    )
    (line_es,) = ax.plot(
        freqs,
        wrap_fn(ctf1d.envelope.spatial_1d(freqs)),
        label="Spatial Envelope",
        linestyle="dashed",
        linewidth=ctrl.ui.linewidth,
    )
    (line_ed,) = ax.plot(
        freqs,
        wrap_fn(ctf1d.envelope.detector(freqs)),
        label="Detector Envelope",
        linestyle="dashed",
        linewidth=ctrl.ui.linewidth,
    )
    (line_te,) = ax.plot(
        freqs,
        wrap_fn(ctf1d.envelope.total_1d(freqs)),
        label="Total Envelope",
        linewidth=ctrl.ui.linewidth,
    )
    (line_dc,) = ax.plot(
        freqs,
        wrap_fn(ctf1d.ctf(freqs)),
        label="Microscope CTF",
        linewidth=ctrl.ui.linewidth,
    )
    ctrl.line_et = line_et
    ctrl.line_es = line_es
    ctrl.line_ed = line_ed
    ctrl.line_te = line_te
    ctrl.line_dc = line_dc

    # draw legend once
    ctrl.legend_1d = ax.legend(fontsize=ctrl.ui.font_sizes["medium"])


def setup_2d_plot(ctrl):
    """
    Set up the 2D‐CTF image for the “2D” tab:
      - set title, axis labels
      - call imshow with initial data
      - add colorbar
      - store the returned AxesImage on `ctrl.image_2d`
    """
    fig = ctrl.ui.canvas_2d.fig
    fig.subplots_adjust(hspace=0.3, top=0.875, bottom=0.125, left=0.125, right=0.9)

    ax = ctrl.ui.canvas_2d.axes[1]
    ax.set_title(
        "2-D Contrast Transfer Function",
        fontsize=ctrl.ui.font_sizes["large"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["large"],
    )

    # initial image: use fx_fix, fy_fix meshgrids
    ctf2d = ctrl.ctf_2d
    img = ax.imshow(
        ctf2d.ctf(ctrl.fx_fix, ctrl.fy_fix),
        extent=(-0.5, 0.5, -0.5, 0.5),
        cmap="Greys",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    ctrl.image_2d = img

    ax.set_xlabel("Spatial Frequency X (Å⁻¹)", fontsize=ctrl.ui.font_sizes["medium"])
    ax.set_ylabel("Spatial Frequency Y (Å⁻¹)", fontsize=ctrl.ui.font_sizes["medium"])
    ax.tick_params(axis="both", labelsize=ctrl.ui.font_sizes["medium"])

    # colorbar on the same figure, but tied to the ice‐figure’s fig for layout reasons
    cbar_2d = ctrl.ui.canvas_ice.fig.colorbar(
        img,
        ax=ax,
        orientation="vertical",
    )
    cbar_2d.ax.tick_params(labelsize=ctrl.ui.font_sizes["small"])


def setup_ice_plot(ctrl):
    """
    Build the “Ice / Thickness” tab exactly once:
      - two 1D curves (no‐ice vs. with‐ice)
      - two 2D imshow plots (no‐ice, with‐ice)
      - two colorbars
      - store references on ctrl (e.g. ctrl.line_ice_ref, ctrl.line_ice, ctrl.ice_image_ref, ctrl.ice_image)
    """
    fig = ctrl.ui.canvas_ice.fig
    axes = ctrl.ui.canvas_ice.axes

    fig.suptitle(
        "CTF Modulation by Ice / Sample Thickness",
        fontsize=ctrl.ui.font_sizes["large"],
        fontweight="bold",
    )

    fig.subplots_adjust(hspace=0.3, top=0.93, bottom=0.1, left=0.125, right=0.9)

    ax1 = axes[1]  # 1D plot
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(-1, 1)
    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=ctrl.ui.font_sizes["small"],
        pad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax1.set_xlabel(
        "Spatial Frequency (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax1.set_ylabel(
        "Contrast Transfer Function",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax1.axhline(y=0, color="grey", linestyle="--", alpha=0.8)

    freqs = ctrl.freqs_1d
    ctf1d = ctrl.ctf_1d
    ctf1d_ice = ctrl.ctf_1d_ice

    # 1D lines
    (line_ref,) = ax1.plot(
        freqs,
        ctf1d.ctf(freqs),
        label="CTF without ice",
        color="grey",
        linewidth=0.5,
    )
    (line_ice,) = ax1.plot(
        freqs,
        ctf1d_ice.ctf(freqs),
        label="CTF with ice",
        color="purple",
        linewidth=1,
    )
    ax1.legend(fontsize=ctrl.ui.font_sizes["small"])
    ctrl.line_ice_ref = line_ref
    ctrl.line_ice = line_ice

    # 2D images
    ax_noice = axes[2]
    ax_withice = axes[3]
    # no‐ice 2D
    ax_noice.set_title(
        "CTF without Ice",
        fontsize=ctrl.ui.font_sizes["small"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_noice.set_xlabel(
        "Spatial Frequency X (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_noice.set_ylabel(
        "Spatial Frequency Y (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )  # only on one plot
    ax_noice.tick_params(
        axis="both", labelsize=ctrl.ui.font_sizes["small"], pad=ctrl.ui.font_sizes["tiny"] // 2
    )
    img_ref = ax_noice.imshow(
        ctrl.ctf_2d.ctf(ctrl.fx_fix, ctrl.fy_fix),
        extent=(-0.5, 0.5, -0.5, 0.5),
        cmap="Greys",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    # with‐ice 2D
    ax_withice.set_title(
        "CTF with Ice",
        fontsize=ctrl.ui.font_sizes["small"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_withice.set_xlabel(
        "Spatial Frequency X (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_withice.tick_params(
        axis="both", labelsize=ctrl.ui.font_sizes["small"], pad=ctrl.ui.font_sizes["tiny"] // 2
    )

    img_ice = ax_withice.imshow(
        ctrl.ctf_2d_ice.ctf(ctrl.fx_fix, ctrl.fy_fix),
        extent=(-0.5, 0.5, -0.5, 0.5),
        cmap="Greys",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    ctrl.ice_image_ref = img_ref
    ctrl.ice_image = img_ice

    # colorbars
    cbar_left = fig.colorbar(img_ref, ax=ax_noice, orientation="vertical")
    cbar_left.ax.tick_params(labelsize=ctrl.ui.font_sizes["small"])
    cbar_right = fig.colorbar(img_ice, ax=ax_withice, orientation="vertical")
    cbar_right.ax.tick_params(labelsize=ctrl.ui.font_sizes["small"])


def setup_tomo_plot(ctrl):
    """
    Build the “Tomo” tab once:
      - draw schematic (beam + rectangle)
      - two 2D images (no-tilt, with-tilt), each with colorbars
      - save references: ctrl.beam_line, ctrl.sample_rect, ctrl.tomo_image_ref, ctrl.tomo_image
    """
    fig = ctrl.ui.canvas_tomo.fig
    axes = ctrl.ui.canvas_tomo.axes

    # first, resample freqs based on current pixel_size
    ctrl.fx, ctrl.fy, ctrl.nyquist = compute_resampled_freqs(
        ctrl.ui.pixel_size_slider.get_value(), ctrl.image_size
    )
    ctrl.pixel_size_changed = False

    fig.suptitle(
        "CTF Modulation by Sample Tilting",
        fontsize=ctrl.ui.font_sizes["large"],
        fontweight="bold",
    )

    fig.subplots_adjust(hspace=0.3, top=0.9, bottom=0.1, left=0.125, right=0.9)

    # Diagram (axes[1])
    ax_diag = axes[1]
    ax_diag.set_xlim(-1500, 1500)
    ax_diag.set_ylim(-1500, 1500)
    ax_diag.set_aspect("equal")
    ax_diag.set_title(
        "Schematic Diagram",
        fontsize=ctrl.ui.font_sizes["small"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_diag.set_xlabel(
        "Length (µm)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )

    ax_diag.set_ylabel(
        "Length (µm)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_diag.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x)/1000:.1f}"))
    ax_diag.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x)/1000:.1f}"))
    ax_diag.tick_params(
        axis="both", labelsize=ctrl.ui.font_sizes["small"], pad=ctrl.ui.font_sizes["tiny"] // 2
    )

    # Beam line
    (beam_line,) = ax_diag.plot(
        [0, 0],
        [0, 1500],
        "b",
        linewidth=(ctrl.ui.linewidth - 1) if ctrl.ui.linewidth > 1 else 1,
        label="Beam direction",
    )
    ctrl.beam = beam_line

    # Sample rectangle (centered at 0,0)
    ctrl.width_tomo, ctrl.height_tomo = 1000.0, 50  # in nm
    sample_rect = Rectangle(
        (-ctrl.width_tomo / 2, -ctrl.height_tomo / 2),
        ctrl.width_tomo,
        ctrl.height_tomo,
        fc="gray",
        edgecolor="black",
        linewidth=1 if ctrl.ui.linewidth > 1 else 0.5,
        label="Illuminated area",
    )
    ax_diag.add_patch(sample_rect)
    ctrl.sample_rect = sample_rect
    ctrl.center_x_tomo, ctrl.center_y_tomo = 0, 0

    # legend for diagram
    ax_diag.legend(handles=[beam_line, sample_rect], fontsize=ctrl.ui.font_sizes["small"])

    # "CTF without Tilt" (axes[3]) and "CTF with Tilt" (axes[4])
    ax_ref = axes[3]
    ax_tilt = axes[4]

    ax_ref.set_title(
        "CTF without Tilt",
        fontsize=ctrl.ui.font_sizes["small"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_ref.set_xlabel(
        "Spatial Frequency X (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_ref.set_ylabel(
        "Spatial Frequency Y (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_ref.tick_params(
        axis="both", labelsize=ctrl.ui.font_sizes["small"], pad=ctrl.ui.font_sizes["tiny"] // 2
    )

    ax_tilt.set_title(
        "CTF with Tilted Sample",
        fontsize=ctrl.ui.font_sizes["medium"],
        fontweight="bold",
        pad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    ax_tilt.set_xlabel(
        "Spatial Frequency X (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )
    # if not (ctrl.ui.is_square_screen and ctrl.ui.need_resize or ctrl.ui.is_tight_space):
    ax_tilt.set_ylabel(
        "Spatial Frequency Y (Å⁻¹)",
        fontsize=ctrl.ui.font_sizes["small"],
        labelpad=ctrl.ui.font_sizes["tiny"] // 2,
    )

    ax_tilt.tick_params(
        axis="both", labelsize=ctrl.ui.font_sizes["small"], pad=ctrl.ui.font_sizes["tiny"] // 2
    )

    # now place two initial images
    nyq = ctrl.nyquist
    img_ref = ax_ref.imshow(
        ctrl.ctf_tomo_ref.ctf(ctrl.fx, ctrl.fy),
        extent=(-nyq, nyq, -nyq, nyq),
        cmap="Greys",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    img_tilt = ax_tilt.imshow(
        ctrl.ctf_tomo_tilt.ctf(ctrl.fx, ctrl.fy),
        extent=(-nyq, nyq, -nyq, nyq),
        cmap="Greys",
        vmin=-1,
        vmax=1,
        origin="lower",
    )
    ctrl.tomo_image_ref = img_ref
    ctrl.tomo_image = img_tilt

    # horizontal colorbar below the tilted‐CTF
    cbar = fig.colorbar(
        img_ref,
        ax=ax_tilt,
        orientation="horizontal",
    )
    # adjust the colorbar’s position manually
    image_pos = ax_tilt.get_position()
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([cbar_pos.x0, image_pos.y0 - 0.18, image_pos.width, 0.03])
    cbar.ax.set_title("Gray Scale", fontsize=ctrl.ui.font_sizes["small"])
    cbar.ax.tick_params(labelsize=ctrl.ui.font_sizes["small"])


def setup_image_plot(ctrl):
    """
    Build the “Image” tab once:
      - load default image via load_and_prepare_image(...)
      - compute FFT, CTF matrix, convolved image
      - call imshow for original, FFT, CTF, convolved
      - store references: ctrl.image_original, ctrl.image_fft, ctrl.image_ctf_convolve, ctrl.image_convolved
      - add titles, labels, and 4 colorbars
    """
    fig = ctrl.ui.canvas_image.fig
    axes = ctrl.ui.canvas_image.axes

    fig.suptitle(
        "Image Modulation by CTF",
        fontsize=ctrl.ui.font_sizes["large"],
        fontweight="bold",
    )

    fig.subplots_adjust(hspace=0.3, top=0.9, bottom=0.1, left=0.125, right=0.9)

    # 1) load default image into ctrl.image_data
    ctrl.image_data = load_and_prepare_image(ctrl.default_image, ctrl.image_size)
    ctrl.image_contrast_inverted = False

    # 2) compute FFT
    ctrl.image_data_fft = np.fft.fftshift(np.fft.fft2(ctrl.image_data))
    ctrl.scaled_fft = np.abs(ctrl.image_data_fft)

    # 3) compute CTF matrix and convolved image
    ctf_matrix = ctrl.ctf_2d.ctf(ctrl.fx, ctrl.fy)
    ctrl.scaled_convolved = np.abs(np.fft.ifft2(ctrl.image_data_fft * ctf_matrix))

    # 4) place four imshow’s
    img_orig = axes[1].imshow(ctrl.image_data, cmap="Greys")
    vmin, vmax = np.percentile(
        ctrl.scaled_fft,
        [100 - ctrl.ui.contrast_scale_fft.value(), ctrl.ui.contrast_scale_fft.value()],
    )
    img_fft = axes[2].imshow(
        ctrl.scaled_fft,
        vmin=vmin,
        vmax=vmax,
        cmap="Greys",
        origin="lower",
    )
    img_ctf = axes[4].imshow(
        ctf_matrix,
        cmap="Greys",
        vmin=-1,
        vmax=1,
        origin="lower",
        extent=(-ctrl.nyquist, ctrl.nyquist, -ctrl.nyquist, ctrl.nyquist),
    )
    img_conv = axes[3].imshow(ctrl.scaled_convolved, cmap="Greys")

    ctrl.image_original = img_orig
    ctrl.image_fft = img_fft
    ctrl.image_ctf_convolve = img_ctf
    ctrl.image_convolved = img_conv

    # 5) add titles + labels to each of the 4 axes
    titles = [
        "Original Image",
        "Fourier Transform",
        "Convolved Image",
        "Contrast Transfer Function",
    ]
    for i, ax in enumerate((axes[1], axes[2], axes[3], axes[4]), start=1):
        ax.set_title(
            titles[i - 1],
            fontsize=ctrl.ui.font_sizes["small"],
            fontweight="bold",
            pad=ctrl.ui.font_sizes["tiny"] // 2,
        )
        ax.set_ylabel(
            "Pixel Y" if i in (1, 3) else "Spatial Frequency Y (Å⁻¹)",
            fontsize=ctrl.ui.font_sizes["small"],
            labelpad=ctrl.ui.font_sizes["tiny"] // 2,
        )
        ax.set_xlabel(
            "Pixel X" if i in (1, 3) else "Spatial Frequency X (Å⁻¹)",
            fontsize=ctrl.ui.font_sizes["small"],
            labelpad=ctrl.ui.font_sizes["tiny"] // 2,
        )
        ax.tick_params(
            axis="both", labelsize=ctrl.ui.font_sizes["small"], pad=ctrl.ui.font_sizes["tiny"] // 2
        )

    # 6) colorbars for each subplot
    cbar1 = fig.colorbar(img_orig, ax=axes[1])
    cbar2 = fig.colorbar(img_fft, ax=axes[2])
    cbar3 = fig.colorbar(img_conv, ax=axes[3])
    cbar4 = fig.colorbar(img_ctf, ax=axes[4])
    for cbar in (cbar1, cbar2, cbar3, cbar4):
        cbar.ax.tick_params(labelsize=ctrl.ui.font_sizes["small"])


def setup_annotations(ctrl):
    """
    Create—and immediately hide—all Matplotlib annotation artists
    for 1D, 2D, Ice, Tomo, and Image tabs.
    """
    # disable annotation flag
    ctrl.show_annotation = False

    # 1D annotation
    ctrl.annotation_1d = ctrl.ui.canvas_1d.axes[1].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_1d.set_visible(False)

    # 2D annotation
    ctrl.annotation_2d = ctrl.ui.canvas_2d.axes[1].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_2d.set_visible(False)

    # Ice annotations
    ctrl.annotation_ice_1d = ctrl.ui.canvas_ice.axes[1].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_ice_1d.set_visible(False)

    ctrl.annotation_ice_ref = ctrl.ui.canvas_ice.axes[2].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_ice_ref.set_visible(False)

    ctrl.annotation_ice_ctf = ctrl.ui.canvas_ice.axes[3].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_ice_ctf.set_visible(False)

    # Tomo annotations
    ctrl.annotation_tomo_diagram_state = ctrl.ui.canvas_tomo.axes[1].annotate(
        "",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(3, -33),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_tomo_diagram_state.set_visible(False)

    ctrl.annotation_tomo_tilt_ctf = ctrl.ui.canvas_tomo.axes[4].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_tomo_tilt_ctf.set_visible(False)

    ctrl.annotation_tomo_ref_ctf = ctrl.ui.canvas_tomo.axes[3].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_tomo_ref_ctf.set_visible(False)

    # Image‐tab annotations
    ctrl.annotation_image_original = ctrl.ui.canvas_image.axes[1].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_image_original.set_visible(False)

    ctrl.annotation_image_fft = ctrl.ui.canvas_image.axes[2].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_image_fft.set_visible(False)

    ctrl.annotation_image_convolved = ctrl.ui.canvas_image.axes[3].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_image_convolved.set_visible(False)

    ctrl.annotation_image_ctf_convolve = ctrl.ui.canvas_image.axes[4].annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        fontsize=ctrl.ui.font_sizes["tiny"],
        zorder=10,
    )
    ctrl.annotation_image_ctf_convolve.set_visible(False)
