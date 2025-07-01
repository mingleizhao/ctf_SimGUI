import math
from matplotlib.ticker import FixedLocator
from matplotlib import transforms
import numpy as np
from utils.frequency_helpers import compute_resampled_freqs


def update_plot(gui, tab_idx) -> None:
    """
    Redraw the plot depending on which tab and CTF format are currently selected.
    """
    if tab_idx == 0:
        update_1d_plot(gui)
    elif tab_idx == 1:
        update_2d_plot(gui)
    elif tab_idx == 2:
        update_ice_plot(gui)
    elif tab_idx == 3:
        update_tomo_plot(gui)
    elif tab_idx == 4:
        update_image_plot(gui)
    else:
        return


def update_1d_plot(ctrl):
    """
    Called whenever any 1D‐related parameter changes (voltage, cs, cc, defocus, wrap_fn, etc.).
    It simply recomputes the five curves using ctrl.freqs_1d and updates the Line2D data.
    """
    if ctrl.ui.plot_tabs.currentIndex() != 0:
        return

    freqs = ctrl.freqs_1d
    ctf1d = ctrl.ctf_1d
    wrap_fn = ctrl.wrap_func

    ctrl.line_et.set_data(freqs, wrap_fn(ctf1d.envelope.temporal(freqs)))
    ctrl.line_es.set_data(freqs, wrap_fn(ctf1d.envelope.spatial_1d(freqs)))
    ctrl.line_ed.set_data(freqs, wrap_fn(ctf1d.envelope.detector(freqs)))
    ctrl.line_te.set_data(freqs, wrap_fn(ctf1d.envelope.total_1d(freqs)))
    ctrl.line_dc.set_data(freqs, wrap_fn(ctf1d.ctf(freqs)))

    ctrl.ui.canvas_1d.draw_idle()


def update_2d_plot(ctrl):
    """
    Called whenever any 2D‐CTF parameter changes (voltage, cs, cc, defocus, etc.),
    or when the user zooms/changes wrap_fn on the 2D tab.
    """
    if ctrl.ui.plot_tabs.currentIndex() != 1:
        return

    ctf2d = ctrl.ctf_2d
    wrap_fn = ctrl.wrap_func

    new_data = wrap_fn(ctf2d.ctf(ctrl.fx_fix, ctrl.fy_fix))
    ctrl.image_2d.set_data(new_data)
    ctrl.ui.canvas_2d.draw_idle()


def update_ice_plot(ctrl):
    """
    Called when any Ice‐tab parameter changes (ice thickness, defocus, wrap_fn, etc.),
    or when the user zooms the Ice 2D CTF.
    """
    if ctrl.ui.plot_tabs.currentIndex() != 2:
        return

    freqs = ctrl.freqs_1d
    ctf1d = ctrl.ctf_1d
    ctf2d = ctrl.ctf_2d
    ctf1d_ice = ctrl.ctf_1d_ice
    ctf2d_ice = ctrl.ctf_2d_ice
    wrap_fn = ctrl.wrap_func

    ctrl.line_ice_ref.set_data(freqs, wrap_fn(ctf1d.ctf(freqs)))
    ctrl.line_ice.set_data(freqs, wrap_fn(ctf1d_ice.ctf(freqs)))

    ctrl.ice_image_ref.set_data(wrap_fn(ctf2d.ctf(ctrl.fx_fix, ctrl.fy_fix)))
    ctrl.ice_image.set_data(wrap_fn(ctf2d_ice.ctf(ctrl.fx_fix, ctrl.fy_fix)))

    ctrl.ui.canvas_ice.draw_idle()


def update_tomo_plot(ctrl):
    """
    Called when any Tomo‐tab parameter changes (pixel_size, tilt, thickness, defocus, wrap_fn, etc.).
    Handles re‐sampling frequencies if pixel_size changed, updating both images.
    """
    if ctrl.ui.plot_tabs.currentIndex() != 3:
        return

    # If pixel_size changed, recompute freq grid and adjust extents
    if ctrl.pixel_size_changed:
        ctrl.fx, ctrl.fy, ctrl.nyquist = compute_resampled_freqs(
            ctrl.ui.pixel_size_slider.get_value(), ctrl.image_size
        )
        nyq = ctrl.nyquist
        ctrl.tomo_image_ref.set_extent((-nyq, nyq, -nyq, nyq))
        ctrl.tomo_image.set_extent((-nyq, nyq, -nyq, nyq))

    wrap_fn = ctrl.wrap_func
    # “no tilt” image
    ctrl.tomo_image_ref.set_data(wrap_fn(ctrl.ctf_tomo_ref.ctf(ctrl.fx, ctrl.fy)))
    # “with tilt” image
    ctrl.tomo_image.set_data(wrap_fn(ctrl.ctf_tomo_tilt.ctf(ctrl.fx, ctrl.fy)))

    ctrl.ui.canvas_tomo.draw_idle()


def update_image_plot(ctrl):
    """
    Called when any Image‐tab parameter changes (pixel_size, wrap_fn, contrast, etc.).
    - If pixel_size changed: resample freq grid and update CTF‐imshow extent
    - Always recompute CTF matrix and convolved image, then update imshows
    - Finally adjust contrast if needed
    """
    if ctrl.ui.plot_tabs.currentIndex() != 4:
        return

    # 1) If pixel size changed, update freq grid and CTF‐extent
    if ctrl.pixel_size_changed:
        ctrl.fx, ctrl.fy, ctrl.nyquist = compute_resampled_freqs(
            ctrl.ui.pixel_size_slider.get_value(), ctrl.image_size
        )
        nyq = ctrl.nyquist
        ctrl.image_ctf_convolve.set_extent((-nyq, nyq, -nyq, nyq))

    update_ticks_for_fft(ctrl)

    wrap_fn = ctrl.wrap_func
    ctf_matrix = ctrl.ctf_2d.ctf(ctrl.fx, ctrl.fy)
    ctrl.image_ctf_convolve.set_data(wrap_fn(ctf_matrix))

    # 2) recompute convolved image every time
    new_conv = np.abs(np.fft.ifft2(ctrl.image_data_fft * ctf_matrix))
    ctrl.scaled_convolved = new_conv
    ctrl.image_convolved.set_data(new_conv)

    # 3) adjust contrast (might have changed sync state or slider)
    adjust_contrast_image(ctrl)

    ctrl.ui.canvas_image.draw_idle()


def update_plot_range(ctrl):
    """
    Whenever the user drags the “x min/max” or “y min/max” spinboxes,
    update the axis limits on whichever tab is active.
    """
    idx = ctrl.ui.plot_tabs.currentIndex()
    if idx == 0:
        ax = ctrl.ui.canvas_1d.axes[1]
        ax.set_xlim(ctrl.ui.plot_1d_x_min.value(), ctrl.ui.plot_1d_x_max.value())
        ax.set_ylim(ctrl.ui.plot_1d_y_min.value(), ctrl.ui.plot_1d_y_max.value())

        ctrl.ui.xlim_slider_1d.blockSignals(True)
        ctrl.ui.xlim_slider_1d.set_value(ctrl.ui.plot_1d_x_max.value())
        ctrl.ui.xlim_slider_1d.blockSignals(False)

        ctrl.ui.ylim_slider_1d.blockSignals(True)
        ctrl.ui.ylim_slider_1d.set_value(ctrl.ui.plot_1d_y_max.value())
        ctrl.ui.ylim_slider_1d.blockSignals(False)

        ctrl.ui.canvas_1d.draw_idle()

    elif idx == 1:
        ax = ctrl.ui.canvas_2d.axes[1]
        ax.set_xlim(ctrl.ui.plot_2d_x_min.value(), ctrl.ui.plot_2d_x_max.value())
        ax.set_ylim(ctrl.ui.plot_2d_y_min.value(), ctrl.ui.plot_2d_y_max.value())
        ctrl.ui.canvas_2d.draw_idle()

    elif idx == 2:
        ax = ctrl.ui.canvas_ice.axes[1]
        ax.set_xlim(0, ctrl.ui.xlim_slider_ice.get_value())
        ctrl.ui.canvas_ice.draw_idle()


def update_ticks_for_fft(ctrl):
    """
    When the user zooms the CTF extent, the tick labels on the FFT subplot
    must be recalculated so they match the freq‐axes from the CTF imshow.
    """
    ax_ctf = ctrl.ui.canvas_image.axes[4]
    ax_fft = ctrl.ui.canvas_image.axes[2]

    xticks = ax_ctf.get_xticks()
    xticklabels = [label.get_text() for label in ax_ctf.get_xticklabels()]
    yticks = ax_ctf.get_yticks()
    yticklabels = [label.get_text() for label in ax_ctf.get_yticklabels()]

    nyq = ctrl.ctf_2d.envelope.nyquist
    size = ctrl.image_size
    xpos = [(tick + nyq) / (2 * nyq) * size for tick in xticks]
    ypos = [(tick + nyq) / (2 * nyq) * size for tick in yticks]

    ax_fft.xaxis.set_major_locator(FixedLocator(xpos))
    ax_fft.set_xticklabels(xticklabels)
    ax_fft.yaxis.set_major_locator(FixedLocator(ypos))
    ax_fft.set_yticklabels(yticklabels)

    ctrl.ui.canvas_image.draw_idle()


def invert_contrast(ctrl):
    """
    Toggle the image‐tab’s grayscale colormap between normal (Greys) and inverted (Greys_r),
    then redraw only the image tab.
    """
    # Flip the state
    ctrl.image_contrast_inverted = not ctrl.image_contrast_inverted

    # Update colormap on the two relevant AxesImage objects
    if ctrl.image_contrast_inverted:
        ctrl.image_original.set_cmap("Greys_r")
        ctrl.image_convolved.set_cmap("Greys_r")
    else:
        ctrl.image_original.set_cmap("Greys")
        ctrl.image_convolved.set_cmap("Greys")

    # Finally, only redraw the image tab’s canvas
    ctrl.ui.canvas_image.draw_idle()


def update_ylim_1d(ctrl, value) -> None:
    """
    Update Y limits of 1D CTF plot using slider.
    """
    ctrl.ui.plot_1d_y_max.setValue(value)
    if (
        ctrl.ui.radio_button_group.checkedButton() == ctrl.ui.radio_abs_ctf
        or ctrl.ui.radio_button_group.checkedButton() == ctrl.ui.radio_ctf_squared
    ):
        ctrl.ui.plot_1d_y_min.setValue(0)
    else:
        ctrl.ui.plot_1d_y_min.setValue(-value)


def zoom_2d_ctf(ctrl) -> None:
    """
    Zoom on 2D CTF.
    """
    if ctrl.ui.plot_tabs.currentIndex() == 1:
        ctrl.ui.canvas_2d.axes[1].set_xlim(
            -ctrl.ui.freq_scale_2d.value(), ctrl.ui.freq_scale_2d.value()
        )
        ctrl.ui.canvas_2d.axes[1].set_ylim(
            -ctrl.ui.freq_scale_2d.value(), ctrl.ui.freq_scale_2d.value()
        )
        ctrl.ui.canvas_2d.axes[1].set_xlim(
            -ctrl.ui.freq_scale_2d.value(), ctrl.ui.freq_scale_2d.value()
        )
        ctrl.ui.canvas_2d.axes[1].set_ylim(
            -ctrl.ui.freq_scale_2d.value(), ctrl.ui.freq_scale_2d.value()
        )
        ctrl.ui.canvas_2d.draw_idle()
    elif ctrl.ui.plot_tabs.currentIndex() == 2:
        ctrl.ui.canvas_ice.axes[2].set_xlim(
            -ctrl.ui.freq_scale_ice.value(), ctrl.ui.freq_scale_ice.value()
        )
        ctrl.ui.canvas_ice.axes[2].set_ylim(
            -ctrl.ui.freq_scale_ice.value(), ctrl.ui.freq_scale_ice.value()
        )
        ctrl.ui.canvas_ice.axes[3].set_xlim(
            -ctrl.ui.freq_scale_ice.value(), ctrl.ui.freq_scale_ice.value()
        )
        ctrl.ui.canvas_ice.axes[3].set_ylim(
            -ctrl.ui.freq_scale_ice.value(), ctrl.ui.freq_scale_ice.value()
        )
        ctrl.ui.canvas_ice.draw_idle()


def zoom_2d_image(ctrl, key: str, value: float | int) -> None:
    """
    Zoom on 2D image by updating axes limits.

    Args:
        key (str): The name of the image being updated.
        value (float | int): The new value for the zoom factor.
    """
    center = ctrl.image_size / 2

    # Compute zoom factor
    new_size = ctrl.image_size * 100 / value

    new_extent = [
        center - new_size / 2,
        center + new_size / 2,
    ]

    # Update axes limits
    if key == "image":
        ctrl.ui.canvas_image.axes[1].set_xlim(new_extent[0], new_extent[1])
        ctrl.ui.canvas_image.axes[1].set_ylim(new_extent[1], new_extent[0])
        ctrl.ui.canvas_image.axes[3].set_xlim(new_extent[0], new_extent[1])
        ctrl.ui.canvas_image.axes[3].set_ylim(new_extent[1], new_extent[0])
    elif key == "fft":
        ctrl.ui.canvas_image.axes[2].set_xlim(new_extent[0], new_extent[1])
        ctrl.ui.canvas_image.axes[2].set_ylim(new_extent[0], new_extent[1])

    ctrl.ui.canvas_image.draw_idle()


def adjust_contrast_image(ctrl) -> None:
    """
    Adjust the contrast of both original and convolved images.
    """
    val = ctrl.ui.contrast_scale_image.value()
    rng = [100 - val, val]

    if ctrl.ui.contrast_sync_checkbox.isChecked():
        vmin = min(
            np.percentile(ctrl.image_data, 100 - val),
            np.percentile(ctrl.scaled_convolved, 100 - val),
        )
        vmax = max(
            np.percentile(ctrl.image_data, val),
            np.percentile(ctrl.scaled_convolved, val),
        )
        ctrl.image_original.set_clim(vmin=vmin, vmax=vmax)
        ctrl.image_convolved.set_clim(vmin=vmin, vmax=vmax)
    else:
        ctrl.image_original.set_clim(*np.percentile(ctrl.image_data, rng))
        ctrl.image_convolved.set_clim(*np.percentile(ctrl.scaled_convolved, rng))

    ctrl.ui.canvas_image.draw_idle()


def adjust_contrast_fft(ctrl, value: float | int) -> None:
    """
    Adjust the contrast of fft image.

    Args:
        value (float | int): The new contrast percentile.
    """
    vmin, vmax = np.percentile(ctrl.scaled_fft, [100 - value, value])
    ctrl.image_fft.set_clim(vmin=vmin, vmax=vmax)

    ctrl.ui.canvas_image.draw_idle()


def update_grayness(ctrl) -> None:
    """
    Update the max scales on 2D CTF.
    """
    if ctrl.ui.plot_tabs.currentIndex() == 1:
        current_vmin, _ = ctrl.image_2d.get_clim()
        if current_vmin != 0:
            vmin = -ctrl.ui.gray_scale_2d.value()
        else:
            vmin = 0
        vmax = ctrl.ui.gray_scale_2d.value()
        ctrl.image_2d.set_clim(vmin=vmin, vmax=vmax)
        ctrl.ui.canvas_2d.draw_idle()
    elif ctrl.ui.plot_tabs.currentIndex() == 2:
        current_vmin, _ = ctrl.ice_image.get_clim()
        if current_vmin != 0:
            vmin = -ctrl.ui.gray_scale_ice.value()
        else:
            vmin = 0
        vmax = ctrl.ui.gray_scale_ice.value()
        ctrl.ice_image.set_clim(vmin=vmin, vmax=vmax)
        ctrl.ice_image_ref.set_clim(vmin=vmin, vmax=vmax)
        ctrl.ui.canvas_ice.draw_idle()
    elif ctrl.ui.plot_tabs.currentIndex() == 3:
        current_vmin, _ = ctrl.tomo_image.get_clim()
        if current_vmin != 0:
            vmin = -ctrl.ui.gray_scale_tomo.value()
        else:
            vmin = 0
        vmax = ctrl.ui.gray_scale_tomo.value()
        ctrl.tomo_image.set_clim(vmin=vmin, vmax=vmax)
        ctrl.tomo_image_ref.set_clim(vmin=vmin, vmax=vmax)
        ctrl.ui.canvas_tomo.draw_idle()


def update_display_1d(ctrl) -> None:
    """
    Update the display of 1D CTF
    """
    ctrl.line_et.set_visible(ctrl.ui.show_temp.isChecked())
    ctrl.line_es.set_visible(ctrl.ui.show_spatial.isChecked())
    ctrl.line_ed.set_visible(ctrl.ui.show_detector.isChecked())
    ctrl.line_te.set_visible(ctrl.ui.show_total.isChecked())
    ctrl.line_y0.set_visible(ctrl.ui.show_y0.isChecked())
    ctrl.legend_1d = ctrl.ui.canvas_1d.axes[1].legend(fontsize=ctrl.ui.font_sizes["medium"])
    ctrl.legend_1d.set_visible(ctrl.ui.show_legend.isChecked())
    ctrl.ui.canvas_1d.draw_idle()


def update_tomo(ctrl, key: str | None = None, value: float | int | None = None) -> None:
    """
    Redraw the tomo diagram and CTF plots for the tomo tab depending on the parameters.
    """
    if key == "thickness":
        ctrl.height_tomo = value
        _update_sample_rectangle(ctrl)
        ctrl.ctf_tomo_ref.envelope.ice_thickness = value
        tilt_angle_rad = abs(math.radians(ctrl.ui.tilt_slider_tomo.get_value()))
        ctrl.ctf_tomo_tilt.envelope.ice_thickness = ctrl.width_tomo * math.sin(
            tilt_angle_rad
        ) + ctrl.height_tomo * math.cos(tilt_angle_rad)
        update_tomo_plot(ctrl)
    elif key == "tilt_angle":
        _rotate_sample_rectangle(ctrl, value)
        height_tomo = ctrl.ui.sample_thickness_slider_tomo.get_value()  # in nm
        tilt_angle_rad = abs(math.radians(value))
        ctrl.ctf_tomo_tilt.envelope.ice_thickness = ctrl.width_tomo * math.sin(
            tilt_angle_rad
        ) + height_tomo * math.cos(tilt_angle_rad)
        update_tomo_plot(ctrl)
    elif key == "df_diff":
        ctrl.ctf_tomo_ref.defocus_diff = value
        ctrl.ctf_tomo_tilt.defocus_diff = value
        update_tomo_plot(ctrl)
    elif key == "df_az":
        ctrl.ctf_tomo_ref.defocus_az = value
        ctrl.ctf_tomo_tilt.defocus_az = value
        update_tomo_plot(ctrl)
    elif key == "sample_size":
        ctrl.width_tomo = value * 1000
        _update_sample_rectangle(ctrl)
        height_tomo = ctrl.ui.sample_thickness_slider_tomo.get_value()  # in nm
        tilt_angle_rad = abs(math.radians(ctrl.ui.tilt_slider_tomo.get_value()))
        ctrl.ctf_tomo_tilt.envelope.ice_thickness = ctrl.width_tomo * math.sin(
            tilt_angle_rad
        ) + height_tomo * math.cos(tilt_angle_rad)
        update_tomo_plot(ctrl)

    ctrl.ui.canvas_tomo.draw_idle()


def _update_sample_rectangle(ctrl):
    """Change the size of the sample rectangle."""
    ctrl.sample_rect.set_xy((-ctrl.width_tomo / 2.0, -ctrl.height_tomo / 2.0))
    ctrl.sample_rect.set_width(ctrl.width_tomo)
    ctrl.sample_rect.set_height(ctrl.height_tomo)


def _rotate_sample_rectangle(ctrl, angle: float) -> None:
    """Apply a rotation transformation around the center of the sample rectangle."""
    # Create a transformation: first translate to (0,0), then rotate, then translate back
    transform = (
        transforms.Affine2D().rotate_deg_around(ctrl.center_x_tomo, ctrl.center_y_tomo, angle)
        + ctrl.ui.canvas_tomo.axes[1].transData
    )
    # Apply the transformation
    ctrl.sample_rect.set_transform(transform)
