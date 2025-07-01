import math
import numpy as np


def annotate_1d(ctrl, event):
    """Called when hovering over the 1D axes (canvas_1d.axes[1])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / x
    value = ctrl.wrap_func(ctrl.ctf_1d.ctf(np.array([x])))[0]
    text = f"x: {x:.3f} Å⁻¹\ny: {y:.3f}\nres: {res:.2f} Å\nctf: {value:.4f}"
    ctrl.annotation_1d.xy = (x, y)
    ctrl.annotation_1d.set_text(text)
    ctrl.annotation_1d.set_visible(True)
    ctrl.ui.canvas_1d.draw_idle()


def annotate_2d(ctrl, event):
    """Called when hovering over the 2D CTF axes (canvas_2d.axes[1])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / math.sqrt(x**2 + y**2)
    value = ctrl.wrap_func(ctrl.ctf_2d.ctf(np.array([x]), np.array([y])))[0]
    text = f"x: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {value:.4f}"
    ctrl.annotation_2d.xy = (x, y)
    ctrl.annotation_2d.set_text(text)
    ctrl.annotation_2d.set_visible(True)
    ctrl.ui.canvas_2d.draw_idle()


def annotate_ice_1d(ctrl, event):
    """Called when hovering over the ice-tab 1D plot (canvas_ice.axes[1])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / x
    no_ice = ctrl.wrap_func(ctrl.ctf_1d.ctf(np.array([x])))[0]
    with_ice = ctrl.wrap_func(ctrl.ctf_1d_ice.ctf(np.array([x])))[0]
    text = (
        f"x: {x:.3f} Å⁻¹\n" f"res: {res:.2f} Å\n" f"gray: {no_ice:.4f}\n" f"purple: {with_ice:.4f}"
    )
    ctrl.annotation_ice_1d.xy = (x, y)
    ctrl.annotation_ice_1d.set_text(text)
    ctrl.annotation_ice_1d.set_visible(True)
    ctrl.ui.canvas_ice.draw_idle()


def annotate_ice_2d_noice(ctrl, event):
    """Called when hovering over the ice-tab ‘no-ice’ 2D plot (canvas_ice.axes[2])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / math.sqrt(x**2 + y**2)
    value = ctrl.wrap_func(ctrl.ctf_2d.ctf(np.array([x]), np.array([y])))[0]
    text = f"x: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {value:.4f}"
    ctrl.annotation_ice_ref.xy = (x, y)
    ctrl.annotation_ice_ref.set_text(text)
    ctrl.annotation_ice_ref.set_visible(True)
    ctrl.ui.canvas_ice.draw_idle()


def annotate_ice_2d_withice(ctrl, event):
    """Called when hovering over the ice-tab ‘with-ice’ 2D plot (canvas_ice.axes[3])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / math.sqrt(x**2 + y**2)
    val = ctrl.wrap_func(ctrl.ctf_2d_ice.ctf(np.array([x]), np.array([y])))[0]
    text = f"x: {x:.3f} Å⁻¹\n" f"y: {y:.3f} Å⁻¹\n" f"res: {res:.2f} Å\n" f"ctf: {val:.4f}"
    ctrl.annotation_ice_ctf.xy = (x, y)
    ctrl.annotation_ice_ctf.set_text(text)
    ctrl.annotation_ice_ctf.set_visible(True)
    ctrl.ui.canvas_ice.draw_idle()


def annotate_tomo_diagram(ctrl, event):
    """Called when hovering over the tomo-diagram (canvas_tomo.axes[1])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    tilt = abs(math.radians(ctrl.ui.tilt_slider_tomo.get_value()))
    thickness = ctrl.width_tomo * math.sin(tilt) + ctrl.height_tomo * math.cos(tilt)
    text = (
        f"size: {ctrl.ui.sample_size_tomo.value():.2f} µm\n"
        f"tilt angle: {ctrl.ui.tilt_slider_tomo.get_value():.1f}°\n"
        f"thk.: {thickness:.1f} nm"
    )
    ctrl.annotation_tomo_diagram_state.set_text(text)
    ctrl.annotation_tomo_diagram_state.set_visible(True)
    ctrl.ui.canvas_tomo.draw_idle()


def annotate_tomo_ref_ctf(ctrl, event):
    """Called when hovering over the tomo-tab ‘no-tilt’ CTF (canvas_tomo.axes[3])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / math.sqrt(x**2 + y**2)
    val = ctrl.wrap_func(ctrl.ctf_tomo_ref.ctf(np.array([x]), np.array([y])))[0]
    text = (
        f"tilt angle: 0°\n"
        f"x: {x:.3f} Å⁻¹\n"
        f"y: {y:.3f} Å⁻¹\n"
        f"res: {res:.2f} Å\n"
        f"ctf: {val:.4f}"
    )
    ctrl.annotation_tomo_ref_ctf.xy = (x, y)
    ctrl.annotation_tomo_ref_ctf.set_text(text)
    ctrl.annotation_tomo_ref_ctf.set_visible(True)
    ctrl.ui.canvas_tomo.draw_idle()


def annotate_tomo_tilt_ctf(ctrl, event):
    """Called when hovering over the tomo-tab ‘with-tilt’ CTF (canvas_tomo.axes[4])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / math.sqrt(x**2 + y**2)
    val = ctrl.wrap_func(ctrl.ctf_tomo_tilt.ctf(np.array([x]), np.array([y])))[0]
    text = (
        f"tilt angle: {ctrl.ui.tilt_slider_tomo.get_value():.1f}°\n"
        f"x: {x:.3f} Å⁻¹\n"
        f"y: {y:.3f} Å⁻¹\n"
        f"res: {res:.2f} Å\n"
        f"ctf: {val:.4f}"
    )
    ctrl.annotation_tomo_tilt_ctf.xy = (x, y)
    ctrl.annotation_tomo_tilt_ctf.set_text(text)
    ctrl.annotation_tomo_tilt_ctf.set_visible(True)
    ctrl.ui.canvas_tomo.draw_idle()


def annotate_image_original(ctrl, event):
    """Called when hovering over the original-image panel (canvas_image.axes[1])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    val = ctrl.image_data[int(y), int(x)]
    ctrl.annotation_image_original.xy = (x, y)
    ctrl.annotation_image_original.set_text(f"{val:.2f}")
    ctrl.annotation_image_original.set_visible(True)
    ctrl.ui.canvas_image.draw_idle()


def annotate_image_fft(ctrl, event):
    """Called when hovering over the FFT panel (canvas_image.axes[2])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    x_freq = (2 * x / ctrl.image_size - 1) * ctrl.nyquist
    y_freq = (2 * y / ctrl.image_size - 1) * ctrl.nyquist
    res = 1.0 / math.sqrt(x_freq**2 + y_freq**2)
    val = ctrl.scaled_fft[int(y), int(x)]
    text = f"res: {res:.2f} Å\namp: {val:.2f}"
    ctrl.annotation_image_fft.xy = (x, y)
    ctrl.annotation_image_fft.set_text(text)
    ctrl.annotation_image_fft.set_visible(True)
    ctrl.ui.canvas_image.draw_idle()


def annotate_image_convolved(ctrl, event):
    """Called when hovering over the convolved-image panel (canvas_image.axes[3])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    val = ctrl.scaled_convolved[int(y), int(x)]
    ctrl.annotation_image_convolved.xy = (x, y)
    ctrl.annotation_image_convolved.set_text(f"{val:.2f}")
    ctrl.annotation_image_convolved.set_visible(True)
    ctrl.ui.canvas_image.draw_idle()


def annotate_image_ctf(ctrl, event):
    """Called when hovering over the CTF-image panel (canvas_image.axes[4])."""
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    res = 1.0 / math.sqrt(x**2 + y**2)
    val = ctrl.wrap_func(ctrl.ctf_2d.ctf(np.array([x]), np.array([y])))[0]
    text = f"res: {res:.2f} Å\nctf: {val:.4f}"
    ctrl.annotation_image_ctf_convolve.xy = (x, y)
    ctrl.annotation_image_ctf_convolve.set_text(text)
    ctrl.annotation_image_ctf_convolve.set_visible(True)
    ctrl.ui.canvas_image.draw_idle()


def hide_all_annotations(ctrl):
    """When the cursor leaves any of the axes, hide every annotation."""
    for ann in (
        ctrl.annotation_1d,
        ctrl.annotation_2d,
        ctrl.annotation_ice_1d,
        ctrl.annotation_ice_ref,
        ctrl.annotation_ice_ctf,
        ctrl.annotation_tomo_diagram_state,
        ctrl.annotation_tomo_ref_ctf,
        ctrl.annotation_tomo_tilt_ctf,
        ctrl.annotation_image_original,
        ctrl.annotation_image_fft,
        ctrl.annotation_image_convolved,
        ctrl.annotation_image_ctf_convolve,
    ):
        ann.set_visible(False)
    # Also redraw all canvases so hidden state takes effect
    for canvas in (
        ctrl.ui.canvas_1d,
        ctrl.ui.canvas_2d,
        ctrl.ui.canvas_ice,
        ctrl.ui.canvas_tomo,
        ctrl.ui.canvas_image,
    ):
        canvas.draw_idle()
