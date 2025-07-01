"""
left_panel.py
--------------

This module defines build functions for constructing and configuring the left panel GUI widgets of
the CTFSimGUI application's interface.

Functions:
    - build_group_control(...):
        Builds a QGroupBox and layout from a set of widget factories.

    - build_microscope_group()
        Constructs the microscope control panel group.

    - build_imaging_group()
        Constructs the imaging control panel group.

    - build_detector_group()
        Constructs the detector control panel group.

    - build_plotting_group()
        Constructs the plotting control panel group.

    - build_button_group()
        Constructs the button control panel group.
"""

from typing import Callable, Dict, Tuple, Union
from PyQt5.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QLayout,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)
from models.detector import ALL_DETECTORS
from .customized_widgets import LabeledSlider
from .styles import (
    left_panel_qgroupbox_style,
    button_style,
    default_button_style,
    check_box_style,
    radio_box_style,
    combo_box_style,
    get_ui_scale,
)


WidgetOrLayout = Union[QWidget, QLayout]
FactoryResult = Union[WidgetOrLayout, Tuple[WidgetOrLayout, int]]
Factory = Callable[[], FactoryResult]


def build_group_control(
    title: str,
    style: str,
    factories: Dict[str, Factory],
    *,
    margin: int = 0,
    spacing: int = 0,
) -> Tuple[QGroupBox, Dict[str, WidgetOrLayout]]:
    """
    Build a QGroupBox + layout from a set of factories.

    Each factory may return:
      - A single QWidget or QLayout → that widget/layout is both added
        to the box *and* recorded in `created[name]`.
      - A 2-tuple (w_or_lay, extra_px) → add w_or_lay, then addSpacing(extra_px);
        record w_or_lay in `created[name]`.
      - A 3-tuple (w_or_lay, to_expose: QWidget, extra_px) → add w_or_lay,
        then addSpacing(extra_px); record *to_expose* in `created[name]`.
    """
    box = QGroupBox(title)
    box.setStyleSheet(style)

    layout = QVBoxLayout(box)
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(spacing)

    created: Dict[str, QWidget] = {}

    for name, factory in factories.items():
        result = factory()

        # unpack 1-, 2-, or 3-tuple
        if isinstance(result, tuple):
            if len(result) == 1:
                (w_or_lay,) = result
                expose = w_or_lay
                extra_px = None
            elif len(result) == 2:
                w_or_lay, extra_px = result
                expose = w_or_lay
            elif len(result) == 3:
                w_or_lay, expose, extra_px = result
            else:
                raise ValueError(f"factory for {name!r} returned wrong tuple size")
        else:
            w_or_lay = result
            expose = result
            extra_px = None

        # actually add it to the group‐box
        if isinstance(w_or_lay, QLayout):
            layout.addLayout(w_or_lay)
        else:
            layout.addWidget(w_or_lay)

        # optional post‐spacing
        if extra_px is not None:
            layout.addSpacing(extra_px)

        # expose only the *widget* you asked for
        if isinstance(expose, QWidget):
            created[name] = expose
        else:
            # if they exposed a layout by mistake, skip it
            pass

    return box, created


def build_microscope_group():
    """
    Create a QGroupBox for 'Microscope Parameters' (voltage, aberrations, stability, etc.).
    This section uses custom sliders from 'customized_widgets' for parameter control.

    Returns:
      - box: QGroupBox already populated with all the sliders
      - sliders: a dict of each slider widget, so controller can hook signals
    """
    factories = {
        "voltage": lambda: LabeledSlider(
            "Voltage (KV)", min_value=80, max_value=1000, step=20, value_format="{:.0f}"
        ),
        "stability": lambda: LabeledSlider(
            "Voltage Stability",
            min_value=1e-9,
            max_value=1e-4,
            step=1e-9,
            value_format="{:.2e}",
            log_scale=True,
        ),
        "angle": lambda: LabeledSlider(
            "<b>e⁻</b> Angle Spread (rad)",
            min_value=1e-5,
            max_value=1e-2,
            step=1e-5,
            value_format="{:.1e}",
            log_scale=True,
        ),
        "energy": lambda: LabeledSlider(
            "<b>e⁻</b> Energy Spread (eV)",
            min_value=0,
            max_value=10,
            step=0.1,
            value_format="{:.1f}",
        ),
        "chrom_aberr": lambda: LabeledSlider(
            "Chromatic Aberration (mm)",
            min_value=0.0,
            max_value=10,
            step=0.1,
            value_format="{:.1f}",
        ),
        "sph_aberr": lambda: LabeledSlider(
            "Spherical Aberration (mm)",
            min_value=0.0,
            max_value=10,
            step=0.1,
            value_format="{:.1f}",
        ),
        "obj_stab": lambda: LabeledSlider(
            "Objective Lens Stability",
            min_value=1e-9,
            max_value=1e-4,
            step=1e-9,
            value_format="{:.2e}",
            log_scale=True,
        ),
    }

    return build_group_control(
        "Microscope Parameters",
        left_panel_qgroupbox_style(),
        factories,
        margin=int(12 * get_ui_scale()),
        spacing=int(6 * get_ui_scale()),
    )


def build_imaging_group():
    """
    Create a QGroupBox for 'Imaging Parameters' (defocus, amplitude contrast, and additional phase
    shift).

    Returns:
      - box: QGroupBox already populated with all the sliders
      - sliders: a dict of each slider widget, so controller can hook signals
    """
    factories = {
        "defocus": lambda: LabeledSlider(
            "Avg. Defocus (µm)", min_value=-5, max_value=10, step=0.01, value_format="{:.4f}"
        ),
        "amplitude_contrast": lambda: LabeledSlider(
            "Amplitude Contrast", min_value=0, max_value=1, step=0.01, value_format="{:.2f}"
        ),
        "additional_phase": lambda: LabeledSlider(
            "Additional Phase Shift (°)", min_value=0, max_value=180, step=1, value_format="{:.0f}"
        ),
    }

    return build_group_control(
        "Imaging Parameters",
        left_panel_qgroupbox_style(),
        factories,
        margin=int(12 * get_ui_scale()),
        spacing=int(6 * get_ui_scale()),
    )


def build_detector_group():
    """
    Create a QGroupBox for 'Detector Parameters' (detector and pixel size).

    Returns:
      - box: QGroupBox already populated with all the sliders
      - widgets: a dict of each widget, so controller can hook signals
    """

    def make_detector_row():
        row = QHBoxLayout()
        combo = QComboBox()
        combo.addItems([d.name for d in ALL_DETECTORS])
        combo.setStyleSheet(combo_box_style())
        row.addWidget(combo)
        # return (layout, widget_to_expose, extra_spacing)
        return row, combo, int(6 * get_ui_scale())

    factories = {
        "detector_label": lambda: QLabel("Detector"),
        "detectors": make_detector_row,
        "pixel_size": lambda: LabeledSlider(
            "Pixel Size (Å)", min_value=0.5, max_value=5.0, step=0.1, value_format="{:.3f}"
        ),
    }

    return build_group_control(
        "Detector Parameters",
        left_panel_qgroupbox_style(),
        factories,
        margin=int(12 * get_ui_scale()),
        spacing=int(6 * get_ui_scale()),
    )


def build_plotting_group():
    """
    Create a QGroupBox for 'Plotting Parameters' (envelope function toggles, CTFs, etc.).

    Returns:
      - box: QGroupBox already populated with all the sliders
      - widgets: a dict of each widget, so controller can hook signals
    """
    # ——— prepare the actual widgets to expose ———
    temporal = QCheckBox("Temporal")
    temporal.setStyleSheet(check_box_style())
    spatial = QCheckBox("Spatial")
    spatial.setStyleSheet(check_box_style())
    detector = QCheckBox("Detector")
    detector.setStyleSheet(check_box_style())

    radio_ctf = QRadioButton("CTF")
    radio_ctf.setStyleSheet(radio_box_style())
    radio_abs_ctf = QRadioButton("|CTF|")
    radio_abs_ctf.setStyleSheet(radio_box_style())
    radio_ctf_squared = QRadioButton("CTF²")
    radio_ctf_squared.setStyleSheet(radio_box_style())

    # group them
    radio_group = QButtonGroup()
    radio_group.addButton(radio_ctf)
    radio_group.addButton(radio_abs_ctf)
    radio_group.addButton(radio_ctf_squared)
    radio_ctf.setChecked(True)

    # ——— small helpers to build each “row” as a QLayout ———
    def _make_envelope_row():
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(temporal)
        row.addStretch()
        row.addWidget(spatial)
        row.addStretch()
        row.addWidget(detector)
        return row

    def _make_radio_row():
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(radio_ctf)
        row.addWidget(radio_abs_ctf)
        row.addWidget(radio_ctf_squared)
        return row

    # ——— now dispatch into our generic factory ———
    factories = {
        # just a plain label
        "envelope_label": lambda: QLabel("Envelope Function"),
        # envelope‐checkbox row + a bit of spacing afterward
        "envelope_checks": lambda: (_make_envelope_row(), int(6 * get_ui_scale())),
        # the CTF‐mode label
        "ctf_label": lambda: QLabel("CTF Display Mode"),
        # the radio‐buttons row
        "ctf_radios": lambda: (_make_radio_row(), None),
    }

    box, widgets = build_group_control(
        "CTF Calculation Options",
        left_panel_qgroupbox_style(),
        factories,
        margin=int(12 * get_ui_scale()),
        spacing=int(6 * get_ui_scale()),
    )

    widgets.update(
        {
            "temporal_check": temporal,
            "spatial_check": spatial,
            "detector_check": detector,
            "radio_group": radio_group,
            "radio_ctf": radio_ctf,
            "radio_abs_ctf": radio_abs_ctf,
            "radio_ctf_squared": radio_ctf_squared,
        }
    )

    return box, widgets


def build_button_group():
    """
    Create a box for all the push buttons.

    Returns:
      - box: QGroupBox already populated with all the sliders
      - buttons: a dict of each button, so controller can hook signals
    """
    box = QHBoxLayout()

    # Create buttons
    reset_button = QPushButton("Reset")
    reset_button.setStyleSheet(button_style())
    save_img_button = QPushButton("Save Plot")
    save_img_button.setStyleSheet(default_button_style())
    save_csv_button = QPushButton("Save CSV")
    save_csv_button.setStyleSheet(default_button_style())

    box.addWidget(reset_button)
    box.addWidget(save_img_button)
    box.addWidget(save_csv_button)

    buttons = {
        "reset": reset_button,
        "save_plot": save_img_button,
        "save_csv": save_csv_button,
    }

    return box, buttons
