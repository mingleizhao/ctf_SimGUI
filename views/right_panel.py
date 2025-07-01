"""
right_panel.py
--------------

This module defines build functions for constructing and configuring the right panel GUI widgets of
the CTFSimGUI application's interface.

Functions:
    - build_plot_tabs(parent=None)
        Constructs a QTabWidget with tabs for 1D-CTF, 2D-CTF, ICE-CTF, TOMO-CTF, and IMAGE-CTF
        plots.

    - build_1d_ctf_tab(parent=None)
        Constructs the 1D-CTF tab with a canvas and control panel.

    - build_2d_ctf_tab(parent=None)
        Constructs the 2D-CTF tab with a canvas and control panel.

    - build_ice_ctf_tab(parent=None)
        Constructs the ICE-CTF tab with a canvas and control panel.

    - build_tomo_ctf_tab(parent=None)
        Constructs the TOMO-CTF tab with a canvas and control panel.

    - build_image_ctf_tab(parent=None)
        Constructs the IMAGE-CTF tab with a canvas and control panel.

    - build_ctf_tab(parent, canvas_kwargs, build_controls):
        Constructs a tab containing a canvas and control panel, injecting the canvas into the
        controls dict.

    - build_axis_control(...):
        Constructs a QGridLayout containing labeled QDoubleSpinBox widgets for axis range control.

    - create_info_button():
        Creates a styled info QPushButton for displaying additional information.

    - create_toggle_button():
        Creates a styled toggle QPushButton for showing/hiding annotations.

    - create_tab_buttons(info_button, toggle_button):
        Groups info and toggle buttons vertically for use in tab layouts.

    - setup_horizontal_scrollable_area(control):
        Wraps a control widget in a horizontally scrollable QScrollArea for responsive layouts.
"""

from typing import Callable, Tuple
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QSizePolicy,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QGridLayout,
    QDoubleSpinBox,
    QTabWidget,
    QScrollArea,
)
from .customized_widgets import LabeledSlider, MplCanvas
from .styles import (
    right_panel_qgroupbox_style,
    button_style,
    qtabwidget_style,
    default_button_style,
    check_box_style,
    double_spin_box_style,
    get_ui_scale,
    info_button_style,
    scroll_area_style,
)


def build_plot_tabs(parent=None):
    """
    Create a QTabWidget with three tabs for 1D-CTF, 2D-CTF, ICE-CTF plots.
    Each tab holds its own MplCanvas.

    Returns:
        tabs: QWidget containing the tabs.
        widgets: a list of widget dict for each tab, so controller can hook signals
    """
    tabs = QTabWidget(parent)
    tabs.setStyleSheet(qtabwidget_style())

    tab1, tab1_widgets = build_1d_ctf_tab(parent)
    tab2, tab2_widgets = build_2d_ctf_tab(parent)
    tab3, tab3_widgets = build_ice_ctf_tab(parent)
    tab4, tab4_widgets = build_tomo_ctf_tab(parent)
    tab5, tab5_widgets = build_image_ctf_tab(parent)

    tabs.addTab(tab1, "1D")
    tabs.addTab(tab2, "2D")
    tabs.addTab(tab3, "Ice")
    tabs.addTab(tab4, "Tilt")
    tabs.addTab(tab5, "Image")

    widgets = [tab1_widgets, tab2_widgets, tab3_widgets, tab4_widgets, tab5_widgets]

    return tabs, widgets


def build_1d_ctf_tab(parent=None):
    """
    Build the 1D-CTF tab.

    Returns:
        tab: QWidget containing the canvas and controls.
        widgets: a dict of each widget, so controller can hook signals
    """

    def _build_controls():
        margin = int(12 * get_ui_scale())
        # sliders + axis spinboxes
        xlim = LabeledSlider(
            "X-axis Limit (Å⁻¹)", min_value=0.001, max_value=1.1, step=0.001, value_format="{:.3f}"
        )
        ylim = LabeledSlider(
            "y-axis Limit (Å⁻¹)", min_value=0.001, max_value=1.1, step=0.001, value_format="{:.3f}"
        )

        axis_grid, axis_boxes = build_axis_control(
            x_min_range=(-0.1, 1),
            x_min_value=0,
            x_max_range=(0.001, 1.1),
            x_max_value=0.5,
            y_min_range=(-1.1, 1),
            y_min_value=-1,
            y_max_range=(0.001, 1.1),
            y_max_value=1,
        )
        # envelope checks
        t = QCheckBox("Temporal Envelope")
        t.setStyleSheet(check_box_style())
        s = QCheckBox("Spatial Envelope")
        s.setStyleSheet(check_box_style())
        d = QCheckBox("Detector Envelope")
        s.setStyleSheet(check_box_style())
        total = QCheckBox("Total Envelope")
        total.setStyleSheet(check_box_style())
        y0 = QCheckBox("y=0…")
        y0.setStyleSheet(check_box_style())
        lg = QCheckBox("Legend")
        lg.setStyleSheet(check_box_style())

        grid = QGridLayout()
        grid.addWidget(t, 0, 0)
        grid.addWidget(s, 1, 0)
        grid.addWidget(d, 0, 1)
        grid.addWidget(total, 1, 1)
        grid.addWidget(y0, 0, 2)
        grid.addWidget(lg, 1, 2)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(margin)

        info_btn = create_info_button()
        toggle_btn = create_toggle_button()

        h = QHBoxLayout()
        h.addWidget(xlim)
        h.addStretch()
        h.addWidget(ylim)
        h.addStretch()
        h.addLayout(axis_grid)
        h.addStretch()
        h.addLayout(grid)
        h.addStretch()
        h.addLayout(create_tab_buttons(info_btn, toggle_btn))
        h.setContentsMargins(margin, margin, margin, margin)

        box = QGroupBox()  # this outer box will get styled later by the helper
        box.setLayout(h)

        # finally return the box *and* only those widgets the controller cares about
        return box, {
            "xlim_slider": xlim,
            "ylim_slider": ylim,
            "x_min_box": axis_boxes["x_min"],
            "x_max_box": axis_boxes["x_max"],
            "y_min_box": axis_boxes["y_min"],
            "y_max_box": axis_boxes["y_max"],
            "show_temp": t,
            "show_spatial": s,
            "show_detector": d,
            "show_total": total,
            "show_y0": y0,
            "show_legend": lg,
            "info": info_btn,
            "annotation": toggle_btn,
        }

    return build_ctf_tab(
        parent, canvas_kwargs={"width": 5, "height": 4}, build_controls=_build_controls
    )


def build_2d_ctf_tab(parent=None):
    """
    Build the 2D-CTF tab.

    Returns:
        tab: QWidget containing the canvas and controls.
        widgets: a dict of each widget, so controller can hook signals
    """

    def _build_controls():
        scale = get_ui_scale()
        spacing = int(6 * scale)
        margin = int(12 * scale)

        # --- Spin‐boxes for scaling ---
        grid = QGridLayout()
        freq_scale = QDoubleSpinBox()
        freq_scale.setRange(0.1, 0.5)
        freq_scale.setValue(0.5)
        freq_scale.setSingleStep(0.02)
        freq_scale.setDecimals(2)
        freq_scale.setStyleSheet(double_spin_box_style())

        gray_scale = QDoubleSpinBox()
        gray_scale.setRange(0.05, 1)
        gray_scale.setValue(1)
        gray_scale.setSingleStep(0.02)
        gray_scale.setDecimals(2)
        gray_scale.setStyleSheet(double_spin_box_style())

        grid.addWidget(QLabel("|Spatial Frequency|:"), 0, 0)
        grid.addWidget(freq_scale, 0, 1)
        grid.addWidget(QLabel("Max Gray Scale:"), 1, 0)
        grid.addWidget(gray_scale, 1, 1)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(spacing)

        # --- Axis‐control spinboxes ---
        axis_grid, axis_boxes = build_axis_control(
            x_min_range=(-0.5, 0.5),
            x_min_value=-0.5,
            x_max_range=(-0.5, 0.5),
            x_max_value=0.5,
            y_min_range=(-0.5, 0.5),
            y_min_value=-1,
            y_max_range=(-0.5, 0.5),
            y_max_value=1,
        )

        # --- Defocus sliders ---
        defocus_diff = LabeledSlider(
            "Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.2f}"
        )
        defocus_az = LabeledSlider(
            "Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}"
        )

        # --- Info & toggle buttons ---
        info_btn = create_info_button()
        toggle_btn = create_toggle_button()

        # --- assemble the H‐layout ---
        h = QHBoxLayout()
        h.addLayout(grid)
        h.addStretch()
        h.addLayout(axis_grid)
        h.addStretch()
        h.addWidget(defocus_diff)
        h.addStretch()
        h.addWidget(defocus_az)
        h.addStretch()
        h.addLayout(create_tab_buttons(info_btn, toggle_btn))
        h.setContentsMargins(margin, margin, margin, margin)

        # wrap in a QGroupBox (styling applied by the helper)
        box = QGroupBox()
        box.setLayout(h)

        return box, {
            "freq_box": freq_scale,
            "gray_box": gray_scale,
            "x_min_box": axis_boxes["x_min"],
            "x_max_box": axis_boxes["x_max"],
            "y_min_box": axis_boxes["y_min"],
            "y_max_box": axis_boxes["y_max"],
            "defocus_diff": defocus_diff,
            "defocus_az": defocus_az,
            "info": info_btn,
            "annotation": toggle_btn,
        }

    # now hand off to the generic tab‐maker
    return build_ctf_tab(
        parent=parent, canvas_kwargs=dict(width=5, height=4), build_controls=_build_controls
    )


def build_ice_ctf_tab(parent=None):
    """
    Build the ICE-CTF tab.

    Returns:
        tab: QWidget containing the canvas and controls.
        widgets: a dict of each widget, so controller can hook signals
    """

    def _build_controls():
        scale = get_ui_scale()
        spacing = int(6 * scale)
        margin = int(12 * scale)

        # --- frequency & gray‐scale spinboxes in a grid ---
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(spacing)

        freq = QDoubleSpinBox()
        freq.setRange(0.1, 0.5)
        freq.setValue(0.5)
        freq.setSingleStep(0.02)
        freq.setDecimals(2)
        freq.setStyleSheet(double_spin_box_style())

        gray = QDoubleSpinBox()
        gray.setRange(0.05, 1)
        gray.setValue(1)
        gray.setSingleStep(0.02)
        gray.setDecimals(2)
        gray.setStyleSheet(double_spin_box_style())

        grid.addWidget(QLabel("|Spatial Frequency|:"), 0, 0)
        grid.addWidget(freq, 0, 1)
        grid.addWidget(QLabel("Max Gray Scale:"), 1, 0)
        grid.addWidget(gray, 1, 1)

        # --- the four sliders ---
        thickness = LabeledSlider(
            "Ice Thickness (nm)", min_value=1, max_value=1000, step=1, value_format="{:.0f}"
        )
        xlim = LabeledSlider(
            "X-axis Limit (Å⁻¹)", min_value=0.1, max_value=1.1, step=0.01, value_format="{:.2f}"
        )
        df_diff = LabeledSlider(
            "Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.2f}"
        )
        df_az = LabeledSlider(
            "Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}"
        )

        # --- info & toggle buttons ---
        info_btn = create_info_button()
        toggle_btn = create_toggle_button()

        # --- assemble horizontal layout ---
        h = QHBoxLayout()
        h.setContentsMargins(margin, margin, margin, margin)
        h.addLayout(grid)
        h.addStretch()
        h.addWidget(thickness)
        h.addStretch()
        h.addWidget(xlim)
        h.addStretch()
        h.addWidget(df_diff)
        h.addStretch()
        h.addWidget(df_az)
        h.addStretch()
        h.addLayout(create_tab_buttons(info_btn, toggle_btn))

        # wrap in a styled QGroupBox
        box = QGroupBox()
        box.setLayout(h)
        box.setStyleSheet(right_panel_qgroupbox_style())

        # return the group-box and the dict of controls
        return box, {
            "freq_box": freq,
            "gray_box": gray,
            "thickness": thickness,
            "xlim": xlim,
            "defocus_diff": df_diff,
            "defocus_az": df_az,
            "info": info_btn,
            "annotation": toggle_btn,
        }

    # canvas layout parameters
    subplot_args = {
        1: {"rowspan": slice(0, 1), "colspan": slice(0, 2)},
        2: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},
        3: {"rowspan": slice(1, 2), "colspan": slice(1, 2)},
    }

    return build_ctf_tab(
        parent=parent,
        canvas_kwargs={
            "subplot_grid": (2, 2),
            "subplot_args": subplot_args,
            "width": 5,
            "height": 4,
        },
        build_controls=_build_controls,
    )


def build_tomo_ctf_tab(parent=None):
    """
    Build the TOMO-CTF tab.

    Returns:
        tab: QWidget containing the canvas and controls.
        widgets: a dict of each widget, so controller can hook signals
    """

    def _build_controls():
        scale = get_ui_scale()
        spacing = int(6 * scale)
        margin = int(12 * scale)

        # spinboxes grid
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(spacing)

        size_scale = QDoubleSpinBox()
        size_scale.setRange(0.4, 2)
        size_scale.setValue(1)
        size_scale.setSingleStep(0.02)
        size_scale.setDecimals(2)
        size_scale.setStyleSheet(double_spin_box_style())

        gray_scale = QDoubleSpinBox()
        gray_scale.setRange(0.05, 1)
        gray_scale.setValue(1)
        gray_scale.setSingleStep(0.02)
        gray_scale.setDecimals(2)
        gray_scale.setStyleSheet(double_spin_box_style())

        grid.addWidget(QLabel("Sample Size (µm):"), 0, 0)
        grid.addWidget(size_scale, 0, 1)
        grid.addWidget(QLabel("Max Gray Scale:"), 1, 0)
        grid.addWidget(gray_scale, 1, 1)

        # sliders
        thickness_slider = LabeledSlider(
            "Sample Thickness (nm)", min_value=50, max_value=1000, step=1, value_format="{:.0f}"
        )
        tilt_slider = LabeledSlider(
            "Tilt Angle (°)", min_value=-70, max_value=70, step=0.1, value_format="{:.1f}"
        )
        df_diff_slider = LabeledSlider(
            "Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.2f}"
        )
        df_az_slider = LabeledSlider(
            "Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}"
        )

        # info & toggle buttons
        info_btn = create_info_button()
        toggle_btn = create_toggle_button()

        # assemble horizontal layout
        h = QHBoxLayout()
        h.setContentsMargins(margin, margin, margin, margin)
        h.addLayout(grid)
        h.addStretch()
        h.addWidget(thickness_slider)
        h.addStretch()
        h.addWidget(tilt_slider)
        h.addStretch()
        h.addWidget(df_diff_slider)
        h.addStretch()
        h.addWidget(df_az_slider)
        h.addStretch()
        h.addLayout(create_tab_buttons(info_btn, toggle_btn))

        # wrap in a styled QGroupBox
        box = QGroupBox()
        box.setLayout(h)
        box.setStyleSheet(right_panel_qgroupbox_style())

        return box, {
            "size_box": size_scale,
            "gray_box": gray_scale,
            "thickness": thickness_slider,
            "tilt": tilt_slider,
            "defocus_diff": df_diff_slider,
            "defocus_az": df_az_slider,
            "info": info_btn,
            "annotation": toggle_btn,
        }

    # 2) supply the canvas parameters:
    subplot_args = {
        1: {"rowspan": slice(0, 1), "colspan": slice(0, 1)},
        3: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},
        4: {"rowspan": slice(0, 2), "colspan": slice(1, 2)},
    }

    return build_ctf_tab(
        parent=parent,
        canvas_kwargs={
            "subplot_grid": (2, 2),
            "subplot_args": subplot_args,
            "width": 5,
            "height": 4,
        },
        build_controls=_build_controls,
    )


def build_image_ctf_tab(parent=None):
    """
    Build the IMAGE-CTF tab.

    Returns:
        tab: QWidget containing the canvas and controls.
        widgets: a dict of each widget, so controller can hook signals
    """

    def _build_controls():
        scale = get_ui_scale()
        margin = int(12 * scale)
        spacing = int(6 * scale)

        # --- upload button ---
        upload_btn = QPushButton("Upload Image")
        upload_btn.setStyleSheet(button_style())

        # --- zoom spinboxes grid ---
        zoom_boxes = QGridLayout()
        zoom_boxes.setContentsMargins(0, 0, 0, 0)
        zoom_boxes.setSpacing(spacing)

        image_zoom = QDoubleSpinBox()
        image_zoom.setRange(100, 500)
        image_zoom.setValue(100)
        image_zoom.setSingleStep(5)
        image_zoom.setDecimals(0)
        image_zoom.setStyleSheet(double_spin_box_style())

        fft_zoom = QDoubleSpinBox()
        fft_zoom.setRange(100, 500)
        fft_zoom.setValue(100)
        fft_zoom.setSingleStep(5)
        fft_zoom.setDecimals(0)
        fft_zoom.setStyleSheet(double_spin_box_style())

        zoom_boxes.addWidget(QLabel("Image Zoom (%):"), 0, 0)
        zoom_boxes.addWidget(image_zoom, 0, 1)
        zoom_boxes.addWidget(QLabel("FFT Zoom (%):"), 1, 0)
        zoom_boxes.addWidget(fft_zoom, 1, 1)

        # --- contrast spinboxes grid ---
        contrast_boxes = QGridLayout()
        contrast_boxes.setContentsMargins(0, 0, 0, 0)
        contrast_boxes.setSpacing(spacing)

        image_contrast = QDoubleSpinBox()
        image_contrast.setRange(51, 100)
        image_contrast.setValue(100)
        image_contrast.setSingleStep(0.1)
        image_contrast.setDecimals(1)
        image_contrast.setStyleSheet(double_spin_box_style())

        fft_contrast = QDoubleSpinBox()
        fft_contrast.setRange(51, 100)
        fft_contrast.setValue(100)
        fft_contrast.setSingleStep(0.1)
        fft_contrast.setDecimals(1)
        fft_contrast.setStyleSheet(double_spin_box_style())

        contrast_boxes.addWidget(QLabel("Image Contrast (%):"), 0, 0)
        contrast_boxes.addWidget(image_contrast, 0, 1)
        contrast_boxes.addWidget(QLabel("FFT Contrast (%):"), 1, 0)
        contrast_boxes.addWidget(fft_contrast, 1, 1)

        # --- defocus sliders ---
        defocus_diff_slider = LabeledSlider(
            "Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.2f}"
        )
        defocus_az_slider = LabeledSlider(
            "Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}"
        )

        # --- sync & invert controls ---
        sync_cb = QCheckBox("Sync Greyscale")
        sync_cb.setChecked(False)
        sync_cb.setStyleSheet(check_box_style())
        sync_cb.setToolTip("Synchronize greyscale between original and convolved images")

        invert_btn = QPushButton("Invert Image")
        invert_btn.setStyleSheet(default_button_style())
        invert_btn.setToolTip("Invert greyscale of original and convolved images")

        contrast_group = QVBoxLayout()
        contrast_group.setContentsMargins(0, 0, 0, 0)
        contrast_group.setSpacing(spacing)
        contrast_group.addWidget(sync_cb)
        contrast_group.addWidget(invert_btn)

        # --- info / annotation toggles ---
        info_btn = create_info_button()
        toggle_btn = create_toggle_button()

        # --- assemble horizontal control panel ---
        h = QHBoxLayout()
        h.setContentsMargins(margin, margin, margin, margin)
        h.addWidget(upload_btn)
        h.addStretch()
        h.addLayout(zoom_boxes)
        h.addStretch()
        h.addLayout(contrast_boxes)
        h.addStretch()
        h.addLayout(contrast_group)
        h.addStretch()
        h.addWidget(defocus_diff_slider)
        h.addStretch()
        h.addWidget(defocus_az_slider)
        h.addStretch()
        h.addLayout(create_tab_buttons(info_btn, toggle_btn))

        # wrap in a styled QGroupBox
        box = QGroupBox()
        box.setLayout(h)
        box.setStyleSheet(right_panel_qgroupbox_style())

        return box, {
            "upload_button": upload_btn,
            "image_zoom": image_zoom,
            "fft_zoom": fft_zoom,
            "image_contrast": image_contrast,
            "fft_contrast": fft_contrast,
            "sync_checkbox": sync_cb,
            "invert_button": invert_btn,
            "defocus_diff": defocus_diff_slider,
            "defocus_az": defocus_az_slider,
            "info": info_btn,
            "annotation": toggle_btn,
        }

    # subplot arrangement exactly as before
    subplot_args = {
        1: {"rowspan": slice(0, 1), "colspan": slice(0, 1)},
        2: {"rowspan": slice(0, 1), "colspan": slice(1, 2)},
        3: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},
        4: {"rowspan": slice(1, 2), "colspan": slice(1, 2)},
    }

    return build_ctf_tab(
        parent=parent,
        canvas_kwargs={
            "subplot_grid": (2, 2),
            "subplot_args": subplot_args,
            "width": 5,
            "height": 4,
        },
        build_controls=_build_controls,
    )


def build_ctf_tab(
    parent, canvas_kwargs: dict, build_controls: Callable[[], Tuple[QGroupBox, dict]]
) -> Tuple[QWidget, dict]:
    """
    Generic “CTF tab” builder.

    - canvas_kwargs is passed to MplCanvas(...)
    - build_controls() must return (control_groupbox, control_widgets_dict)
    - we then wrap the canvas + control panel + scroll, inject the canvas
      into that dict, and return (tab_widget, widgets_dict).
    """
    # 1) make the canvas
    canvas = MplCanvas(parent, **canvas_kwargs)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # 2) make the tab and put canvas at the top
    tab = QWidget()
    layout = QVBoxLayout(tab)
    m = int(12 * get_ui_scale())
    layout.setContentsMargins(m, m, m, m)
    layout.addWidget(canvas)

    # 3) ask the client to build its unique controls
    controls_box, controls_dict = build_controls()
    # inject the canvas reference
    controls_dict["canvas"] = canvas

    # 4) wrap & style the control panel exactly once
    controls_box.setStyleSheet(right_panel_qgroupbox_style())
    layout.addWidget(setup_horizontal_scrollable_area(controls_box))

    return tab, controls_dict


def build_axis_control(
    x_min_range: tuple[float, float],
    x_min_value: float,
    x_max_range: tuple[float, float],
    x_max_value: float,
    y_min_range: tuple[float, float],
    y_min_value: float,
    y_max_range: tuple[float, float],
    y_max_value: float,
    single_step: float = 0.01,
    decimals: int = 2,
):
    """
    Create a grid layout with labeled QDoubleSpinBox widgets for controlling X and Y axis limits.
    Args:
        x_min_range (tuple): (min, max) range for x_min.
        x_min_value (float): Default value for x_min.
        x_max_range (tuple): (min, max) range for x_max.
        x_max_value (float): Default value for x_max.
        y_min_range (tuple): (min, max) range for y_min.
        y_min_value (float): Default value for y_min.
        y_max_range (tuple): (min, max) range for y_max.
        y_max_value (float): Default value for y_max.
        single_step (float, optional): Increment step for all spin boxes. Defaults to 0.005.
        decimals (int, optional): Number of decimal places. Defaults to 3.

    Returns:
        QVBoxLayout: Layout containing X and Y axis controls.
        spin_boxes: a dict of each spin box.
    """

    def configure_spinbox(spinbox, value, value_range):
        """Helper function to configure a QDoubleSpinBox."""
        spinbox.setRange(*value_range)
        spinbox.setValue(value)
        spinbox.setSingleStep(single_step)
        spinbox.setDecimals(decimals)
        spinbox.setStyleSheet(double_spin_box_style())

    x_min = QDoubleSpinBox()
    x_max = QDoubleSpinBox()
    y_min = QDoubleSpinBox()
    y_max = QDoubleSpinBox()

    configure_spinbox(x_min, x_min_value, x_min_range)
    configure_spinbox(x_max, x_max_value, x_max_range)
    configure_spinbox(y_min, y_min_value, y_min_range)
    configure_spinbox(y_max, y_max_value, y_max_range)

    # Create labels
    labels = {
        "x_axis": QLabel("X-Axis:"),
        "x_min": QLabel("Min"),
        "x_max": QLabel("Max"),
        "y_axis": QLabel("Y-Axis:"),
        "y_min": QLabel("Min"),
        "y_max": QLabel("Max"),
    }

    # Create Grid Layout
    grid_layout = QGridLayout()

    # X-Axis Controls (Row 0)
    grid_layout.addWidget(labels["x_axis"], 0, 0)
    grid_layout.addWidget(labels["x_min"], 0, 1)
    grid_layout.addWidget(x_min, 0, 2)
    grid_layout.addWidget(labels["x_max"], 0, 3)
    grid_layout.addWidget(x_max, 0, 4)

    # Y-Axis Controls (Row 1)
    grid_layout.addWidget(labels["y_axis"], 1, 0)
    grid_layout.addWidget(labels["y_min"], 1, 1)
    grid_layout.addWidget(y_min, 1, 2)
    grid_layout.addWidget(labels["y_max"], 1, 3)
    grid_layout.addWidget(y_max, 1, 4)
    grid_layout.setContentsMargins(0, 0, 0, 0)
    grid_layout.setSpacing(int(6 * get_ui_scale()))

    spin_boxes = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    return grid_layout, spin_boxes


def create_info_button():
    """
    Create a styled info QPushButton for displaying additional information.

    Returns:
        QPushButton: The info button widget.
    """
    info_button = QPushButton("i")
    info_button.setFixedSize(int(18 * get_ui_scale()), int(18 * get_ui_scale()))
    info_button.setToolTip("Additional info")
    info_button.setStyleSheet(info_button_style())

    return info_button


def create_toggle_button():
    """
    Create a styled toggle QPushButton for showing or hiding annotations.

    Returns:
        QPushButton: The toggle button widget.
    """
    toggle_button = QPushButton("V")
    toggle_button.setCheckable(True)
    toggle_button.setFixedSize(int(18 * get_ui_scale()), int(18 * get_ui_scale()))
    toggle_button.setToolTip("Show/Hide annotation")
    toggle_button.setStyleSheet(info_button_style())

    return toggle_button


def create_tab_buttons(info_button, toggle_button):
    """
    Group info and toggle buttons vertically for use in tab layouts.

    Args:
        info_button (QPushButton): The info button to include.
        toggle_button (QPushButton): The toggle button to include.

    Returns:
        QVBoxLayout: Layout containing the provided buttons.
    """
    button_group = QVBoxLayout()
    button_group.setSpacing(int(6 * get_ui_scale()))
    button_group.addWidget(info_button)
    button_group.addWidget(toggle_button)

    return button_group


def setup_horizontal_scrollable_area(control):
    """
    Wrap a control widget in a horizontally scrollable QScrollArea.

    Args:
        control (QWidget): The widget to make horizontally scrollable.

    Returns:
        QScrollArea: The scroll area containing the control widget.
    """
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll_area.setStyleSheet(scroll_area_style())

    scroll_content = QWidget()
    scroll_layout = QHBoxLayout(scroll_content)
    scroll_layout.setContentsMargins(0, 0, 0, 0)
    scroll_layout.addWidget(control)
    scroll_area.setWidget(scroll_content)

    return scroll_area
