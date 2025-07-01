"""
ui_main_window.py
--------------

Defines the Ui_MainWindow class responsible for building and laying out
all GUI components of the CTF Simulation application. This includes
parameter panels, tabbed plots, and the exposure of widget references
for use by the application's controller.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QScrollArea,
    QDesktopWidget,
    QApplication,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QTabWidget,
    QDoubleSpinBox,
)
from PyQt5.QtCore import Qt
from .customized_widgets import LabeledSlider, MplCanvas
from .left_panel import (
    build_microscope_group,
    build_imaging_group,
    build_detector_group,
    build_plotting_group,
    build_button_group,
)
from .right_panel import (
    build_plot_tabs,
)
from .styles import scroll_area_style, get_ui_scale


class Ui_MainWindow:
    """
    The view layer of the application — responsible for laying out and styling
    all widgets in the main window, including:

      - A scrollable left panel of parameter groups (microscope, imaging, detector, plotting,
      buttons),
      - A right panel containing the tabbed Matplotlib canvases for each CTF view (1D, 2D, ICE,
      TOMO, IMAGE),
      - Runtime UI scaling based on screen resolution and DPI,
      - Exposure of all child widgets as attributes so the controller can hook up event handlers.

    This class does *not* implement any behavior; it merely builds and arranges the UI
    and hands off widget references for the controller layer to wire up.
    """

    # ——————————————————————————————————————————————————————————
    # class‐level annotations for everything setupUi will populate
    # — left panel —
    voltage_slider: LabeledSlider
    voltage_stability_slider: LabeledSlider
    electron_source_angle_slider: LabeledSlider
    electron_source_spread_slider: LabeledSlider
    chromatic_aberr_slider: LabeledSlider
    spherical_aberr_slider: LabeledSlider
    obj_lens_stability_slider: LabeledSlider

    defocus_slider: LabeledSlider
    amplitude_contrast_slider: LabeledSlider
    additional_phase_slider: LabeledSlider

    detector_combo: QComboBox
    pixel_size_slider: LabeledSlider

    temporal_env_check: QCheckBox
    spatial_env_check: QCheckBox
    detector_env_check: QCheckBox
    radio_button_group: QButtonGroup
    radio_ctf: QRadioButton
    radio_abs_ctf: QRadioButton
    radio_ctf_squared: QRadioButton

    reset_button: QPushButton
    save_img_button: QPushButton
    save_csv_button: QPushButton

    # — right panel tabs —
    plot_tabs: QTabWidget

    # 1D tab
    canvas_1d: MplCanvas
    xlim_slider_1d: LabeledSlider
    ylim_slider_1d: LabeledSlider
    plot_1d_x_min: QDoubleSpinBox
    plot_1d_x_max: QDoubleSpinBox
    plot_1d_y_min: QDoubleSpinBox
    plot_1d_y_max: QDoubleSpinBox
    show_temp: QCheckBox
    show_spatial: QCheckBox
    show_detector: QCheckBox
    show_total: QCheckBox
    show_y0: QCheckBox
    show_legend: QCheckBox
    info_button_1d: QPushButton
    toggle_button_1d: QPushButton

    # 2D tab
    canvas_2d: MplCanvas
    freq_scale_2d: QDoubleSpinBox
    gray_scale_2d: QDoubleSpinBox
    plot_2d_x_min: QDoubleSpinBox
    plot_2d_x_max: QDoubleSpinBox
    plot_2d_y_min: QDoubleSpinBox
    plot_2d_y_max: QDoubleSpinBox
    defocus_diff_slider_2d: LabeledSlider
    defocus_az_slider_2d: LabeledSlider
    info_button_2d: QPushButton
    toggle_button_2d: QPushButton

    # ICE tab
    canvas_ice: MplCanvas
    freq_scale_ice: QDoubleSpinBox
    gray_scale_ice: QDoubleSpinBox
    ice_thickness_slider: LabeledSlider
    xlim_slider_ice: LabeledSlider
    defocus_diff_slider_ice: LabeledSlider
    defocus_az_slider_ice: LabeledSlider
    info_button_ice: QPushButton
    toggle_button_ice: QPushButton

    # TOMO tab
    canvas_tomo: MplCanvas
    sample_size_tomo: QDoubleSpinBox
    gray_scale_tomo: QDoubleSpinBox
    sample_thickness_slider_tomo: LabeledSlider
    tilt_slider_tomo: LabeledSlider
    defocus_diff_slider_tomo: LabeledSlider
    defocus_az_slider_tomo: LabeledSlider
    info_button_tomo: QPushButton
    toggle_button_tomo: QPushButton

    # IMAGE tab
    canvas_image: MplCanvas
    upload_btn: QPushButton
    size_scale_image: QDoubleSpinBox
    size_scale_fft: QDoubleSpinBox
    contrast_scale_image: QDoubleSpinBox
    contrast_scale_fft: QDoubleSpinBox
    contrast_sync_checkbox: QCheckBox
    invert_btn: QPushButton
    defocus_diff_slider_image: LabeledSlider
    defocus_az_slider_image: LabeledSlider
    info_button_image: QPushButton
    toggle_button_image: QPushButton

    # font and line sizes
    font_sizes: dict
    linewidth: int
    # — end of annotations ————————————————————————————————————————

    def setupUi(self, MainWindow):
        """
        Lay out and style all widgets in the main application window.

        This method performs the following steps:
          1. Sets window title and initial size (with DPI/resolution-based scaling).
          2. Adjusts the application font according to the computed UI scale.
          3. Creates the central QWidget and QHBoxLayout to host two panels.
          4. Builds the left scrollable panel of parameter groups:
             - Microscope parameters
             - Imaging parameters
             - Detector parameters
             - Plotting options
             - Action buttons (Reset, Save Plot, Save CSV)
          5. Wraps the left panel in a QScrollArea for vertical scrolling.
          6. Builds the right panel as a QTabWidget containing:
             - 1D CTF plot
             - 2D CTF plot
             - ICE thickness plot
             - TOMO tilt plot
             - Image + FFT plot
          7. Exposes every relevant widget (sliders, buttons, canvases, etc.) as attributes
             on `self` so that the controller layer can attach event handlers.

        Note:
            This is purely a “view” class—no application logic is defined here.
        """
        # Window settings
        MainWindow.setWindowTitle("CTF Simulation")

        screen_height = QDesktopWidget().availableGeometry().height()
        # screen_width = QDesktopWidget().availableGeometry().width()

        desired_w, desired_h = 1620, 1080

        MainWindow.resize(int(desired_w * get_ui_scale()), int(desired_h * get_ui_scale()))

        base_fonts = {"tiny": 10, "small": 12, "medium": 14, "large": 16}

        self.font_sizes = {k: max(1, int(v * get_ui_scale())) for k, v in base_fonts.items()}
        if screen_height < 800:
            self.linewidth = 1
        elif screen_height < 1080:
            self.linewidth = 2
        else:
            self.linewidth = 3

        app = QApplication.instance()
        font = app.font()  # get the default font
        font.setPointSizeF(font.pointSizeF() * get_ui_scale())
        app.setFont(font)

        # Central widget + main layout
        central = QWidget(MainWindow)
        MainWindow.setCentralWidget(central)
        main_l = QHBoxLayout(central)

        # Left panel container
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(0)

        # Build parameter groups
        mic_box, mic_sliders = build_microscope_group()
        img_box, img_sliders = build_imaging_group()
        det_box, det_widgets = build_detector_group()
        plot_box, plot_widgets = build_plotting_group()
        btn_layout, btns = build_button_group()

        # Scrollable container for left panel
        scroll_cont = QWidget()
        scroll_l = QVBoxLayout(scroll_cont)
        scroll_l.setContentsMargins(0, 0, 0, 0)
        scroll_l.setSpacing(int(10 * get_ui_scale()))
        for w in (mic_box, img_box, det_box, plot_box):
            scroll_l.addWidget(w)
        scroll_l.addLayout(btn_layout)
        scroll_l.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_cont)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll.setStyleSheet(scroll_area_style())
        left_l.addWidget(scroll)
        main_l.addWidget(left_w, stretch=0)

        # Right panel: tabbed plots
        tabs, tab_widgets = build_plot_tabs(MainWindow)
        main_l.addWidget(tabs, stretch=1)

        # Expose all widgets for controller
        self.voltage_slider = mic_sliders["voltage"]
        self.voltage_stability_slider = mic_sliders["stability"]
        self.electron_source_angle_slider = mic_sliders["angle"]
        self.electron_source_spread_slider = mic_sliders["energy"]
        self.chromatic_aberr_slider = mic_sliders["chrom_aberr"]
        self.spherical_aberr_slider = mic_sliders["sph_aberr"]
        self.obj_lens_stability_slider = mic_sliders["obj_stab"]

        self.defocus_slider = img_sliders["defocus"]
        self.amplitude_contrast_slider = img_sliders["amplitude_contrast"]
        self.additional_phase_slider = img_sliders["additional_phase"]

        self.detector_combo = det_widgets["detectors"]
        self.pixel_size_slider = det_widgets["pixel_size"]

        self.temporal_env_check = plot_widgets["temporal_check"]
        self.spatial_env_check = plot_widgets["spatial_check"]
        self.detector_env_check = plot_widgets["detector_check"]
        self.radio_button_group = plot_widgets["radio_group"]
        self.radio_ctf = plot_widgets["radio_ctf"]
        self.radio_abs_ctf = plot_widgets["radio_abs_ctf"]
        self.radio_ctf_squared = plot_widgets["radio_ctf_squared"]

        self.reset_button = btns["reset"]
        self.save_img_button = btns["save_plot"]
        self.save_csv_button = btns["save_csv"]

        self.plot_tabs = tabs

        self.canvas_1d = tab_widgets[0]["canvas"]
        self.xlim_slider_1d = tab_widgets[0]["xlim_slider"]
        self.ylim_slider_1d = tab_widgets[0]["ylim_slider"]
        self.plot_1d_x_min = tab_widgets[0]["x_min_box"]
        self.plot_1d_x_max = tab_widgets[0]["x_max_box"]
        self.plot_1d_y_min = tab_widgets[0]["y_min_box"]
        self.plot_1d_y_max = tab_widgets[0]["y_max_box"]
        self.show_temp = tab_widgets[0]["show_temp"]
        self.show_spatial = tab_widgets[0]["show_spatial"]
        self.show_detector = tab_widgets[0]["show_detector"]
        self.show_total = tab_widgets[0]["show_total"]
        self.show_y0 = tab_widgets[0]["show_y0"]
        self.show_legend = tab_widgets[0]["show_legend"]
        self.info_button_1d = tab_widgets[0]["info"]
        self.toggle_button_1d = tab_widgets[0]["annotation"]

        self.canvas_2d = tab_widgets[1]["canvas"]
        self.freq_scale_2d = tab_widgets[1]["freq_box"]
        self.gray_scale_2d = tab_widgets[1]["gray_box"]
        self.plot_2d_x_min = tab_widgets[1]["x_min_box"]
        self.plot_2d_x_max = tab_widgets[1]["x_max_box"]
        self.plot_2d_y_min = tab_widgets[1]["y_min_box"]
        self.plot_2d_y_max = tab_widgets[1]["y_max_box"]
        self.defocus_diff_slider_2d = tab_widgets[1]["defocus_diff"]
        self.defocus_az_slider_2d = tab_widgets[1]["defocus_az"]
        self.info_button_2d = tab_widgets[1]["info"]
        self.toggle_button_2d = tab_widgets[1]["annotation"]

        self.canvas_ice = tab_widgets[2]["canvas"]
        self.freq_scale_ice = tab_widgets[2]["freq_box"]
        self.gray_scale_ice = tab_widgets[2]["gray_box"]
        self.ice_thickness_slider = tab_widgets[2]["thickness"]
        self.xlim_slider_ice = tab_widgets[2]["xlim"]
        self.defocus_diff_slider_ice = tab_widgets[2]["defocus_diff"]
        self.defocus_az_slider_ice = tab_widgets[2]["defocus_az"]
        self.info_button_ice = tab_widgets[2]["info"]
        self.toggle_button_ice = tab_widgets[2]["annotation"]

        self.canvas_tomo = tab_widgets[3]["canvas"]
        self.sample_size_tomo = tab_widgets[3]["size_box"]
        self.gray_scale_tomo = tab_widgets[3]["gray_box"]
        self.sample_thickness_slider_tomo = tab_widgets[3]["thickness"]
        self.tilt_slider_tomo = tab_widgets[3]["tilt"]
        self.defocus_diff_slider_tomo = tab_widgets[3]["defocus_diff"]
        self.defocus_az_slider_tomo = tab_widgets[3]["defocus_az"]
        self.info_button_tomo = tab_widgets[3]["info"]
        self.toggle_button_tomo = tab_widgets[3]["annotation"]

        self.canvas_image = tab_widgets[4]["canvas"]
        self.upload_btn = tab_widgets[4]["upload_button"]
        self.size_scale_image = tab_widgets[4]["image_zoom"]
        self.size_scale_fft = tab_widgets[4]["fft_zoom"]
        self.contrast_scale_image = tab_widgets[4]["image_contrast"]
        self.contrast_scale_fft = tab_widgets[4]["fft_contrast"]
        self.contrast_sync_checkbox = tab_widgets[4]["sync_checkbox"]
        self.invert_btn = tab_widgets[4]["invert_button"]
        self.defocus_diff_slider_image = tab_widgets[4]["defocus_diff"]
        self.defocus_az_slider_image = tab_widgets[4]["defocus_az"]
        self.info_button_image = tab_widgets[4]["info"]
        self.toggle_button_image = tab_widgets[4]["annotation"]
