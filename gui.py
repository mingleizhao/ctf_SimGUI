import sys
import matplotlib
matplotlib.use("Qt5Agg")  # Use the Qt5 backend
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, QDoubleSpinBox,
    QCheckBox, QComboBox, QRadioButton, QButtonGroup, QSizePolicy,
    QGridLayout, QSpacerItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from customized_widgets import LabeledSlider
from models import DetectorConfigs
from styles import LEFT_PANEL_QGROUPBOX_STYLE, RIGHT_PANEL_QGROUPBOX_STYLE, QTABWIDGET_STYLE, BUTTON_STYLE, INFO_BUTTON_STYLE


class MplCanvas(FigureCanvasQTAgg):
    """
    A custom Matplotlib canvas widget for embedding plots in a PyQt application.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 8.0,
        height: float = 6.0,
        dpi: int = 100,
        subplot_grid: tuple[int, int] = (1, 1), 
        subplot_args: dict | None = None,
    ) -> None:
        """
        Initialize the Matplotlib canvas with support for custom subplot layouts.

        Args:
            parent (QWidget | None, optional): Optional parent widget for this canvas. Defaults to None.
            width (float, optional): Width of the plot in inches. Defaults to 8.0.
            height (float, optional): Height of the plot in inches. Defaults to 6.0.
            dpi (int, optional): Resolution of the plot in dots per inch. Defaults to 100.
            subplot_grid (tuple[int, int], optional): Grid layout for subplots (rows, cols). Defaults to (2, 2).
            subplot_args (dict | None, optional): Arguments to customize specific subplots.
                Example: {1: {"colspan": 2}} for subplot 1 spanning 2 columns.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        nrows, ncols = subplot_grid

        # Initialize the layout for subplots
        self.axes = {}
        if subplot_args is None:
            subplot_args = {}

        # Use GridSpec for advanced layout
        grid_spec = self.fig.add_gridspec(nrows=nrows, ncols=ncols)

        # Define subplots
        if not subplot_args:
            # Create default subplots when no specific args are provided
            for idx in range(1, nrows * ncols + 1):
                row, col = divmod(idx - 1, ncols)  # 1-based index
                self.axes[idx] = self.fig.add_subplot(grid_spec[row, col])
        else:
            # Apply custom subplot arguments
            for idx, args in subplot_args.items():
                if "rowspan" in args or "colspan" in args:
                    self.axes[idx] = self.fig.add_subplot(
                        grid_spec[args.pop("rowspan"), args.pop("colspan")], **args
                    )
                else:
                    row, col = divmod(idx - 1, ncols)  # 1-based index
                    self.axes[idx] = self.fig.add_subplot(grid_spec[row, col], **args)

        super().__init__(self.fig)
        self.setParent(parent)


class CTFSimGUI(QMainWindow):
    """
    A main window class providing a GUI for simulating the Contrast Transfer Function (CTF) in electron microscopy.
    It consists of parameter sections (microscope, imaging, plotting) on the left panel, and Matplotlib plots on the right panel.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the CTFSimGUI, setting up layouts, panels, and Matplotlib canvases.

        Args:
            parent (QWidget | None, optional): Optional parent widget for this main window.
        """
        super().__init__(parent)

        self.setWindowTitle("CTF Simulation")
        self.setMinimumSize(1620, 1080)

        # 1) Create main container widget and a main layout
        container = QWidget()
        self.setCentralWidget(container)
        self.main_layout = QHBoxLayout(container)

        # Left panel will hold the parameter widgets
        self.left_panel = QVBoxLayout()
        self.main_layout.addLayout(self.left_panel, stretch=0)

        # Right panel will hold the plots
        self.right_panel = QVBoxLayout()
        self.main_layout.addLayout(self.right_panel, stretch=1)

        # 2) Build each section of the left panel (microscope, imaging, plotting, and buttons.)
        self._build_microscope_section()
        self._build_imaging_section()
        self._build_detector_section()
        self._build_plotting_section()
        self._build_button_section()

        # 3) Create the tabbed plots for the right panel
        self._build_plot_tabs()

        # 4) Populate the left panel
        self.left_panel.addWidget(self.microscope_box)
        self.left_panel.addWidget(self.imaging_box)
        self.left_panel.addWidget(self.detector_box)
        self.left_panel.addWidget(self.plotting_box)
        self.left_panel.addLayout(self.button_box)
        self.left_panel.addStretch()

        # 5) Put tabbed plots in the right panel
        self.right_panel.addWidget(self.plot_tabs)

    def _build_microscope_section(self) -> None:
        """
        Create a QGroupBox for 'Microscope Parameters' (voltage, aberrations, stability, etc.).
        This section uses custom sliders from 'customized_widgets' for parameter control.
        """
        self.microscope_box = QGroupBox("Microscope Parameters")
        
        self.voltage_slider = LabeledSlider("Voltage (KV)", min_value=80, max_value=1000, step=20, value_format="{:.0f}")       
        self.voltage_stability_slider = LabeledSlider("Voltage Stability", min_value=1e-9, max_value=1e-4, step=1e-9, value_format="{:.2e}", log_scale=True)       
        self.electron_source_angle_slider = LabeledSlider("Electron Source Angle (rad)", min_value=1e-5, max_value=1e-2, step=1e-5, value_format="{:.1e}", log_scale=True)        
        self.electron_source_spread_slider = LabeledSlider("Electron Source Spread (eV)", min_value=0, max_value=10, step=0.1, value_format="{:.1f}")
        self.chromatic_aberr_slider = LabeledSlider("Chromatic Aberration (mm)", min_value=0., max_value=10, step=0.1, value_format="{:.1f}")
        self.spherical_aberr_slider = LabeledSlider("Spherical Aberration (mm)", min_value=0., max_value=10, step=0.1, value_format="{:.1f}")
        self.obj_lens_stability_slider = LabeledSlider("Objective Lens Stability", min_value=1e-9, max_value=1e-4, step=1e-9, value_format="{:.2e}", log_scale=True)

        layout = QVBoxLayout()
        layout.addWidget(self.voltage_slider)
        layout.addWidget(self.voltage_stability_slider)
        layout.addWidget(self.electron_source_angle_slider)
        layout.addWidget(self.electron_source_spread_slider)
        layout.addWidget(self.chromatic_aberr_slider)
        layout.addWidget(self.spherical_aberr_slider)
        layout.addWidget(self.obj_lens_stability_slider)

        self.microscope_box.setLayout(layout)
        self.microscope_box.setStyleSheet(LEFT_PANEL_QGROUPBOX_STYLE)

    def _build_imaging_section(self) -> None:
        """
        Create a QGroupBox for 'Imaging Parameters' (defocus, amplitude contrast, and additional phase shift).
        """
        self.imaging_box = QGroupBox("Imaging Parameters")

        self.defocus_slider = LabeledSlider("Avg. Defocus (µm)", min_value=-5, max_value=10, step=0.01, value_format="{:.4f}")
        self.amplitude_contrast_slider = LabeledSlider("Amplitude Contrast", min_value=0, max_value=1, step=0.01, value_format="{:.2f}")
        self.additional_phase_slider = LabeledSlider("Additional Phase Shift (°)", min_value=0, max_value=180, step=1, value_format="{:.0f}") 

        layout = QVBoxLayout()
        layout.addWidget(self.defocus_slider)
        layout.addWidget(self.amplitude_contrast_slider)
        layout.addWidget(self.additional_phase_slider)

        self.imaging_box.setLayout(layout)
        self.imaging_box.setStyleSheet(LEFT_PANEL_QGROUPBOX_STYLE)

    def _build_detector_section(self) -> None:
        """
        Create a QGroupBox for 'Detector Parameters' (detector and pixel size).
        """
        self.detector_box = QGroupBox("Detector Parameters")

        # Detector dropdown
        self.detector_label = QLabel("Detector")
        self.detector_combo = QComboBox()
        # Populate the combo box with values from DETECTOR_REGISTERS
        self.detector_combo.addItems([detector.value["name"] for detector in DetectorConfigs])

        self.pixel_size_slider = LabeledSlider("Pixel Size (Å)", min_value=0.5, max_value=5., step=0.1, value_format="{:.3f}" )
  
        layout = QVBoxLayout()
        layout.addWidget(self.detector_label)
        layout.addWidget(self.detector_combo)
        layout.addWidget(self.pixel_size_slider)

        self.detector_box.setLayout(layout)
        self.detector_box.setStyleSheet(LEFT_PANEL_QGROUPBOX_STYLE)

    def _build_plotting_section(self) -> None:
        """
        Create a QGroupBox for 'Plotting Parameters' (envelope function toggles, CTFs, etc.).
        """
        self.plotting_box = QGroupBox("CTF Calculation Options")

        # Create checkbox widgets
        self.envelope_label = QLabel("Envelope Function")
        self.temporal_env_check = QCheckBox("Temporal")   
        self.spatial_env_check = QCheckBox("Spatial")
        self.detector_env_check = QCheckBox("Detector")

        # Create a horizontal layout for the checkboxes
        checkbox_layout = QHBoxLayout()

        # Add checkboxes to the horizontal layout
        checkbox_layout.addWidget(self.temporal_env_check)
        checkbox_layout.addWidget(self.spatial_env_check)
        checkbox_layout.addWidget(self.detector_env_check)
  
        # Create a horizontal layout for the radio buttons
        button_layout = QHBoxLayout()

        # Create radio buttons for different CTF formats
        self.ctf_label = QLabel("CTF Display Mode")
        self.radio_ctf = QRadioButton("CTF")
        self.radio_abs_ctf = QRadioButton("|CTF|")
        self.radio_ctf_squared = QRadioButton("CTF²")

        # Add radio buttons to the horizontal layout
        button_layout.addWidget(self.radio_ctf)
        button_layout.addWidget(self.radio_abs_ctf)
        button_layout.addWidget(self.radio_ctf_squared)

        # Group the radio buttons
        self.radio_button_group = QButtonGroup()
        self.radio_button_group.addButton(self.radio_ctf)
        self.radio_button_group.addButton(self.radio_abs_ctf)
        self.radio_button_group.addButton(self.radio_ctf_squared)

        # Set default selection
        self.radio_ctf.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.envelope_label)
        layout.addLayout(checkbox_layout)
        layout.addWidget(self.ctf_label)
        layout.addLayout(button_layout)

        self.plotting_box.setLayout(layout)
        self.plotting_box.setStyleSheet(LEFT_PANEL_QGROUPBOX_STYLE)

    def _build_button_section(self) -> None:
        """
        Create a box for all the push buttons.
        """
        self.button_box = QHBoxLayout()

        # Create buttons
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(BUTTON_STYLE)
        self.save_img_button = QPushButton("Save Plot")
        self.save_csv_button = QPushButton("Save CSV")
        
        self.button_box.addWidget(self.reset_button)
        self.button_box.addWidget(self.save_img_button)
        self.button_box.addWidget(self.save_csv_button)
 
    def _build_plot_tabs(self) -> None:
        """
        Create a QTabWidget with three tabs for 1D-CTF, 2D-CTF, ICE-CTF plots.
        Each tab holds its own MplCanvas.
        """
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet(QTABWIDGET_STYLE)
        self.plot_tabs.addTab(self._build_1d_ctf_tab(), "1D")
        self.plot_tabs.addTab(self._build_2d_ctf_tab(), "2D")
        self.plot_tabs.addTab(self._build_ice_ctf_tab(), "Thickness")
        self.plot_tabs.addTab(self._build_tomo_ctf_tab(), "Tilt")
        self.plot_tabs.addTab(self._build_image_ctf_tab(), "Image")

    def _build_1d_ctf_tab(self):
        """
        Build the 1D-CTF tab.

        Returns:
            QWidget: Widget containing the canvas and controls.
        """
        self.canvas_1d = MplCanvas(self, width=5, height=4)
        self.canvas_1d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        widget_1d = QWidget()
        layout_1d = QVBoxLayout(widget_1d)
        layout_1d.addWidget(self.canvas_1d)

        display_1d = QHBoxLayout()
        # display_1d.addStretch()

        display_1d.addLayout(self._build_axis_control(
            "plot_1d",
            x_min_range=(-0.1, 1), x_min_value=0,
            x_max_range=(0, 1.1), x_max_value=0.5,
            y_min_range=(-1.1, 1), y_min_value=-1,
            y_max_range=(-1, 1.1), y_max_value=1
        ))

        self.show_temp = QCheckBox("Temporal Envelope    ")
        self.show_spatial = QCheckBox("Spatial Envelope")
        self.show_detector = QCheckBox("Detector Envelope    ")
        self.show_total = QCheckBox("Total Envelope")
        self.show_y0 = QCheckBox("y=0 dotted line")
        self.show_legend = QCheckBox("Legend")

        display_1d.addSpacerItem(QSpacerItem(30, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)) 

        plotting_options = QGridLayout()

        plotting_options.addWidget(QLabel("Plotting    "), 0, 0)
        plotting_options.addWidget(QLabel("Options    "), 1, 0)
        plotting_options.addWidget(self.show_temp, 0, 1)
        plotting_options.addWidget(self.show_spatial, 1, 1)
        plotting_options.addWidget(self.show_detector, 0, 2)
        plotting_options.addWidget(self.show_total, 1, 2)
        plotting_options.addWidget(self.show_y0, 0, 3)
        plotting_options.addWidget(self.show_legend, 1, 3)

        display_1d.addLayout(plotting_options)
        display_1d.addStretch()

        self.info_button_1d = self._create_info_button()
        self.toggle_button_1d = self._create_toggle_button()
        display_1d.addLayout(self._create_tab_buttons(
            self.info_button_1d,
            self.toggle_button_1d
        ))

        display_control_1d = QGroupBox()
        display_control_1d.setLayout(display_1d)
        display_control_1d.setStyleSheet(RIGHT_PANEL_QGROUPBOX_STYLE)

        layout_1d.addWidget(display_control_1d)

        return widget_1d
    
    def _build_2d_ctf_tab(self):
        """
        Build the 2D-CTF tab.

        Returns:
            QWidget: Widget containing the canvas and controls.
        """
        self.canvas_2d = MplCanvas(self, width=5, height=4)
        self.canvas_2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        widget_2d = QWidget()
        layout_2d = QVBoxLayout(widget_2d)
        layout_2d.addWidget(self.canvas_2d)

        display_2d = QHBoxLayout()
        # display_2d.addStretch()
        display_2d.addLayout(self._build_axis_control(
            "plot_2d",
            x_min_range=(-0.5, 0.5), x_min_value=-0.5,
            x_max_range=(-0.5, 0.5), x_max_value=0.5,
            y_min_range=(-0.5, 0.5), y_min_value=-1,
            y_max_range=(-0.5, 0.5), y_max_value=1
        ))

        scale_2d = QGridLayout()
        self.freq_scale_2d = QDoubleSpinBox()
        self.freq_scale_2d.setRange(0.1, 0.5)
        self.freq_scale_2d.setValue(0.5)
        self.freq_scale_2d.setSingleStep(0.02)
        self.freq_scale_2d.setDecimals(3)
        self.freq_scale_2d.setFixedWidth(70)

        self.gray_scale_2d = QDoubleSpinBox()
        self.gray_scale_2d.setRange(0.05, 1)
        self.gray_scale_2d.setValue(1)
        self.gray_scale_2d.setSingleStep(0.02)
        self.gray_scale_2d.setDecimals(3)
        self.gray_scale_2d.setFixedWidth(70)

        scale_2d.addWidget(QLabel("|Spatial Frequency|:"), 0, 0)
        scale_2d.addWidget(self.freq_scale_2d, 0, 1)
        scale_2d.addWidget(QLabel("Max Gray Scale: "), 1, 0)
        scale_2d.addWidget(self.gray_scale_2d, 1, 1)

        display_2d.addStretch()
        display_2d.addLayout(scale_2d)
        display_2d.addStretch()

        self.defocus_diff_slider_2d = LabeledSlider("Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.defocus_az_slider_2d = LabeledSlider("Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}")
        
        display_2d.addWidget(self.defocus_diff_slider_2d)
        display_2d.addStretch()
        display_2d.addWidget(self.defocus_az_slider_2d)
        display_2d.addStretch()
        
        self.info_button_2d = self._create_info_button()
        self.toggle_button_2d = self._create_toggle_button()
        display_2d.addLayout(self._create_tab_buttons(
            self.info_button_2d,
            self.toggle_button_2d
        ))

        display_control_2d = QGroupBox()
        display_control_2d.setLayout(display_2d)
        display_control_2d.setStyleSheet(RIGHT_PANEL_QGROUPBOX_STYLE)
        
        layout_2d.addWidget(display_control_2d)
        
        return widget_2d

    def _build_ice_ctf_tab(self):
        """
        Build the ICE-CTF tab.

        Returns:
            QWidget: Widget containing the canvas and controls.
        """
        # Define subplot arguments for the layout
        subplot_args = {
            1: {"rowspan": slice(0, 1), "colspan": slice(0, 2)},  # Top row spanning two columns
            2: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},  # Bottom-left
            3: {"rowspan": slice(1, 2), "colspan": slice(1, 2)},  # Bottom-right
        }
        self.canvas_ice = MplCanvas(self, subplot_grid=(2, 2), subplot_args=subplot_args, width=5, height=4)
        self.canvas_ice.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        self.ice_thickness_slider = LabeledSlider("Ice Thickness (nm)", min_value=1, max_value=1000, step=1, value_format="{:.0f}" )
        self.xlim_slider_ice = LabeledSlider("1D X-axis Limit (Å⁻¹)", min_value=0.1, max_value=1.1, step=0.01, value_format="{:.2f}" )
        self.defocus_diff_slider_ice = LabeledSlider("Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.defocus_az_slider_ice = LabeledSlider("Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}") 

        widget_ice = QWidget()
        layout_ice = QVBoxLayout(widget_ice)
        layout_ice.addWidget(self.canvas_ice)

        display_ice = QHBoxLayout()

        scale_ice = QGridLayout()
        self.freq_scale_ice = QDoubleSpinBox()
        self.freq_scale_ice.setRange(0.1, 0.5)
        self.freq_scale_ice.setValue(0.5)
        self.freq_scale_ice.setSingleStep(0.02)
        self.freq_scale_ice.setDecimals(3)
        self.freq_scale_ice.setFixedWidth(70)

        self.gray_scale_ice = QDoubleSpinBox()
        self.gray_scale_ice.setRange(0.05, 1)
        self.gray_scale_ice.setValue(1)
        self.gray_scale_ice.setSingleStep(0.02)
        self.gray_scale_ice.setDecimals(3)
        self.gray_scale_ice.setFixedWidth(70)

        scale_ice.addWidget(QLabel("|Spatial Frequency|:"), 0, 0)
        scale_ice.addWidget(self.freq_scale_ice, 0, 1)
        scale_ice.addWidget(QLabel("Max Gray Scale: "), 1, 0)
        scale_ice.addWidget(self.gray_scale_ice, 1, 1)

        display_ice.addLayout(scale_ice)
        display_ice.addStretch()
        display_ice.addWidget(self.xlim_slider_ice)
        display_ice.addStretch()
        display_ice.addWidget(self.ice_thickness_slider)
        display_ice.addStretch() 
        display_ice.addWidget(self.defocus_diff_slider_ice)
        display_ice.addStretch()
        display_ice.addWidget(self.defocus_az_slider_ice)
        display_ice.addStretch()

        self.info_button_ice = self._create_info_button()
        self.toggle_button_ice = self._create_toggle_button()
        display_ice.addLayout(self._create_tab_buttons(
            self.info_button_ice,
            self.toggle_button_ice
        ))

        display_control_ice = QGroupBox()
        display_control_ice.setLayout(display_ice)
        display_control_ice.setStyleSheet(RIGHT_PANEL_QGROUPBOX_STYLE)

        layout_ice.addWidget(display_control_ice)

        return widget_ice

    def _build_tomo_ctf_tab(self):
        """
        Build the TOMO-CTF tab.

        Returns:
            QWidget: Widget containing the canvas and controls.
        """
        # Define subplot arguments for the layout
        subplot_args = {
            1: {"rowspan": slice(0, 1), "colspan": slice(0, 1)},  # Top-left
            3: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},  # Bottom-left
            4: {"rowspan": slice(0, 2), "colspan": slice(1, 2)},  # Right column
        }
        self.canvas_tomo = MplCanvas(self, subplot_grid=(2, 2), subplot_args=subplot_args, width=5, height=4)
        self.canvas_tomo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        # Create sliders for tilt angle and sample_thickness
        self.sample_thickness_slider_tomo = LabeledSlider("Sample Thickness (nm)", min_value=50, max_value=1000, step=1, value_format="{:.0f}" )
        self.tilt_slider_tomo = LabeledSlider("Tilt Angle (°)", min_value=-70, max_value=70, step=0.1, value_format="{:.1f}")
        self.defocus_diff_slider_tomo = LabeledSlider("Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.defocus_az_slider_tomo = LabeledSlider("Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}") 

        widget_tomo = QWidget()
        layout_tomo = QVBoxLayout(widget_tomo)
        layout_tomo.addWidget(self.canvas_tomo)

        display_tomo = QHBoxLayout()

        scale_tomo = QGridLayout()
        self.sample_size_tomo = QDoubleSpinBox()
        self.sample_size_tomo.setRange(0.4, 2)
        self.sample_size_tomo.setValue(1)
        self.sample_size_tomo.setSingleStep(0.02)
        self.sample_size_tomo.setDecimals(2)
        self.sample_size_tomo.setFixedWidth(70)

        self.gray_scale_tomo = QDoubleSpinBox()
        self.gray_scale_tomo.setRange(0.05, 1)
        self.gray_scale_tomo.setValue(1)
        self.gray_scale_tomo.setSingleStep(0.02)
        self.gray_scale_tomo.setDecimals(3)
        self.gray_scale_tomo.setFixedWidth(70)

        scale_tomo.addWidget(QLabel("Sample Size (µm):"), 0, 0)
        scale_tomo.addWidget(self.sample_size_tomo, 0, 1)
        scale_tomo.addWidget(QLabel("Max Gray Scale: "), 1, 0)
        scale_tomo.addWidget(self.gray_scale_tomo, 1, 1)

        display_tomo.addLayout(scale_tomo)
        display_tomo.addStretch()

        display_tomo.addWidget(self.sample_thickness_slider_tomo)
        display_tomo.addStretch() 
        display_tomo.addWidget(self.tilt_slider_tomo)
        display_tomo.addStretch()
        display_tomo.addWidget(self.defocus_diff_slider_tomo)
        display_tomo.addStretch()
        display_tomo.addWidget(self.defocus_az_slider_tomo)
        display_tomo.addStretch()

        self.info_button_tomo = self._create_info_button()
        self.toggle_button_tomo = self._create_toggle_button()
        display_tomo.addLayout(self._create_tab_buttons(
            self.info_button_tomo,
            self.toggle_button_tomo
        ))

        display_control_tomo = QGroupBox()
        display_control_tomo.setLayout(display_tomo)
        display_control_tomo.setStyleSheet(RIGHT_PANEL_QGROUPBOX_STYLE)

        layout_tomo.addWidget(display_control_tomo)

        return widget_tomo

    def _build_image_ctf_tab(self):
        """
        Build the IMAGE-CTF tab.

        Returns:
            QWidget: Widget containing the canvas and controls.
        """
        # Define subplot arguments for the layout
        subplot_args = {
            1: {"rowspan": slice(0, 1), "colspan": slice(0, 1)},  # Top-left
            2: {"rowspan": slice(0, 1), "colspan": slice(1, 2)},  # Top-right
            3: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},  # Bottom-left
            4: {"rowspan": slice(1, 2), "colspan": slice(1, 2)},  # Bottom-right
        }
        self.canvas_image = MplCanvas(self, subplot_grid=(2, 2), subplot_args=subplot_args, width=5, height=4)
        self.canvas_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        # Create a button for uploading image
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setStyleSheet(BUTTON_STYLE)
        
        # Create spin boxes for size scaling 
        scale_image = QGridLayout()
        self.size_scale_image = QDoubleSpinBox()
        self.size_scale_image.setRange(100, 500)
        self.size_scale_image.setValue(100)
        self.size_scale_image.setSingleStep(5)
        self.size_scale_image.setDecimals(0)
        self.size_scale_image.setFixedWidth(70)

        self.size_scale_fft = QDoubleSpinBox()
        self.size_scale_fft.setRange(100, 500)
        self.size_scale_fft.setValue(100)
        self.size_scale_fft.setSingleStep(5)
        self.size_scale_fft.setDecimals(0)
        self.size_scale_fft.setFixedWidth(70)

        scale_image.addWidget(QLabel("Image Zoom (%):"), 0, 0)
        scale_image.addWidget(self.size_scale_image, 0, 1)
        scale_image.addWidget(QLabel("FFT Zoom (%): "), 1, 0)
        scale_image.addWidget(self.size_scale_fft, 1, 1)

        # Create spin boxes for contrast adjustment
        contrast_image = QGridLayout()
        self.contrast_scale_image = QDoubleSpinBox()
        self.contrast_scale_image.setRange(51, 100)
        self.contrast_scale_image.setValue(100)
        self.contrast_scale_image.setSingleStep(0.1)
        self.contrast_scale_image.setDecimals(1)
        self.contrast_scale_image.setFixedWidth(70)

        self.contrast_scale_fft = QDoubleSpinBox()
        self.contrast_scale_fft.setRange(51, 100)
        self.contrast_scale_fft.setValue(100)
        self.contrast_scale_fft.setSingleStep(0.1)
        self.contrast_scale_fft.setDecimals(1)
        self.contrast_scale_fft.setFixedWidth(70)

        contrast_image.addWidget(QLabel("Image Contrast (%):"), 0, 0)
        contrast_image.addWidget(self.contrast_scale_image, 0, 1)
        contrast_image.addWidget(QLabel("FFT Contrast (%): "), 1, 0)
        contrast_image.addWidget(self.contrast_scale_fft, 1, 1)

        # Create sliders
        self.defocus_diff_slider_image = LabeledSlider("Defocus Ast. (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.defocus_az_slider_image = LabeledSlider("Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}") 

        widget_image = QWidget()
        layout_image = QVBoxLayout(widget_image)
        layout_image.addWidget(self.canvas_image)

        display_image = QHBoxLayout()

        display_image.addWidget(self.upload_btn)
        display_image.addStretch()
        display_image.addLayout(scale_image)
        display_image.addStretch()
        display_image.addLayout(contrast_image)
        display_image.addStretch()

        # Create a checkbox for contrast sync
        self.contrast_sync_checkbox = QCheckBox("Sync Greyscale")
        self.contrast_sync_checkbox.setChecked(False)
        self.contrast_sync_checkbox.setStyleSheet("""
            QCheckBox {
                padding: 0px 0px;
            }
        """)
        self.contrast_sync_checkbox.setToolTip("Synchronize greyscale between original and convolved images")
        
        # Create a button for contrast inversion
        self.invert_btn = QPushButton("Invert Image")
        self.invert_btn.setFixedHeight(26)
        self.invert_btn.setToolTip("Invert greyscale of original and convolved images")

        contrast_group = QVBoxLayout()
        contrast_group.addWidget(self.contrast_sync_checkbox)
        contrast_group.addWidget(self.invert_btn)        
        
        display_image.addLayout(contrast_group)
        display_image.addStretch()
                 
        display_image.addWidget(self.defocus_diff_slider_image)
        display_image.addStretch()
        display_image.addWidget(self.defocus_az_slider_image)
        display_image.addStretch()

        self.info_button_image = self._create_info_button()
        self.toggle_button_image = self._create_toggle_button()
        display_image.addLayout(self._create_tab_buttons(
            self.info_button_image,
            self.toggle_button_image
        ))

        display_control_image = QGroupBox()
        display_control_image.setLayout(display_image)
        display_control_image.setStyleSheet(RIGHT_PANEL_QGROUPBOX_STYLE)

        layout_image.addWidget(display_control_image)

        return widget_image

    def _build_axis_control(
            self,
            attr_prefix: str,
            x_min_range: tuple[float, float], x_min_value: float,
            x_max_range: tuple[float, float], x_max_value: float,
            y_min_range: tuple[float, float], y_min_value: float,
            y_max_range: tuple[float, float], y_max_value: float,
            single_step: float = 0.01, decimals: int = 3, width: int = 70
        ):
        """
        Generalized function to create an axis control layout with configurable QDoubleSpinBox widgets using QGridLayout.

        Args:
            attr_prefix (str): Prefix for instance variables (e.g., "xlim_1d" → creates self.xlim_1d_min, self.xlim_1d_max).
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
            width (int, optional): Fixed width for all spin boxes. Defaults to 70.

        Returns:
            QVBoxLayout: Layout containing X and Y axis controls.
        """

        def configure_spinbox(spinbox, value, value_range):
            """Helper function to configure a QDoubleSpinBox."""
            spinbox.setRange(*value_range)
            spinbox.setValue(value)
            spinbox.setSingleStep(single_step)
            spinbox.setDecimals(decimals)
            spinbox.setFixedWidth(width)

        # Dynamically create instance variables using the prefix
        for axis in ["x_min", "x_max", "y_min", "y_max"]:
            setattr(self, f"{attr_prefix}_{axis}", QDoubleSpinBox())

        x_min = getattr(self, f"{attr_prefix}_x_min")
        x_max = getattr(self, f"{attr_prefix}_x_max")
        y_min = getattr(self, f"{attr_prefix}_y_min")
        y_max = getattr(self, f"{attr_prefix}_y_max")

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
            "y_max": QLabel("Max")
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

        return grid_layout 
    
    def _create_info_button(self):
        info_button = QPushButton("?")
        info_button.setFixedSize(18, 18)
        info_button.setToolTip("Additional info")
        info_button.setStyleSheet(INFO_BUTTON_STYLE)
        
        return info_button
    
    def _create_toggle_button(self):
        # toggle_button = QPushButton("𝒱")
        toggle_button = QPushButton("V")
        toggle_button.setCheckable(True)
        toggle_button.setFixedSize(18, 18)
        toggle_button.setToolTip("Show/Hide annotation")
        toggle_button.setStyleSheet(INFO_BUTTON_STYLE)

        return toggle_button
    
    def _create_tab_buttons(self, info_button, toggle_button):
        button_group = QVBoxLayout()
        button_group.setSpacing(4)
        button_group.addWidget(info_button)
        button_group.addWidget(toggle_button)

        return button_group


def test_gui() -> None:
    """
    Initialize and run the CTFSimGUI as a standalone application.
    """
    app = QApplication(sys.argv)
    window = CTFSimGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_gui()
