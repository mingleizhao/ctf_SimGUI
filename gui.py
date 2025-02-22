import sys
import matplotlib
matplotlib.use("Qt5Agg")  # Use the Qt5 backend
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, QDoubleSpinBox,
    QCheckBox, QComboBox, QRadioButton, QButtonGroup, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from customized_widgets import LabeledSlider
from models import DetectorConfigs
from styles import SHARED_QGROUPBOX_STYLESHEET, SHARED_QTABWIDGET_STYLESHEET


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
        sharex: bool = False,
        sharey: bool = False,
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
            sharex (bool, optional): Whether to share the x-axis among subplots. Defaults to False.
            sharey (bool, optional): Whether to share the y-axis among subplots. Defaults to False.
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
        self.setMinimumSize(1500, 1000)

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
        self._build_plotting_section()
        self._build_button_section()

        # 3) Create the tabbed plots for the right panel
        self._build_plot_tabs()

        # 4) Populate the left panel
        self.left_panel.addWidget(self.microscope_box)
        self.left_panel.addWidget(self.imaging_box)
        self.left_panel.addWidget(self.plotting_box)
        self.left_panel.addLayout(self.button_box)

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
        self.electron_source_angle_slider = LabeledSlider("E-Source Angle (rad)", min_value=1e-5, max_value=1e-2, step=1e-5, value_format="{:.1e}", log_scale=True)        
        self.electron_source_spread_slider = LabeledSlider("E-Source Spread (eV)", min_value=0, max_value=10, step=0.1, value_format="{:.1f}")
        self.chromatic_aberr_slider = LabeledSlider("Chromatic Aberration (mm)", min_value=0., max_value=10, step=0.1, value_format="{:.1f}")
        self.spherical_aberr_slider = LabeledSlider("Spherical Aberration (mm)", min_value=0., max_value=10, step=0.1, value_format="{:.1f}")
        self.obj_lens_stability_slider = LabeledSlider("Obj. Lens Stability", min_value=1e-9, max_value=1e-4, step=1e-9, value_format="{:.2e}", log_scale=True)

        layout = QVBoxLayout()
        layout.addWidget(self.voltage_slider)
        layout.addWidget(self.voltage_stability_slider)
        layout.addWidget(self.electron_source_angle_slider)
        layout.addWidget(self.electron_source_spread_slider)
        layout.addWidget(self.chromatic_aberr_slider)
        layout.addWidget(self.spherical_aberr_slider)
        layout.addWidget(self.obj_lens_stability_slider)

        self.microscope_box.setLayout(layout)
        self.microscope_box.setStyleSheet(SHARED_QGROUPBOX_STYLESHEET)

    def _build_imaging_section(self) -> None:
        """
        Create a QGroupBox for 'Imaging Parameters' (detector, pixel size, defocus, etc.).
        """
        self.imaging_box = QGroupBox("Imaging Parameters")

        # Detector dropdown
        self.detector_label = QLabel("Detector:")
        self.detector_combo = QComboBox()
        # Populate the combo box with values from DETECTOR_REGISTERS
        self.detector_combo.addItems([detector.value["name"] for detector in DetectorConfigs])

        self.pixel_size_slider = LabeledSlider("Pixel Size (Å)", min_value=0.2, max_value=5., step=0.1, value_format="{:.3f}" )
        self.defocus_slider = LabeledSlider("Avg. Defocus (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.amplitude_contrast_slider = LabeledSlider("Amplitude Contrast", min_value=0, max_value=1, step=0.01, value_format="{:.2f}")
        self.additional_phase_slider = LabeledSlider("Additional Phase Shift (°)", min_value=0, max_value=180, step=1, value_format="{:.0f}") 

        layout = QVBoxLayout()
        layout.addWidget(self.detector_label)
        layout.addWidget(self.detector_combo)
        layout.addWidget(self.pixel_size_slider)
        layout.addWidget(self.defocus_slider)
        layout.addWidget(self.amplitude_contrast_slider)
        layout.addWidget(self.additional_phase_slider)

        self.imaging_box.setLayout(layout)
        self.imaging_box.setStyleSheet(SHARED_QGROUPBOX_STYLESHEET)

    def _build_plotting_section(self) -> None:
        """
        Create a QGroupBox for 'Plotting Parameters' (envelope function toggles, CTFs, etc.).
        """
        self.plotting_box = QGroupBox("Plotting Parameters")

        # Create widgets
        self.temporal_env_check = QCheckBox("Temporal Envelope")   
        self.spatial_env_check = QCheckBox("Spatial Envelope")
        self.detector_env_check = QCheckBox("Detector Envelope")
  
        # Create a horizontal layout for the radio buttons
        button_layout = QHBoxLayout()

        # Create radio buttons for different CTF formats
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
        layout.addWidget(self.temporal_env_check)
        layout.addWidget(self.spatial_env_check)
        layout.addWidget(self.detector_env_check)
        layout.addLayout(button_layout)

        self.plotting_box.setLayout(layout)
        self.plotting_box.setStyleSheet(SHARED_QGROUPBOX_STYLESHEET)

    def _build_button_section(self) -> None:
        """
        Create a box for all the push buttons.
        """
        self.button_box = QHBoxLayout()

        # Create buttons
        self.reset_button = QPushButton("Reset")
        self.save_img_button = QPushButton("Save Plot")
        self.save_csv_button = QPushButton("Save CSV")
        
        self.button_box.addWidget(self.reset_button)
        self.button_box.addWidget(self.save_img_button)
        self.button_box.addWidget(self.save_csv_button)

 
    def _build_plot_tabs(self) -> None:
        """
        Create a QTabWidget with two tabs for 1D-CTF and 2D-CTF plots.
        Each tab holds its own MplCanvas.
        """
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet(SHARED_QTABWIDGET_STYLESHEET)

        # 1D Plot Tab
        self.canvas_1d = MplCanvas(self, width=5, height=4)
        self.canvas_1d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        widget_1d = QWidget()
        layout_1d = QVBoxLayout(widget_1d)

        layout_1d.addWidget(self.canvas_1d)
        layout_1d.addLayout(self._build_axis_control(
            "plot_1d",
            x_min_range=(-0.1, 1), x_min_value=0,
            x_max_range=(0, 1.1), x_max_value=0.5,
            y_min_range=(-1.1, 1), y_min_value=-1,
            y_max_range=(-1, 1.1), y_max_value=1
        ))

        # 2D Plot Tab
        self.canvas_2d = MplCanvas(self, width=5, height=4)
        self.canvas_2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow canvas to expand fully

        self.defocus_diff_slider_2d = LabeledSlider("𝛥Defocus (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.defocus_az_slider_2d = LabeledSlider("Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}")

        layout_2d_sliders = QHBoxLayout()
        layout_2d_sliders.addLayout(self._build_axis_control(
            "plot_2d",
            x_min_range=(-0.5, 0.5), x_min_value=-0.5,
            x_max_range=(-0.5, 0.5), x_max_value=0.5,
            y_min_range=(-0.5, 0.5), y_min_value=-1,
            y_max_range=(-0.5, 0.5), y_max_value=1
        ))
        layout_2d_sliders.addWidget(self.defocus_diff_slider_2d)
        layout_2d_sliders.addWidget(self.defocus_az_slider_2d)

        widget_2d = QWidget()
        layout_2d = QVBoxLayout(widget_2d)
        layout_2d.addWidget(self.canvas_2d)
        layout_2d.addLayout(layout_2d_sliders)
        
        # ice thickness Tab
        # Define subplot arguments for the layout
        subplot_args = {
            1: {"rowspan": slice(0, 1), "colspan": slice(0, 2)},  # Top row spanning two columns
            2: {"rowspan": slice(1, 2), "colspan": slice(0, 1)},  # Bottom-left
            3: {"rowspan": slice(1, 2), "colspan": slice(1, 2)},  # Bottom-right
        }
        self.canvas_ice = MplCanvas(self, subplot_grid=(2, 2), subplot_args=subplot_args, width=5, height=4)
        self.ice_thickness_slider = LabeledSlider("Ice Thickness (nm)", min_value=1, max_value=1000, step=1, value_format="{:.0f}" )
        self.xlim_slider_ice = LabeledSlider("X-axis Limit (Å⁻¹)", min_value=0.1, max_value=1.1, step=0.01, value_format="{:.2f}" )
        self.defocus_diff_slider_ice = LabeledSlider("𝛥Defocus (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.4f}")
        self.defocus_az_slider_ice = LabeledSlider("Defocus Azimuth (°)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}")        
        
        layout_ice_sliders = QHBoxLayout()
        layout_ice_sliders.addWidget(self.ice_thickness_slider)
        layout_ice_sliders.addWidget(self.xlim_slider_ice)
        layout_ice_sliders.addWidget(self.defocus_diff_slider_ice)
        layout_ice_sliders.addWidget(self.defocus_az_slider_ice)

        widget_ice = QWidget()
        layout_ice = QVBoxLayout(widget_ice)
        layout_ice.addWidget(self.canvas_ice)
        layout_ice.addLayout(layout_ice_sliders)

        # Add widgets to each tab
        self.plot_tabs.addTab(widget_1d, "1D-CTF")
        self.plot_tabs.addTab(widget_2d, "2D-CTF")
        self.plot_tabs.addTab(widget_ice, "ICE-CTF")

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
        Generalized function to create an axis control layout with configurable QDoubleSpinBox widgets.

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
        setattr(self, f"{attr_prefix}_x_min", QDoubleSpinBox())
        setattr(self, f"{attr_prefix}_x_max", QDoubleSpinBox())
        setattr(self, f"{attr_prefix}_y_min", QDoubleSpinBox())
        setattr(self, f"{attr_prefix}_y_max", QDoubleSpinBox())

        x_min = getattr(self, f"{attr_prefix}_x_min")
        x_max = getattr(self, f"{attr_prefix}_x_max")
        y_min = getattr(self, f"{attr_prefix}_y_min")
        y_max = getattr(self, f"{attr_prefix}_y_max")

        configure_spinbox(x_min, x_min_value, x_min_range)
        configure_spinbox(x_max, x_max_value, x_max_range)
        configure_spinbox(y_min, y_min_value, y_min_range)
        configure_spinbox(y_max, y_max_value, y_max_range)

        # Create labels
        x_label_min = QLabel("X-Axis:    Min")
        x_label_min.setMinimumHeight(23)
        x_label_max = QLabel(" Max")
        y_label_min = QLabel("Y-Axis:    Min")
        y_label_min.setMinimumHeight(23)
        y_label_max = QLabel(" Max")

        # Find the widest label
        max_width = max(x_label_min.sizeHint().width(), y_label_min.sizeHint().width())

        # Apply the same width to all labels
        x_label_min.setFixedWidth(max_width)
        y_label_min.setFixedWidth(max_width)

        # X-Axis Layout
        xlim_control = QHBoxLayout()
        xlim_control.addWidget(x_label_min)
        xlim_control.addWidget(x_min)
        xlim_control.addWidget(x_label_max)
        xlim_control.addWidget(x_max)
        xlim_control.addStretch()

        # Y-Axis Layout
        ylim_control = QHBoxLayout()
        ylim_control.addWidget(y_label_min)
        ylim_control.addWidget(y_min)
        ylim_control.addWidget(y_label_max)
        ylim_control.addWidget(y_max)
        ylim_control.addStretch()

        # Main Layout
        layout_control = QVBoxLayout()
        layout_control.addLayout(xlim_control)
        layout_control.addLayout(ylim_control)

        return layout_control    


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
