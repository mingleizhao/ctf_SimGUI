import sys
import matplotlib
matplotlib.use("Qt5Agg")  # Use the Qt5 backend
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget,
    QCheckBox, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from customized_widgets import SelectionSlider, FloatLogSlider, FloatSlider, SHARED_QGROUPBOX_STYLESHEET
from models import DETECTOR_REGISTERS

class MplCanvas(FigureCanvasQTAgg):
    """
    A custom Matplotlib canvas widget for embedding plots in a PyQt application.
    """
    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 5.0,
        height: float = 4.0,
        dpi: int = 100
    ) -> None:
        """Initialize the Matplotlib canvas.

        Args:
            parent (QWidget | None, optional): Optional parent widget for this main window. Defaults to None.
            width (float, optional): Width of the plot. Defaults to 5.0.
            height (float, optional): Height of the plot. Defaults to 4.0.
            dpi (int, optional): Resolution of the plot. Defaults to 100.
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
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

        # 2) Build each section of the left panel (microscope, imaging, plotting, and reset button.)
        self._build_microscope_section()
        self._build_imaging_section()
        self._build_plotting_section()
        self._build_reset_button()

        # 3) Create the tabbed plots for the right panel
        self._build_plot_tabs()

        # 4) Populate the left panel
        self.left_panel.addWidget(self.microscope_box)
        self.left_panel.addWidget(self.imaging_box)
        self.left_panel.addWidget(self.plotting_box)
        self.left_panel.addWidget(self.reset_button)

        # 5) Put tabbed plots in the right panel
        self.right_panel.addWidget(self.plot_tabs)

    def _build_microscope_section(self) -> None:
        """
        Create a QGroupBox for 'Microscope Parameters' (voltage, aberrations, stability, etc.).
        This section uses custom sliders from 'customized_widgets' for parameter control.
        """
        self.microscope_box = QGroupBox("Microscope Parameters")
        
        self.voltage_slider = SelectionSlider("Voltage (KV)", [80, 100, 120, 200, 300, 500, 1000])       
        self.voltage_stability_slider = FloatLogSlider("Voltage Stability", min_exp=-9, max_exp=-4, step_exp=0.01, value_format="{:.2e}")       
        self.electron_source_angle_slider = FloatLogSlider("E-Source Angle (rad)", min_exp=-5, max_exp=-2, step_exp=0.01, value_format="{:.1e}")        
        self.electron_source_spread_slider = FloatSlider("E-Source Spread (eV)", min_value=0, max_value=10, step=0.1, value_format="{:.1f}")
        self.chromatic_aberr_slider = FloatSlider("Chromatic Aberration (mm)", min_value=0., max_value=10, step=0.1, value_format="{:.1f}")
        self.spherical_aberr_slider = FloatSlider("Spherical Aberration (mm)", min_value=0., max_value=10, step=0.1, value_format="{:.1f}")
        self.obj_lens_stability_slider = FloatLogSlider("Objective Lens Stability", min_exp=-9, max_exp=-4, step_exp=0.01, value_format="{:.2e}")

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
        self.detector_combo.addItems([detector for detector in DETECTOR_REGISTERS.values()])

        self.pixel_size_slider = FloatSlider("Pixel Size (Å)", min_value=0.2, max_value=5., step=0.1, value_format="{:.1f}" )
        self.defocus_slider = FloatSlider("Defocus (µm)", min_value=-5, max_value=5, step=0.01, value_format="{:.2f}")
        self.defocus_diff_slider = FloatSlider("Defocus Diff. (µm, 2D)", min_value=-5, max_value=5, step=0.01, value_format="{:.2f}")
        self.defocus_az_slider = FloatSlider("Defocus Az. (°, 2D)", min_value=0, max_value=180, step=0.1, value_format="{:.1f}")
        self.amplitude_contrast_slider = FloatSlider("Amplitude Contrast", min_value=0, max_value=1, step=0.01, value_format="{:.2f}")
        self.additional_phase_slider = FloatSlider("Additional phase shift (°)", min_value=0, max_value=180, step=1, value_format="{:.0f}") 

        layout = QVBoxLayout()
        layout.addWidget(self.detector_label)
        layout.addWidget(self.detector_combo)
        layout.addWidget(self.pixel_size_slider)
        layout.addWidget(self.defocus_slider)
        layout.addWidget(self.defocus_diff_slider)
        layout.addWidget(self.defocus_az_slider)
        layout.addWidget(self.amplitude_contrast_slider)
        layout.addWidget(self.additional_phase_slider)

        self.imaging_box.setLayout(layout)
        self.imaging_box.setStyleSheet(SHARED_QGROUPBOX_STYLESHEET)

    def _build_plotting_section(self) -> None:
        """
        Create a QGroupBox for 'Plotting Parameters' (X-axis limit, envelope function toggles, etc.).
        """
        self.plotting_box = QGroupBox("Plotting Parameters")

        self.xlim_slider = FloatSlider("X-axis Limit (Å^-1, 1D)", min_value=0.1, max_value=1.1, step=0.01, value_format="{:.1f}" )
        self.temporal_env_check = QCheckBox("Temporal Envelope")   
        self.spatial_env_check = QCheckBox("Spatial Envelope")
        self.detector_env_check = QCheckBox("Detector Envelope")
        
        layout = QVBoxLayout()
        layout.addWidget(self.xlim_slider)
        layout.addWidget(self.temporal_env_check)
        layout.addWidget(self.spatial_env_check)
        layout.addWidget(self.detector_env_check)

        self.plotting_box.setLayout(layout)
        self.plotting_box.setStyleSheet(SHARED_QGROUPBOX_STYLESHEET)

    def _build_reset_button(self) -> None:
        """
        Create the Reset button.
        """
        self.reset_button = QPushButton("Reset")

    def _build_plot_tabs(self) -> None:
        """
        Create a QTabWidget with two tabs for 1D-CTF and 2D-CTF plots.
        Each tab holds its own MplCanvas.
        """
        self.plot_tabs = QTabWidget()

        # 1D Plot Canvas
        self.canvas_1d = MplCanvas(self, width=5, height=4)
        # 2D Plot Canvas
        self.canvas_2d = MplCanvas(self, width=5, height=4)

        # Wrap them in QWidget for QTabWidget
        widget_1d = QWidget()
        layout_1d = QVBoxLayout(widget_1d)
        layout_1d.addWidget(self.canvas_1d)

        widget_2d = QWidget()
        layout_2d = QVBoxLayout(widget_2d)
        layout_2d.addWidget(self.canvas_2d)

        self.plot_tabs.addTab(widget_1d, "1D-CTF")
        self.plot_tabs.addTab(widget_2d, "2D-CTF")


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
