import numpy as np
from gui import CTFSimGUI
from models import CTF1D, CTF2D

class AppController(CTFSimGUI):
    def __init__(self, line_points=10000, image_size=400):
        super().__init__()
        # Initialize frequency data
        self.line_points = line_points
        self.image_size = image_size
        self._initialize_data()
        
        # Initialize models
        self.ctf_1d = CTF1D()
        self.ctf_2d = CTF2D()

        # Initialize GUI
        self.setup_default_gui_values()

        # Inital plots
        self._setup_initial_plots()
        
        # Event handler
        self._setup_event_handlers()

    def setup_default_gui_values(self):
        self.voltage_slider.set_value(300)
        self.voltage_stability_slider.set_value(3.3333e-8)
        self.electron_source_angle_slider.set_value(1e-4)
        self.electron_source_spread_slider.set_value(0.7)
        self.chromatic_aberr_slider.set_value(3.4)
        self.spherical_aberr_slider.set_value(2.7)
        self.obj_lens_stability_slider.set_value(1.6666e-8)
        self.detector_combo.setCurrentIndex(1)  # Default to "DDD counting"
        self.pixel_size_slider.set_value(1.0)
        self.defocus_slider.set_value(1.0)
        self.defocus_diff_slider.set_value(0)
        self.defocus_az_slider.set_value(0)
        self.amplitude_contrast_slider.set_value(0.1)
        self.additional_phase_slider.set_value(0)
        self.xlim_slider.set_value(0.5)
        self.temporal_env_check.setChecked(True)
        self.spatial_env_check.setChecked(True)
        self.detector_env_check.setChecked(True)

    def _setup_initial_plots(self):
        self.canvas_1d.axes.set_title("1-D Contrast Transfer Function", fontsize=18, fontweight='bold', pad=20)
        self.canvas_1d.axes.set_xlim(0, 0.5)
        self.canvas_1d.axes.tick_params(axis='both', which='major', labelsize=14)
        self.canvas_1d.axes.set_ylim(-1, 1)
        self.canvas_1d.axes.axhline(y=0, color='grey', linestyle='--', alpha=0.8)
        self.canvas_1d.axes.set_xlabel("Spatial Frequency (1/Å)", fontsize=16)
        self.line_et = self.canvas_1d.axes.plot(self.freqs_1d, self.ctf_1d.Et(self.freqs_1d), 
                                                label="Temporal Envelope", 
                                                linestyle="dashed",
                                                linewidth=3)
        self.line_es = self.canvas_1d.axes.plot(self.freqs_1d, self.ctf_1d.Es_1d(self.freqs_1d), 
                                                label="Spacial Envelope", 
                                                linestyle="dashed",
                                                linewidth=3)
        self.line_ed = self.canvas_1d.axes.plot(self.freqs_1d, self.ctf_1d.Ed(self.freqs_1d), 
                                                label="Detector Envelope", 
                                                linestyle="dashed",
                                                linewidth=3)
        self.line_te = self.canvas_1d.axes.plot(self.freqs_1d, self.ctf_1d.Etotal_1d(self.freqs_1d), 
                                                label="Total Envelope",
                                                linewidth=3)
        self.line_dc = self.canvas_1d.axes.plot(self.freqs_1d, self.ctf_1d.dampened_ctf_1d(self.freqs_1d), 
                                                label="Microscope CTF",
                                                linewidth=3)
        self.canvas_1d.axes.legend(fontsize=16)

        self.canvas_2d.axes.set_title("2-D Contrast Transfer Function", fontsize=18, fontweight='bold', pad=20)
        self.image = self.canvas_2d.axes.imshow(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy), cmap='Greys')
        self.annotation_1d = self.canvas_1d.axes.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"), fontsize=10
        )
        self.annotation_1d.set_visible(False)
        self.annotation_2d = self.canvas_2d.axes.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"), fontsize=10
        )
        self.annotation_2d.set_visible(False)

    def _initialize_data(self):
        self.freqs_1d = np.linspace(0.001, 1, self.line_points)
        freq_x = np.linspace(-0.5, 0.5, self.image_size)
        freq_y = np.linspace(-0.5, 0.5, self.image_size)
        self.fx, self.fy = np.meshgrid(freq_x, freq_y, sparse=True)

    def _setup_event_handlers(self):
        """
        Connect signals from PyQt widgets to your update or reset methods.
        """
        self.voltage_slider.valueChanged.connect(lambda value, key="voltage": self.update_ctf(key, value))
        self.voltage_stability_slider.valueChanged.connect(lambda value, key="voltage_stability": self.update_ctf(key, value))
        self.electron_source_angle_slider.valueChanged.connect(lambda value, key="es_angle": self.update_ctf(key, value))
        self.electron_source_spread_slider.valueChanged.connect(lambda value, key="es_spread": self.update_ctf(key, value))
        self.chromatic_aberr_slider.valueChanged.connect(lambda value, key="cc": self.update_ctf(key, value))
        self.spherical_aberr_slider.valueChanged.connect(lambda value, key="cs": self.update_ctf(key, value))
        self.obj_lens_stability_slider.valueChanged.connect(lambda value, key="obj_stability": self.update_ctf(key, value))
        self.detector_combo.currentIndexChanged.connect(lambda value, key="detector": self.update_ctf(key, value))
        self.pixel_size_slider.valueChanged.connect(lambda value, key="pixel_size": self.update_ctf(key, value))
        self.defocus_slider.valueChanged.connect(lambda value, key="df": self.update_ctf(key, value))
        self.defocus_diff_slider.valueChanged.connect(lambda value, key="df_diff": self.update_ctf(key, value))
        self.defocus_az_slider.valueChanged.connect(lambda value, key="df_az": self.update_ctf(key, value))
        self.amplitude_contrast_slider.valueChanged.connect(lambda value, key="ac": self.update_ctf(key, value))
        self.additional_phase_slider.valueChanged.connect(lambda value, key="phase": self.update_ctf(key, value))
        self.xlim_slider.valueChanged.connect(self.update_ctf)
        self.temporal_env_check.stateChanged.connect(lambda value, key="temporal_env": self.update_ctf(key, value))
        self.spatial_env_check.stateChanged.connect(lambda value, key="spatial_env": self.update_ctf(key, value))
        self.detector_env_check.stateChanged.connect(lambda value, key="detector_env": self.update_ctf(key, value))
        self.plot_tabs.currentChanged.connect(self.update_ctf)
        self.reset_button.clicked.connect(self.reset_parameters)
        self.canvas_1d.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas_2d.mpl_connect("motion_notify_event", self.on_hover)

    # ---------------------------------------------------------------------
    # Domain-Specific Logic
    # ---------------------------------------------------------------------
    def update_ctf(self, key=None, value=None):
        if key == "voltage":
            self.ctf_1d.microscope.voltage = value
            self.ctf_2d.microscope.voltage = value
        elif key == "voltage_stability":
            self.ctf_1d.microscope.voltage_stability = value
            self.ctf_2d.microscope.voltage_stability = value
        elif key == "es_angle":
            self.ctf_1d.microscope.electron_source_angle = value
            self.ctf_2d.microscope.electron_source_angle = value
        elif key == "es_spread":
            self.ctf_1d.microscope.electron_source_spread = value
            self.ctf_2d.microscope.electron_source_spread = value
        elif key == "cc":
            self.ctf_1d.microscope.cc = value
            self.ctf_2d.microscope.cc = value
        elif key == "cs":
            self.ctf_1d.microscope.cs = value
            self.ctf_2d.microscope.cs = value
        elif key == "obj_stability":
            self.ctf_1d.microscope.obj_lens_stability = value
            self.ctf_2d.microscope.obj_lens_stability = value
        elif key == "detector":
            self.ctf_1d.detector.detector_type = value
            self.ctf_2d.detector.detector_type = value
        elif key == "pixel_size":
            self.ctf_1d.detector.pixel_size = value
            self.ctf_2d.detector.pixel_size = value
        elif key == "df":
            self.ctf_1d.defocus_um = value
            self.ctf_2d.df = value
        elif key == "df_diff":
            self.ctf_2d.df_diff = value
        elif key == "df_az":
            self.ctf_2d.df_az = value 
        elif key == "ac":
            self.ctf_1d.amplitude_contrast = value
            self.ctf_2d.amplitude_contrast = value
        elif key == "phase":
            self.ctf_1d.phase_shift_deg = value
            self.ctf_2d.phase_shift_deg = value
        elif key == "temporal_env":
            self.ctf_1d.include_temporal_env = value
            self.ctf_2d.include_temporal_env = value
        elif key == "spatial_env":
            self.ctf_1d.include_spatial_env = value
            self.ctf_2d.include_spatial_env = value
        elif key == "detector_env":
            self.ctf_1d.include_detector_env = value
            self.ctf_2d.include_detector_env = value

        self.update_plot()

    def update_plot(self):
        if self.plot_tabs.currentIndex() == 0:
            self.line_et[0].set_data(self.freqs_1d, self.ctf_1d.Et(self.freqs_1d))
            self.line_es[0].set_data(self.freqs_1d, self.ctf_1d.Es_1d(self.freqs_1d))
            self.line_ed[0].set_data(self.freqs_1d, self.ctf_1d.Ed(self.freqs_1d))
            self.line_te[0].set_data(self.freqs_1d, self.ctf_1d.Etotal_1d(self.freqs_1d))
            self.line_dc[0].set_data(self.freqs_1d, self.ctf_1d.dampened_ctf_1d(self.freqs_1d))
            self.canvas_1d.axes.set_xlim(0, self.xlim_slider.get_value())
            self.canvas_1d.draw_idle()
        else:
            self.image.set_data(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy))
            self.canvas_2d.draw_idle()

    def reset_parameters(self):
        self.setup_default_gui_values()
        self.update_ctf()

    def on_hover(self, event):
        """
        Handles the hover event over the Matplotlib canvas.
        """
        if event.inaxes == self.canvas_1d.axes:  # Check if the mouse is over the plot
            x, y = event.xdata, event.ydata  # Get data coordinates
            if x is not None and y is not None:
                # Update annotation position and text
                self.annotation_1d.xy = (x, y)
                self.annotation_1d.set_text(f"x: {x:.2f}, y: {y:.2f}")
                self.annotation_1d.set_visible(True)
                self.canvas_1d.draw_idle()  # Redraw the canvas for updates
        elif event.inaxes == self.canvas_2d.axes:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # col, row = int(round(x)), int(round(y))
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    # Update annotation position and text
                    self.annotation_2d.xy = (x, y)
                    x_freq = (x - 200)/400.
                    y_freq = (y - 200)/400.
                    value = self.ctf_2d.dampened_ctf_2d(np.array([x_freq]), np.array([y_freq]))
                    self.annotation_2d.set_text(f"x: {x_freq:.2f}, y: {y_freq:.2f}, value: {float(value):.2f}")
                    self.annotation_2d.set_visible(True)
                    self.canvas_2d.draw_idle()  # Redraw the canvas for updates
        else:
            self.annotation_1d.set_visible(False)  # Hide annotation if not hovering over the plot
            self.annotation_2d.set_visible(False)
            self.canvas_1d.draw_idle()
            self.canvas_2d.draw_idle()