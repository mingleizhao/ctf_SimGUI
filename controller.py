import numpy as np
from typing import Optional
from gui import CTFSimGUI
from models import CTF1D, CTF2D, CTFIce1D, CTFIce2D


class AppController(CTFSimGUI):
    """
    A controller class that extends the `CTFSimGUI` main window to manage 
    the Contrast Transfer Function (CTF) simulation in both 1D and 2D modes.
    
    This class sets up frequency data, initializes CTF models, configures default GUI values,
    and handles user interactions (e.g., slider changes, resets, tab switching).
    """

    def __init__(self, line_points: int = 10000, image_size: int = 400) -> None:
        """
        Initialize the AppController by creating CTF models, setting up the GUI,
        and establishing event handlers.

        Args:
            line_points (int, optional): Number of sampling points for the 1D plot. Defaults to 10000.
            image_size (int, optional): Size of the 2D plot in pixels. Defaults to 400.
        """
        super().__init__()
        self.line_points: int = line_points
        self.image_size: int = image_size

        # Initialize frequency data
        self._initialize_data()
        
        # Initialize models
        self.ctf_1d: CTFIce1D = CTFIce1D()
        self.ctf_2d: CTFIce2D = CTFIce2D()
        # self.ice_1d: CTFIce1D = CTFIce1D()
        # self.ice_2d: CTFIce2D = CTFIce2D()

        # Initialize GUI
        self.setup_default_gui_values()

        # Initial plots
        self._setup_initial_plots()
        
        # Event handlers
        self._setup_event_handlers()

    def setup_default_gui_values(self) -> None:
        """
        Set default values for all GUI sliders, combo boxes, and checkboxes.
        """
        self.voltage_slider.set_value(300)
        self.voltage_stability_slider.set_value(3.3333e-8)
        self.electron_source_angle_slider.set_value(1e-4)
        self.electron_source_spread_slider.set_value(0.7)
        self.chromatic_aberr_slider.set_value(3.4)
        self.spherical_aberr_slider.set_value(2.7)
        self.obj_lens_stability_slider.set_value(1.6666e-8)
        self.detector_combo.setCurrentIndex(0)  # Default to "DDD counting"
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
        self.ice_thickness_slider.set_value(50)

    def _setup_initial_plots(self) -> None:
        """
        Configure the initial state of the 1D and 2D Matplotlib plots,
        including titles, limits, lines, and annotations.
        """
        # 1D Plot
        self.canvas_1d.axes[1].set_title("1-D Contrast Transfer Function", fontsize=18, fontweight='bold', pad=20)
        self.canvas_1d.axes[1].set_xlim(0, 0.5)
        self.canvas_1d.axes[1].tick_params(axis='both', which='major', labelsize=14)
        self.canvas_1d.axes[1].set_ylim(-1, 1)
        self.canvas_1d.axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.8)
        self.canvas_1d.axes[1].set_xlabel("Spatial Frequency (1/Å)", fontsize=16)

        self.line_et = self.canvas_1d.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.Et(self.freqs_1d),
            label="Temporal Envelope",
            linestyle="dashed",
            linewidth=3
        )
        self.line_es = self.canvas_1d.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.Es_1d(self.freqs_1d),
            label="Spacial Envelope",
            linestyle="dashed",
            linewidth=3
        )
        self.line_ed = self.canvas_1d.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.Ed(self.freqs_1d),
            label="Detector Envelope",
            linestyle="dashed",
            linewidth=3
        )
        self.line_te = self.canvas_1d.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.Etotal_1d(self.freqs_1d),
            label="Total Envelope",
            linewidth=3
        )
        self.line_dc = self.canvas_1d.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.dampened_ctf_1d(self.freqs_1d),
            label="Microscope CTF",
            linewidth=3
        )
        self.canvas_1d.axes[1].legend(fontsize=16)

        # 2D Plot
        self.canvas_2d.axes[1].set_title("2-D Contrast Transfer Function", fontsize=18, fontweight='bold', pad=20)
        self.image = self.canvas_2d.axes[1].imshow(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy), cmap='Greys')

        # Ice Plots
        self.canvas_ice.fig.subplots_adjust(hspace=0.25, top=0.9, bottom=0.05)
        self.canvas_ice.axes[1].set_title("CTF in the presence of ice", fontsize=18, fontweight='bold', pad=20)
        self.canvas_ice.axes[1].set_xlim(0, 0.5)
        self.canvas_ice.axes[1].tick_params(axis='both', which='major', labelsize=12)
        self.canvas_ice.axes[1].set_ylim(-1, 1)
        self.canvas_ice.axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.8)
        self.canvas_ice.axes[1].set_xlabel("Spatial Frequency (1/Å)", fontsize=12)
        self.canvas_ice.axes[1].set_ylabel("Contrast Transfer Function", fontsize=12)
        self.line_ice_ref = self.canvas_ice.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.dampened_ctf_1d(self.freqs_1d),
            label="CTF without ice",
            color="grey",
            linewidth=0.5,
        )
        self.line_ice = self.canvas_ice.axes[1].plot(
            self.freqs_1d,
            self.ctf_1d.dampened_ctf_ice(self.freqs_1d),
            label="CTF with ice",
            color="purple",
            linewidth=1,
        )
        self.canvas_ice.axes[1].legend(fontsize=12)
        self.ice_image_ref = self.canvas_ice.axes[2].imshow(
            self.ctf_2d.dampened_ctf_2d(self.fx, self.fy), 
            extent=(-1, 1, -1, 1), 
            cmap='Greys', 
            vmin=-1, 
            vmax=1
        )
        self.ice_image = self.canvas_ice.axes[3].imshow(
            self.ctf_2d.dampened_ctf_ice(self.fx, self.fy), 
            extent=(-1, 1, -1, 1),
            cmap='Greys', 
            vmin=-1, 
            vmax=1
        )
        cbar = self.canvas_ice.fig.colorbar(
            self.ice_image_ref, 
            ax=self.canvas_ice.axes[3], 
            orientation='vertical',  
            shrink=0.8,  # Adjust the size of the color bar
            pad=0.05  # Adjust the spacing between the color bar and the plot
        )
        cbar.ax.set_title('CTF')
        # cbar.ax.xaxis.set_label_position('top')
        self.canvas_ice.axes[2].set_xticks(np.linspace(-1, 1, 5))
        self.canvas_ice.axes[2].set_yticks(np.linspace(-1, 1, 5))
        self.canvas_ice.axes[3].set_xticks(np.linspace(-1, 1, 5))
        self.canvas_ice.axes[3].set_yticks(np.linspace(-1, 1, 5))

        # Annotations for 1D CTF
        self.annotation_1d = self.canvas_1d.axes[1].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_1d.set_visible(False)

        # Annotations for 2D CTF
        self.annotation_2d = self.canvas_2d.axes[1].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_2d.set_visible(False)

        self.canvas_ice.axes[2].annotate(
            "without ice",
            xy=(-1, 1),
            xytext=(0, -11),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        self.canvas_ice.axes[3].annotate(
            "with ice",
            xy=(-1, 1),
            xytext=(0, -11),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

    def _initialize_data(self) -> None:
        """
        Create 1D and 2D frequency data for plotting the CTF.

        freqs_1d (NDArray): 1D frequency array from 0.001 to 1, for line_points samples.
        fx, fy (NDArray): 2D grids in the range [-0.5, 0.5], used for the 2D CTF.
        """
        self.freqs_1d = np.linspace(0.001, 1, self.line_points)
        freq_x = np.linspace(-0.5, 0.5, self.image_size)
        freq_y = np.linspace(-0.5, 0.5, self.image_size)
        self.fx, self.fy = np.meshgrid(freq_x, freq_y, sparse=True)

    def _setup_event_handlers(self) -> None:
        """
        Connect signals from PyQt widgets to appropriate callbacks
        to update or reset the CTF parameters and refresh the plots.
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
        self.ice_thickness_slider.valueChanged.connect(lambda value, key="ice": self.update_ctf(key, value))

    def update_ctf(self, key: str | None = None, value: float | int | None = None) -> None:
        """
        Update the 1D and 2D CTF models based on parameter changes, then refresh the plots.

        Args:
            key (str | None, optional): The name of the parameter being updated. Defaults to None.
            value (float | int | None, optional): The new value for the parameter. Defaults to None.
        """
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
            self.ctf_1d.detector = value
            self.ctf_2d.detector = value
        elif key == "pixel_size":
            self.ctf_1d.pixel_size = value
            self.ctf_2d.pixel_size = value
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
        elif key == "ice":
            self.ctf_1d.ice_thickness = value
            self.ctf_2d.ice_thickness = value

        self.update_plot()

    def update_plot(self) -> None:
        """
        Redraw the 1D or 2D CTF plot depending on which tab is currently selected.
        """
        if self.plot_tabs.currentIndex() == 0:
            # Update 1D
            self.line_et[0].set_data(self.freqs_1d, self.ctf_1d.Et(self.freqs_1d))
            self.line_es[0].set_data(self.freqs_1d, self.ctf_1d.Es_1d(self.freqs_1d))
            self.line_ed[0].set_data(self.freqs_1d, self.ctf_1d.Ed(self.freqs_1d))
            self.line_te[0].set_data(self.freqs_1d, self.ctf_1d.Etotal_1d(self.freqs_1d))
            self.line_dc[0].set_data(self.freqs_1d, self.ctf_1d.dampened_ctf_1d(self.freqs_1d))
            self.canvas_1d.axes[1].set_xlim(0, self.xlim_slider.get_value())
            self.canvas_1d.draw_idle()
        elif self.plot_tabs.currentIndex() == 1:
            # Update 2D
            self.image.set_data(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy))
            self.canvas_2d.draw_idle()
        else:
            self.line_ice_ref[0].set_data(self.freqs_1d, self.ctf_1d.dampened_ctf_1d(self.freqs_1d))
            self.line_ice[0].set_data(self.freqs_1d, self.ctf_1d.dampened_ctf_ice(self.freqs_1d))
            self.canvas_ice.axes[1].set_xlim(0, self.xlim_slider.get_value())
            self.ice_image_ref.set_data(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy))
            self.ice_image.set_data(self.ctf_2d.dampened_ctf_ice(self.fx, self.fy))
            self.canvas_2d.draw_idle()
            self.canvas_ice.draw_idle()

    def reset_parameters(self) -> None:
        """
        Restore default GUI values and re-compute the CTF plots.
        """
        self.setup_default_gui_values()
        self.update_ctf()

    def on_hover(self, event) -> None:
        """
        Display coordinates and/or values on hover over the 1D or 2D plot.

        Args:
            event: A Matplotlib MouseEvent with xdata, ydata, and inaxes.
        """
        if event.inaxes == self.canvas_1d.axes[1]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.annotation_1d.xy = (x, y)
                self.annotation_1d.set_text(f"x: {x:.2f}, y: {y:.2f}")
                self.annotation_1d.set_visible(True)
                self.canvas_1d.draw_idle()
        elif event.inaxes == self.canvas_2d.axes[1]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    self.annotation_2d.xy = (x, y)
                    x_freq = (x - 200) / 400.0
                    y_freq = (y - 200) / 400.0
                    value = self.ctf_2d.dampened_ctf_2d(np.array([x_freq]), np.array([y_freq]))
                    self.annotation_2d.set_text(f"x: {x_freq:.2f}, y: {y_freq:.2f}, value: {float(value):.2f}")
                    self.annotation_2d.set_visible(True)
                    self.canvas_2d.draw_idle()
        else:
            self.annotation_1d.set_visible(False)
            self.annotation_2d.set_visible(False)
            self.canvas_1d.draw_idle()
            self.canvas_2d.draw_idle()