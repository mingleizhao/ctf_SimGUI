import math
import numpy as np
import pandas as pd
from typing import Optional
from gui import CTFSimGUI
from models import CTFIce1D, CTFIce2D
import matplotlib.transforms as transforms
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PyQt5.QtWidgets import QRadioButton, QFileDialog


class AppController(CTFSimGUI):
    """
    A controller class that extends the `CTFSimGUI` main window to manage 
    the Contrast Transfer Function (CTF) simulation in both 1D and 2D modes.
    
    This class sets up frequency data, initializes CTF models, configures default GUI values,
    and handles user interactions (e.g., slider changes, resets, tab switching, saving plots).
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
        
        # Initialize models
        self.ctf_1d: CTFIce1D = CTFIce1D()
        self.ctf_2d: CTFIce2D = CTFIce2D()
        self.ctf_tomo_ref: CTFIce2D = CTFIce2D()
        self.ctf_tomo_tilt: CTFIce2D = CTFIce2D()

        # Initialize GUI
        self.setup_default_gui_values()

        # Initialize frequency data
        self._initialize_data()

        # Initial plots
        self._setup_initial_plots()

        # Initialize wrap_func
        self.wrap_func = lambda x: x

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
        self.defocus_diff_slider_2d.set_value(0)
        self.defocus_diff_slider_ice.set_value(0)
        self.defocus_diff_slider_tomo.set_value(0)
        self.defocus_az_slider_2d.set_value(0)
        self.defocus_az_slider_ice.set_value(0)
        self.defocus_az_slider_tomo.set_value(0)
        self.amplitude_contrast_slider.set_value(0.1)
        self.additional_phase_slider.set_value(0)
        self.xlim_slider_ice.set_value(0.5)
        self.temporal_env_check.setChecked(True)
        self.spatial_env_check.setChecked(True)
        self.detector_env_check.setChecked(True)
        self.show_temp.setChecked(True)
        self.show_spatial.setChecked(True)
        self.show_detector.setChecked(True)
        self.show_total.setChecked(True)
        self.show_y0.setChecked(True)
        self.show_legend.setChecked(True)
        self.ice_thickness_slider.set_value(50)
        self.plot_1d_x_min.setValue(0)
        self.plot_1d_x_max.setValue(0.5)
        self.plot_1d_y_min.setValue(self._setup_default_ylim()[0])
        self.plot_1d_y_max.setValue(1)
        self.plot_2d_x_min.setValue(-0.5)
        self.plot_2d_x_max.setValue(0.5)
        self.plot_2d_y_min.setValue(-0.5)
        self.plot_2d_y_max.setValue(0.5)
        self.freq_scale_2d.setValue(0.5)
        self.gray_scale_2d.setValue(1)
        self.freq_scale_ice.setValue(0.5)
        self.gray_scale_ice.setValue(1)
        self.sample_size_tomo.setValue(1)
        self.gray_scale_tomo.setValue(1)
        self.sample_thickness_slider_tomo.set_value(50)
        self.tilt_slider_tomo.set_value(0)

    def _setup_initial_plots(self) -> None:
        """
        Configure the initial plots,
        including titles, limits, lines, and annotations.
        """
        self._setup_1d_plot()
        self._setup_2d_plot()
        self._setup_ice_plot()
        self._setup_tomo_plot()
        self._setup_annotations()
  
    def _setup_1d_plot(self):
        # 1D Plot
        self.canvas_1d.axes[1].set_title("1-D Contrast Transfer Function", fontsize=18, fontweight='bold', pad=20)
        self.canvas_1d.axes[1].set_xlim(0, 0.5)
        self.canvas_1d.axes[1].tick_params(axis='both', which='major', labelsize=14)
        self.canvas_1d.axes[1].set_ylim(-1, 1)
        self.canvas_1d.axes[1].set_xlabel("Spatial Frequency (Å⁻¹)", fontsize=16)

        self.line_y0 = self.canvas_1d.axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.8)

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
        self.legend_1d = self.canvas_1d.axes[1].legend(fontsize=16)

    def _setup_2d_plot(self):
        # 2D Plot
        self.canvas_2d.axes[1].set_title("2-D Contrast Transfer Function", fontsize=18, fontweight='bold', pad=20)
        self.image = self.canvas_2d.axes[1].imshow(
            self.ctf_2d.dampened_ctf_2d(self.fx, self.fy), 
            extent=(-0.5, 0.5, -0.5, 0.5), 
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )

        self.canvas_2d.axes[1].tick_params(axis='both', labelsize=14)
        self.canvas_2d.axes[1].set_xlabel("Spatial Frequency X (Å⁻¹)", fontsize=14)
        self.canvas_2d.axes[1].set_ylabel("Spatial Frequency Y (Å⁻¹)", fontsize=14)

        cbar_2D = self.canvas_ice.fig.colorbar(
            self.image, 
            ax=self.canvas_2d.axes[1], 
            orientation='horizontal',  
            shrink=0.5,  # Adjust the size of the color bar
            pad=0.01  # Adjust the spacing between the color bar and the plot
        )
        cbar_2D.ax.set_xlabel('CTF', fontsize=12) 
        cbar_2D.ax.tick_params(labelsize=12)
        cbar_2D.ax.set_position([0.25, 0.12, 0.5, 0.02])

    def _setup_ice_plot(self):
        # Ice Plots
        self.canvas_ice.fig.subplots_adjust(hspace=0.25, top=0.9, bottom=0.05)
        self.canvas_ice.axes[1].set_title("Impact of Sample Thickness on CTF", fontsize=18, fontweight='bold', pad=20)
        self.canvas_ice.axes[1].set_xlim(0, 0.5)
        self.canvas_ice.axes[1].tick_params(axis='both', which='major', labelsize=12)
        self.canvas_ice.axes[1].set_ylim(-1, 1)
        self.canvas_ice.axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.8)
        self.canvas_ice.axes[1].set_xlabel("Spatial Frequency (Å⁻¹)", fontsize=12)
        self.canvas_ice.axes[1].set_ylabel("Contrast Transfer Function", fontsize=12)
        self.canvas_ice.axes[2].set_ylabel("Spatial Frequency (Å⁻¹)")
        # self.canvas_ice.axes[3].set_ylabel("Spatial Frequency (Å⁻¹)")
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
            extent=(-0.5, 0.5, -0.5, 0.5), 
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        self.ice_image = self.canvas_ice.axes[3].imshow(
            self.ctf_2d.dampened_ctf_ice(self.fx, self.fy), 
            extent=(-0.5, 0.5, -0.5, 0.5),
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        cbar_ice = self.canvas_ice.fig.colorbar(
            self.ice_image_ref, 
            ax=self.canvas_ice.axes[3], 
            orientation='vertical',  
            shrink=0.8,  # Adjust the size of the color bar
            pad=0.05  # Adjust the spacing between the color bar and the plot
        )
        cbar_ice.ax.set_title('CTF')

    def _setup_tomo_plot(self):
        self._setup_tomo_data()

        # Draw initial tomo plot 
        self.canvas_tomo.fig.suptitle("Tomography Simulation", fontsize=18, fontweight='bold')
          
        self.canvas_tomo.axes[1].set_xlim(-1500, 1500)
        self.canvas_tomo.axes[1].set_ylim(-1500, 1500)
        self.canvas_tomo.axes[1].set_aspect("equal")
        self.canvas_tomo.axes[1].set_title("Schematic Diagram")
        self.canvas_tomo.axes[1].set_xlabel("Size (µm)")
        self.canvas_tomo.axes[1].set_ylabel("Size (µm)")
        self.canvas_tomo.axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x)/1000:.1f}"))
        self.canvas_tomo.axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x)/1000:.1f}"))

        # Beam
        self.beam = self.canvas_tomo.axes[1].plot([0, 0], [0, 1500], 'b', linewidth=2, label="Beam direction")

        # Sample rectangle centered at (0, 0)
        self.width_tomo, self.height_tomo = 1000.0, 50  # in nanometer
        self.sample_rect = Rectangle(
            (-self.width_tomo / 2, -self.height_tomo / 2), 
            self.width_tomo, 
            self.height_tomo, 
            fc='gray', 
            edgecolor='black',
            label='Illuminated area'
        )
        self.canvas_tomo.axes[1].add_patch(self.sample_rect)

        # Save the center point of rotation
        self.center_x_tomo, self.center_y_tomo = 0, 0

        # Legend
        self.canvas_tomo.axes[1].legend(handles=[
            self.beam[0],
            self.sample_rect])

        # Setup axis labels        
        self.canvas_tomo.axes[3].set_xlabel("Spatial Frequency X (Å⁻¹)")
        self.canvas_tomo.axes[3].set_ylabel("Spatial Frequency Y (Å⁻¹)")
        self.canvas_tomo.axes[4].set_title("CTF with Tilted Sample", fontsize=14, pad=15)
        self.canvas_tomo.axes[4].set_xlabel("Spatial Frequency X (Å⁻¹)")
        self.canvas_tomo.axes[4].set_ylabel("Spatial Frequency Y (Å⁻¹)")

        # CTF
        self.tomo_image_ref = self.canvas_tomo.axes[3].imshow(
            self.ctf_tomo_ref.dampened_ctf_ice(self.fx_tomo, self.fy_tomo), 
            extent=(-self.nyquist_tomo, self.nyquist_tomo, -self.nyquist_tomo, self.nyquist_tomo), 
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        self.tomo_image = self.canvas_tomo.axes[4].imshow(
            self.ctf_tomo_tilt.dampened_ctf_ice(self.fx_tomo, self.fy_tomo), 
            extent=(-self.nyquist_tomo, self.nyquist_tomo, -self.nyquist_tomo, self.nyquist_tomo),
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        cbar_tomo = self.canvas_tomo.fig.colorbar(
            self.tomo_image_ref, 
            ax=self.canvas_tomo.axes[4], 
            orientation='horizontal',  
            shrink=0.8,  # Adjust the size of the color bar
            pad=0.05  # Adjust the spacing between the color bar and the plot
        )
        cbar_tomo.ax.set_xlabel('CTF')
        cbar_tomo.ax.set_position([0.48, 0.15, 0.5, 0.02])

    def _setup_annotations(self):
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

        # Annotations for Ice tab
        self.annotation_ice_1d = self.canvas_ice.axes[1].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_ice_1d.set_visible(False)

        self.annotation_ice_ref = self.canvas_ice.axes[2].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_ice_ref.set_visible(False)

        self.annotation_ice_ctf = self.canvas_ice.axes[3].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_ice_ctf.set_visible(False)

        self.canvas_ice.axes[2].annotate(
            "without ice",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, -11),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        self.canvas_ice.axes[3].annotate(
            "with ice",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, -11),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        # Annotations for Tomo tab
        self.annotation_tomo_diagram_note = self.canvas_tomo.axes[1].annotate(
            (
                "This simulation assumes: \n"
                "1) The electron beam remains parallel to the optical axis.\n"
                "2) Sample tilting does not introduce astigmatism.\n"
                "3) The CTF is affected solely by the apparent sample thickness."
            ),
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(80, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )
        # self.annotation_tomo_diagram_note.set_visible(False)

        self.annotation_tomo_diagram_state = self.canvas_tomo.axes[1].annotate(
            "",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, -33),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )
        self.annotation_tomo_diagram_state.set_visible(False)

        self.annotation_tomo_tilt_ctf = self.canvas_tomo.axes[4].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_tomo_tilt_ctf.set_visible(False)

        self.annotation_tomo_ref_ctf = self.canvas_tomo.axes[3].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )
        self.annotation_tomo_ref_ctf.set_visible(False)

        self.canvas_tomo.axes[3].annotate(
            "without tilt",
            xy=(0, 1),
            xycoords="axes fraction",
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

    def _setup_tomo_data(self) -> None:
        """Setup tomo specific data
        """        
        self.nyquist_tomo = 0.5 / self.pixel_size_slider.get_value()

        freq_x = np.linspace(-self.nyquist_tomo, self.nyquist_tomo, self.image_size)
        freq_y = np.linspace(-self.nyquist_tomo, self.nyquist_tomo, self.image_size)
        self.fx_tomo, self.fy_tomo = np.meshgrid(freq_x, freq_y, sparse=True)
        self.update_tomo_data = False

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
        self.sample_size_tomo.valueChanged.connect(lambda value, key="sample_size": self.update_tomo(key, value))
        self.defocus_slider.valueChanged.connect(lambda value, key="df": self.update_ctf(key, value))
        self.defocus_diff_slider_2d.valueChanged.connect(lambda value, key="df_diff": self.update_ctf(key, value))
        self.defocus_diff_slider_ice.valueChanged.connect(lambda value, key="df_diff": self.update_ctf(key, value))
        self.defocus_az_slider_2d.valueChanged.connect(lambda value, key="df_az": self.update_ctf(key, value))
        self.defocus_az_slider_ice.valueChanged.connect(lambda value, key="df_az": self.update_ctf(key, value))
        self.amplitude_contrast_slider.valueChanged.connect(lambda value, key="ac": self.update_ctf(key, value))
        self.additional_phase_slider.valueChanged.connect(lambda value, key="phase": self.update_ctf(key, value))
        self.plot_1d_x_min.valueChanged.connect(self.update_plot_range)
        self.plot_1d_x_max.valueChanged.connect(self.update_plot_range)
        self.plot_1d_y_min.valueChanged.connect(self.update_plot_range)
        self.plot_1d_y_max.valueChanged.connect(self.update_plot_range)
        self.plot_2d_x_min.valueChanged.connect(self.update_plot_range)
        self.plot_2d_x_max.valueChanged.connect(self.update_plot_range)
        self.plot_2d_y_min.valueChanged.connect(self.update_plot_range)
        self.plot_2d_y_max.valueChanged.connect(self.update_plot_range)
        self.freq_scale_2d.valueChanged.connect(self.zoom_2d_ctf)
        self.xlim_slider_ice.valueChanged.connect(self.update_plot_range)
        self.freq_scale_ice.valueChanged.connect(self.zoom_2d_ctf)
        self.gray_scale_2d.valueChanged.connect(self.update_grayness)
        self.gray_scale_ice.valueChanged.connect(self.update_grayness)
        self.gray_scale_tomo.valueChanged.connect(self.update_grayness)
        self.temporal_env_check.stateChanged.connect(lambda value, key="temporal_env": self.update_ctf(key, value))
        self.spatial_env_check.stateChanged.connect(lambda value, key="spatial_env": self.update_ctf(key, value))
        self.detector_env_check.stateChanged.connect(lambda value, key="detector_env": self.update_ctf(key, value))
        self.show_temp.stateChanged.connect(self.update_display_1d)
        self.show_spatial.stateChanged.connect(self.update_display_1d)
        self.show_detector.stateChanged.connect(self.update_display_1d)
        self.show_total.stateChanged.connect(self.update_display_1d)
        self.show_y0.stateChanged.connect(self.update_display_1d)
        self.show_legend.stateChanged.connect(self.update_display_1d)
        self.plot_tabs.currentChanged.connect(lambda value, key="tab_switch": self.update_ctf(key, value))
        self.reset_button.clicked.connect(self.reset_parameters)
        self.save_img_button.clicked.connect(self.save_plot)
        self.save_csv_button.clicked.connect(self.save_csv)
        self.canvas_1d.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas_2d.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas_ice.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas_tomo.mpl_connect("motion_notify_event", self.on_hover)
        self.ice_thickness_slider.valueChanged.connect(lambda value, key="ice": self.update_ctf(key, value))
        self.radio_button_group.buttonToggled.connect(self.update_wrap_func)
        self.sample_thickness_slider_tomo.valueChanged.connect(lambda value, key="thickness": self.update_tomo(key, value))
        self.tilt_slider_tomo.valueChanged.connect(lambda value, key="tilt_angle": self.update_tomo(key, value))
        self.defocus_diff_slider_tomo.valueChanged.connect(lambda value, key="df_diff": self.update_tomo(key, value))
        self.defocus_az_slider_tomo.valueChanged.connect(lambda value, key="df_az": self.update_tomo(key, value))

    def _setup_ctf_wrap_func(self):
        if self.radio_button_group.checkedButton() == self.radio_ctf:
            return lambda x: x
        elif self.radio_button_group.checkedButton() == self.radio_abs_ctf:
            return lambda x: np.abs(x)
        elif self.radio_button_group.checkedButton() == self.radio_ctf_squared:
            return lambda x: x ** 2

    def _setup_default_ylim(self):
        if self.radio_button_group.checkedButton() == self.radio_ctf:
            return (-1, 1)
        elif self.radio_button_group.checkedButton() == self.radio_abs_ctf:
            return (0, 1)
        elif self.radio_button_group.checkedButton() == self.radio_ctf_squared:
            return (0, 1)

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
            self.ctf_tomo_ref.microscope.voltage = value
            self.ctf_tomo_tilt.microscope.voltage = value
        elif key == "voltage_stability":
            self.ctf_1d.microscope.voltage_stability = value
            self.ctf_2d.microscope.voltage_stability = value
            self.ctf_tomo_ref.microscope.voltage_stability = value
            self.ctf_tomo_tilt.microscope.voltage_stability = value
        elif key == "es_angle":
            self.ctf_1d.microscope.electron_source_angle = value
            self.ctf_2d.microscope.electron_source_angle = value
            self.ctf_tomo_ref.microscope.electron_source_angle = value
            self.ctf_tomo_tilt.microscope.electron_source_angle = value
        elif key == "es_spread":
            self.ctf_1d.microscope.electron_source_spread = value
            self.ctf_2d.microscope.electron_source_spread = value
            self.ctf_tomo_ref.microscope.electron_source_spread = value
            self.ctf_tomo_tilt.microscope.electron_source_spread = value
        elif key == "cc":
            self.ctf_1d.microscope.cc = value
            self.ctf_2d.microscope.cc = value
            self.ctf_tomo_ref.microscope.cc = value
            self.ctf_tomo_tilt.microscope.cc = value
        elif key == "cs":
            self.ctf_1d.microscope.cs = value
            self.ctf_2d.microscope.cs = value
            self.ctf_tomo_ref.microscope.cs = value
            self.ctf_tomo_tilt.microscope.cs = value
        elif key == "obj_stability":
            self.ctf_1d.microscope.obj_lens_stability = value
            self.ctf_2d.microscope.obj_lens_stability = value
            self.ctf_tomo_ref.microscope.obj_lens_stability = value
            self.ctf_tomo_tilt.microscope.obj_lens_stability = value
        elif key == "detector":
            self.ctf_1d.detector = value
            self.ctf_2d.detector = value
            self.ctf_tomo_ref.detector = value
            self.ctf_tomo_tilt.detector = value
        elif key == "pixel_size":
            self.ctf_1d.pixel_size = value
            self.ctf_2d.pixel_size = value
            self.ctf_tomo_ref.pixel_size = value
            self.ctf_tomo_tilt.pixel_size = value
            self.update_tomo_data = True
        elif key == "df":
            self.ctf_1d.defocus_um = value
            self.ctf_2d.df = value
            self.ctf_tomo_ref.df = value
            self.ctf_tomo_tilt.df = value
        elif key == "df_diff":
            self.ctf_2d.df_diff = value
        elif key == "df_az":
            self.ctf_2d.df_az = value
        elif key == "ac":
            self.ctf_1d.amplitude_contrast = value
            self.ctf_2d.amplitude_contrast = value
            self.ctf_tomo_ref.amplitude_contrast = value
            self.ctf_tomo_tilt.amplitude_contrast = value
        elif key == "phase":
            self.ctf_1d.phase_shift_deg = value
            self.ctf_2d.phase_shift_deg = value
            self.ctf_tomo_ref.phase_shift_deg = value
            self.ctf_tomo_tilt.phase_shift_deg = value
        elif key == "temporal_env":
            self.ctf_1d.include_temporal_env = value
            self.ctf_2d.include_temporal_env = value
            self.ctf_tomo_ref.include_temporal_env = value
            self.ctf_tomo_tilt.include_temporal_env = value
        elif key == "spatial_env":
            self.ctf_1d.include_spatial_env = value
            self.ctf_2d.include_spatial_env = value
            self.ctf_tomo_ref.include_spatial_env = value
            self.ctf_tomo_tilt.include_spatial_env = value
        elif key == "detector_env":
            self.ctf_1d.include_detector_env = value
            self.ctf_2d.include_detector_env = value
            self.ctf_tomo_ref.include_detector_env = value
            self.ctf_tomo_tilt.include_detector_env = value
        elif key == "ice":
            self.ctf_1d.ice_thickness = value
            self.ctf_2d.ice_thickness = value
        elif key == "tab_switch":
            self.update_plot_range()
            if value == 0:  # 1D tab
                pass
            elif value == 1:  # 2D tab
                self.defocus_diff_slider_2d.set_value(self.ctf_2d.df_diff)
                self.defocus_az_slider_2d.set_value(self.ctf_2d.df_az)
            elif value == 2:  # Ice tab
                self.defocus_diff_slider_ice.set_value(self.ctf_2d.df_diff)
                self.defocus_az_slider_ice.set_value(self.ctf_2d.df_az)

        self.update_plot()

    def update_plot(self) -> None:
        """
        Redraw the 1D or 2D CTF plot depending on which tab and CTF format are currently selected.
        """
        if self.plot_tabs.currentIndex() == 0:
            # Update 1D
            self.line_et[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.Et(self.freqs_1d)))
            self.line_es[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.Es_1d(self.freqs_1d)))
            self.line_ed[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.Ed(self.freqs_1d)))
            self.line_te[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.Etotal_1d(self.freqs_1d)))
            self.line_dc[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.dampened_ctf_1d(self.freqs_1d)))
            self.canvas_1d.draw_idle()
        elif self.plot_tabs.currentIndex() == 1:
            # Update 2D
            self.image.set_data(self.wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy)))
            self.canvas_2d.draw_idle()
        elif self.plot_tabs.currentIndex() == 2:
            # Update ice tab plots
            self.line_ice_ref[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.dampened_ctf_1d(self.freqs_1d)))
            self.line_ice[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.dampened_ctf_ice(self.freqs_1d)))
            self.canvas_ice.axes[1].set_xlim(0, self.xlim_slider_ice.get_value())
            self.ice_image_ref.set_data(self.wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy)))
            self.ice_image.set_data(self.wrap_func(self.ctf_2d.dampened_ctf_ice(self.fx, self.fy)))
            self.canvas_ice.draw_idle()
        elif self.plot_tabs.currentIndex() == 3:
            # Update tomo tab plots
            if self.update_tomo_data:
                self._setup_tomo_data()
            self.tomo_image_ref.set_data(self.wrap_func(self.ctf_tomo_ref.dampened_ctf_ice(self.fx_tomo, self.fy_tomo)))
            self.tomo_image_ref.set_extent((-self.nyquist_tomo, self.nyquist_tomo, -self.nyquist_tomo, self.nyquist_tomo))
            self.tomo_image.set_data(self.wrap_func(self.ctf_tomo_tilt.dampened_ctf_ice(self.fx_tomo, self.fy_tomo)))
            self.tomo_image.set_extent((-self.nyquist_tomo, self.nyquist_tomo, -self.nyquist_tomo, self.nyquist_tomo))
            self.canvas_tomo.draw_idle()

    def update_plot_range(self) -> None:
        """
        Update X Y limits of the plot without setting data.
        """        
        if self.plot_tabs.currentIndex() == 0:
            self.canvas_1d.axes[1].set_xlim(self.plot_1d_x_min.value(), self.plot_1d_x_max.value())
            self.canvas_1d.axes[1].set_ylim(self.plot_1d_y_min.value(), self.plot_1d_y_max.value())
            self.canvas_1d.draw_idle()
        elif self.plot_tabs.currentIndex() == 1:
            self.canvas_2d.axes[1].set_xlim(self.plot_2d_x_min.value(), self.plot_2d_x_max.value())
            self.canvas_2d.axes[1].set_ylim(self.plot_2d_y_min.value(), self.plot_2d_y_max.value())
            self.canvas_2d.draw_idle()
        elif self.plot_tabs.currentIndex() == 2:
            self.canvas_ice.axes[1].set_xlim(0, self.xlim_slider_ice.get_value())
            self.canvas_ice.draw_idle()

    def zoom_2d_ctf(self) -> None:
        """
        Zoom on 2D CTF.
        """
        if self.plot_tabs.currentIndex() == 1:
            self.canvas_2d.axes[1].set_xlim(-self.freq_scale_2d.value(), self.freq_scale_2d.value())
            self.canvas_2d.axes[1].set_ylim(-self.freq_scale_2d.value(), self.freq_scale_2d.value())
            self.canvas_2d.axes[1].set_xlim(-self.freq_scale_2d.value(), self.freq_scale_2d.value())
            self.canvas_2d.axes[1].set_ylim(-self.freq_scale_2d.value(), self.freq_scale_2d.value())
            self.canvas_2d.draw_idle()
        elif self.plot_tabs.currentIndex() == 2:
            self.canvas_ice.axes[2].set_xlim(-self.freq_scale_ice.value(), self.freq_scale_ice.value())
            self.canvas_ice.axes[2].set_ylim(-self.freq_scale_ice.value(), self.freq_scale_ice.value())
            self.canvas_ice.axes[3].set_xlim(-self.freq_scale_ice.value(), self.freq_scale_ice.value())
            self.canvas_ice.axes[3].set_ylim(-self.freq_scale_ice.value(), self.freq_scale_ice.value())
            self.canvas_ice.draw_idle()

    def update_grayness(self) -> None:
        """
        Update the max scales on 2D CTF.
        """
        if self.plot_tabs.currentIndex() == 1:
            current_vmin, _ = self.image.get_clim()
            if current_vmin != 0:
                vmin = -self.gray_scale_2d.value()               
            else:
                vmin = 0
            vmax = self.gray_scale_2d.value()
            self.image.set_clim(vmin=vmin, vmax=vmax)
            self.canvas_2d.draw_idle()
        elif self.plot_tabs.currentIndex() == 2:
            current_vmin, _ = self.ice_image.get_clim()
            if current_vmin != 0:
                vmin = -self.gray_scale_ice.value()               
            else:
                vmin = 0
            vmax = self.gray_scale_ice.value()
            self.ice_image.set_clim(vmin=vmin, vmax=vmax)
            self.ice_image_ref.set_clim(vmin=vmin, vmax=vmax)
            self.canvas_ice.draw_idle()
        elif self.plot_tabs.currentIndex() == 3:
            current_vmin, _ = self.tomo_image.get_clim()
            if current_vmin != 0:
                vmin = -self.gray_scale_tomo.value()               
            else:
                vmin = 0
            vmax = self.gray_scale_tomo.value()
            self.tomo_image.set_clim(vmin=vmin, vmax=vmax)
            self.tomo_image_ref.set_clim(vmin=vmin, vmax=vmax)
            self.canvas_tomo.draw_idle()

    def update_display_1d(self) -> None:
        """
        Update the display of 1D CTF
        """
        self.line_et[0].set_visible(self.show_temp.isChecked())
        self.line_es[0].set_visible(self.show_spatial.isChecked())
        self.line_ed[0].set_visible(self.show_detector.isChecked())
        self.line_te[0].set_visible(self.show_total.isChecked())
        self.line_y0.set_visible(self.show_y0.isChecked())
        self.legend_1d = self.canvas_1d.axes[1].legend(fontsize=16)
        self.legend_1d.set_visible(self.show_legend.isChecked())
        self.canvas_1d.draw_idle()

    def update_wrap_func(self) -> None:
        """
        Update wraper function without setting data.
        """
        self.wrap_func = self._setup_ctf_wrap_func()

        ylim = self._setup_default_ylim()

        # Rescale 1D plots
        self.canvas_1d.axes[1].set_ylim(*ylim)
        self.plot_1d_y_min.setValue(ylim[0])
        self.plot_1d_y_max.setValue(ylim[1])
        self.canvas_ice.axes[1].set_ylim(*ylim)
        
        # Renormalize the 2D images
        self.image.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.ice_image_ref.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.ice_image.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.tomo_image_ref.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.tomo_image.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))

        # Reset gray scales
        self.gray_scale_2d.setValue(1)
        self.gray_scale_ice.setValue(1)
        self.gray_scale_tomo.setValue(1)

        # Reset spatial frequency
        self.freq_scale_2d.setValue(0.5)
        self.freq_scale_ice.setValue(0.5)
        
        self.update_plot()

    def reset_parameters(self) -> None:
        """
        Restore default GUI values and re-compute the CTF plots.
        """
        self.setup_default_gui_values()
        self.update_plot_range()
        self.update_ctf()

    def save_plot(self) -> None:
        """Opens a file dialog and saves the plots in the current tab as an image."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )

        if not file_path:
            return  # Exit if no file is selected

        if self.plot_tabs.currentIndex() == 0:
            self.canvas_1d.fig.savefig(file_path, dpi=300)
        elif self.plot_tabs.currentIndex() == 1:
            self.canvas_2d.fig.savefig(file_path, dpi=300)
        elif self.plot_tabs.currentIndex() == 2:
            self.canvas_ice.fig.savefig(file_path, dpi=300)
        elif self.plot_tabs.currentIndex() == 3:
            self.canvas_tomo.fig.savefig(file_path, dpi=300)

    def save_csv(self) -> None:
        """Opens a file dialog and saves the plotted data in the current tab as a CSV file."""
        wrap_func = self._setup_ctf_wrap_func()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "", "CSV Files (*.csv);;All Files (*)",
            options=options
        )

        if not file_path:
            return  # Exit if no file is selected

        tab_index = self.plot_tabs.currentIndex()

        if tab_index == 0:  # 1D CTF
            df = self._generate_ctf_1d_data(wrap_func)
            df.to_csv(file_path, index=False)

        elif tab_index == 1:  # 2D CTF
            x_full, y_full = self._get_full_meshgrid(self.fx, self.fy)
            df = self._generate_ctf_2d_data(wrap_func, x_full, y_full, ice=False)
            df.to_csv(file_path, index=False)

        elif tab_index == 2:  # Ice CTF
            x_full, y_full = self._get_full_meshgrid(self.fx, self.fy)
            with open(file_path, "w") as f:
                f.write("# 1D CTF\n")
                self._generate_ctf_1d_data(wrap_func).to_csv(f, index=False)

                f.write("\n# 2D CTF\n")
                self._generate_ctf_2d_data(wrap_func, x_full, y_full, ice=True).to_csv(f, index=False)
        elif tab_index == 3:  # Tomo CTF
            x_full, y_full =self._get_full_meshgrid(self.fx_tomo, self.fy_tomo)
            df = self._generate_ctf_tomo_data(wrap_func, x_full, y_full)
            df.to_csv(file_path, index=False)

    def _generate_ctf_1d_data(self, wrap_func) -> pd.DataFrame:
        """Generates 1D CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "freqs_1d": self.freqs_1d,
            "ctf": wrap_func(self.ctf_1d.ctf_1d(self.freqs_1d)),
            "ctf_dampened": wrap_func(self.ctf_1d.dampened_ctf_1d(self.freqs_1d)),
            "temporal_env": wrap_func(self.ctf_1d.Et(self.freqs_1d)),
            "spatial_env": wrap_func(self.ctf_1d.Es_1d(self.freqs_1d)),
            "detector_env": wrap_func(self.ctf_1d.Ed(self.freqs_1d)),
            "total_env": wrap_func(self.ctf_1d.Etotal_1d(self.freqs_1d)),
        })

    def _generate_ctf_2d_data(self, wrap_func, x_full, y_full, ice: bool) -> pd.DataFrame:
        """Generates 2D CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "freqs_x": x_full.flatten(),
            "freqs_y": y_full.flatten(),
            "ctf_no_ice": wrap_func(self.ctf_2d.ctf_2d(self.fx, self.fy)).flatten(),
            "ctf_no_ice_dampened": wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy)).flatten(),
            "ctf_ice": wrap_func(self.ctf_2d.ctf_ice(self.fx, self.fy)).flatten() if ice else None,
            "ctf_ice_dampened": wrap_func(self.ctf_2d.dampened_ctf_ice(self.fx, self.fy)).flatten() if ice else None,
            "total_env": wrap_func(self.ctf_2d.Etotal_2d(self.fx, self.fy)).flatten(),
        })
    
    def _generate_ctf_tomo_data(self, wrap_func, x_full, y_full) -> pd.DataFrame:
        """Generates 2D CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "freqs_x": x_full.flatten(),
            "freqs_y": y_full.flatten(),
            "ctf_no_tilt": wrap_func(self.ctf_tomo_ref.ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_no_tilt_dampened": wrap_func(self.ctf_tomo_ref.dampened_ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_tilt": wrap_func(self.ctf_tomo_tilt.ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_tilt_dampened": wrap_func(self.ctf_tomo_tilt.dampened_ctf_ice(self.fx, self.fy)).flatten(),
            "total_env": wrap_func(self.ctf_2d.Etotal_2d(self.fx, self.fy)).flatten(),
        })

    def _get_full_meshgrid(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Generates a full meshgrid for 2D frequency data."""
        x_full = np.broadcast_to(x, (self.image_size, self.image_size))
        y_full = np.broadcast_to(y, (self.image_size, self.image_size))
        return x_full, y_full

    def on_hover(self, event) -> None:
        """
        Display coordinates and/or values on hover over the 1D or 2D plot.

        Args:
            event: A Matplotlib MouseEvent with xdata, ydata, and inaxes.
        """
        wrap_func = self._setup_ctf_wrap_func()

        if event.inaxes == self.canvas_1d.axes[1]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / x
                value = wrap_func(self.ctf_1d.dampened_ctf_1d(np.array([x])))
                self.annotation_1d.xy = (x, y)
                self.annotation_1d.set_text(f"x: {x:.3f} Å⁻¹\ny: {y:.3f}\nres: {res:.2f} Å\nctf: {float(value):.4f}")
                self.annotation_1d.set_visible(True)
                self.canvas_1d.draw_idle()
        elif event.inaxes == self.canvas_2d.axes[1]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / math.sqrt(x**2 + y**2)
                # angle = math.degrees(math.atan2(y, x))
                self.annotation_2d.xy = (x, y)
                value = wrap_func(self.ctf_2d.dampened_ctf_2d(np.array([x]), np.array([y])))
                self.annotation_2d.set_text(f"x: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {float(value):.4f}")
                self.annotation_2d.set_visible(True)
                self.canvas_2d.draw_idle()
        elif event.inaxes == self.canvas_ice.axes[1]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / x
                value_no_ice = wrap_func(self.ctf_1d.dampened_ctf_1d(np.array([x])))
                value_ice = wrap_func(self.ctf_1d.dampened_ctf_ice(np.array([x])))
                self.annotation_ice_1d.xy = (x, y)
                self.annotation_ice_1d.set_text(f"x: {x:.3f} Å⁻¹\nres: {res:.2f} Å\ngray: {float(value_no_ice):.4f}\npurple: {float(value_ice):.4f}")
                self.annotation_ice_1d.set_visible(True)
                self.canvas_ice.draw_idle()
        elif event.inaxes == self.canvas_ice.axes[2]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / math.sqrt(x**2 + y**2)
                self.annotation_ice_ref.xy = (x, y)
                value = wrap_func(self.ctf_2d.dampened_ctf_2d(np.array([x]), np.array([y])))
                self.annotation_ice_ref.set_text(f"x: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {float(value):.4f}")
                self.annotation_ice_ref.set_visible(True)
                self.canvas_ice.draw_idle()
        elif event.inaxes == self.canvas_ice.axes[3]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / math.sqrt(x**2 + y**2)
                self.annotation_ice_ctf.xy = (x, y)
                value = wrap_func(self.ctf_2d.dampened_ctf_ice(np.array([x]), np.array([y])))
                self.annotation_ice_ctf.set_text(f"x: {x:.3f} Å⁻¹\n"
                                                 f"y: {y:.3f} Å⁻¹\n"
                                                 f"res: {res:.2f} Å\n"
                                                 f"ctf: {float(value):.4f}")
                self.annotation_ice_ctf.set_visible(True)
                self.canvas_ice.draw_idle()
        elif event.inaxes == self.canvas_tomo.axes[1]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # self.annotation_tomo_diagram_note.set_text(
                #     "This simulation assumes: \n"
                #     "1) The electron beam remains parallel to the optical axis.\n"
                #     "2) Sample tilting does not introduce astigmatism.\n"
                #     "3) The CTF is affected solely by the apparent sample thickness."
                # )
                # self.annotation_tomo_diagram_note.set_visible(True)
                tilt_angle_rad = abs(math.radians(self.tilt_slider_tomo.get_value()))
                thickness = self.width_tomo * math.sin(tilt_angle_rad) +  self.height_tomo * math.cos(tilt_angle_rad)
                self.annotation_tomo_diagram_state.set_text(f"size: {self.sample_size_tomo.value():.2f} µm\n"
                                                            f"tilt angle: {self.tilt_slider_tomo.get_value():.1f}°\n"
                                                            f"thk.: {thickness:.1f} nm")
                self.annotation_tomo_diagram_state.set_visible(True)
                self.canvas_tomo.draw_idle()
        elif event.inaxes == self.canvas_tomo.axes[3]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / math.sqrt(x**2 + y**2)
                # angle = math.degrees(math.atan2(y, x))
                self.annotation_tomo_ref_ctf.xy = (x, y)
                value = wrap_func(self.ctf_tomo_ref.dampened_ctf_ice(np.array([x]), np.array([y])))
                self.annotation_tomo_ref_ctf.set_text(f"tilt angle: 0°\n"
                                                      f"x: {x:.3f} Å⁻¹\n"
                                                      f"y: {y:.3f} Å⁻¹\n"
                                                      f"res: {res:.2f} Å\n"
                                                      f"ctf: {float(value):.4f}")
                self.annotation_tomo_ref_ctf.set_visible(True)
                self.canvas_tomo.draw_idle()
        elif event.inaxes == self.canvas_tomo.axes[4]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                res = 1. / math.sqrt(x**2 + y**2)
                # angle = math.degrees(math.atan2(y, x))
                self.annotation_tomo_tilt_ctf.xy = (x, y)
                value = wrap_func(self.ctf_tomo_tilt.dampened_ctf_ice(np.array([x]), np.array([y])))
                self.annotation_tomo_tilt_ctf.set_text(f"tilt angle: {self.tilt_slider_tomo.get_value():.1f}°\nx: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {float(value):.4f}")
                self.annotation_tomo_tilt_ctf.set_visible(True)
                self.canvas_tomo.draw_idle()
        else:
            self.annotation_1d.set_visible(False)
            self.annotation_2d.set_visible(False)
            self.annotation_ice_1d.set_visible(False)
            self.annotation_ice_ref.set_visible(False)
            self.annotation_ice_ctf.set_visible(False)
            # self.annotation_tomo_diagram_note.set_visible(False)
            self.annotation_tomo_diagram_state.set_visible(False)
            self.annotation_tomo_ref_ctf.set_visible(False)
            self.annotation_tomo_tilt_ctf.set_visible(False)
            self.canvas_1d.draw_idle()
            self.canvas_2d.draw_idle()
            self.canvas_ice.draw_idle()
            self.canvas_tomo.draw_idle()

    def update_tomo(self, key: str | None = None, value: float | int | None = None) -> None:
        """
        Redraw the tomo diagram and CTF plots for the tomo tab depending on the parameters.
        """
        if key == "thickness":
            self.height_tomo = value
            self._update_sample_tomo()
            self.ctf_tomo_ref.ice_thickness = value
            tilt_angle_rad = abs(math.radians(self.tilt_slider_tomo.get_value()))
            self.ctf_tomo_tilt.ice_thickness = self.width_tomo * math.sin(tilt_angle_rad) +  self.height_tomo * math.cos(tilt_angle_rad)
            self.update_plot()
        elif key == "tilt_angle":
            self._rotate_sample(value)
            height_tomo = self.sample_thickness_slider_tomo.get_value()  # in nm
            tilt_angle_rad = abs(math.radians(value))
            self.ctf_tomo_tilt.ice_thickness = self.width_tomo * math.sin(tilt_angle_rad) +  height_tomo * math.cos(tilt_angle_rad)
            self.update_plot()
        elif key == "df_diff":
            self.ctf_tomo_ref.df_diff = value
            self.ctf_tomo_tilt.df_diff = value
            self.update_plot()
        elif key == "df_az":
            self.ctf_tomo_ref.df_az = value
            self.ctf_tomo_tilt.df_az = value
            self.update_plot()
        elif key == "sample_size":
            self.width_tomo = value * 1000
            self._update_sample_tomo()
            height_tomo = self.sample_thickness_slider_tomo.get_value()  # in nm
            tilt_angle_rad = abs(math.radians(self.tilt_slider_tomo.get_value()))
            self.ctf_tomo_tilt.ice_thickness = self.width_tomo * math.sin(tilt_angle_rad) +  height_tomo * math.cos(tilt_angle_rad)
            self.update_plot()

        self.canvas_tomo.draw_idle()

    def _update_sample_tomo(self):
        """Change the size of the sample rectangle."""
        self.sample_rect.set_xy((-self.width_tomo / 2., -self.height_tomo / 2.))
        self.sample_rect.set_width(self.width_tomo)
        self.sample_rect.set_height(self.height_tomo)

    def _rotate_sample(self, angle: float) -> None:
        """Apply a rotation transformation around the center of the sample rectangle."""
        # Create a transformation: first translate to (0,0), then rotate, then translate back
        transform = (transforms.Affine2D()
                     .rotate_deg_around(self.center_x_tomo, self.center_y_tomo, angle)
                     + self.canvas_tomo.axes[1].transData)

        # Apply the transformation
        self.sample_rect.set_transform(transform)
