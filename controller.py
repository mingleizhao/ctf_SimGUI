import math
import numpy as np
import pandas as pd
from typing import Optional
from gui import CTFSimGUI
from models import CTFIce1D, CTFIce2D
import matplotlib.transforms as transforms
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.image import imread
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PIL import Image


class AppController(CTFSimGUI):
    """
    A controller class that extends the `CTFSimGUI` main window to manage 
    the Contrast Transfer Function (CTF) simulation in both 1D and 2D modes.
    
    This class sets up frequency data, initializes CTF models, configures default GUI values,
    and handles user interactions (e.g., slider changes, resets, tab switching, saving plots).
    """

    def __init__(self, 
                 line_points: int = 10000, 
                 image_size: int = 400, 
                 default_image: str = "sample_images/sample_image.png") -> None:
        """
        Initialize the AppController by creating CTF models, setting up the GUI,
        and establishing event handlers.

        Args:
            line_points (int, optional): Number of sampling points for the 1D plot. Defaults to 10000.
            image_size (int, optional): Size of the 2D plot in pixels. Defaults to 400.
            default_image (str, optional): Defaults to sample_images/sample_image.png
        """
        super().__init__()
        self.line_points: int = line_points
        self.image_size: int = image_size
        self.default_image: str = default_image
        
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
        self.defocus_diff_slider_image.set_value(0)
        self.defocus_az_slider_2d.set_value(0)
        self.defocus_az_slider_ice.set_value(0)
        self.defocus_az_slider_tomo.set_value(0)
        self.defocus_az_slider_image.set_value(0)
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
        self.size_scale_image.setValue(100)
        self.size_scale_fft.setValue(100)
        self.contrast_scale_image.setValue(100)
        self.contrast_scale_fft.setValue(99)
        self.contrast_sync_checkbox.setChecked(False)

    def _setup_initial_plots(self) -> None:
        """
        Configure the initial plots,
        including titles, limits, lines, and annotations.
        """
        self._setup_1d_plot()
        self._setup_2d_plot()
        self._setup_ice_plot()
        self._setup_tomo_plot()
        self._setup_image_plot()
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
            self.ctf_2d.dampened_ctf_2d(self.fx_fix, self.fy_fix), 
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
        self.canvas_ice.fig.suptitle("CTF Modulation by Sample Thickness", fontsize=18, fontweight='bold')
        self.canvas_ice.fig.subplots_adjust(hspace=0.2, top=0.93, bottom=0.08)
        # Ice Plots
        self.canvas_ice.axes[1].set_xlim(0, 0.5)
        self.canvas_ice.axes[1].tick_params(axis='both', which='major', labelsize=12)
        self.canvas_ice.axes[1].set_ylim(-1, 1)
        self.canvas_ice.axes[1].axhline(y=0, color='grey', linestyle='--', alpha=0.8)
        self.canvas_ice.axes[1].set_xlabel("Spatial Frequency (Å⁻¹)", fontsize=12)
        self.canvas_ice.axes[1].set_ylabel("Contrast Transfer Function", fontsize=12)
        self.canvas_ice.axes[2].set_xlabel("Spatial Frequency x (Å⁻¹)")
        self.canvas_ice.axes[2].set_ylabel("Spatial Frequency Y (Å⁻¹)")
        self.canvas_ice.axes[3].set_xlabel("Spatial Frequency X (Å⁻¹)")

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
            self.ctf_2d.dampened_ctf_2d(self.fx_fix, self.fy_fix), 
            extent=(-0.5, 0.5, -0.5, 0.5), 
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        self.ice_image = self.canvas_ice.axes[3].imshow(
            self.ctf_2d.dampened_ctf_ice(self.fx_fix, self.fy_fix), 
            extent=(-0.5, 0.5, -0.5, 0.5),
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        self.canvas_ice.fig.colorbar(
            self.ice_image_ref, 
            ax=self.canvas_ice.axes[2], 
            orientation='vertical',  
        )
        self.canvas_ice.fig.colorbar(
            self.ice_image, 
            ax=self.canvas_ice.axes[3], 
            orientation='vertical',  
        )

    def _setup_tomo_plot(self):
        self._resample_data_points()

        # Draw initial tomo plot 
        self.canvas_tomo.fig.suptitle("CTF Modulation by Sample Tilting", fontsize=18, fontweight='bold')
        self.canvas_tomo.fig.subplots_adjust(hspace=0.28, top=0.9, bottom=0.08)
          
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

        # Setup plot and axis labels  
        self.canvas_tomo.axes[3].set_title("CTF without Tilt")      
        self.canvas_tomo.axes[3].set_xlabel("Spatial Frequency X (Å⁻¹)")
        self.canvas_tomo.axes[3].set_ylabel("Spatial Frequency Y (Å⁻¹)")
        self.canvas_tomo.axes[4].set_title("CTF with Tilted Sample", fontsize=14, pad=10)
        self.canvas_tomo.axes[4].set_xlabel("Spatial Frequency X (Å⁻¹)")
        self.canvas_tomo.axes[4].set_ylabel("Spatial Frequency Y (Å⁻¹)")

        # CTF
        self.tomo_image_ref = self.canvas_tomo.axes[3].imshow(
            self.ctf_tomo_ref.dampened_ctf_ice(self.fx, self.fy), 
            extent=(-self.nyquist, self.nyquist, -self.nyquist, self.nyquist), 
            cmap='Greys', 
            vmin=-1, 
            vmax=1,
            origin = 'lower',
        )
        self.tomo_image = self.canvas_tomo.axes[4].imshow(
            self.ctf_tomo_tilt.dampened_ctf_ice(self.fx, self.fy), 
            extent=(-self.nyquist, self.nyquist, -self.nyquist, self.nyquist),
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

    def _setup_image_plot(self):
        self.canvas_image.fig.suptitle("Image Modulation by CTF", fontsize=18, fontweight='bold')
        self.canvas_image.fig.subplots_adjust(hspace=0.28, top=0.9, bottom=0.08)

        # Load default image
        self.image_data = self._load_and_prepare_image(self.default_image)
        self.image_contrast_inverted = False  # initialize instance variable 

        # FFT
        self.image_data_fft = np.fft.fftshift(np.fft.fft2(self.image_data))
        self.scaled_fft = np.abs(self.image_data_fft)

        # CTF
        ctf_matrix = self.ctf_2d.dampened_ctf_2d(self.fx, self.fy)
        self.scaled_convolved = np.abs(np.fft.ifft2(self.image_data_fft * ctf_matrix))

        # Setup axes
        self.image_original = self.canvas_image.axes[1].imshow(self.image_data, cmap='Greys')
        vmin, vmax = np.percentile(self.scaled_fft,
                                   [100 - self.contrast_scale_fft.value(), self.contrast_scale_fft.value()])
        self.image_fft = self.canvas_image.axes[2].imshow(self.scaled_fft, 
                                                          vmin=vmin,
                                                          vmax=vmax,
                                                          cmap='Greys', 
                                                          origin='lower')
        self.image_ctf_convolve = self.canvas_image.axes[4].imshow(ctf_matrix, cmap='Greys', vmin=-1, vmax=1, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5))
        self.image_convolved = self.canvas_image.axes[3].imshow(self.scaled_convolved, cmap='Greys')

        # Add titles and labels
        self.canvas_image.axes[1].set_title("Original Image")
        self.canvas_image.axes[2].set_title("Fast Fourier Transform")
        self.canvas_image.axes[3].set_title("Convolved Image")
        self.canvas_image.axes[4].set_title("Contrast Transfer Function")

        for i in [1, 2, 3, 4]:
            self.canvas_image.axes[i].set_xlabel("Pixel X" if i in [1, 3] else "Spatial Frequency X (Å⁻¹)")
            self.canvas_image.axes[i].set_ylabel("Pixel Y" if i in [1, 3] else "Spatial Frequency Y (Å⁻¹)")

        # Colorbars
        self.canvas_image.fig.colorbar(self.image_original, ax=self.canvas_image.axes[1])
        self.canvas_image.fig.colorbar(self.image_fft, ax=self.canvas_image.axes[2])
        self.canvas_image.fig.colorbar(self.image_convolved, ax=self.canvas_image.axes[3])
        self.canvas_image.fig.colorbar(self.image_ctf_convolve, ax=self.canvas_image.axes[4])

        self._update_ticks_for_fft()

    def _setup_annotations(self):
        # An instance variable to control the annotation
        self.show_annotation = False

        # Annotations for 1D CTF
        self.annotation_1d = self.canvas_1d.axes[1].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
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
            fontsize=10,
            zorder=10
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
            fontsize=10,
            zorder=10
        )
        self.annotation_ice_1d.set_visible(False)

        self.canvas_ice.axes[1].annotate(
            "1D",
            xy=(1, 0),
            xycoords="axes fraction",
            xytext=(-17, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        self.annotation_ice_ref = self.canvas_ice.axes[2].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_ice_ref.set_visible(False)

        self.annotation_ice_ctf = self.canvas_ice.axes[3].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_ice_ctf.set_visible(False)

        self.canvas_ice.axes[2].annotate(
            "without ice",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(3, -11),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        self.canvas_ice.axes[2].annotate(
            "2D",
            xy=(1, 0),
            xycoords="axes fraction",
            xytext=(-17, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        self.canvas_ice.axes[3].annotate(
            "with ice",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(3, -11),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        self.canvas_ice.axes[3].annotate(
            "2D",
            xy=(1, 0),
            xycoords="axes fraction",
            xytext=(-17, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10
        )

        # Annotations for Tomo tab
        self.annotation_tomo_diagram_state = self.canvas_tomo.axes[1].annotate(
            "",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(3, -33),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            fontsize=10,
            zorder=10
        )
        self.annotation_tomo_diagram_state.set_visible(False)

        self.annotation_tomo_tilt_ctf = self.canvas_tomo.axes[4].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_tomo_tilt_ctf.set_visible(False)

        self.annotation_tomo_ref_ctf = self.canvas_tomo.axes[3].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_tomo_ref_ctf.set_visible(False)

        # Annotations for the image tab
        self.annotation_image_original = self.canvas_image.axes[1].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_image_original.set_visible(False)

        self.annotation_image_fft = self.canvas_image.axes[2].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_image_fft.set_visible(False)

        self.annotation_image_convolved = self.canvas_image.axes[3].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_image_convolved.set_visible(False)

        self.annotation_image_ctf_convolve = self.canvas_image.axes[4].annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
            zorder=10
        )
        self.annotation_image_ctf_convolve.set_visible(False)

    def _initialize_data(self) -> None:
        """
        Create 1D and 2D frequency data for plotting the CTF.

        freqs_1d (NDArray): 1D frequency array from 0.001 to 1, for line_points samples.
        fx, fy (NDArray): 2D grids in the range [-0.5, 0.5], used for the 2D CTF.
        """
        self.freqs_1d = np.linspace(0.001, 1, self.line_points)
        freq_x = np.linspace(-0.5, 0.5, self.image_size)
        freq_y = np.linspace(-0.5, 0.5, self.image_size)
        self.fx_fix, self.fy_fix = np.meshgrid(freq_x, freq_y, sparse=True)

    def _resample_data_points(self) -> None:
        """Resample data points up to nyquist based on current pixel size
        """        
        self.nyquist = 0.5 / self.pixel_size_slider.get_value()

        freq_x = np.linspace(-self.nyquist, self.nyquist, self.image_size)
        freq_y = np.linspace(-self.nyquist, self.nyquist, self.image_size)
        self.fx, self.fy = np.meshgrid(freq_x, freq_y, sparse=True)
        self.pixel_size_changed = False

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
        self.defocus_diff_slider_image.valueChanged.connect(lambda value, key="df_diff": self.update_ctf(key, value))
        self.defocus_az_slider_2d.valueChanged.connect(lambda value, key="df_az": self.update_ctf(key, value))
        self.defocus_az_slider_ice.valueChanged.connect(lambda value, key="df_az": self.update_ctf(key, value))
        self.defocus_az_slider_image.valueChanged.connect(lambda value, key="df_az": self.update_ctf(key, value))
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
        self.canvas_image.mpl_connect("motion_notify_event", self.on_hover)
        self.ice_thickness_slider.valueChanged.connect(lambda value, key="ice": self.update_ctf(key, value))
        self.radio_button_group.buttonToggled.connect(self.update_wrap_func)
        self.tilt_slider_tomo.valueChanged.connect(lambda value, key="tilt_angle": self.update_tomo(key, value))
        self.sample_thickness_slider_tomo.valueChanged.connect(lambda value, key="thickness": self.update_tomo(key, value))
        self.defocus_diff_slider_tomo.valueChanged.connect(lambda value, key="df_diff": self.update_tomo(key, value))
        self.defocus_az_slider_tomo.valueChanged.connect(lambda value, key="df_az": self.update_tomo(key, value))
        self.upload_btn.clicked.connect(self._handle_upload_image)
        self.invert_btn.clicked.connect(self._invert_contrast)
        self.size_scale_image.valueChanged.connect(lambda value, key="image": self.zoom_2d_image(key, value))
        self.size_scale_fft.valueChanged.connect(lambda value, key="fft": self.zoom_2d_image(key, value))
        self.contrast_scale_image.valueChanged.connect(self.adjust_contrast_image)
        self.contrast_scale_fft.valueChanged.connect(self.adjust_contrast_fft)
        self.info_button_1d.clicked.connect(self.show_info)
        self.info_button_2d.clicked.connect(self.show_info)
        self.info_button_ice.clicked.connect(self.show_info)
        self.info_button_tomo.clicked.connect(self.show_info)
        self.info_button_image.clicked.connect(self.show_info)
        self.annotation_toggle_buttons = [
            self.toggle_button_1d,
            self.toggle_button_2d,
            self.toggle_button_ice,
            self.toggle_button_tomo,
            self.toggle_button_image,
        ]
        for button in self.annotation_toggle_buttons:
            button.clicked.connect(self._handle_annotation_toggle)
        self.contrast_sync_checkbox.stateChanged.connect(self.adjust_contrast_image)

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
            self.pixel_size_changed = True
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
            elif value == 4:  # Image tab
                self.defocus_diff_slider_image.set_value(self.ctf_2d.df_diff)
                self.defocus_az_slider_image.set_value(self.ctf_2d.df_az)

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
            self.image.set_data(self.wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx_fix, self.fy_fix)))
            self.canvas_2d.draw_idle()
        elif self.plot_tabs.currentIndex() == 2:
            # Update ice tab plots
            self.line_ice_ref[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.dampened_ctf_1d(self.freqs_1d)))
            self.line_ice[0].set_data(self.freqs_1d, self.wrap_func(self.ctf_1d.dampened_ctf_ice(self.freqs_1d)))
            self.canvas_ice.axes[1].set_xlim(0, self.xlim_slider_ice.get_value())
            self.ice_image_ref.set_data(self.wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx_fix, self.fy_fix)))
            self.ice_image.set_data(self.wrap_func(self.ctf_2d.dampened_ctf_ice(self.fx_fix, self.fy_fix)))
            self.canvas_ice.draw_idle()
        elif self.plot_tabs.currentIndex() == 3:
            # Update tomo tab plots
            if self.pixel_size_changed:
                self._resample_data_points()
                self.tomo_image_ref.set_extent((-self.nyquist, self.nyquist, -self.nyquist, self.nyquist))
                self.tomo_image.set_extent((-self.nyquist, self.nyquist, -self.nyquist, self.nyquist))
            self.tomo_image_ref.set_data(self.wrap_func(self.ctf_tomo_ref.dampened_ctf_ice(self.fx, self.fy)))  
            self.tomo_image.set_data(self.wrap_func(self.ctf_tomo_tilt.dampened_ctf_ice(self.fx, self.fy)))    
            self.canvas_tomo.draw_idle()
        elif self.plot_tabs.currentIndex() == 4:
            if self.pixel_size_changed:
                self._resample_data_points()
                self.image_ctf_convolve.set_extent((-self.nyquist, self.nyquist, -self.nyquist, self.nyquist))
                self._update_ticks_for_fft()

            # update ctf
            ctf_matrix = self.ctf_2d.dampened_ctf_2d(self.fx, self.fy)
            self.image_ctf_convolve.set_data(self.wrap_func(ctf_matrix))

            # update convolved image
            image_data_convolved = np.fft.ifft2(self.image_data_fft * ctf_matrix)
            self.scaled_convolved = np.abs(image_data_convolved)
            self.image_convolved.set_data(self.scaled_convolved)
            self.adjust_contrast_image()

            self.canvas_image.draw_idle()

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

    def zoom_2d_image(self, key: str, value: float | int ) -> None:
        """
        Zoom on 2D image by updating axes limits.

        Args:
            key (str): The name of the image being updated. 
            value (float | int): The new value for the zoom factor.
        """
        center = self.image_size / 2

        # Compute zoom factor
        new_size = self.image_size * 100 / value

        new_extent = [
            center - new_size / 2,
            center + new_size / 2,
        ]

        # Update axes limits
        if key == "image":
            self.canvas_image.axes[1].set_xlim(new_extent[0], new_extent[1])
            self.canvas_image.axes[1].set_ylim(new_extent[1], new_extent[0])
            self.canvas_image.axes[3].set_xlim(new_extent[0], new_extent[1])
            self.canvas_image.axes[3].set_ylim(new_extent[1], new_extent[0])
        elif key == "fft":
            self.canvas_image.axes[2].set_xlim(new_extent[0], new_extent[1])
            self.canvas_image.axes[2].set_ylim(new_extent[0], new_extent[1])

        self.canvas_image.draw_idle()

    def adjust_contrast_image(self) -> None:
        """
        Adjust the contrast of both original and convolved images.
        """
        value = self.contrast_scale_image.value()
        range = [100-value, value]

        if self.contrast_sync_checkbox.isChecked():           
            vmin = min(np.percentile(self.image_data, 100 - value), np.percentile(self.scaled_convolved, 100 - value)) 
            vmax = max(np.percentile(self.image_data, value), np.percentile(self.scaled_convolved, value))
            self.image_original.set_clim(vmin=vmin, vmax=vmax)
            self.image_convolved.set_clim(vmin=vmin, vmax=vmax)
        else:
            self.image_original.set_clim(*np.percentile(self.image_data, range))
            self.image_convolved.set_clim(*np.percentile(self.scaled_convolved, range))

        self.canvas_image.draw_idle()
        
    def adjust_contrast_fft(self, value: float | int) -> None:
        """
        Adjust the contrast of fft image.

        Args:
            value (float | int): The new contrast percentile. 
        """
        vmin, vmax = np.percentile(self.scaled_fft, [100-value, value])
        self.image_fft.set_clim(vmin=vmin, vmax=vmax)

        self.canvas_image.draw_idle()

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
        self.image_ctf_convolve.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))

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
        elif self.plot_tabs.currentIndex() == 4:
            self.canvas_image.fig.savefig(file_path, dpi=300)

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
            x_full, y_full = self._get_full_meshgrid(self.fx_fix, self.fy_fix)
            df = self._generate_ctf_2d_data(wrap_func, x_full, y_full, ice=False)
            df.to_csv(file_path, index=False)

        elif tab_index == 2:  # Ice CTF
            x_full, y_full = self._get_full_meshgrid(self.fx_fix, self.fy_fix)
            with open(file_path, "w") as f:
                f.write("# 1D CTF\n")
                self._generate_ctf_1d_data(wrap_func).to_csv(f, index=False)

                f.write("\n# 2D CTF\n")
                self._generate_ctf_2d_data(wrap_func, x_full, y_full, ice=True).to_csv(f, index=False)
        elif tab_index == 3:  # Tomo CTF
            x_full, y_full =self._get_full_meshgrid(self.fx, self.fy)
            df = self._generate_ctf_tomo_data(wrap_func, x_full, y_full)
            df.to_csv(file_path, index=False)
        elif tab_index == 4:  # Image CTF
            x_full, y_full =self._get_full_meshgrid(self.fx, self.fy)
            df = self._generate_ctf_image_data(wrap_func, x_full, y_full)
            df.to_csv(file_path, index=False)

    def _generate_ctf_1d_data(self, wrap_func) -> pd.DataFrame:
        """Generates 1D CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "freqs_1d": self.freqs_1d,
            "ctf": wrap_func(self.ctf_1d.ctf_1d(self.freqs_1d)),
            "ctf_dampened": wrap_func(self.ctf_1d.dampened_ctf_1d(self.freqs_1d)),
            "applied_temporal_env": wrap_func(self.ctf_1d.Et(self.freqs_1d)),
            "applied_spatial_env": wrap_func(self.ctf_1d.Es_1d(self.freqs_1d)),
            "applied_detector_env": wrap_func(self.ctf_1d.Ed(self.freqs_1d)),
            "applied_total_env": wrap_func(self.ctf_1d.Etotal_1d(self.freqs_1d)),
        })

    def _generate_ctf_2d_data(self, wrap_func, x_full, y_full, ice: bool) -> pd.DataFrame:
        """Generates 2D CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "freqs_x": x_full.flatten(),
            "freqs_y": y_full.flatten(),
            "ctf_no_ice": wrap_func(self.ctf_2d.ctf_2d(self.fx_fix, self.fy_fix)).flatten(),
            "ctf_no_ice_dampened": wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx_fix, self.fy_fix)).flatten(),
            "ctf_ice": wrap_func(self.ctf_2d.ctf_ice(self.fx_fix, self.fy_fix)).flatten() if ice else None,
            "ctf_ice_dampened": wrap_func(self.ctf_2d.dampened_ctf_ice(self.fx_fix, self.fy_fix)).flatten() if ice else None,
            "applied_total_env": wrap_func(self.ctf_2d.Etotal_2d(self.fx_fix, self.fy_fix)).flatten(),
        })
    
    def _generate_ctf_tomo_data(self, wrap_func, x_full, y_full) -> pd.DataFrame:
        """Generates tomo CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "freqs_x": x_full.flatten(),
            "freqs_y": y_full.flatten(),
            "ctf_no_tilt": wrap_func(self.ctf_tomo_ref.ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_no_tilt_dampened": wrap_func(self.ctf_tomo_ref.dampened_ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_tilt": wrap_func(self.ctf_tomo_tilt.ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_tilt_dampened": wrap_func(self.ctf_tomo_tilt.dampened_ctf_ice(self.fx, self.fy)).flatten(),
            "applied_total_env": wrap_func(self.ctf_tomo_ref.Etotal_2d(self.fx, self.fy)).flatten(),
        })
    
    def _generate_ctf_image_data(self, wrap_func, x_full, y_full) -> pd.DataFrame:
        """Generates image CTF data as a pandas DataFrame."""
        return pd.DataFrame({
            "original_image": self.image_data.flatten(),
            "image_fft": self.image_data_fft.flatten(),
            "image_fft_amplitude": self.scaled_fft.flatten(),
            "freqs_x": x_full.flatten(),
            "freqs_y": y_full.flatten(),
            "ctf": wrap_func(self.ctf_2d.ctf_ice(self.fx, self.fy)).flatten(),
            "ctf_dampened": wrap_func(self.ctf_2d.dampened_ctf_2d(self.fx, self.fy)).flatten(),
            "applied_total_env": wrap_func(self.ctf_2d.Etotal_2d(self.fx_fix, self.fy_fix)).flatten(),
            "convolved_image": self.scaled_convolved.flatten(),
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
        if self.show_annotation:
            wrap_func = self._setup_ctf_wrap_func()

            if event.inaxes == self.canvas_1d.axes[1]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / x
                    value = wrap_func(self.ctf_1d.dampened_ctf_1d(np.array([x])))[0]
                    self.annotation_1d.xy = (x, y)
                    self.annotation_1d.set_text(f"x: {x:.3f} Å⁻¹\ny: {y:.3f}\nres: {res:.2f} Å\nctf: {value:.4f}")
                    self.annotation_1d.set_visible(True)
                    self.canvas_1d.draw_idle()
            elif event.inaxes == self.canvas_2d.axes[1]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / math.sqrt(x**2 + y**2)
                    # angle = math.degrees(math.atan2(y, x))
                    self.annotation_2d.xy = (x, y)
                    value = wrap_func(self.ctf_2d.dampened_ctf_2d(np.array([x]), np.array([y])))[0]
                    self.annotation_2d.set_text(f"x: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {value:.4f}")
                    self.annotation_2d.set_visible(True)
                    self.canvas_2d.draw_idle()
            elif event.inaxes == self.canvas_ice.axes[1]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / x
                    value_no_ice = wrap_func(self.ctf_1d.dampened_ctf_1d(np.array([x])))[0]
                    value_ice = wrap_func(self.ctf_1d.dampened_ctf_ice(np.array([x])))[0]
                    self.annotation_ice_1d.xy = (x, y)
                    self.annotation_ice_1d.set_text(f"x: {x:.3f} Å⁻¹\nres: {res:.2f} Å\ngray: {value_no_ice:.4f}\npurple: {value_ice:.4f}")
                    self.annotation_ice_1d.set_visible(True)
                    self.canvas_ice.draw_idle()
            elif event.inaxes == self.canvas_ice.axes[2]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / math.sqrt(x**2 + y**2)
                    self.annotation_ice_ref.xy = (x, y)
                    value = wrap_func(self.ctf_2d.dampened_ctf_2d(np.array([x]), np.array([y])))[0]
                    self.annotation_ice_ref.set_text(f"x: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {value:.4f}")
                    self.annotation_ice_ref.set_visible(True)
                    self.canvas_ice.draw_idle()
            elif event.inaxes == self.canvas_ice.axes[3]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / math.sqrt(x**2 + y**2)
                    self.annotation_ice_ctf.xy = (x, y)
                    value = wrap_func(self.ctf_2d.dampened_ctf_ice(np.array([x]), np.array([y])))[0]
                    self.annotation_ice_ctf.set_text(f"x: {x:.3f} Å⁻¹\n"
                                                    f"y: {y:.3f} Å⁻¹\n"
                                                    f"res: {res:.2f} Å\n"
                                                    f"ctf: {value:.4f}")
                    self.annotation_ice_ctf.set_visible(True)
                    self.canvas_ice.draw_idle()
            elif event.inaxes == self.canvas_tomo.axes[1]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
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
                    value = wrap_func(self.ctf_tomo_ref.dampened_ctf_ice(np.array([x]), np.array([y])))[0]
                    self.annotation_tomo_ref_ctf.set_text(f"tilt angle: 0°\n"
                                                        f"x: {x:.3f} Å⁻¹\n"
                                                        f"y: {y:.3f} Å⁻¹\n"
                                                        f"res: {res:.2f} Å\n"
                                                        f"ctf: {value:.4f}")
                    self.annotation_tomo_ref_ctf.set_visible(True)
                    self.canvas_tomo.draw_idle()
            elif event.inaxes == self.canvas_tomo.axes[4]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / math.sqrt(x**2 + y**2)
                    # angle = math.degrees(math.atan2(y, x))
                    self.annotation_tomo_tilt_ctf.xy = (x, y)
                    value = wrap_func(self.ctf_tomo_tilt.dampened_ctf_ice(np.array([x]), np.array([y])))[0]
                    self.annotation_tomo_tilt_ctf.set_text(f"tilt angle: {self.tilt_slider_tomo.get_value():.1f}°\nx: {x:.3f} Å⁻¹\ny: {y:.3f} Å⁻¹\nres: {res:.2f} Å\nctf: {value:.4f}")
                    self.annotation_tomo_tilt_ctf.set_visible(True)
                    self.canvas_tomo.draw_idle()
            elif event.inaxes == self.canvas_image.axes[1]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    self.annotation_image_original.xy = (x, y)
                    value = self.image_data[int(y), int(x)]
                    self.annotation_image_original.set_text(f"{value:.2f}")
                    self.annotation_image_original.set_visible(True)
                    self.canvas_image.draw_idle()
            elif event.inaxes == self.canvas_image.axes[2]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    x_freq = (2 * x / self.image_size - 1) * self.nyquist
                    y_freq = (2 * y / self.image_size - 1) * self.nyquist
                    res = 1. / math.sqrt(x_freq**2 + y_freq**2)
                    self.annotation_image_fft.xy = (x, y)
                    value = self.scaled_fft[int(y), int(x)]
                    self.annotation_image_fft.set_text(f"res: {res:.2f} Å\n"
                                                    f"amp: {value:.2f}")
                    self.annotation_image_fft.set_visible(True)
                    self.canvas_image.draw_idle()
            elif event.inaxes == self.canvas_image.axes[3]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    self.annotation_image_convolved.xy = (x, y)
                    value = self.scaled_convolved[int(y), int(x)]
                    self.annotation_image_convolved.set_text(f"{value:.2f}")
                    self.annotation_image_convolved.set_visible(True)
                    self.canvas_image.draw_idle()
            elif event.inaxes == self.canvas_image.axes[4]:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    res = 1. / math.sqrt(x**2 + y**2)
                    self.annotation_image_ctf_convolve.xy = (x, y)
                    value = wrap_func(self.ctf_2d.dampened_ctf_2d(np.array([x]), np.array([y])))[0]
                    self.annotation_image_ctf_convolve.set_text(f"res: {res:.2f} Å\n"
                                                                f"ctf: {value:.4f}")
                    self.annotation_image_ctf_convolve.set_visible(True)
                    self.canvas_image.draw_idle()
            else:
                self.annotation_1d.set_visible(False)
                self.annotation_2d.set_visible(False)
                self.annotation_ice_1d.set_visible(False)
                self.annotation_ice_ref.set_visible(False)
                self.annotation_ice_ctf.set_visible(False)
                self.annotation_tomo_diagram_state.set_visible(False)
                self.annotation_tomo_ref_ctf.set_visible(False)
                self.annotation_tomo_tilt_ctf.set_visible(False)
                self.annotation_image_original.set_visible(False)
                self.annotation_image_fft.set_visible(False)
                self.annotation_image_convolved.set_visible(False)
                self.annotation_image_ctf_convolve.set_visible(False)
                self.canvas_1d.draw_idle()
                self.canvas_2d.draw_idle()
                self.canvas_ice.draw_idle()
                self.canvas_tomo.draw_idle()
                self.canvas_image.draw_idle()
        else:
            pass

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

    def _handle_upload_image(self) -> None:
        """
        Process user supplied image and update the image tab. Supports common formats (PNG, JPG, TIFF, etc.).
        """
        supported_formats = "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", supported_formats)

        if not file_path:
            return

        try:
            self.image_data = self._load_and_prepare_image(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Image Load Error", f"Failed to load image:\n{str(e)}")
            return

        # Reset sliders
        self.size_scale_image.setValue(100)
        self.size_scale_fft.setValue(100)
        self.contrast_scale_image.setValue(100)
        self.contrast_scale_fft.setValue(99)

        self.image_data_fft = np.fft.fftshift(np.fft.fft2(self.image_data))
        self.scaled_fft = np.abs(self.image_data_fft)

        vmin, vmax = np.percentile(self.scaled_fft, 
                                [100 - self.contrast_scale_fft.value(), self.contrast_scale_fft.value()])

        self.image_original.set_data(self.image_data)
        self.image_fft.set_data(self.scaled_fft)
        self.image_fft.set_clim(vmin=vmin, vmax=vmax)

        self._update_ticks_for_fft()
        self.update_plot()

    def _update_ticks_for_fft(self) -> None:
        # Get tick positions and labels from the CTF plot
        xticks = self.canvas_image.axes[4].get_xticks()
        xticklabels = [label.get_text() for label in self.canvas_image.axes[4].get_xticklabels()]
        yticks = self.canvas_image.axes[4].get_yticks()
        yticklabels = [label.get_text() for label in self.canvas_image.axes[4].get_yticklabels()]
        # Map tick locations from the CTF plot
        xpos = [(tick + self.nyquist) / (2 * self.nyquist) * self.image_size for tick in xticks]
        ypos = [(tick + self.nyquist) / (2 * self.nyquist) * self.image_size for tick in yticks]
        # Update ticks in FFT plot
        self.canvas_image.axes[2].xaxis.set_major_locator(FixedLocator(xpos))
        self.canvas_image.axes[2].set_xticklabels(xticklabels)
        self.canvas_image.axes[2].yaxis.set_major_locator(FixedLocator(ypos))
        self.canvas_image.axes[2].set_yticklabels(yticklabels)

    def _invert_contrast(self) -> None:
        # Toggle the contrast state
        self.image_contrast_inverted = not self.image_contrast_inverted
        
        # Flip colormap accordingly
        if self.image_contrast_inverted:
            self.image_original.set_cmap('Greys_r')  # Inverted colormap
            self.image_convolved.set_cmap('Greys_r')
        else:
            self.image_original.set_cmap('Greys')    # Normal colormap
            self.image_convolved.set_cmap('Greys')

        self.canvas_image.draw_idle()

    def _handle_annotation_toggle(self, checked) -> None:
        self.show_annotation = checked  # Sync instance variable

        # Sync all buttons to this state
        sender = self.sender()
        for btn in self.annotation_toggle_buttons:
            if btn is not sender:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)

    def _load_and_prepare_image(self, path: str) -> np.ndarray:
        """
        Load an image, convert to grayscale, crop to square, resize to self.image_size, and normalize.

        Args:
            path (str): Path to the image file.

        Returns:
            np.ndarray: A (self.image_size x self.image_size) normalized grayscale image.
        """
        image = Image.open(path).convert("L")

        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        upper = (height - crop_size) // 2
        right = left + crop_size
        lower = upper + crop_size
        image = image.crop((left, upper, right, lower))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
    
        return np.array(image) / 255.0

    def show_info(self) -> None:
        """Show additonal info for the specific tab
        """
        if self.plot_tabs.currentIndex() == 0:
            QMessageBox.information(self, "", (
                    "<span style='font-weight:normal;'>"
                    "<b>Notes:</b><br><br>"
                    "1) The controls on the left panel affect CTF calculations across all tabs.<br>"
                    "2) Tab-specific controls are located at the bottom of each tab.<br>"
                    "3) Tabs '2D', 'Thickness', 'Image' share the same 2-D CTF model. <br>"
                    "3) The detector envelope models only the attenuation of the CTF at higher spatial frequencies due to the detector's DQE. "
                    "The actual impact of different detectors on image signal-to-noise ratio, based on varying DQE characteristics, is not simulated. "
                    "DQEs are modeled as polynomial functions of spatial frequency. <br><br>"
                    "<b>Equations used:</b>"
                    "<pre style='font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;'>"
                    "1. λ = h / √[ 2 * mₑ * eV * (1 + eV / (2 * mₑ * c²)) ]<br>"
                    "2. γ(f) = -(π / 2) · C<sub>s</sub> · λ³ · f⁴ + π · d<sub>f</sub> · λ · f² + p<br>"
                    "3. f<sub>s</sub> = C<sub>c</sub> · √((ΔV / V)² + (2ΔI / I)² + (ΔE / eV)²)<br>"
                    "4. E<sub>T</sub>(f) = exp(–(π² λ² f<sub>s</sub>² f⁴) / 2)<br>"
                    "5. E<sub>S</sub>(f) = exp(–( (π · eₐ / λ)² · (C<sub>s</sub> · λ³ · f³ + d<sub>f</sub> · λ · f)² ))<br>"
                    "6. E<sub>D</sub>(f) = DQE(f / Nyquist) / max(DQE)<br>"
                    "7. E<sub>total</sub>(f) = E<sub>T</sub>(f) · E<sub>S</sub>(f) · E<sub>D</sub>(f)<br>"
                    "8. CTF(f) = –sin(γ(f) + arcsin(A<sub>c</sub>)) · E<sub>total</sub>(f)"
                    "</pre>"
                    "<b>Tips: </b><br><br>"
                    "1) Scroll on the spin boxes to adjust values quickly.<br>"
                    "2) You can also type values directly into the input fields.<br>"
                    "3) Click the 'v' button to display or hide values on the plot.<br>"
                    "4) Save the entire tab as a PNG image using the save plot button.<br>"
                    "5) Export all simulation data to a CSV file for further analysis."
                    "</span>"
                ))
        elif self.plot_tabs.currentIndex() == 1:
            QMessageBox.information(self, "", (
                    "<span style='font-weight:normal;'>"
                    "<b>Equations used:</b>"
                    "<pre style='font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;'>"
                    "1. φ = arctan(f<sub>y</sub> / f<sub>x</sub>)<br>"
                    "2. d<sub>f</sub> = 0.5 · [ d<sub>u</sub> + d<sub>v</sub> + (d<sub>u</sub> – d<sub>v</sub>) · cos(2 · (φ – φ<sub>a</sub>)) ]<br>"
                    "3. γ(f) = -(π / 2) · C<sub>s</sub> · λ³ · f⁴ + π · d<sub>f</sub> · λ · f² + p<br>"
                    "4. CTF(f) = –sin(γ(f) + arcsin(A<sub>c</sub>)) · E<sub>total</sub>(f)"
                    "</pre>"
                    "<b>Tips: </b><br><br>"
                    "1) Scroll on the spin boxes to adjust values quickly.<br>"
                    "2) You can also type values directly into the input fields.<br>"
                    "3) Click the 'v' button to display or hide values on the plot.<br>"
                    "4) Save the entire tab as a PNG image using the save plot button.<br>"
                    "5) Export all simulation data to a CSV file for further analysis."
                    "</span>"
                ))
        elif self.plot_tabs.currentIndex() == 2:
            QMessageBox.information(self, "", (
                    "<span style='font-weight:normal;'>"
                    "<b>Notes:</b><br><br>"
                    "The equation below is derived by integrating the CTF over a range of defocus values to "
                    "account for the effect of sample thickness. "
                    "It can be interpreted as the classical CTF function modulated by a sinc function, "
                    "and it applies to both 1D and 2D CTF cases.<br><br>"
                    "<b>Equation used:</b>"
                    "<pre style='font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;'>"
                    "CTF(f) = -2 / (π · λ · f²) · sin(π · λ · f² · thickness / 2) · sin(γ(f) + arcsin(A<sub>c</sub>)) / thickness"
                    "</pre>"
                    "<b>Tips: </b><br><br>"
                    "1) Scroll on the spin boxes to adjust values quickly.<br>"
                    "2) You can also type values directly into the input fields.<br>"
                    "3) Click the 'v' button to display or hide values on the plot.<br>"
                    "4) Save the entire tab as a PNG image using the save plot button.<br>"
                    "5) Export all simulation data to a CSV file for further analysis."
                    "</span>"
                ))
        elif self.plot_tabs.currentIndex() == 3:
            QMessageBox.information(self, "", (
                    "<span style='font-weight:normal;'>"
                    "<b>This simulation assumes:</b><br><br>"
                    "1) The electron beam remains parallel to the optical axis.<br>"
                    "2) Sample tilting does not introduce astigmatism.<br>"
                    "3) The CTF is affected solely by the apparent sample thickness.<br><br>"
                    "<b>Equation used:</b>"
                    "<pre style='font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;'>"
                    "CTF(f) = -2 / (π · λ · f²) · sin(π · λ · f² · thickness / 2) · sin(γ(f) + arcsin(A<sub>c</sub>)) / thickness"
                    "</pre>"
                    "<b>Tips: </b><br><br>"
                    "1) Scroll on the spin boxes to adjust values quickly.<br>"
                    "2) You can also type values directly into the input fields.<br>"
                    "3) Click the 'v' button to display or hide values on the plot.<br>"
                    "4) Save the entire tab as a PNG image using the save plot button.<br>"
                    "5) Export all simulation data to a CSV file for further analysis."
                    "</span>"
                ))
        elif self.plot_tabs.currentIndex() == 4:
            QMessageBox.information(self, "", (
                    "<span style='font-weight:normal;'>"
                    "<b>Notes:</b><br><br>"
                    "1) Uploaded images are cropped and resized to the default image size.<br>"
                    "2) The FFT and convolved images display the amplitude values.<br>"
                    "3) Enable 'Sync Greyscale' to better visualize signal attenuation.<br>"
                    "4) A collection of sample images is provided for educational purposes.<br>"
                    "5) For the most accurate simulation of CTF effects, use a pixel size close to the actual imaging conditions. "
                    "This may not be applicable or achievable for a random image.<br><br>"
                    "<b>Equations used:</b>"
                    "<pre style='font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;'>"
                    "1. φ = arctan(f<sub>y</sub> / f<sub>x</sub>)<br>"
                    "2. d<sub>f</sub> = 0.5 · [ d<sub>u</sub> + d<sub>v</sub> + (d<sub>u</sub> – d<sub>v</sub>) · cos(2 · (φ – φ<sub>a</sub>)) ]<br>"
                    "3. γ(f) = -(π / 2) · C<sub>s</sub> · λ³ · f⁴ + π · d<sub>f</sub> · λ · f² + p<br>"
                    "4. CTF(f) = –sin(γ(f) + arcsin(A<sub>c</sub>)) · E<sub>total</sub>(f)<br>"
                    "5. Colvolved_image = FFT<sup>-1</sup>(FFT(Image) · CTF)"
                    "</pre>"
                    "<b>Tips: </b><br><br>"
                    "1) Scroll on the spin boxes to adjust values quickly.<br>"
                    "2) You can also type values directly into the input fields.<br>"
                    "3) Click the 'v' button to display or hide values on the plot.<br>"
                    "4) Save the entire tab as a PNG image using the save plot button.<br>"
                    "5) Export all simulation data to a CSV file for further analysis."
                    "</span>"
                ))


if __name__ == "__main__":
    app: QApplication = QApplication([])
    gui: AppController = AppController()
    gui.show()
    app.exec_()
