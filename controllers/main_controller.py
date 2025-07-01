import os
import numpy as np
from matplotlib.colors import Normalize
from PyQt5.QtWidgets import QMessageBox

from models.ctf_1d import CTF1D
from models.ctf_2d import CTF2D
from models.detector import ALL_DETECTORS
from views.main_window import MainWindow
from utils.file_io import prompt_and_save_figure, prompt_and_save_csv
from utils.ctf_parameter_utils import apply_ctf_parameter, get_ctf_wrap_func
from controllers.plot_setup import (
    setup_1d_plot,
    setup_2d_plot,
    setup_ice_plot,
    setup_tomo_plot,
    setup_image_plot,
    setup_annotations,
)
from controllers.plot_updates import (
    update_plot_range,
    update_plot,
    update_grayness,
)
from controllers.gui_handelers import setup_event_handlers
from controllers.annotation_handlers import (
    annotate_1d,
    annotate_2d,
    annotate_ice_1d,
    annotate_ice_2d_noice,
    annotate_ice_2d_withice,
    annotate_tomo_diagram,
    annotate_tomo_ref_ctf,
    annotate_tomo_tilt_ctf,
    annotate_image_original,
    annotate_image_fft,
    annotate_image_convolved,
    annotate_image_ctf,
    hide_all_annotations,
)
from controllers import defaults


class AppController:
    """
    A controller class that extends the `CTFSimGUI` main window to manage
    the Contrast Transfer Function (CTF) simulation in both 1D and 2D modes.

    This class sets up frequency data, initializes CTF models, configures default GUI values,
    and handles user interactions (e.g., slider changes, resets, tab switching, saving plots).
    """

    def __init__(
        self,
        line_points: int = 10000,
        image_size: int = 400,
        default_image: str = "sample_images/sample_image.png",
    ) -> None:
        """
        Initialize the AppController by creating CTF models, setting up the GUI,
        and establishing event handlers.

        Args:
            line_points (int, optional): Number of sampling points for the 1D plot. Defaults to 10000.
            image_size (int, optional): Size of the 2D plot in pixels. Defaults to 400.
            default_image (str, optional): Defaults to sample_images/sample_image.png
        """
        # ─── Predeclare all attributes that plot_setup.py will create ───
        self.line_y0 = None  # type: matplotlib.lines.Line2D
        self.line_et = None
        self.line_es = None
        self.line_ed = None
        self.line_te = None
        self.line_dc = None
        self.legend_1d = None

        self.image_2d = None

        self.line_ice_ref = None
        self.line_ice = None
        self.ice_image_ref = None
        self.ice_image = None

        self.beam = None
        self.sample_rect = None
        self.tomo_image_ref = None
        self.tomo_image = None

        self.image_data = None  # will hold NumPy array
        self.image_original = None
        self.image_fft = None
        self.image_ctf_convolve = None
        self.image_convolved = None
        self.image_contrast_inverted = False  # bool

        self.pixel_size_changed = False  # pixel_size flag
        self.show_annotation = False  # annotation flag
        self.width_tomo = None  # will hold number
        self.height_tomo = None  # will hold number

        # Initialize attributes
        self.window: MainWindow = MainWindow()
        self.ui = self.window.ui
        self.line_points: int = line_points
        self.image_size: int = image_size
        self.default_image: str = default_image

        # Initialize models
        self.ctf_1d: CTF1D = CTF1D(detector_model=ALL_DETECTORS[0])
        self.ctf_2d: CTF2D = CTF2D(detector_model=ALL_DETECTORS[0])
        self.ctf_1d_ice: CTF1D = CTF1D(
            detector_model=ALL_DETECTORS[0], include_ice=True, ice_thickness=50
        )
        self.ctf_2d_ice: CTF2D = CTF2D(
            detector_model=ALL_DETECTORS[0], include_ice=True, ice_thickness=50
        )
        self.ctf_tomo_ref: CTF2D = CTF2D(
            detector_model=ALL_DETECTORS[0], include_ice=True, ice_thickness=50
        )
        self.ctf_tomo_tilt: CTF2D = CTF2D(
            detector_model=ALL_DETECTORS[0], include_ice=True, ice_thickness=50
        )

        # Initialize GUI
        self.setup_default_gui_values()

        # Precompute base frequency data:
        self.freqs_1d = np.linspace(0.001, 1, self.line_points)
        self.fx_fix, self.fy_fix = np.meshgrid(
            np.linspace(-0.5, 0.5, self.image_size),
            np.linspace(-0.5, 0.5, self.image_size),
            sparse=True,
        )

        # Initial plot setup
        setup_1d_plot(self)
        setup_2d_plot(self)
        setup_ice_plot(self)
        setup_tomo_plot(self)
        setup_image_plot(self)
        setup_annotations(self)

        # Initialize wrap_func
        self.wrap_func = lambda x: x

        # Hook up signals -> slots
        self.annotation_toggle_buttons = [
            self.ui.toggle_button_1d,
            self.ui.toggle_button_2d,
            self.ui.toggle_button_ice,
            self.ui.toggle_button_tomo,
            self.ui.toggle_button_image,
        ]
        setup_event_handlers(self)

    def setup_default_gui_values(self) -> None:
        """
        Set default values for all GUI sliders, combo boxes, and checkboxes.
        """
        for widget_name, method_name, default_value in defaults.DEFAULT_WIDGET_VALUES:
            widget = getattr(self.ui, widget_name)
            getattr(widget, method_name)(default_value)

        # Handle the “plot_1d_y_min” special case (based on radio button state)
        self.ui.plot_1d_y_min.setValue(self._setup_default_ylim()[0])

    def on_ctf_param_changed(self, key, value):
        """
        Called whenever ANY CTF parameter slider/spinbox changes.
        1) Update the underlying models via apply_ctf_parameter()
        2) If pixel_size changed, set pixel_size_changed flag
        3) Let the plot‐updater functions redraw the current tab
        """
        # update all model attributes
        pixel_flag = apply_ctf_parameter(
            key,
            value,
            [
                self.ctf_1d,
                self.ctf_2d,
                self.ctf_1d_ice,
                self.ctf_2d_ice,
                self.ctf_tomo_ref,
                self.ctf_tomo_tilt,
            ],
        )
        if pixel_flag:
            self.pixel_size_changed = True

        # call the appropriate update function
        update_plot(self, self.ui.plot_tabs.currentIndex())

    def on_tab_switched(self, new_index):
        """
        Called when the user clicks to a different tab.
        Reset any tab‐specific sliders to match the underlying model values.
        """
        update_plot_range(self)
        update_grayness(self)

        # Sync defocus sliders for 2D/ice/image tabs
        if new_index == 1:  # 2D tab
            self.ui.defocus_diff_slider_2d.set_value(self.ctf_2d.defocus_diff)
            self.ui.defocus_az_slider_2d.set_value(self.ctf_2d.defocus_az)
        elif new_index == 2:  # Ice tab
            self.ui.defocus_diff_slider_ice.set_value(self.ctf_2d.defocus_diff)
            self.ui.defocus_az_slider_ice.set_value(self.ctf_2d.defocus_az)
        elif new_index == 4:  # Image tab
            self.ui.defocus_diff_slider_image.set_value(self.ctf_2d.defocus_diff)
            self.ui.defocus_az_slider_image.set_value(self.ctf_2d.defocus_az)

        # Redraw the newly‐selected tab
        update_plot(self, new_index)

    def update_wrap_func(self) -> None:
        """
        Update wraper function without setting data.
        """
        self.wrap_func = get_ctf_wrap_func(
            self.ui.radio_button_group,
            self.ui.radio_ctf,
            self.ui.radio_abs_ctf,
            self.ui.radio_ctf_squared,
        )

        ylim = self._setup_default_ylim()

        # Rescale 1D plots
        self.ui.canvas_1d.axes[1].set_ylim(*ylim)
        self.ui.plot_1d_y_min.setValue(ylim[0])
        self.ui.plot_1d_y_max.setValue(ylim[1])
        self.ui.canvas_ice.axes[1].set_ylim(*ylim)

        # Renormalize the 2D images
        self.image_2d.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.ice_image_ref.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.ice_image.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.tomo_image_ref.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.tomo_image.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))
        self.image_ctf_convolve.set_norm(Normalize(vmin=ylim[0], vmax=ylim[1]))

        # Reset gray scales
        self.ui.gray_scale_2d.setValue(1)
        self.ui.gray_scale_ice.setValue(1)
        self.ui.gray_scale_tomo.setValue(1)

        # Reset spatial frequency
        self.ui.freq_scale_2d.setValue(0.5)
        self.ui.freq_scale_ice.setValue(0.5)

        update_plot(self, self.ui.plot_tabs.currentIndex())

    def _setup_default_ylim(self):
        """
        Return the (ymin, ymax) pair based on which wrap‐function (CTF, |CTF|, or CTF²) is checked.
        Called by self.update_wrap_func().
        """
        checked = self.ui.radio_button_group.checkedButton()
        if checked == self.ui.radio_ctf:
            return (-1, 1)
        elif checked == self.ui.radio_abs_ctf:
            return (0, 1)
        elif checked == self.ui.radio_ctf_squared:
            return (0, 1)
        else:
            # fallback
            return (-1, 1)

    def reset_parameters(self) -> None:
        """
        Restore default GUI values and re-compute the CTF plots.
        """
        self.setup_default_gui_values()
        update_plot_range(self)
        update_plot(self, self.ui.plot_tabs.currentIndex())

    def save_plot(self) -> None:
        """Opens a file dialog and saves the plots in the current tab as an image."""
        current_fig = {
            0: self.ui.canvas_1d.fig,
            1: self.ui.canvas_2d.fig,
            2: self.ui.canvas_ice.fig,
            3: self.ui.canvas_tomo.fig,
            4: self.ui.canvas_image.fig,
        }[self.ui.plot_tabs.currentIndex()]
        prompt_and_save_figure(self, current_fig)

    def save_csv(self) -> None:
        """Opens a file dialog and saves the plotted data in the current tab as a CSV file."""
        prompt_and_save_csv(self)

    def on_hover(self, event) -> None:
        """
        Display coordinates and/or values on hover over the 1D or 2D plot.

        Args:
            event: A Matplotlib MouseEvent with xdata, ydata, and inaxes.
        """
        if not self.show_annotation:
            return

        ax = event.inaxes
        if ax == self.ui.canvas_1d.axes[1]:
            annotate_1d(self, event)
        elif ax == self.ui.canvas_2d.axes[1]:
            annotate_2d(self, event)
        elif ax == self.ui.canvas_ice.axes[1]:
            annotate_ice_1d(self, event)
        elif ax == self.ui.canvas_ice.axes[2]:
            annotate_ice_2d_noice(self, event)
        elif ax == self.ui.canvas_ice.axes[3]:
            annotate_ice_2d_withice(self, event)
        elif ax == self.ui.canvas_tomo.axes[1]:
            annotate_tomo_diagram(self, event)
        elif ax == self.ui.canvas_tomo.axes[3]:
            annotate_tomo_ref_ctf(self, event)
        elif ax == self.ui.canvas_tomo.axes[4]:
            annotate_tomo_tilt_ctf(self, event)
        elif ax == self.ui.canvas_image.axes[1]:
            annotate_image_original(self, event)
        elif ax == self.ui.canvas_image.axes[2]:
            annotate_image_fft(self, event)
        elif ax == self.ui.canvas_image.axes[3]:
            annotate_image_convolved(self, event)
        elif ax == self.ui.canvas_image.axes[4]:
            annotate_image_ctf(self, event)
        else:
            hide_all_annotations(self)

    def handle_annotation_toggle(self, checked) -> None:
        """Sync status across tabs.

        Args:
            checked (bool): value returned by pushing the button
        """
        self.show_annotation = checked  # Sync instance variable

        # Sync all buttons to this state
        sender = self.window.sender()
        for btn in self.annotation_toggle_buttons:
            if btn is not sender:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)

    def show_info(self) -> None:
        """Show additional HTML info text for the current tab by loading from info/*.html."""
        tab_index = self.ui.plot_tabs.currentIndex()
        html_files = {
            0: "../info/1d_info.html",
            1: "../info/2d_info.html",
            2: "../info/ice_info.html",
            3: "../info/tomo_info.html",
            4: "../info/image_info.html",
        }

        filepath = html_files.get(tab_index)
        if filepath is None:
            return

        # Construct absolute path relative to the script’s location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, filepath)

        try:
            with open(full_path, "r", encoding="utf-8") as fh:
                html_text = fh.read()
            # Show the HTML content in a QMessageBox
            QMessageBox.information(self.window, "", html_text)
        except Exception as e:
            QMessageBox.critical(self.window, "Help Load Error", f"Could not load help text:\n{e}")
