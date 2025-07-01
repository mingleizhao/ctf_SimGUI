from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import numpy as np
from matplotlib.figure import Figure


def prompt_and_save_figure(parent, fig: Figure):
    """
    Open a file dialog (PNG/JPG/PDF) and save `fig` to disk with dpi=300.
    `parent` is the AppController instance for modality.
    """
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(
        parent.window,
        "Save Plot",
        "",
        "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
        options=options,
    )
    if not file_path:
        return

    try:
        fig.savefig(file_path, dpi=300)
    except Exception as e:
        QMessageBox.critical(parent.window, "Save Error", f"Failed to save plot:\n{e}")


def prompt_and_save_csv(parent):
    """
    Open CSV save dialog, then call the correct “generate DataFrame” method based on current tab.
    """
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(
        parent.window, "Save Data", "", "CSV Files (*.csv);;All Files (*)", options=options
    )
    if not file_path:
        return

    tab_index = parent.window.plot_tabs.currentIndex()

    try:
        if tab_index == 0:
            df = generate_ctf_1d_data(parent, parent.ctf_1d)
            df.to_csv(file_path, index=False)

        elif tab_index == 1:
            df = generate_ctf_2d_data(parent, parent.ctf_2d)
            df.to_csv(file_path, index=False)

        elif tab_index == 2:
            with open(file_path, "w") as f:
                f.write("# 1D CTF\n")
                generate_ctf_1d_data(parent, parent.ctf_1d_ice).to_csv(f, index=False)
                f.write("\n# 2D CTF\n")
                generate_ctf_2d_data(parent, parent.ctf_2d_ice).to_csv(f, index=False)

        elif tab_index == 3:
            df = generate_ctf_tomo_data(parent)
            df.to_csv(file_path, index=False)

        elif tab_index == 4:
            df = generate_ctf_image_data(parent)
            df.to_csv(file_path, index=False)

    except Exception as e:
        QMessageBox.critical(parent.window, "Save Error", f"Failed to save CSV:\n{e}")


def generate_ctf_1d_data(ctrl, model) -> pd.DataFrame:
    """Generates 1D CTF data as a pandas DataFrame."""
    wrap_func = ctrl.wrap_func
    return pd.DataFrame(
        {
            "freqs_1d": ctrl.freqs_1d,
            "ctf": wrap_func(model.ctf_1d(ctrl.freqs_1d)),
            "ctf_dampened": wrap_func(model.ctf(ctrl.freqs_1d)),
            "temporal_env": wrap_func(model.envelope.temporal(ctrl.freqs_1d)),
            "spatial_env": wrap_func(model.envelope.spatial_1d(ctrl.freqs_1d)),
            "detector_env": wrap_func(model.envelope.detector(ctrl.freqs_1d)),
            "ice_env": wrap_func(model.envelope.ice(ctrl.freqs_1d)),
            "total_env": wrap_func(model.envelope.total_1d(ctrl.freqs_1d)),
        }
    )


def generate_ctf_2d_data(ctrl, model) -> pd.DataFrame:
    """Generates 2D CTF data as a pandas DataFrame."""
    freqs_radial = np.sqrt(ctrl.fx_fix**2 + ctrl.fy_fix**2)
    wrap_func = ctrl.wrap_func
    return pd.DataFrame(
        {
            "freqs_x": np.repeat(ctrl.fx_fix, ctrl.image_size, axis=0).flatten(),
            "freqs_y": np.repeat(ctrl.fy_fix, ctrl.image_size).flatten(),
            "ctf": wrap_func(model.ctf_2d(ctrl.fx_fix, ctrl.fy_fix)).flatten(),
            "ctf_dampened": wrap_func(model.ctf(ctrl.fx_fix, ctrl.fy_fix)).flatten(),
            "temporal_env": wrap_func(model.envelope.temporal(freqs_radial)).flatten(),
            "spatial_env": wrap_func(model.envelope.spatial_2d(ctrl.fx_fix, ctrl.fy_fix)).flatten(),
            "detector_env": wrap_func(model.envelope.detector(freqs_radial)).flatten(),
            "ice_env": wrap_func(model.envelope.ice(freqs_radial)).flatten(),
            "total_env": wrap_func(model.envelope.total_2d(ctrl.fx_fix, ctrl.fy_fix)).flatten(),
        }
    )


def generate_ctf_tomo_data(ctrl) -> pd.DataFrame:
    """Generates tomo CTF data as a pandas DataFrame."""
    wrap_func = ctrl.wrap_func
    return pd.DataFrame(
        {
            "freqs_x": np.repeat(ctrl.fx, ctrl.image_size, axis=0).flatten(),
            "freqs_y": np.repeat(ctrl.fy, ctrl.image_size).flatten(),
            "ctf_no_tilt": wrap_func(ctrl.ctf_tomo_ref.ctf_2d(ctrl.fx, ctrl.fy)).flatten(),
            "ctf_no_tilt_dampened": wrap_func(ctrl.ctf_tomo_ref.ctf(ctrl.fx, ctrl.fy)).flatten(),
            "ctf_tilt": wrap_func(ctrl.ctf_tomo_tilt.ctf_2d(ctrl.fx, ctrl.fy)).flatten(),
            "ctf_tilt_dampened": wrap_func(ctrl.ctf_tomo_tilt.ctf(ctrl.fx, ctrl.fy)).flatten(),
            "total_env_non_tilt": wrap_func(
                ctrl.ctf_tomo_ref.envelope.total_2d(ctrl.fx, ctrl.fy)
            ).flatten(),
            "total_env_tilt": wrap_func(
                ctrl.ctf_tomo_tilt.envelope.total_2d(ctrl.fx, ctrl.fy)
            ).flatten(),
        }
    )


def generate_ctf_image_data(ctrl) -> pd.DataFrame:
    """Generates image CTF data as a pandas DataFrame."""
    wrap_func = ctrl.wrap_func
    return pd.DataFrame(
        {
            "original_image": ctrl.image_data.flatten(),
            "image_fft": ctrl.image_data_fft.flatten(),
            "image_fft_amplitude": ctrl.scaled_fft.flatten(),
            "freqs_x": np.repeat(ctrl.fx, ctrl.image_size, axis=0).flatten(),
            "freqs_y": np.repeat(ctrl.fy, ctrl.image_size).flatten(),
            "ctf": wrap_func(ctrl.ctf_2d.ctf_2d(ctrl.fx, ctrl.fy)).flatten(),
            "ctf_dampened": wrap_func(ctrl.ctf_2d.ctf(ctrl.fx, ctrl.fy)).flatten(),
            "total_env": wrap_func(ctrl.ctf_2d.envelope.total_2d(ctrl.fx, ctrl.fy)).flatten(),
            "convolved_image": ctrl.scaled_convolved.flatten(),
        }
    )
