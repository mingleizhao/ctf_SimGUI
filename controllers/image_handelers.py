import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from utils.image_processing import load_and_prepare_image
from controllers.plot_updates import update_plot


def handle_upload_image(ctrl) -> None:
    """
    Process user supplied image and update the image tab. Supports common formats (PNG, JPG, TIFF, etc.).
    """
    supported_formats = "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
    file_path, _ = QFileDialog.getOpenFileName(ctrl.window, "Select Image", "", supported_formats)

    if not file_path:
        return

    try:
        ctrl.image_data = load_and_prepare_image(file_path, ctrl.image_size)
    except Exception as e:
        QMessageBox.critical(ctrl.window, "Image Load Error", f"Failed to load image:\n{str(e)}")
        return

    # Reset sliders
    ctrl.ui.size_scale_image.setValue(100)
    ctrl.ui.size_scale_fft.setValue(100)
    ctrl.ui.contrast_scale_image.setValue(100)
    ctrl.ui.contrast_scale_fft.setValue(99)

    ctrl.image_data_fft = np.fft.fftshift(np.fft.fft2(ctrl.image_data))
    ctrl.scaled_fft = np.abs(ctrl.image_data_fft)

    vmin, vmax = np.percentile(
        ctrl.scaled_fft,
        [100 - ctrl.ui.contrast_scale_fft.value(), ctrl.ui.contrast_scale_fft.value()],
    )

    ctrl.image_original.set_data(ctrl.image_data)
    ctrl.image_fft.set_data(ctrl.scaled_fft)
    ctrl.image_fft.set_clim(vmin=vmin, vmax=vmax)

    update_plot(ctrl, ctrl.ui.plot_tabs.currentIndex())
