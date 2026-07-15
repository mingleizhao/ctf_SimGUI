"""
ui_utils.py
--------------

This module provides utility functions for the CTFSimGUI application's PyQt5-based interface.
It includes helpers for UI scaling.

Functions:
    - compute_ui_scale(design_size=(1620, 1080)):
        Computes a UI scaling factor based on the current screen size and a reference design size.
    - show_html_info(parent, html, title=""):
        Displays HTML help text in a scrollable, resizable dialog with a light
        content background (so the black equation images stay legible in dark mode).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
)


def compute_ui_scale(design_size=(1620, 1080)):
    """
    Compute a UI scaling factor based on the available screen size and a reference design size.

    Args:
        design_size (tuple): The reference (width, height) for which the UI was designed.

    Returns:
        float: The scaling factor to apply to UI elements for consistent appearance across screens.
    """
    w, h = (
        QDesktopWidget().availableGeometry().size().width(),
        QDesktopWidget().availableGeometry().size().height(),
    )
    ref_w, ref_h = design_size
    res = min(w / ref_w, h / ref_h)
    return res


def show_html_info(parent, html: str, title: str = "") -> None:
    """
    Display HTML help text in a scrollable, resizable dialog.

    Unlike QMessageBox, this expands to fit (and scrolls) so wide/tall equation
    content is never clipped, and it forces a light content background with dark
    text so the (black) pre-rendered equation images remain legible regardless of
    the system's light/dark theme.

    Args:
        parent: Parent widget (for modality/positioning).
        html (str): Rich-text HTML to display.
        title (str): Optional window title.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)

    layout = QVBoxLayout(dialog)

    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)

    label = QLabel()
    label.setTextFormat(Qt.RichText)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    label.setText(html)
    # Force a light "page" background so black equation images and text stay
    # readable even when the OS/Qt theme is dark.
    label.setStyleSheet("background-color: #ffffff; color: #000000; padding: 12px;")

    scroll.setWidget(label)
    layout.addWidget(scroll)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok, parent=dialog)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)

    # Sensible starting size; the dialog is resizable and the content scrolls.
    scale = min(compute_ui_scale(), 1.0)
    dialog.resize(int(680 * scale), int(620 * scale))
    dialog.exec_()
