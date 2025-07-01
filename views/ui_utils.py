"""
ui_utils.py
--------------

This module provides utility functions for the CTFSimGUI application's PyQt5-based interface.
It includes helpers for UI scaling.

Functions:
    - compute_ui_scale(design_size=(1620, 1080)):
        Computes a UI scaling factor based on the current screen size and a reference design size.
"""

from PyQt5.QtWidgets import (
    QDesktopWidget,
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
