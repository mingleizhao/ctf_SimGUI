"""
main_window.py
--------------

Defines the MainWindow class, a QMainWindow subclass that initializes
and configures the application's main user interface using Ui_MainWindow.
It also computes and applies UI scaling factors before setting up the UI.

"""

from PyQt5.QtWidgets import QMainWindow
from .ui_main_window import Ui_MainWindow
from .styles import init_ui_scale
from .ui_utils import compute_ui_scale


class MainWindow(QMainWindow):
    """
    QMainWindow subclass that holds the generated UI and exposes
    widget references for the controller.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        scale = min(compute_ui_scale(), 1.0)
        init_ui_scale(scale)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
