"""
Entry point for the CTF Simulation application.

This module creates and runs the PyQt application, instantiating the
AppController class which handles the GUI and related logic.
"""

from typing import NoReturn
from PyQt5.QtWidgets import QApplication
from controller import AppController

def main() -> NoReturn:
    """
    Initialize the PyQt application, create the main controller (GUI), and start the event loop.
    """
    app: QApplication = QApplication([])
    gui: AppController = AppController()
    gui.show()
    app.exec_()

if __name__ == "__main__":
    main()