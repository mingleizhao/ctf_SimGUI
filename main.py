"""
Entry point for the CTF Simulation application.

This module creates and runs the PyQt application, instantiating the
AppController class which handles the GUI and related logic.
"""

import argparse
from typing import NoReturn
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from controllers.main_controller import AppController


def main() -> NoReturn:
    """
    Initialize the PyQt application, parse user arguments,
    create the main controller (GUI), and start the event loop.
    """
    parser = argparse.ArgumentParser(description="CTF Simulation GUI Options")
    parser.add_argument(
        "--line_points",
        type=int,
        default=10000,
        help="Number of sampling points for 1D plot (default: 10000)",
    )
    parser.add_argument(
        "--image_size", type=int, default=400, help="Size of the 2D image in pixels (default: 400)"
    )
    parser.add_argument(
        "--default_image",
        type=str,
        default="sample_images/sample_image.png",
        help="Path to default sample image",
    )

    args = parser.parse_args()

    app: QApplication = QApplication([])

    # app.setStyle("Fusion")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    gui: AppController = AppController(
        line_points=args.line_points, image_size=args.image_size, default_image=args.default_image
    )
    gui.window.show()
    app.exec_()


if __name__ == "__main__":
    main()
