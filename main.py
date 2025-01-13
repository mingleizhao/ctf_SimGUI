from PyQt5.QtWidgets import QApplication
from controller import AppController

if __name__ == "__main__":
    app = QApplication([])

    # Initialize Controller
    gui = AppController()

    # Show GUI
    gui.show()
    app.exec_()