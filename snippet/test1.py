import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Hover Example")

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create Matplotlib canvas
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        # Generate some example data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.line, = self.canvas.ax.plot(x, y, label="sin(x)")
        self.canvas.ax.set_title("Hover to display coordinates")
        self.canvas.ax.set_xlabel("X-axis")
        self.canvas.ax.set_ylabel("Y-axis")

        # Enable the interactive connection
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Add a text annotation for displaying the coordinates
        self.annotation = self.canvas.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"), fontsize=10
        )
        self.annotation.set_visible(False)

    def on_hover(self, event):
        """
        Handles the hover event over the Matplotlib canvas.
        """
        if event.inaxes == self.canvas.ax:  # Check if the mouse is over the plot
            x, y = event.xdata, event.ydata  # Get data coordinates
            if x is not None and y is not None:
                # Update annotation position and text
                self.annotation.xy = (x, y)
                self.annotation.set_text(f"x: {x:.2f}, y: {y:.2f}")
                self.annotation.set_visible(True)
                self.canvas.draw_idle()  # Redraw the canvas for updates
        else:
            self.annotation.set_visible(False)  # Hide annotation if not hovering over the plot
            self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec_())