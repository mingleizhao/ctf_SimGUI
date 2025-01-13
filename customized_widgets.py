SHARED_SLIDER_STYLESHEET = """
QSlider::handle:horizontal {
    width: 20px; /* Adjust handle size */
    height: 10px;
    background-color: lightblue;
    border-radius: 10px;
    margin: -4px 0;
}
QSlider::groove:horizontal {
    height: 4px; /* Adjust groove size */
    background-color: lightgray;
    margin: 0px;
}
"""
SHARED_QGROUPBOX_STYLESHEET ="""
QGroupBox {
    font-size: 16px; /* Set font size */
    font-weight: bold; /* Make it bold */
    margin-top: 30px
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px; /* Add some padding around the label */
    margin-left: 0px;
}
"""
SHARED_QGROUPBOX_BACKUP ="""
QGroupBox {
    font-size: 16px; /* Set font size */
    font-weight: bold; /* Make it bold */
    color: darkblue; /* Change text color */
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center; /* Center the label */
    padding: 5px; /* Add some padding around the label */
}
"""
SHARED_SLIDER_STYLESHEET_BACKUP = """
QSlider::handle:horizontal {
    width: 20px; /* Adjust handle size */
    height: 20px;
    background-color: lightblue;
    border: 2px solid gray;
    border-radius: 10px;
    margin: -6px 0;
}
QSlider::groove:horizontal {
    height: 6px; /* Adjust groove size */
    background-color: lightgray;
    border: 1px solid gray;
    margin: 0px;
}
"""

import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

class LabeledSlider(QWidget):
    """
    Base class for labeled sliders, managing shared functionality and layout.
    """
    valueChanged = pyqtSignal(object)  # Emit the actual value, could be int or float

    def __init__(self, label_text="", orientation=Qt.Horizontal, value_format="{}", parent=None):
        super().__init__(parent)

        self.label_text = label_text
        self.value_format = value_format  # Number formatting string

        # Main layout
        self.layout = QVBoxLayout(self)
        # self.layout.setContentsMargins(5, 10, 5, 5)
        self.layout.setContentsMargins(5, 5, 5, 5)
        # self.layout.setSpacing(15)
        self.layout.setSpacing(5)
        # self.layout.addStretch()

        # Combined label to show description and current value
        self.combined_label = QLabel(self)
        self.combined_label.setAlignment(Qt.AlignLeft)
        self.combined_label.setMinimumHeight(20)
        self.layout.addWidget(self.combined_label)

        # Slider widget
        self.slider = QSlider(orientation, self)
        self.slider.setStyleSheet(SHARED_SLIDER_STYLESHEET)
        self.slider.setMinimumHeight(20)
        self.layout.addWidget(self.slider)

        # Connect slider signal
        self.slider.valueChanged.connect(self._on_slider_value_changed)

    def _on_slider_value_changed(self):
        """
        Placeholder for updating label and emitting valueChanged signal in derived classes.
        """
        pass

    def update_combined_label(self, value):
        """
        Update the label with the current value, formatted as per `value_format`.
        """
        formatted_value = self.value_format.format(value)
        self.combined_label.setText(f"{self.label_text}: {formatted_value}")

class SelectionSlider(LabeledSlider):
    def __init__(self, label_text="", values=None, orientation=Qt.Horizontal, value_format="{}", parent=None):
        super().__init__(label_text, orientation, value_format, parent)

        if values is None:
            values = [0, 50, 100]

        self.values = sorted(values)
        self.slider.setRange(0, len(self.values) - 1)

        # Initialize the label with the first value
        self._on_slider_value_changed()

    def _on_slider_value_changed(self):
        index = self.slider.value()
        value = self.values[index]
        self.update_combined_label(value)
        self.valueChanged.emit(value)

    def set_value(self, value):
        """
        Set the slider to the position corresponding to the given value.
        """
        if value in self.values:
            index = self.values.index(value)
            self.slider.setValue(index)
        else:
            raise ValueError(f"Value {value} not in predefined values: {self.values}")

    def get_value(self):
        """
        Get the current value corresponding to the slider's position.
        """
        index = self.slider.value()
        return self.values[index]

class FloatSlider(LabeledSlider):
    def __init__(self, label_text="", min_value=0.0, max_value=1.0, step=0.1, orientation=Qt.Horizontal, value_format="{:.2f}", parent=None):
        super().__init__(label_text, orientation, value_format, parent)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.num_steps = round((max_value - min_value) / step)
        self.slider.setRange(0, self.num_steps)

        # Initialize the label with the first value
        self._on_slider_value_changed()

    def _on_slider_value_changed(self):
        slider_value = self.slider.value()
        value = self.min_value + (slider_value / self.num_steps) * (self.max_value - self.min_value)
        self.update_combined_label(value)
        self.valueChanged.emit(value)

    def set_value(self, value):
        """
        Set the slider to the position corresponding to the given float value.
        """
        if self.min_value <= value <= self.max_value:
            slider_value = int((value - self.min_value) / (self.max_value - self.min_value) * self.num_steps)
            self.slider.setValue(slider_value)
        else:
            raise ValueError(f"Value {value} out of range: [{self.min_value}, {self.max_value}]")

    def get_value(self):
        """
        Get the current float value corresponding to the slider's position.
        """
        slider_value = self.slider.value()
        return self.min_value + (slider_value / float(self.num_steps)) * (self.max_value - self.min_value)
    
class FloatLogSlider(LabeledSlider):
    def __init__(self, label_text="", min_exp=0.0, max_exp=2.0, step_exp=0.1, orientation=Qt.Horizontal, value_format="{:.2e}", parent=None):
        super().__init__(label_text, orientation, value_format, parent)

        self.min_exp = min_exp
        self.max_exp = max_exp
        self.step_exp = step_exp
        self.num_steps = round((max_exp - min_exp) / step_exp)
        self.slider.setRange(0, self.num_steps)

        # Initialize the label with the first value
        self._on_slider_value_changed()

    def _on_slider_value_changed(self):
        slider_value = self.slider.value()
        exponent = self.min_exp + (slider_value / self.num_steps) * (self.max_exp - self.min_exp)
        value = 10 ** exponent
        self.update_combined_label(value)
        self.valueChanged.emit(value)

    def set_value(self, value):
        """
        Set the slider to the position corresponding to the given float value in log scale.
        """
        if 10 ** self.min_exp <= value <= 10 ** self.max_exp:
            exponent = math.log10(value)
            slider_value = int((exponent - self.min_exp) / (self.max_exp - self.min_exp) * self.num_steps)
            self.slider.setValue(slider_value)
        else:
            raise ValueError(f"Value {value} out of range: [10^{self.min_exp}, 10^{self.max_exp}]")

    def get_value(self):
        """
        Get the current float value corresponding to the slider's position in log scale.
        """
        slider_value = self.slider.value()
        exponent = self.min_exp + (slider_value / self.num_steps) * (self.max_exp - self.min_exp)
        return 10 ** exponent


def test():
    a = FloatLogSlider(min_exp=-2, max_exp=2, step_exp=0.1)
    a.set_value(10)
    print(a.get_value())

    b= FloatSlider(min_value=0.2, max_value=5, step=0.1)
    b.set_value(1)
    print(b.get_value())
    print(b.min_value)
    print(b.max_value)
    print(b.num_steps)
    print(b.slider.value())

    c= FloatSlider(min_value=-5, max_value=5, step=0.01)
    c.set_value(1)
    print(c.get_value())

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slider Test Implementation")
        self.setGeometry(100, 100, 500, 400)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Test SelectionSlider
        self.selection_slider = SelectionSlider("Brightness", values=[10, 50, 100, 200, 500])
        layout.addWidget(self.selection_slider)

        # Button to set and get SelectionSlider value
        selection_set_button = QPushButton("Set Brightness to 100")
        selection_set_button.clicked.connect(lambda: self.selection_slider.set_value(100))
        layout.addWidget(selection_set_button)

        selection_get_button = QPushButton("Get Brightness Value")
        selection_get_button.clicked.connect(
            lambda: print(f"Current Brightness: {self.selection_slider.get_value()}")
        )
        layout.addWidget(selection_get_button)

        # Test FloatSlider
        self.float_slider = FloatSlider("Temperature", min_value=-20.0, max_value=100.0, step=0.5, value_format="{:.1f}")
        layout.addWidget(self.float_slider)

        # Button to set and get FloatSlider value
        float_set_button = QPushButton("Set Temperature to 25.5")
        float_set_button.clicked.connect(lambda: self.float_slider.set_value(25.5))
        layout.addWidget(float_set_button)

        float_get_button = QPushButton("Get Temperature Value")
        float_get_button.clicked.connect(
            lambda: print(f"Current Temperature: {self.float_slider.get_value():.1f}")
        )
        layout.addWidget(float_get_button)

        # Test FloatLogSlider
        self.log_slider = FloatLogSlider("Log Scale", min_exp=-2, max_exp=2, step_exp=0.1, value_format="{:.3e}")
        layout.addWidget(self.log_slider)

        # Button to set and get FloatLogSlider value
        log_set_button = QPushButton("Set Logarithmic Value to 100")
        log_set_button.clicked.connect(lambda: self.log_slider.set_value(100))
        layout.addWidget(log_set_button)

        log_get_button = QPushButton("Get Logarithmic Value")
        log_get_button.clicked.connect(
            lambda: print(f"Current Logarithmic Value: {self.log_slider.get_value():.3e}")
        )
        layout.addWidget(log_get_button)

if __name__ == "__main__":
    app = QApplication([])
    test()
    # window = TestWindow()
    # window.show()
    # app.exec_()