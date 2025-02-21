"""
Shared style sheets for QSlider, QLineEdit, and QGroupBox to maintain a consistent UI theme.
"""

SHARED_SLIDER_STYLESHEET = """
LabeledSlider {
    max-height: 55px;
    min-height: 46px;
    max-width: 320px;
}
"""

SHARED_QSLIDER_STYLESHEET = """
QSlider::handle:horizontal {
    width: 18px; /* Adjust handle size */
    height: 18px;
    background-color: lightblue; /* Blue handle */
    border-radius: 9px;
    margin: -6px 0;
}
QSlider::groove:horizontal {
    height: 6px; /* Groove size */
    background-color: #d3d3d3; /* Light gray */
    border-radius: 3px;
    margin: 0px;
}
QSlider::sub-page:horizontal {
    /* background-color: #2980b9; /* Darker blue for filled portion */ */
    border-radius: 3px;
}
QSlider::add-page:horizontal {
    background-color: #e0e0e0;
    border-radius: 3px;
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

SHARED_QTABWIDGET_STYLESHEET = """
/* Tabs */
QTabBar::tab {
    padding: 11px 15px;
    min-width: 100px;
    font-size: 14px;
    font-weight: bold;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border-bottom-left-radius: 4px;
    border-bottom-right-radius: 4px;
    border: 1px solid #4A4A4A;
}

QTabBar::tab:selected {
    background: lightblue;
}

"""


import math
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit, QApplication, QMainWindow, QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFontMetrics


import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFontMetrics

class LabeledSlider(QWidget):
    """
    A labeled slider with a QLineEdit for precise numerical input.
    Supports both linear and logarithmic scaling.

    Attributes:
        label (QLabel): Static label describing the slider.
        value_input (QLineEdit): Allows manual numerical input.
        slider (QSlider): Enables quick graphical adjustments.
        log_scale (bool): If True, the slider operates on a logarithmic scale.
        min_exp (float): Minimum exponent in logarithmic mode.
        max_exp (float): Maximum exponent in logarithmic mode.
        valueChanged (pyqtSignal): Emits the updated value when changed.
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label_text: str = "",
        min_value: float = 0.1,
        max_value: float = 100.0,
        step: float = 1.0,
        log_scale: bool = False,
        orientation: Qt.Orientation = Qt.Horizontal,
        value_format: str = "{:.2f}",
        parent: QWidget | None = None
    ) -> None:
        """
        Initialize the labeled slider with an optional logarithmic mode.

        Args:
            label_text (str, optional): Static label text. Defaults to "".
            min_value (float, optional): Minimum value. Defaults to 0.1.
            max_value (float, optional): Maximum value. Defaults to 100.0.
            step (float, optional): Step size for linear scaling. Defaults to 1.0.
            log_scale (bool, optional): If True, use logarithmic scaling. Defaults to False.
            orientation (Qt.Orientation, optional): Slider orientation. Defaults to Qt.Horizontal.
            value_format (str, optional): Format string for displaying the value. Defaults to "{:.2f}".
            parent (QWidget, optional): Optional parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.value_format = value_format
        self.log_scale = log_scale

        if log_scale:
            if min_value <= 0 or max_value <= 0:
                raise ValueError("Log scale requires min_value and max_value > 0")

            self.min_exp = math.log10(min_value)
            self.max_exp = math.log10(max_value)
            self.num_steps = 100  # Fixed steps for logarithmic scaling
        else:
            self.num_steps = int((max_value - min_value) / step)

        # Layout: Row 1 (Label + Input), Row 2 (Slider)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Label + Input Row
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        self.label = QLabel(label_text, self)
        top_layout.addWidget(self.label)

        self.value_input = QLineEdit(self)
        self.value_input.setAlignment(Qt.AlignRight)
        self.value_input.setValidator(QDoubleValidator(min_value, max_value, 10))
        self.value_input.textChanged.connect(self._adjust_input_width)
        self.value_input.returnPressed.connect(self._on_text_input_changed)
        top_layout.addWidget(self.value_input)

        main_layout.addLayout(top_layout)

        # Slider Row
        self.slider = QSlider(orientation, self)
        self.slider.setRange(0, self.num_steps)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self._on_slider_changed)
        main_layout.addWidget(self.slider)

        # Apply Styles
        self.setStyleSheet(SHARED_SLIDER_STYLESHEET)
        self.slider.setStyleSheet(SHARED_QSLIDER_STYLESHEET)

        self.set_value(min_value)

    def _adjust_input_width(self) -> None:
        """Dynamically adjust the width of QLineEdit to fit content."""
        text = self.value_input.text()
        metrics = QFontMetrics(self.value_input.font())
        width = metrics.boundingRect(text).width() # + 10  # Add padding
        self.value_input.setMinimumWidth(width)
        self.value_input.setMaximumWidth(width + 10)  # Allow small extra space

    def _slider_to_value(self, slider_value: int) -> float:
        """Convert slider position to actual value based on scale."""
        if self.log_scale:
            return 10 ** (self.min_exp + (slider_value / self.num_steps) * (self.max_exp - self.min_exp))
        return self.min_value + (slider_value / self.num_steps) * (self.max_value - self.min_value)

    def _value_to_slider(self, value: float) -> int:
        """Convert actual value to slider position based on scale."""
        if self.log_scale:
            return int((math.log10(value) - self.min_exp) / (self.max_exp - self.min_exp) * self.num_steps)
        return int((value - self.min_value) / (self.max_value - self.min_value) * self.num_steps)

    def _on_slider_changed(self, value: int) -> None:
        """Sync slider movement with QLineEdit."""
        real_value = self._slider_to_value(value)
        self.value_input.blockSignals(True)
        self.value_input.setText(self.value_format.format(real_value))
        self.value_input.blockSignals(False)
        self._adjust_input_width()  # Ensure width updates dynamically
        self.valueChanged.emit(real_value)

    def _on_text_input_changed(self) -> None:
        """Sync text input with the slider when manually edited."""
        try:
            value = float(self.value_input.text())
            value = max(self.min_value, min(value, self.max_value))  # Clamp value

            slider_value = self._value_to_slider(value)

            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)

            self.value_input.setText(self.value_format.format(value))  # Ensure proper format
            self.valueChanged.emit(value)
        except ValueError:
            self.value_input.setText(self.value_format.format(self.get_value()))  # Reset on invalid input

    def set_value(self, value: float) -> None:
        """
        Set the slider and input box to a specific value.

        Args:
            value (float): The value to set.

        Raises:
            ValueError: If the value is out of range.
        """
        if self.min_value <= value <= self.max_value:
            slider_value = self._value_to_slider(value)
            self.slider.setValue(slider_value)
            self.value_input.setText(self.value_format.format(value))
            self._adjust_input_width()  # Ensure proper width after setting value
            self.valueChanged.emit(value)
        else:
            raise ValueError(f"Value {value} out of range [{self.min_value}, {self.max_value}].")

    def get_value(self) -> float:
        """
        Get the current value of the slider.

        Returns:
            float: The current value, clamped within [min_value, max_value].
        """
        try:
            value = float(self.value_input.text())
            value = max(self.min_value, min(value, self.max_value))  # Clamp value
            self.value_input.setText(self.value_format.format(value))
            self._adjust_input_width()  # Adjust width after getting value
            return value
        except ValueError:
            return self.min_value


class TestWindow(QMainWindow):
    """
    A test PyQt window containing various slider widgets for demonstration.
    """
    def __init__(self) -> None:
        """
        Initialize the test window with selection, float, and float-log sliders plus set/get value buttons.
        """
        super().__init__()
        self.setWindowTitle("Slider Test Implementation")
        self.setGeometry(100, 100, 500, 400)

        central_widget: QWidget = QWidget()
        layout: QVBoxLayout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Test SelectionSlider
        self.selection_slider: LabeledSlider = LabeledSlider("Brightness", min_value=10.0, max_value=500.0, step=70, value_format="{:.0f}")
        layout.addWidget(self.selection_slider)

        # Button to set and get SelectionSlider value
        selection_set_button: QPushButton = QPushButton("Set Brightness to 100")
        selection_set_button.clicked.connect(lambda: self.selection_slider.set_value(100))
        layout.addWidget(selection_set_button)

        selection_get_button: QPushButton = QPushButton("Get Brightness Value")
        selection_get_button.clicked.connect(
            lambda: print(f"Current Brightness: {self.selection_slider.get_value()}")
        )
        layout.addWidget(selection_get_button)

        # Test FloatSlider
        self.float_slider: LabeledSlider = LabeledSlider("Temperature", min_value=-20.0, max_value=100.0, step=0.5, value_format="{:.1f}")
        layout.addWidget(self.float_slider)

        # Button to set and get FloatSlider value
        float_set_button: QPushButton = QPushButton("Set Temperature to 25.5")
        float_set_button.clicked.connect(lambda: self.float_slider.set_value(25.5))
        layout.addWidget(float_set_button)

        float_get_button: QPushButton = QPushButton("Get Temperature Value")
        float_get_button.clicked.connect(
            lambda: print(f"Current Temperature: {self.float_slider.get_value():.1f}")
        )
        layout.addWidget(float_get_button)

        # Test FloatLogSlider
        self.log_slider: LabeledSlider = LabeledSlider("Log Scale", min_value=10**-2, max_value=10**2, step=0.1, value_format="{:.3e}", log_scale=True)
        layout.addWidget(self.log_slider)

        # Button to set and get FloatLogSlider value
        log_set_button: QPushButton = QPushButton("Set Logarithmic Value to 100")
        log_set_button.clicked.connect(lambda: self.log_slider.set_value(100))
        layout.addWidget(log_set_button)

        log_get_button: QPushButton = QPushButton("Get Logarithmic Value")
        log_get_button.clicked.connect(
            lambda: print(f"Current Logarithmic Value: {self.log_slider.get_value():.3e}")
        )
        layout.addWidget(log_get_button)


if __name__ == "__main__":
    app = QApplication([])
    window = TestWindow()
    window.show()
    app.exec_()
