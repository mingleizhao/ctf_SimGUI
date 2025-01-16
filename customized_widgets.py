"""
Shared style sheets for QSlider and QGroupBox to maintain a consistent UI theme.
"""
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


import math
from typing import List, Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

class LabeledSlider(QWidget):
    """
    A base class for labeled sliders, managing shared functionality and layout.

    Attributes:
        label_text (str): The descriptive text for the slider.
        value_format (str): A string format specifier for displaying slider values.
        layout (QVBoxLayout): The primary layout holding the label and slider.
        combined_label (QLabel): A label showing `label_text` plus the current slider value.
        slider (QSlider): The underlying PyQt slider widget.

    Signals:
        valueChanged (pyqtSignal): Emits the current slider value (as an int or float) when it changes.
    """

    valueChanged = pyqtSignal(object)

    def __init__(
        self,
        label_text: str = "",
        orientation: Qt.Orientation = Qt.Horizontal,
        value_format: str = "{}",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the labeled slider with a specified label, orientation, and value format.

        Args:
            label_text (str, optional): Descriptive text for the slider. Defaults to "".
            orientation (Qt.Orientation, optional): Slider orientation (horizontal or vertical). Defaults to Qt.Horizontal.
            value_format (str, optional): Format string for displaying the slider value. Defaults to "{}".
            parent (QWidget, optional): Optional parent widget. Defaults to None.
        """
        super().__init__(parent)

        self.label_text: str = label_text
        self.value_format: str = value_format

        # Main layout
        self.layout: QVBoxLayout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        # Combined label to show description and current value
        self.combined_label: QLabel = QLabel(self)
        self.combined_label.setAlignment(Qt.AlignLeft)
        self.combined_label.setMinimumHeight(20)
        self.layout.addWidget(self.combined_label)

        # Slider widget
        self.slider: QSlider = QSlider(orientation, self)
        self.slider.setStyleSheet(SHARED_SLIDER_STYLESHEET)
        self.slider.setMinimumHeight(20)
        self.layout.addWidget(self.slider)

        # Connect slider signal
        self.slider.valueChanged.connect(self._on_slider_value_changed)

    def _on_slider_value_changed(self) -> None:
        """
        A placeholder for updating the label and emitting the valueChanged signal in derived classes.
        Override this in subclasses to implement actual behavior.
        """
        pass

    def update_combined_label(self, value: float) -> None:
        """
        Update the label with the current value, formatted according to `value_format`.

        Args:
            value (float): The current value to display on the label.
        """
        formatted_value: str = self.value_format.format(value)
        self.combined_label.setText(f"{self.label_text}: {formatted_value}")


class SelectionSlider(LabeledSlider):
    """
    A slider that snaps to predefined discrete values.

    Attributes:
        values (List[float]): Sorted list of discrete values for the slider.
    """

    def __init__(
        self,
        label_text: str = "",
        values: Optional[List[float]] = None,
        orientation: Qt.Orientation = Qt.Horizontal,
        value_format: str = "{}",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize a SelectionSlider with a label, list of discrete values, orientation, and format.

        Args:
            label_text (str, optional): Descriptive label for the slider. Defaults to "".
            values (List[float], optional): List of discrete values the slider can occupy. Defaults to [0, 50, 100].
            orientation (Qt.Orientation, optional): Slider orientation. Defaults to Qt.Horizontal.
            value_format (str, optional): Format string for displaying the slider's current value. Defaults to "{}".
            parent (QWidget, optional): Optional parent widget. Defaults to None.
        """
        super().__init__(label_text, orientation, value_format, parent)

        if values is None:
            values = [0, 50, 100]

        self.values: List[float] = sorted(values)
        self.slider.setRange(0, len(self.values) - 1)

        # Initialize the label with the first value
        self._on_slider_value_changed()

    def _on_slider_value_changed(self) -> None:
        """Handle slider movement by updating the label and emitting the new value."""
        index: int = self.slider.value()
        value: float = self.values[index]
        self.update_combined_label(value)
        self.valueChanged.emit(value)

    def set_value(self, value: float) -> None:
        """
        Set the slider position to a specific value in the predefined list.

        Args:
            value (float): The new value to select.

        Raises:
            ValueError: If the specified value does not exist in `self.values`.
        """
        if value in self.values:
            index: int = self.values.index(value)
            self.slider.setValue(index)
        else:
            raise ValueError(f"Value {value} not in predefined values: {self.values}")

    def get_value(self) -> float:
        """
        Retrieve the current value based on the slider's position.

        Returns:
            float: The currently selected value from `self.values`.
        """
        index: int = self.slider.value()
        return self.values[index]


class FloatSlider(LabeledSlider):
    """
    A slider for continuous float ranges, using linear scaling between a minimum and maximum value.

    Attributes:
        min_value (float): Minimum value of the slider range.
        max_value (float): Maximum value of the slider range.
        step (float): Step size used to compute the number of discrete steps for the slider.
        num_steps (int): Number of internal steps computed from the range and step size.
    """

    def __init__(
        self,
        label_text: str = "",
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.1,
        orientation: Qt.Orientation = Qt.Horizontal,
        value_format: str = "{:.2f}",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize a FloatSlider with a label, min/max values, step size, orientation, and format.

        Args:
            label_text (str, optional): Descriptive label text. Defaults to "".
            min_value (float, optional): Minimum float value. Defaults to 0.0.
            max_value (float, optional): Maximum float value. Defaults to 1.0.
            step (float, optional): Step size for discretizing the slider. Defaults to 0.1.
            orientation (Qt.Orientation, optional): Slider orientation. Defaults to Qt.Horizontal.
            value_format (str, optional): Format string for displaying the float value. Defaults to "{:.2f}".
            parent (QWidget, optional): Optional parent widget. Defaults to None.
        """
        super().__init__(label_text, orientation, value_format, parent)

        self.min_value: float = min_value
        self.max_value: float = max_value
        self.step: float = step
        self.num_steps: int = round((max_value - min_value) / step)
        self.slider.setRange(0, self.num_steps)

        # Initialize the label with the first value
        self._on_slider_value_changed()

    def _on_slider_value_changed(self) -> None:
        """Update the displayed value and emit the current float value."""
        slider_value: int = self.slider.value()
        value: float = self.min_value + (slider_value / self.num_steps) * (self.max_value - self.min_value)
        self.update_combined_label(value)
        self.valueChanged.emit(value)

    def set_value(self, value: float) -> None:
        """
        Set the slider to a specified float value, if within the [min_value, max_value] range.

        Args:
            value (float): The desired float value to set.

        Raises:
            ValueError: If the value is outside of the valid range.
        """
        if self.min_value <= value <= self.max_value:
            slider_value = int((value - self.min_value) / (self.max_value - self.min_value) * self.num_steps)
            self.slider.setValue(slider_value)
        else:
            raise ValueError(f"Value {value} out of range: [{self.min_value}, {self.max_value}]")

    def get_value(self) -> float:
        """
        Get the current float value from the slider.

        Returns:
            float: The value computed by mapping the slider's position back to the [min_value, max_value] range.
        """
        slider_value: int = self.slider.value()
        return self.min_value + (slider_value / float(self.num_steps)) * (self.max_value - self.min_value)


class FloatLogSlider(LabeledSlider):
    """
    A slider for float ranges that are logarithmically scaled between 10^min_exp and 10^max_exp.

    Attributes:
        min_exp (float): Minimum exponent for 10^min_exp.
        max_exp (float): Maximum exponent for 10^max_exp.
        step_exp (float): Step size for discretizing the exponent range.
        num_steps (int): Number of steps derived from (max_exp - min_exp) / step_exp.
    """

    def __init__(
        self,
        label_text: str = "",
        min_exp: float = 0.0,
        max_exp: float = 2.0,
        step_exp: float = 0.1,
        orientation: Qt.Orientation = Qt.Horizontal,
        value_format: str = "{:.2e}",
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize a FloatLogSlider with a label, exponent range, and step size.

        Args:
            label_text (str, optional): Descriptive label text. Defaults to "".
            min_exp (float, optional): Minimum exponent. The value is 10^min_exp. Defaults to 0.0.
            max_exp (float, optional): Maximum exponent. The value is 10^max_exp. Defaults to 2.0.
            step_exp (float, optional): Step size for exponent increments. Defaults to 0.1.
            orientation (Qt.Orientation, optional): Slider orientation. Defaults to Qt.Horizontal.
            value_format (str, optional): Format string for displaying the float value. Defaults to "{:.2e}".
            parent (QWidget, optional): Optional parent widget. Defaults to None.
        """
        super().__init__(label_text, orientation, value_format, parent)

        self.min_exp: float = min_exp
        self.max_exp: float = max_exp
        self.step_exp: float = step_exp
        self.num_steps: int = round((max_exp - min_exp) / step_exp)
        self.slider.setRange(0, self.num_steps)

        # Initialize the label with the first value
        self._on_slider_value_changed()

    def _on_slider_value_changed(self) -> None:
        """Update the displayed log-scaled value and emit it."""
        slider_value: int = self.slider.value()
        exponent: float = self.min_exp + (slider_value / self.num_steps) * (self.max_exp - self.min_exp)
        value: float = 10 ** exponent
        self.update_combined_label(value)
        self.valueChanged.emit(value)

    def set_value(self, value: float) -> None:
        """
        Set the slider to match a given float value in log scale, if within [10^min_exp, 10^max_exp].

        Args:
            value (float): The new float value in normal space.

        Raises:
            ValueError: If the value is outside the valid log range.
        """
        lower_bound: float = 10 ** self.min_exp
        upper_bound: float = 10 ** self.max_exp
        if lower_bound <= value <= upper_bound:
            exponent: float = math.log10(value)
            slider_value: int = int((exponent - self.min_exp) / (self.max_exp - self.min_exp) * self.num_steps)
            self.slider.setValue(slider_value)
        else:
            raise ValueError(f"Value {value} out of range: [10^{self.min_exp}, 10^{self.max_exp}]")

    def get_value(self) -> float:
        """
        Retrieve the current float value in log scale.

        Returns:
            float: The float value corresponding to the slider's exponent position.
        """
        slider_value: int = self.slider.value()
        exponent: float = self.min_exp + (slider_value / self.num_steps) * (self.max_exp - self.min_exp)
        return 10 ** exponent


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
        self.selection_slider: SelectionSlider = SelectionSlider("Brightness", values=[10, 50, 100, 200, 500])
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
        self.float_slider: FloatSlider = FloatSlider("Temperature", min_value=-20.0, max_value=100.0, step=0.5, value_format="{:.1f}")
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
        self.log_slider: FloatLogSlider = FloatLogSlider("Log Scale", min_exp=-2, max_exp=2, step_exp=0.1, value_format="{:.3e}")
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

