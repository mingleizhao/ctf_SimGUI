"""
customized_widgets.py
--------------

This module provides two main components for the CTF Simulation GUI:

LabeledSlider
   A composite QWidget consisting of:
        - a QLabel for a descriptive text
        - a QLineEdit for precise numeric entry
        - a QSlider for quick graphical adjustment
   Supports both linear and optional logarithmic scaling, and emits a `valueChanged(float)` signal.

MplCanvas
    A convenience subclass of Matplotlib’s FigureCanvasQTAgg for embedding one or more Matplotlib
      subplots in a PyQt5 application.

    It supports:
        - Specifying the figure size and DPI.
        - Laying out subplots on a flexible grid via GridSpec.
        - Customizing individual subplot spans (rowspan/colspan) and keyword args.
"""

import math
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFontMetrics
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from .ui_utils import compute_ui_scale
from .styles import qslider_style, labeled_slider_style


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
        parent: QWidget | None = None,
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

        self._scale = min(compute_ui_scale(), 1.0)

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
        margin = int(5 * self._scale)
        main_layout.setContentsMargins(margin, 0, margin, 0)
        main_layout.setSpacing(margin)

        # Label + Input Row
        top_layout = QHBoxLayout()
        top_layout.setSpacing(int(10 * self._scale))

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
        self.setStyleSheet(labeled_slider_style())
        self.slider.setStyleSheet(qslider_style())

        self.set_value(min_value)

    def _adjust_input_width(self) -> None:
        """Dynamically adjust the width of QLineEdit to fit content."""
        text = self.value_input.text()
        metrics = QFontMetrics(self.value_input.font())
        width = metrics.boundingRect(text).width()  # + 10  # Add padding
        self.value_input.setMinimumWidth(width + int(5 * self._scale))
        self.value_input.setMaximumWidth(width + int(10 * self._scale))  # Allow small extra space

    def _slider_to_value(self, slider_value: int) -> float:
        """Convert slider position to actual value based on scale."""
        if self.log_scale:
            return 10 ** (
                self.min_exp + (slider_value / self.num_steps) * (self.max_exp - self.min_exp)
            )
        return self.min_value + (slider_value / self.num_steps) * (self.max_value - self.min_value)

    def _value_to_slider(self, value: float) -> int:
        """Convert actual value to slider position based on scale."""
        if self.log_scale:
            return int(
                (math.log10(value) - self.min_exp) / (self.max_exp - self.min_exp) * self.num_steps
            )
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
            self.value_input.setText(
                self.value_format.format(self.get_value())
            )  # Reset on invalid input

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


class MplCanvas(FigureCanvasQTAgg):
    """
    A custom Matplotlib canvas widget for embedding plots in a PyQt application.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 8.0,
        height: float = 6.0,
        dpi: int = 100,
        subplot_grid: tuple[int, int] = (1, 1),
        subplot_args: dict | None = None,
    ) -> None:
        """
        Initialize the Matplotlib canvas with support for custom subplot layouts.

        Args:
            parent (QWidget | None, optional): Optional parent widget for this canvas.
                Defaults to None.
            width (float, optional): Width of the plot in inches. Defaults to 8.0.
            height (float, optional): Height of the plot in inches. Defaults to 6.0.
            dpi (int, optional): Resolution of the plot in dots per inch. Defaults to 100.
            subplot_grid (tuple[int, int], optional): Grid layout for subplots (rows, cols).
                Defaults to (2, 2).
            subplot_args (dict | None, optional): Arguments to customize specific subplots.
                Example: {1: {"colspan": 2}} for subplot 1 spanning 2 columns.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        nrows, ncols = subplot_grid

        # Initialize the layout for subplots
        self.axes = {}
        if subplot_args is None:
            subplot_args = {}

        # Use GridSpec for advanced layout
        grid_spec = self.fig.add_gridspec(nrows=nrows, ncols=ncols)

        # Define subplots
        if not subplot_args:
            # Create default subplots when no specific args are provided
            for idx in range(1, nrows * ncols + 1):
                row, col = divmod(idx - 1, ncols)  # 1-based index
                self.axes[idx] = self.fig.add_subplot(grid_spec[row, col])
        else:
            # Apply custom subplot arguments
            for idx, args in subplot_args.items():
                if "rowspan" in args or "colspan" in args:
                    self.axes[idx] = self.fig.add_subplot(
                        grid_spec[args.pop("rowspan"), args.pop("colspan")], **args
                    )
                else:
                    row, col = divmod(idx - 1, ncols)  # 1-based index
                    self.axes[idx] = self.fig.add_subplot(grid_spec[row, col], **args)

        super().__init__(self.fig)
        self.setParent(parent)
