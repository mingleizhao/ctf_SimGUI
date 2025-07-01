"""
styles.py
--------------

Shared style sheets for QSlider, QGroupBox, etc., all
scaled by a single UI scale factor.
"""

__all__ = [
    "init_ui_scale",
    "labeled_slider_style",
    "qslider_style",
    "left_panel_qgroupbox_style",
    "right_panel_qgroupbox_style",
    "qtabwidget_style",
    "button_style",
    "info_button_style",
    "scroll_area_style",
    "get_ui_scale",
]

# internal scale value, call init_ui_scale() after QApplication exists
_scale = 1.0


def get_ui_scale() -> float:
    """Getter function"""
    return _scale


def init_ui_scale(scale: float) -> None:
    """Initialize the module-level UI scale factor."""
    global _scale
    _scale = scale


def _px(value: float) -> str:
    """Return a scaled pixel string (e.g. '12px')."""
    size = max(int(value * _scale), 1)
    return f"{size}px"


def labeled_slider_style() -> str:
    """Stylesheet for LabeledSlider, scaled by UI factor."""
    return f"""
LabeledSlider {{
    max-height: {_px(55)};
    min-height: {_px(46)};
    max-width: {_px(320)};
    min-width: {_px(200)};
}}
"""


def qslider_style() -> str:
    """Stylesheet for QSlider components."""
    r = round(3 * _scale)
    return f"""
QSlider::handle:horizontal {{
    width: {6 * r}px;
    height: {6 * r}px;
    background-color: lightblue;
    border-radius: {3 * r}px;
    margin: -{2 * r}px 0px;
}}
QSlider::handle:hover {{
    background-color: #2980b9;
}}
QSlider::groove:horizontal {{
    height: {2 * r}px;
    background-color: #d3d3d3;
    border-radius: {r}px;
    margin: 0;
}}
"""


def left_panel_qgroupbox_style() -> str:
    """Stylesheet for left panel QGroupBox."""
    return f"""
QGroupBox {{
    font-size: {_px(16)};
    font-weight: bold;
    margin-top: {_px(30)};
    border: 1px solid #4A4A4A;
    border-radius: {_px(6)};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: {_px(5)};
    margin-left: 0;
}}
"""


def right_panel_qgroupbox_style() -> str:
    """Stylesheet for right panel QGroupBox."""
    return f"""
QGroupBox {{
    font-size: {_px(16)};
    font-weight: bold;
    border: 1px solid #4A4A4A;
    border-radius: {_px(6)};
}}
"""


def qtabwidget_style() -> str:
    """Stylesheet for QTabWidget and QTabBar."""
    return f"""
QTabWidget::tab-bar {{
    alignment: center;
}}
QTabWidget::pane {{
    border: 1px solid #4A4A4A;
    border-radius: {_px(6)};
    background: transparent;
}}
QTabBar::tab {{
    min-height: {_px(34)};
    max-height: {_px(34)};
    min-width: {_px(90)};
    font-size: {_px(16)};
    font-weight: bold;
    border-top-left-radius: {_px(6)};
    border-top-right-radius: {_px(6)};
    border: 1px solid #4A4A4A;
}}
QTabBar::tab:hover {{
    background: lightblue;
}}
QTabBar::tab:selected {{
    background: #2980b9;
}}
"""


def button_style() -> str:
    """Stylesheet for blue QPushButton."""
    return f"""
QPushButton {{
    border-radius: {_px(6)};
    padding: {_px(4)} {_px(10)};
    background-color: lightblue;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: #2980b9;
}}
"""


def default_button_style() -> str:
    """Stylesheet for default QPushButton."""
    return f"""
QPushButton {{
    border: 1px solid #4A4A4A;
    border-radius: {_px(6)};
    padding: {_px(2)} {_px(10)};
}}
QPushButton:hover {{
    background: #4A4A4A;
}}
"""


def info_button_style() -> str:
    """Stylesheet for small info-toggle buttons."""
    return f"""
QPushButton {{
    border-radius: {_px(4)};
    background-color: lightblue;
    color: white;
    font-weight: bold;
    font-family: "Georgia", "Times New Roman", serif;
    font-style: italic;
}}
QPushButton:hover {{
    background-color: #2980b9;
}}
QPushButton:checked {{
    background-color: #3498db;
    color: white;
}}
"""


def scroll_area_style() -> str:
    """Stylesheet for QScrollArea and its scrollbars."""
    return f"""
QScrollArea {{
    border: none;
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    width: {_px(6)};
    height: {_px(6)};
    background: transparent;
}}
QScrollBar::handle:vertical {{
    background: #aaa;
    border-radius: {_px(3)};
}}
QScrollBar::handle:horizontal {{
    background: #aaa;
    border-radius: {_px(3)};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    background: none;
    height: 0;
    width: 0;
}}
"""


def check_box_style() -> str:
    """Stylesheet for QCheckBox."""
    return f"""
QCheckBox {{
    margin: 0px;
    padding: 0px;
    spacing: {_px(6)};          /* distance between indicator and text */
}}
QCheckBox::indicator {{
    margin: 0px;
    padding: 0px;
    width: {_px(18)};
    height: {_px(18)};
}}
"""


def radio_box_style() -> str:
    """Stylesheet for QRadioButton."""
    return f"""
QRadioButton {{
    margin: 0px;
    padding: 0px;
    spacing: {_px(6)};          /* distance between indicator and text */
}}
QRadioButton::indicator {{
    margin: 0px;
    padding: 0px;
    width: {_px(18)};
    height: {_px(18)};
}}
"""


def combo_box_style() -> str:
    """Stylesheet for QComboBox."""
    return f"""
QComboBox {{
    /* overall widget size */
    min-width: {_px(230)};
    max-width: {_px(320)};
    min-height: {_px(20)};
    max-height: {_px(40)};
    padding-left: {_px(6)};
    border-radius: {_px(6)};
    background-color: palette(WindowText);
}}
QComboBox QAbstractItemView {{
    min-width: {_px(240)};
    max-width: {_px(320)};
}}
"""


def double_spin_box_style() -> str:
    """Stylesheet for QDoubleSpinBox (and QSpinBox)."""
    return f"""
QDoubleSpinBox, QSpinBox {{
    min-width: {_px(70)};
    max-width: {_px(200)};
    min-height: {_px(20)};
    max-height: {_px(20)};
}}
QDoubleSpinBox::drop-down, QSpinBox::drop-down {{
    width: {_px(20)};
}}
"""
