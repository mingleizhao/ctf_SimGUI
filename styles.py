"""
Shared style sheets for QSlider, QLineEdit, and QGroupBox to maintain a consistent UI theme.
"""

LABELED_SLIDER_STYLE = """
LabeledSlider {
    max-height: 55px;
    min-height: 46px;
    max-width: 320px;
    min-width: 200px;
}
"""

QSLIDER_STYLE = """
QSlider::handle:horizontal {
    width: 18px; /* Adjust handle size */
    height: 18px;
    background-color: lightblue; /* Blue handle */
    border-radius: 9px;
    margin: -6px 0;
}

QSlider::handle:hover {
    background-color: #2980b9;
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

LEFT_PANEL_QGROUPBOX_STYLE ="""
QGroupBox {
    font-size: 16px; /* Set font size */
    font-weight: bold; /* Make it bold */
    margin-top: 30px;
    border: 1px solid #4A4A4A;
    border-radius: 6px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px; /* Add some padding around the label */
    margin-left: 0px;
}
"""

RIGHT_PANEL_QGROUPBOX_STYLE ="""
QGroupBox {
    font-size: 16px; /* Set font size */
    font-weight: bold; /* Make it bold */
    border: 1px solid #4A4A4A;
    border-radius: 6px;
}
"""

QTABWIDGET_STYLE = """
/* Tabs */
QTabWidget::tab-bar {
    alignment: center;
}

QTabWidget::pane {
    border: 1px solid #4A4A4A;
    border-radius: 6px;
    background: transparent;
}

QTabBar::tab {
    padding: 6px 15px;
    min-width: 100px;
    font-size: 14px;
    font-weight: bold;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #4A4A4A;
}

QTabBar::tab:hover {
    background: lightblue;
}

QTabBar::tab:selected {
    background: #2980b9;
}

"""

TAB_BUTTON_STYLE = """
QPushButton {
    border-radius: 4px;
    padding: 0px 8px;
    background-color: lightblue;
}

QPushButton:hover {
    background-color: #2980b9;
}
"""

INFO_BUTTON_STYLE = """
QPushButton {
    border-radius: 4px;
    background-color: lightblue;
    color: white;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:checked {
    background-color: #3498db;
    color: white;
}
"""
