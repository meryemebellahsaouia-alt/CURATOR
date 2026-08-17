"""
Central stylesheet for the DICOM Viewer application.

The design language is intentionally data-dense and publication-oriented:
- quiet backgrounds with high legibility text
- clear hierarchy for tabs, cards, and tables
- restrained accent usage to avoid visual noise

This module keeps the public theme constants used elsewhere in the application,
while generating the dark and light Qt stylesheets from a shared builder to
reduce duplication and drift between themes.
"""

from typing import Dict

SPACING_BASE = 10
MARGIN_BASE = 14

# --- Typography ---
FONT_FAMILY = '"Segoe UI Variable", "Segoe UI", "Noto Sans", sans-serif'
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_HEADER = "15px"
HINT_FONT_SIZE = "12px"

# --- Dark Theme Tokens ---
COLOR_BACKGROUND = "#171C25"
COLOR_WIDGET_BACKGROUND = "#222B3A"
COLOR_WIDGET_BACKGROUND_ALT = "#283348"
COLOR_WIDGET_BORDER = "#3A4960"
COLOR_TEXT = "#E4EAF3"
COLOR_TEXT_MUTED = "#9DAAC0"
COLOR_PRIMARY = "#2EA6A6"
COLOR_PRIMARY_HIGHLIGHT = "#48BCBC"
COLOR_PRIMARY_PRESSED = "#238D8D"
COLOR_SCROLLBAR = "#4A5C76"

# --- Light Theme Tokens ---
LIGHT_COLOR_BACKGROUND = "#EEF3F8"
LIGHT_COLOR_WIDGET_BACKGROUND = "#FFFFFF"
LIGHT_COLOR_WIDGET_BACKGROUND_ALT = "#F8FBFF"
LIGHT_COLOR_WIDGET_BORDER = "#C8D4E3"
LIGHT_COLOR_TEXT = "#1F2A39"
LIGHT_COLOR_TEXT_MUTED = "#5B6777"
LIGHT_COLOR_PRIMARY = "#1C8B8B"
LIGHT_COLOR_PRIMARY_HIGHLIGHT = "#2CA5A5"
LIGHT_COLOR_PRIMARY_PRESSED = "#157373"
LIGHT_COLOR_SCROLLBAR = "#9FB2C8"


def _build_theme_tokens(*, dark: bool) -> Dict[str, str]:
    """Returns one flat token dictionary used by the shared stylesheet builder."""
    if dark:
        return {
            "window_bg": COLOR_BACKGROUND,
            "widget_bg": COLOR_BACKGROUND,
            "widget_panel": COLOR_WIDGET_BACKGROUND,
            "widget_panel_alt": COLOR_WIDGET_BACKGROUND_ALT,
            "border": COLOR_WIDGET_BORDER,
            "text": COLOR_TEXT,
            "text_muted": COLOR_TEXT_MUTED,
            "primary": COLOR_PRIMARY,
            "primary_hi": COLOR_PRIMARY_HIGHLIGHT,
            "primary_pressed": COLOR_PRIMARY_PRESSED,
            "scrollbar": COLOR_SCROLLBAR,
            "loading_title": "#F2F6FD",
            "tooltip_bg": "#111722",
            "tooltip_text": "#EAF1FC",
            "menu_bar_bg": "#1A2331",
            "menu_item_hover": "#2A374D",
            "menu_bg": "#1D2636",
            "menu_hover": "#2B3A50",
            "tab_pane_bg": "#1C2432",
            "tab_bg": "#232D3E",
            "tab_text": "#C7D2E5",
            "tab_selected_text": "#F4F8FF",
            "tab_hover_bg": "#2B3850",
            "group_title_bg": COLOR_BACKGROUND,
            "group_title_text": "#DCE5F4",
            "button_disabled_bg": "#3A4558",
            "button_disabled_border": "#4B586E",
            "button_disabled_text": "#8A96AB",
            "input_bg": "#1A2230",
            "selection_text": "#FFFFFF",
            "text_selection_bg": "#355A87",
            "checkbox_border": "#5C6C87",
            "slider_groove_border": "#17202E",
            "slider_groove_bg": "#1A2230",
            "slider_fill": "#2D7C9A",
            "table_bg": "#1C2432",
            "table_alt_bg": "#202A3A",
            "header_bg": "#1A2230",
            "header_text": "#DCE5F6",
            "tree_bg": "#1A2230",
            "tree_alt_bg": "#202A3A",
            "tree_selected_bg": "#2F415B",
            "tree_selected_text": "#FFFFFF",
            "progress_bg": "#1A2230",
            "splitter_bg": "#1E2736",
            "splitter_border": "#2A3448",
            "splitter_hover": "#2A3A52",
            "status_bg": "#1A2230",
            "status_text": "#D6DEEC",
            "combo_selection_bg": COLOR_PRIMARY,
        }

    return {
        "window_bg": LIGHT_COLOR_BACKGROUND,
        "widget_bg": LIGHT_COLOR_BACKGROUND,
        "widget_panel": LIGHT_COLOR_WIDGET_BACKGROUND,
        "widget_panel_alt": LIGHT_COLOR_WIDGET_BACKGROUND_ALT,
        "border": LIGHT_COLOR_WIDGET_BORDER,
        "text": LIGHT_COLOR_TEXT,
        "text_muted": LIGHT_COLOR_TEXT_MUTED,
        "primary": LIGHT_COLOR_PRIMARY,
        "primary_hi": LIGHT_COLOR_PRIMARY_HIGHLIGHT,
        "primary_pressed": LIGHT_COLOR_PRIMARY_PRESSED,
        "scrollbar": LIGHT_COLOR_SCROLLBAR,
        "loading_title": "#102131",
        "tooltip_bg": "#FFFFFF",
        "tooltip_text": "#1B2A3A",
        "menu_bar_bg": LIGHT_COLOR_WIDGET_BACKGROUND,
        "menu_item_hover": "#DDE8F5",
        "menu_bg": LIGHT_COLOR_WIDGET_BACKGROUND,
        "menu_hover": "#DDE8F5",
        "tab_pane_bg": LIGHT_COLOR_WIDGET_BACKGROUND,
        "tab_bg": "#E9F0F8",
        "tab_text": "#415266",
        "tab_selected_text": "#132334",
        "tab_hover_bg": "#DFEAF6",
        "group_title_bg": LIGHT_COLOR_BACKGROUND,
        "group_title_text": "#1F3347",
        "button_disabled_bg": "#D5DFEA",
        "button_disabled_border": "#C4D0DE",
        "button_disabled_text": "#7F8DA1",
        "input_bg": "#FFFFFF",
        "selection_text": "#FFFFFF",
        "text_selection_bg": "#CFE5F2",
        "checkbox_border": "#9EB2C9",
        "slider_groove_border": "#D7E1EC",
        "slider_groove_bg": "#E8EEF6",
        "slider_fill": "#63A9C4",
        "table_bg": "#FFFFFF",
        "table_alt_bg": "#F7FAFE",
        "header_bg": "#EEF3F9",
        "header_text": "#1E3145",
        "tree_bg": "#FFFFFF",
        "tree_alt_bg": "#F7FAFE",
        "tree_selected_bg": "#D8EAF5",
        "tree_selected_text": LIGHT_COLOR_TEXT,
        "progress_bg": "#FFFFFF",
        "splitter_bg": "#D9E3EE",
        "splitter_border": "#CBD7E4",
        "splitter_hover": "#C5D3E2",
        "status_bg": "#F7FAFE",
        "status_text": "#2D3E52",
        "combo_selection_bg": "#D5EAF5",
    }


def _build_stylesheet(tokens: Dict[str, str]) -> str:
    """Builds the full Qt stylesheet from one token dictionary."""
    return f"""
/* ---- Global ---- */
QMainWindow {{
    background-color: {tokens['window_bg']};
}}
QWidget {{
    background-color: {tokens['widget_bg']};
    color: {tokens['text']};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_NORMAL};
}}
QLabel {{
    background: transparent;
}}
QLabel#LoadingTitle {{
    font-size: {FONT_SIZE_HEADER};
    font-weight: 700;
    color: {tokens['loading_title']};
}}
QLabel#HintLabel {{
    color: {tokens['text_muted']};
    font-size: {HINT_FONT_SIZE};
}}
QToolTip {{
    background-color: {tokens['tooltip_bg']};
    color: {tokens['tooltip_text']};
    border: 1px solid {tokens['border']};
    padding: 6px 8px;
}}

/* ---- Menu Bar / Menus ---- */
QMenuBar {{
    background-color: {tokens['menu_bar_bg']};
    border-bottom: 1px solid {tokens['border']};
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 5px;
    margin: 2px 2px;
}}
QMenuBar::item:selected {{
    background-color: {tokens['menu_item_hover']};
    color: {tokens['selection_text']};
}}
QMenu {{
    background-color: {tokens['menu_bg']};
    border: 1px solid {tokens['border']};
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 22px 6px 14px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {tokens['menu_hover']};
    color: {tokens['selection_text']};
}}
QMenu::separator {{
    height: 1px;
    margin: 6px 4px;
    background-color: {tokens['border']};
}}

/* ---- Tab Hierarchy ---- */
QTabWidget::pane {{
    border: 1px solid {tokens['border']};
    border-radius: 8px;
    top: -1px;
    background: {tokens['tab_pane_bg']};
}}
QTabBar::tab {{
    background: {tokens['tab_bg']};
    border: 1px solid {tokens['border']};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    min-width: 108px;
    padding: 10px 14px;
    margin-right: 4px;
    color: {tokens['tab_text']};
    font-size: {FONT_SIZE_HEADER};
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {tokens['tab_pane_bg']};
    color: {tokens['tab_selected_text']};
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    background: {tokens['tab_hover_bg']};
}}

/* ---- Group Panels ---- */
QGroupBox {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {tokens['widget_panel_alt']},
        stop:1 {tokens['widget_panel']}
    );
    border: 1px solid {tokens['border']};
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top center;
    background-color: {tokens['group_title_bg']};
    border-radius: 6px;
    padding: 2px 10px;
    color: {tokens['group_title_text']};
    font-size: {FONT_SIZE_HEADER};
    font-weight: 600;
}}
QGroupBox QLabel, QGroupBox QCheckBox {{
    background: transparent;
    border: none;
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {tokens['primary']};
    color: #FFFFFF;
    border: 1px solid {tokens['primary_hi']};
    border-radius: 7px;
    padding: 8px 14px;
    font-weight: 700;
}}
QPushButton:hover {{
    background-color: {tokens['primary_hi']};
}}
QPushButton:pressed {{
    background-color: {tokens['primary_pressed']};
}}
QPushButton:disabled {{
    background-color: {tokens['button_disabled_bg']};
    border-color: {tokens['button_disabled_border']};
    color: {tokens['button_disabled_text']};
}}
QDialogButtonBox QPushButton {{
    min-width: 96px;
}}

/* ---- Inputs ---- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {{
    background-color: {tokens['input_bg']};
    border: 1px solid {tokens['border']};
    border-radius: 7px;
    padding: 5px 8px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView,
QSpinBox QAbstractItemView,
QDoubleSpinBox QAbstractItemView {{
    background-color: {tokens['input_bg']};
    border: 1px solid {tokens['border']};
    selection-background-color: {tokens['combo_selection_bg']};
    selection-color: {tokens['selection_text']};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QLineEdit:focus, QTextEdit:focus, QPushButton:focus {{
    border: 1px solid {tokens['primary_hi']};
}}
QTextEdit {{
    selection-background-color: {tokens['text_selection_bg']};
}}

/* ---- CheckBox ---- */
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border: 1px solid {tokens['checkbox_border']};
    border-radius: 5px;
    background-color: {tokens['input_bg']};
}}
QCheckBox::indicator:checked {{
    background-color: {tokens['primary']};
    border-color: {tokens['primary_hi']};
}}
QCheckBox::indicator:hover {{
    border-color: {tokens['primary_hi']};
}}

/* ---- Slider ---- */
QSlider::groove:horizontal {{
    border: 1px solid {tokens['slider_groove_border']};
    height: 6px;
    background: {tokens['slider_groove_bg']};
    border-radius: 3px;
    margin: 2px 0;
}}
QSlider::sub-page:horizontal {{
    background: {tokens['slider_fill']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {tokens['primary']};
    border: 3px solid {tokens['primary_hi']};
    width: 16px;
    margin: -7px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {tokens['primary_hi']};
}}

/* ---- Data Widgets ---- */
QTableWidget {{
    background-color: {tokens['table_bg']};
    alternate-background-color: {tokens['table_alt_bg']};
    gridline-color: {tokens['border']};
    border: 1px solid {tokens['border']};
    border-radius: 8px;
}}
QTableWidget::item {{
    padding: 4px 6px;
}}
QHeaderView::section {{
    background-color: {tokens['header_bg']};
    color: {tokens['header_text']};
    border: 1px solid {tokens['border']};
    padding: 6px;
    font-weight: 700;
}}
QTreeView {{
    background-color: {tokens['tree_bg']};
    border: 1px solid {tokens['border']};
    border-radius: 7px;
    alternate-background-color: {tokens['tree_alt_bg']};
}}
QTreeView::item {{
    padding: 4px 3px;
}}
QTreeView::item:selected {{
    background-color: {tokens['tree_selected_bg']};
    color: {tokens['tree_selected_text']};
}}
QProgressBar {{
    background-color: {tokens['progress_bg']};
    border: 1px solid {tokens['border']};
    border-radius: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {tokens['primary']};
    border-radius: 5px;
}}

/* ---- Scroll ---- */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {tokens['scrollbar']};
    min-height: 26px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {tokens['scrollbar']};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: {tokens['scrollbar']};
    min-width: 26px;
    border-radius: 5px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* ---- Splitter / Status ---- */
QSplitter::handle:horizontal {{
    background: {tokens['splitter_bg']};
    width: 6px;
    margin: 2px 0;
    border-left: 1px solid {tokens['splitter_border']};
    border-right: 1px solid {tokens['splitter_border']};
}}
QSplitter::handle:horizontal:hover {{
    background: {tokens['splitter_hover']};
}}
QStatusBar {{
    background-color: {tokens['status_bg']};
    border-top: 1px solid {tokens['border']};
    color: {tokens['status_text']};
    font-weight: 600;
}}
"""


# Exported application stylesheets.
STYLESHEET = _build_stylesheet(_build_theme_tokens(dark=True))
LIGHT_STYLESHEET = _build_stylesheet(_build_theme_tokens(dark=False))
