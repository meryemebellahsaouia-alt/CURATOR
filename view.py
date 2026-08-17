# view.py

import logging
from functools import partial
from typing import Dict
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PyQt6.QtGui import QAction, QActionGroup, QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QFrame, QGroupBox,  
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QTableWidget, QHeaderView, QTableWidgetItem,
                             QAbstractItemView,
                             QTabWidget, QScrollArea, QSlider, QApplication, QSplitter,
                             QVBoxLayout, QWidget, QStatusBar, QFormLayout, QLayout,
                             QComboBox, QPushButton, QTextEdit, QGraphicsOpacityEffect)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import styles
import config
from widgets import DVHPlotWidget, LoadingDialog



class MainWindow(QMainWindow):
    # --- Signals ---
    load_files_requested = pyqtSignal()
    structure_visibility_changed = pyqtSignal()
    dose_slider_changed = pyqtSignal(int)
    units_toggled = pyqtSignal(bool)
    export_dvh_data_requested = pyqtSignal()
    export_dvh_plot_requested = pyqtSignal()
    export_ct_requested = pyqtSignal()
    export_dose_requested = pyqtSignal()
    resample_and_export_mask_requested = pyqtSignal()
    export_screenshot_viewer_requested = pyqtSignal()
    machine_beam_slider_changed = pyqtSignal(int)
    machine_cp_slider_changed = pyqtSignal(int)
    machine_visibility_changed = pyqtSignal(str, bool)
    export_dvh_stats_requested = pyqtSignal()
    slice_slider_changed = pyqtSignal(int)
    window_preset_selected = pyqtSignal(str)
    reset_slice_view_requested = pyqtSignal()
    machine_play_toggled = pyqtSignal(bool)
    machine_stop_requested = pyqtSignal()
    machine_playback_speed_changed = pyqtSignal(int)
    machine_lock_beam_toggled = pyqtSignal(bool)
    structure_palette_toggled = pyqtSignal(bool)
    diagnostics_cleared_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        #self.setWindowTitle("CURATOR")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        self.structure_checkboxes: Dict[int, QCheckBox] = {}
        self.last_slice_status: str = ""
        self.ui_scale_factor: float = 1.0
        self.high_contrast_enabled: bool = False
        self._context_animations: Dict[int, QParallelAnimationGroup] = {}
        self.loading_dialog = LoadingDialog(self)
        
        self._setup_ui()
        self._connect_signals()

    # --- 1. Initialization & UI Setup ---

    def _setup_ui(self):
        """Sets up the main UI layout, menu bar, and status bar."""
        self._create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(styles.SPACING_BASE)
        main_layout.setContentsMargins(
            styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE
        )

        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([1120, 360])

        self.status_label = QLabel("Please load DICOM data to begin.")
        self.setStatusBar(QStatusBar(self))
        self.statusBar().addWidget(self.status_label)

        main_layout.addWidget(self.main_splitter)

    def _create_menu_bar(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu("&File")
        self.load_action = QAction("&Open Patient Folder", self)
        file_menu.addAction(self.load_action)
        file_menu.addSeparator()
        self.export_ct_action = QAction("Resample & Export &CT", self)
        self.export_dose_action = QAction("Resample & Export &Dose", self)
        self.resample_mask_action = QAction("Resample & Export Mask(s)", self)
        file_menu.addAction(self.export_ct_action)
        file_menu.addAction(self.export_dose_action)
        file_menu.addAction(self.resample_mask_action)
        
        view_menu = menu_bar.addMenu("&View")
        self.view_screenshot_action = QAction("3D View Screenshot", self)
        view_menu.addAction(self.view_screenshot_action)
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)  # Make it a toggle switch
        self.dark_mode_action.setChecked(True)   # Start in dark mode by default
        view_menu.addAction(self.dark_mode_action)
        self.high_contrast_action = QAction("High Contrast", self)
        self.high_contrast_action.setCheckable(True)
        view_menu.addAction(self.high_contrast_action)
        self.colorblind_palette_action = QAction("Colorblind-safe Structure Colors", self)
        self.colorblind_palette_action.setCheckable(True)
        view_menu.addAction(self.colorblind_palette_action)

        scale_menu = view_menu.addMenu("UI Scale")
        self.ui_scale_action_group = QActionGroup(self)
        self.ui_scale_action_group.setExclusive(True)
        self.ui_scale_actions = {}
        for label, scale in [("100%", 1.0), ("110%", 1.1), ("125%", 1.25)]:
            action = QAction(label, self)
            action.setCheckable(True)
            if scale == 1.0:
                action.setChecked(True)
            scale_menu.addAction(action)
            self.ui_scale_action_group.addAction(action)
            self.ui_scale_actions[action] = scale

        view_menu.addSeparator()
        self.clear_diagnostics_action = QAction("Clear Diagnostics", self)
        view_menu.addAction(self.clear_diagnostics_action)
   
        
        self.export_dvh_data_action = QAction("Export DVH &Data (CSV)", self)
        self.export_dvh_plot_action = QAction("Export DVH &Plot (PNG)", self)
        self.export_dvh_stats_action = QAction("Export DVH &Stats (CSV)...", self)
        self.export_dvh_data_action.setEnabled(False)
        self.export_dvh_plot_action.setEnabled(False)
        self.export_dvh_stats_action.setEnabled(False)




    def _create_left_panel(self) -> QWidget:
        """Creates the left panel containing the tabbed Axial and DVH viewers."""
        self.tab_widget = QTabWidget()
        
        # Axial Viewer Tab
        vtk_frame = QFrame()
        vtk_frame.setStyleSheet(f"border: 1px solid {styles.COLOR_WIDGET_BORDER};")
        self.vtkWidget = QVTKRenderWindowInteractor(vtk_frame)
        vtk_layout = QVBoxLayout(vtk_frame)
        vtk_layout.setContentsMargins(0,0,0,0)
        vtk_layout.addWidget(self.vtkWidget)
        
        # 3D Viewer Tab
        vtk_3d_frame = QFrame()
        vtk_3d_frame.setStyleSheet(f"border: 1px solid {styles.COLOR_WIDGET_BORDER};")
        self.vtk_3d_Widget = QVTKRenderWindowInteractor(vtk_3d_frame)
        vtk_3d_layout = QVBoxLayout(vtk_3d_frame)
        vtk_3d_layout.setContentsMargins(0,0,0,0)
        vtk_3d_layout.addWidget(self.vtk_3d_Widget)
        
        # DVH Analysis Tab
        dvh_widget = QWidget()
        dvh_layout = QVBoxLayout(dvh_widget)
        self.dvh_plot_widget = DVHPlotWidget()
        self._setup_dvh_stats_table()
        dvh_layout.addWidget(self.dvh_plot_widget)
        dvh_layout.addWidget(self.dvh_stats_table)

        diagnostics_widget = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_widget)
        self.diagnostics_text = QTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setPlaceholderText("Diagnostics messages will appear here.")
        diagnostics_layout.addWidget(self.diagnostics_text)
        
        self.tab_widget.addTab(vtk_frame, "Axial Viewer")
        self.tab_widget.addTab(vtk_3d_frame, "3D Viewer")
        self.tab_widget.addTab(dvh_widget, "DVH Analysis")
        self.tab_widget.addTab(diagnostics_widget, "Diagnostics")
        return self.tab_widget
    
    def _setup_dvh_stats_table(self):
        """Configures the properties and headers of the DVH stats table."""
        self.dvh_stats_table = QTableWidget(0, 9)
        self.dvh_stats_table.setHorizontalHeaderLabels([
            "Structure", "Min Dose (Gy)", "Mean Dose (Gy)", "Max Dose (Gy)",
            "D95 (Gy)", "V20 (%)", "V30 (%)", "HI", "CI"
        ])
        self.dvh_stats_table.horizontalHeaderItem(4).setToolTip("D95 = Dose delivered to 95% of the structure volume")
        self.dvh_stats_table.horizontalHeaderItem(5).setToolTip("V20 = Volume percentage receiving at least 20 Gy")
        self.dvh_stats_table.horizontalHeaderItem(6).setToolTip("V30 = Volume percentage receiving at least 30 Gy")
        self.dvh_stats_table.horizontalHeaderItem(7).setToolTip("Homogeneity Index = (D2% - D98%) / D50%")
        self.dvh_stats_table.horizontalHeaderItem(8).setToolTip("Conformity Index = (TV_PIV)^2 / (TV * PIV)")
        self.dvh_stats_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.dvh_stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dvh_stats_table.horizontalHeader().setHighlightSections(False)
        self.dvh_stats_table.verticalHeader().setVisible(False)
        self.dvh_stats_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.dvh_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dvh_stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.dvh_stats_table.setColumnWidth(0, 150)
        self._resize_dvh_stats_table_height()

    def _create_right_panel(self) -> QWidget:
        """Creates the main right-side control panel by assembling group boxes."""
        controls_host = QWidget()
        controls_host.setMinimumWidth(300)

        controls_content = QWidget()
        controls_layout = QVBoxLayout(controls_content)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(styles.SPACING_BASE * 2)
        controls_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        
        self.navigation_group = self._create_slice_navigation_group()
        self.structures_group = self._create_structures_group()
        self.dose_group = self._create_dose_display_group()
        self.machine_controls_group = self._create_machine_view_controls_group()
        self.dvh_export_group = self._create_dvh_export_group()
        
        controls_layout.addWidget(self.navigation_group)
        controls_layout.addWidget(self.structures_group, 1) # Allow vertical stretch
        controls_layout.addWidget(self.dose_group)
        controls_layout.addWidget(self.machine_controls_group)
        controls_layout.addWidget(self.dvh_export_group)
        controls_layout.addStretch()

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls_content)

        host_layout = QVBoxLayout(controls_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addWidget(controls_scroll)

        self.set_controls_enabled(False)
        self.set_controls_context(self.tab_widget.currentIndex(), animate=False)
        return controls_host

    def _resize_dvh_stats_table_height(self):
        """Resizes the DVH stats table height to fit the current number of rows."""
        self.dvh_stats_table.resizeRowsToContents()
        header_height = self.dvh_stats_table.horizontalHeader().height()
        rows_height = sum(self.dvh_stats_table.rowHeight(row) for row in range(self.dvh_stats_table.rowCount()))
        frame_height = self.dvh_stats_table.frameWidth() * 2
        target_height = header_height + rows_height + frame_height + 4
        min_height = header_height + frame_height + 6
        max_height = 280
        target_height = min(max(target_height, min_height), max_height)
        self.dvh_stats_table.setMinimumHeight(target_height)
        self.dvh_stats_table.setMaximumHeight(target_height)

    def _create_slice_navigation_group(self) -> QGroupBox:
        """Creates controls for 2D slice navigation and CT presets."""
        group = QGroupBox("2D Navigation")
        layout = QFormLayout(group)
        layout.setSpacing(styles.SPACING_BASE)
        layout.setContentsMargins(styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.slice_position_label = QLabel("Slice: N/A")
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setTracking(False)
        self.slice_slider.setEnabled(False)

        self.window_preset_combo = QComboBox()
        self.window_preset_combo.addItems(["Auto", "Soft Tissue", "Lung", "Bone"])
        self.window_preset_combo.setEnabled(False)

        self.reset_slice_view_button = QPushButton("Reset Camera")
        self.reset_slice_view_button.setMinimumHeight(34)
        self.reset_slice_view_button.setEnabled(False)

        shortcuts_label = QLabel("Keys: J/K, Arrow keys, PageUp/PageDown")
        shortcuts_label.setObjectName("HintLabel")
        shortcuts_label.setWordWrap(True)

        layout.addRow(self.slice_position_label)
        layout.addRow("Slice:", self.slice_slider)
        layout.addRow("CT Preset:", self.window_preset_combo)
        layout.addRow(self.reset_slice_view_button)
        layout.addRow(shortcuts_label)
        return group

    def _create_structures_group(self) -> QGroupBox:
        """Creates the 'Structures' group box with a scrollable list."""
        group = QGroupBox("Structures")
        group.setObjectName("AlignTop")
        layout = QVBoxLayout(group)
        self.structures_container = QWidget()
        self.structures_container.setStyleSheet("background-color: transparent;")
        self.structures_layout = QVBoxLayout(self.structures_container)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.structures_container)
        layout.addWidget(scroll_area)
        return group
    
    def _create_machine_view_controls_group(self) -> QGroupBox:
        """Creates the group box for 3D view sliders."""
        group = QGroupBox("3D View Controls")
        layout = QFormLayout(group)
        layout.setSpacing(styles.SPACING_BASE)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        self.machine_beam_label = QLabel("Beam: N/A")
        self.machine_beam_slider = QSlider(Qt.Orientation.Horizontal)
        
        self.machine_cp_label = QLabel("Control Point: N/A")
        self.machine_cp_slider = QSlider(Qt.Orientation.Horizontal)
        self.gantry_angle_label = QLabel("Gantry Angle: N/A")
        self.playback_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_speed_slider.setRange(1, 10)
        self.playback_speed_slider.setValue(5)
        self.playback_speed_label = QLabel("Playback Speed: 5")
        self.machine_play_button = QPushButton("Play")
        self.machine_play_button.setCheckable(True)
        self.machine_play_button.setMinimumHeight(34)
        self.machine_stop_button = QPushButton("Stop")
        self.machine_stop_button.setMinimumHeight(34)
        self.machine_lock_beam_checkbox = QCheckBox("Lock Beam During Playback")
        self.machine_lock_beam_checkbox.setChecked(True)
        
        for slider in [self.machine_beam_slider, self.machine_cp_slider]:
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setTracking(False)
            slider.setEnabled(False)
        self.playback_speed_slider.setEnabled(False)
        self.machine_play_button.setEnabled(False)
        self.machine_stop_button.setEnabled(False)
        self.machine_lock_beam_checkbox.setEnabled(False)

        layout.addRow(self.machine_beam_label)
        layout.addRow(self.machine_beam_slider)
        layout.addRow(self.machine_cp_label)
        layout.addRow(self.machine_cp_slider)
        layout.addRow(self.gantry_angle_label)
        layout.addRow(self.playback_speed_label)
        layout.addRow(self.playback_speed_slider)

        play_controls = QWidget()
        play_controls_layout = QHBoxLayout(play_controls)
        play_controls_layout.setContentsMargins(0, 0, 0, 0)
        play_controls_layout.setSpacing(styles.SPACING_BASE)
        play_controls_layout.addWidget(self.machine_play_button)
        play_controls_layout.addWidget(self.machine_stop_button)
        layout.addRow(play_controls)
        layout.addRow(self.machine_lock_beam_checkbox)
        
        visibility_widget = QWidget()
        visibility_layout = QHBoxLayout(visibility_widget)
        visibility_layout.setContentsMargins(0, 0, 0, 0)
        visibility_layout.setSpacing(styles.SPACING_BASE * 2)
        
        self.mlc_checkbox = QCheckBox("MLC")
        self.ct_3d_checkbox = QCheckBox("CT")
        self.jaws_checkbox = QCheckBox("Jaws")

        for checkbox in [self.mlc_checkbox, self.ct_3d_checkbox, self.jaws_checkbox]:
            checkbox.setChecked(True) # Checked by default
            visibility_layout.addWidget(checkbox)
            
        layout.addRow("Visible:", visibility_widget)
        return group
    
    def _create_dose_display_group(self) -> QGroupBox:
        """Creates the 'Dose Display' group box."""
        group = QGroupBox("Dose Display")
        layout = QFormLayout(group)
        layout.setSpacing(styles.SPACING_BASE)
        layout.setContentsMargins(styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE)

        self.prescription_label = QLabel("Prescription Dose: N/A")
        self.unit_toggle_checkbox = QCheckBox("Show as % of Prescription")
        self.dose_slider = QSlider(Qt.Orientation.Horizontal)
        self.dose_slider.setRange(0, 100)
        self.dose_slider.setTracking(False)
        self.dose_label = QLabel("Min: N/A")
        
        layout.addRow(self.prescription_label)
        layout.addRow(self.unit_toggle_checkbox)
        layout.addRow("Dose Threshold:", self.dose_slider)
        layout.addRow(self.dose_label)
        
        return group

    def _create_dvh_export_group(self) -> QGroupBox:
        """Creates a compact DVH export group for the right panel."""
        group = QGroupBox("DVH Exports")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE, styles.MARGIN_BASE)
        layout.setSpacing(styles.SPACING_BASE)

        self.export_dvh_data_button = QPushButton("Export Data (CSV)")
        self.export_dvh_plot_button = QPushButton("Export Plot (PNG)")
        self.export_dvh_stats_button = QPushButton("Export Stats (CSV)")
        for button in (self.export_dvh_data_button, self.export_dvh_plot_button, self.export_dvh_stats_button):
            button.setEnabled(False)
            button.setMinimumHeight(32)
            layout.addWidget(button)
        return group

    def show_loading_dialog(self):
        """Clears messages and shows the loading dialog."""
        self.loading_dialog.reset_state()
        self.loading_dialog.show()

    def hide_loading_dialog(self):
        """Hides the loading dialog."""
        self.loading_dialog.hide()

    def update_loading_progress(self, percentage: int, message: str):
        """Updates the loading dialog's progress bar and message list."""
        self.loading_dialog.update_progress(percentage, message)
        
    
    # --- 2. Signal Connections & Event Handlers ---

    def _connect_signals(self):
        """Connects widget signals to the main window's public signals or internal slots."""
        # Menu Actions
        self.load_action.triggered.connect(self.load_files_requested.emit)
        self.export_dvh_data_action.triggered.connect(self.export_dvh_data_requested.emit)
        self.export_dvh_plot_action.triggered.connect(self.export_dvh_plot_requested.emit)
        self.export_dvh_stats_action.triggered.connect(self.export_dvh_stats_requested.emit)
        self.view_screenshot_action.triggered.connect(self.export_screenshot_viewer_requested.emit)
        self.dark_mode_action.toggled.connect(self._toggle_dark_mode)
        self.high_contrast_action.toggled.connect(self._toggle_high_contrast)
        self.colorblind_palette_action.toggled.connect(self.structure_palette_toggled.emit)
        self.clear_diagnostics_action.triggered.connect(self.clear_diagnostics)
        self.clear_diagnostics_action.triggered.connect(self.diagnostics_cleared_requested.emit)
        for scale_action, scale_value in self.ui_scale_actions.items():
            scale_action.triggered.connect(partial(self._set_ui_scale, scale_value))
        self.tab_widget.currentChanged.connect(self.set_controls_context)
        
        # Connect export actions to a single helper slot using partial
        self.export_ct_action.triggered.connect(self.export_ct_requested.emit)
        self.export_dose_action.triggered.connect(self.export_dose_requested.emit)
        self.resample_mask_action.triggered.connect(self.resample_and_export_mask_requested.emit)
        self.export_dvh_data_button.clicked.connect(self.export_dvh_data_requested.emit)
        self.export_dvh_plot_button.clicked.connect(self.export_dvh_plot_requested.emit)
        self.export_dvh_stats_button.clicked.connect(self.export_dvh_stats_requested.emit)
        
        # Right Panel Widgets
        self.dose_slider.valueChanged.connect(self.dose_slider_changed.emit)
        self.unit_toggle_checkbox.toggled.connect(self.units_toggled.emit)
        self.slice_slider.valueChanged.connect(self.slice_slider_changed.emit)
        self.window_preset_combo.currentTextChanged.connect(self.window_preset_selected.emit)
        self.reset_slice_view_button.clicked.connect(self.reset_slice_view_requested.emit)
        self.machine_beam_slider.valueChanged.connect(self.machine_beam_slider_changed.emit)
        self.machine_cp_slider.valueChanged.connect(self.machine_cp_slider_changed.emit)
        self.playback_speed_slider.valueChanged.connect(self._on_playback_speed_changed)
        self.machine_play_button.toggled.connect(self._on_machine_play_toggled_ui)
        self.machine_stop_button.clicked.connect(self.machine_stop_requested.emit)
        self.machine_lock_beam_checkbox.toggled.connect(self.machine_lock_beam_toggled.emit)
        self.mlc_checkbox.toggled.connect(partial(self.machine_visibility_changed.emit, "MLC"))
        self.ct_3d_checkbox.toggled.connect(partial(self.machine_visibility_changed.emit, "CT"))
        self.jaws_checkbox.toggled.connect(partial(self.machine_visibility_changed.emit, "Jaws"))

    def _toggle_dark_mode(self, is_checked: bool):
        """Applies or removes the dark stylesheet."""
        self.apply_theme(dark=is_checked)

    def _toggle_high_contrast(self, is_checked: bool):
        """Enables or disables high-contrast overrides."""
        self.high_contrast_enabled = is_checked
        self.apply_theme(dark=self.dark_mode_action.isChecked())

    def _set_ui_scale(self, scale_factor: float):
        """Updates UI font scaling and reapplies the active theme."""
        self.ui_scale_factor = scale_factor
        self.apply_theme(dark=self.dark_mode_action.isChecked())

    def _on_playback_speed_changed(self, speed_value: int):
        """Updates the speed label and emits the playback speed signal."""
        self.playback_speed_label.setText(f"Playback Speed: {speed_value}")
        self.machine_playback_speed_changed.emit(speed_value)

    def _on_machine_play_toggled_ui(self, is_playing: bool):
        """Updates play button text and emits playback toggle signal."""
        self.machine_play_button.setText("Pause" if is_playing else "Play")
        self.machine_play_toggled.emit(is_playing)

    # --- 3. Public API / Slots (called by Controller) ---
    
    def set_controls_enabled(self, enabled: bool):
        """Enables or disables all data-dependent controls."""
        # Menu Actions
        self.export_ct_action.setEnabled(enabled)
        self.export_dose_action.setEnabled(enabled)
        self.resample_mask_action.setEnabled(enabled)
        self.view_screenshot_action.setEnabled(enabled)

        # Right Panel Widgets
        self.unit_toggle_checkbox.setEnabled(enabled)
        self.dose_slider.setEnabled(enabled)
        self.window_preset_combo.setEnabled(enabled)
        self.reset_slice_view_button.setEnabled(enabled)
        self.machine_lock_beam_checkbox.setEnabled(enabled)
        self.mlc_checkbox.setEnabled(enabled)
        self.ct_3d_checkbox.setEnabled(enabled)
        self.jaws_checkbox.setEnabled(enabled)

    def set_slice_controls_enabled(self, enabled: bool):
        """Enables or disables 2D slice slider control."""
        self.slice_slider.setEnabled(enabled)

    def set_machine_play_controls_enabled(self, enabled: bool):
        """Enables or disables machine playback controls."""
        self.playback_speed_slider.setEnabled(enabled)
        self.machine_play_button.setEnabled(enabled)
        self.machine_stop_button.setEnabled(enabled)

    def set_machine_navigation_controls_enabled(self, enabled: bool):
        """Enables or disables 3D machine navigation widgets."""
        self.machine_beam_slider.setEnabled(enabled)
        self.machine_cp_slider.setEnabled(enabled)
        self.machine_lock_beam_checkbox.setEnabled(enabled)
        self.mlc_checkbox.setEnabled(enabled)
        self.ct_3d_checkbox.setEnabled(enabled)
        self.jaws_checkbox.setEnabled(enabled)

    def set_dvh_export_enabled(self, enabled: bool):
        """Enables or disables the DVH-specific export menu actions."""
        self.export_dvh_data_action.setEnabled(enabled)
        self.export_dvh_plot_action.setEnabled(enabled)
        self.export_dvh_stats_action.setEnabled(enabled)
        self.export_dvh_data_button.setEnabled(enabled)
        self.export_dvh_plot_button.setEnabled(enabled)
        self.export_dvh_stats_button.setEnabled(enabled)

    def _ensure_group_opacity_effect(self, group: QWidget) -> QGraphicsOpacityEffect:
        """Ensures each animated group has an opacity effect instance."""
        effect = group.graphicsEffect()
        if isinstance(effect, QGraphicsOpacityEffect):
            return effect
        new_effect = QGraphicsOpacityEffect(group)
        new_effect.setOpacity(1.0)
        group.setGraphicsEffect(new_effect)
        return new_effect

    def _set_group_visibility_immediate(self, group: QWidget, visible: bool):
        """Applies visibility without animation (used at startup and fallbacks)."""
        effect = self._ensure_group_opacity_effect(group)
        if visible:
            group.setVisible(True)
            group.setMaximumHeight(16777215)
            effect.setOpacity(1.0)
        else:
            group.setMaximumHeight(0)
            effect.setOpacity(0.0)
            group.setVisible(False)

    def _animate_group_visibility(self, group: QWidget, visible: bool):
        """Animates control group visibility using height and opacity."""
        effect = self._ensure_group_opacity_effect(group)
        current_animation = self._context_animations.get(id(group))
        if current_animation:
            current_animation.stop()

        is_currently_visible = group.isVisible() and group.maximumHeight() != 0
        if is_currently_visible == visible:
            return

        target_height = max(group.sizeHint().height(), 1)
        start_height = group.height() if group.height() > 0 else (target_height if is_currently_visible else 0)
        end_height = target_height if visible else 0
        start_opacity = effect.opacity()
        end_opacity = 1.0 if visible else 0.0

        if visible:
            group.setVisible(True)

        height_anim = QPropertyAnimation(group, b"maximumHeight", self)
        height_anim.setDuration(170)
        height_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        height_anim.setStartValue(start_height)
        height_anim.setEndValue(end_height)

        opacity_anim = QPropertyAnimation(effect, b"opacity", self)
        opacity_anim.setDuration(140)
        opacity_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        opacity_anim.setStartValue(start_opacity)
        opacity_anim.setEndValue(end_opacity)

        group_anim = QParallelAnimationGroup(self)
        group_anim.addAnimation(height_anim)
        group_anim.addAnimation(opacity_anim)

        def _on_finished():
            if visible:
                group.setMaximumHeight(16777215)
                effect.setOpacity(1.0)
            else:
                group.setMaximumHeight(0)
                effect.setOpacity(0.0)
                group.setVisible(False)
            self._context_animations.pop(id(group), None)

        group_anim.finished.connect(_on_finished)
        self._context_animations[id(group)] = group_anim
        group_anim.start()

    def set_controls_context(self, tab_index: int, animate: bool = True):
        """Shows only context-relevant control groups for the current tab."""
        show_axial = tab_index == 0
        show_3d = tab_index == 1
        show_dvh = tab_index == 2
        show_structures = tab_index in (0, 1, 2)

        visibility_targets = (
            (self.navigation_group, show_axial),
            (self.dose_group, show_axial),
            (self.machine_controls_group, show_3d),
            (self.dvh_export_group, show_dvh),
            (self.structures_group, show_structures),
        )

        use_animation = animate and self.isVisible()
        for group, should_show in visibility_targets:
            if use_animation:
                self._animate_group_visibility(group, should_show)
            else:
                self._set_group_visibility_immediate(group, should_show)

    def populate_structures_legend(self, structures: Dict):
        """Clears and re-populates the structures list in the right panel."""
        checked_state = {roi_number: checkbox.isChecked() for roi_number, checkbox in self.structure_checkboxes.items()}
        # This part for clearing old widgets remains the same
        while self.structures_layout.count():
            child = self.structures_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.structure_checkboxes.clear()

        # This loop is where the changes are
        for roi_number, info in sorted(structures.items()):
            item_widget = QWidget()
            item_widget.setStyleSheet("background-color: transparent;")
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 3, 5, 3)
            item_layout.setSpacing(10) # Added spacing for a cleaner look
            
            # 1. Create a checkbox with NO text
            checkbox = QCheckBox()
            checkbox.setChecked(checked_state.get(roi_number, False))
            checkbox.stateChanged.connect(self.structure_visibility_changed.emit)
            
            # 2. Create the color box (same as before)
            color_box = QLabel()
            color_box.setFixedSize(16, 16)
            rgb = [int(c * 255) for c in info['color']]
            color_box.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #555;")
            
            # 3. Create a separate label for the name
            name_label = QLabel(info['name'])
            name_label.setToolTip(info['name']) # Good for long names that might get cut off
            
            # 4. Add the widgets in the new order
            item_layout.addWidget(checkbox)
            item_layout.addWidget(color_box)
            item_layout.addWidget(name_label, 1) # The '1' makes the label stretch to fill space

            self.structures_layout.addWidget(item_widget)
            self.structure_checkboxes[roi_number] = checkbox
            
        self.structures_layout.addStretch()
        
    def apply_theme(self, dark: bool):
        """Applies the selected theme to the entire application."""
        app = QApplication.instance()
        stylesheet = self._build_effective_stylesheet(dark)
        if dark:
            app.setStyleSheet(stylesheet)
            bg_top = config.COLORS.GetColor3d(config.BACKGROUND_COLOR_TOP)
            bg_bottom = config.COLORS.GetColor3d(config.BACKGROUND_COLOR_BOTTOM)
        else:
            app.setStyleSheet(stylesheet)
            bg_top = config.COLORS.GetColor3d("LightSteelBlue")
            bg_bottom = config.COLORS.GetColor3d("White")

        # This part for custom widgets remains the same
        self.dvh_plot_widget.set_theme(dark)
        
        renderer_2d = self.vtkWidget.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if renderer_2d:
            renderer_2d.SetBackground(bg_bottom)
            renderer_2d.SetBackground2(bg_top)
            self.vtkWidget.GetRenderWindow().Render()

        renderer_3d = self.vtk_3d_Widget.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if renderer_3d:
            renderer_3d.SetBackground(bg_bottom)
            renderer_3d.SetBackground2(bg_top)
            self.vtk_3d_Widget.GetRenderWindow().Render()

    def _build_effective_stylesheet(self, dark: bool) -> str:
        """Builds the active stylesheet with UI scaling and optional contrast overrides."""
        base = styles.STYLESHEET if dark else styles.LIGHT_STYLESHEET
        normal_size = max(10, int(14 * self.ui_scale_factor))
        header_size = max(12, int(16 * self.ui_scale_factor))
        overrides = [
            f"QMainWindow, QWidget {{ font-size: {normal_size}px; }}",
            f"QGroupBox::title, QTabBar::tab, QLabel#LoadingTitle {{ font-size: {header_size}px; }}",
        ]
        if self.high_contrast_enabled:
            if dark:
                overrides.append(
                    "QWidget { color: #F5F7FA; }"
                    "QGroupBox, QTableWidget, QTreeView, QMenu, QMenuBar { border-color: #8EA6BF; }"
                    "QPushButton, QSlider::handle:horizontal, QCheckBox::indicator:checked { background-color: #0BA6D8; }"
                )
            else:
                overrides.append(
                    "QWidget { color: #111111; }"
                    "QGroupBox, QTableWidget, QTreeView, QMenu, QMenuBar { border-color: #5F6F7F; }"
                    "QPushButton, QSlider::handle:horizontal, QCheckBox::indicator:checked { background-color: #006DAA; color: #FFFFFF; }"
                )
        return base + "\n" + "\n".join(overrides)
            
    def _shutdown_vtk_widget(self, widget_name: str):
        """Helper method to safely shut down a single VTK widget."""
        if hasattr(self, widget_name) and getattr(self, widget_name):
            widget = getattr(self, widget_name)
            render_window = widget.GetRenderWindow()
            interactor = render_window.GetInteractor()
            
            
            # Finalize graphics resources first
            if render_window:
                if interactor and interactor.GetInitialized():
                    interactor.TerminateApp()
                render_window.Finalize()
                
            widget.Finalize()
            # Safely schedule the Qt widget for deletion
            widget.deleteLater()
            
            # Break the Python reference to it
            setattr(self, widget_name, None)
        
    def closeEvent(self, event):
        """Handles the window close event by calling the shutdown helper for each VTK widget."""
        try:
            self._shutdown_vtk_widget('vtkWidget')
            self._shutdown_vtk_widget('vtk_3d_Widget')
        except Exception as e:
            # Final safety net
            logging.warning("Ignoring non-critical VTK shutdown error: %s", e)

        event.accept()
        
    def update_status_label(self, text: str):
        """Sets the text of the main status bar."""
        self.status_label.setText(text)

    def update_prescription_label(self, text: str):
        """Sets the text of the prescription dose label."""
        self.prescription_label.setText(text)
        
    def update_dose_display_label(self, text: str):
        """Sets the text of the dose threshold label."""
        self.dose_label.setText(text)

    def update_dose_slider_range(self, max_value: int):
        """Updates the maximum value of the dose slider."""
        self.dose_slider.setRange(0, max_value)

    def update_slice_navigation(self, current_slice: int, max_slice: int):
        """Synchronizes slice slider and label with the active displayed slice."""
        self.slice_position_label.setText(f"Slice: {current_slice + 1} / {max_slice + 1}")
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max_slice)
        self.slice_slider.setValue(current_slice)
        self.slice_slider.blockSignals(False)

    def update_machine_play_state(self, is_playing: bool):
        """Updates play button state without re-emitting playback signals."""
        self.machine_play_button.blockSignals(True)
        self.machine_play_button.setChecked(is_playing)
        self.machine_play_button.setText("Pause" if is_playing else "Play")
        self.machine_play_button.blockSignals(False)

    def append_diagnostic(self, level: str, message: str):
        """Appends one diagnostics line with lightweight truncation."""
        line = f"[{level}] {message}"
        self.diagnostics_text.append(line)
        self._truncate_diagnostics(max_lines=500)

    def clear_diagnostics(self):
        """Clears all diagnostics messages."""
        self.diagnostics_text.clear()

    def _truncate_diagnostics(self, max_lines: int = 500):
        """Keeps diagnostics output bounded for long sessions."""
        document = self.diagnostics_text.document()
        if document.blockCount() <= max_lines:
            return
        cursor = self.diagnostics_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        for _ in range(document.blockCount() - max_lines):
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def populate_dvh_stats_table(self, dvh_results: list):
        """Clears and populates the stats table with data for multiple structures."""
        self.dvh_stats_table.setRowCount(0)
        if not dvh_results:
            self._resize_dvh_stats_table_height()
            return

        self.dvh_stats_table.setRowCount(len(dvh_results))
        for row, result in enumerate(dvh_results):
            name_item = QTableWidgetItem(result['name'])
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter| Qt.AlignmentFlag.AlignVCenter)
            self.dvh_stats_table.setItem(row, 0, name_item)

            # --- Columns 1-5: Stats ---
            stats = result['dvh_data']
            ci_text = f"{result['ci_value']:.2f}" if result['ci_value'] is not None else "N/A"
            values = [
                f"{stats['min_dose']:.2f}",
                f"{stats['mean_dose']:.2f}",
                f"{stats['max_dose']:.2f}",
                f"{stats.get('d95', 0.0):.2f}",
                f"{stats.get('v20', 0.0):.2f}",
                f"{stats.get('v30', 0.0):.2f}",
                f"{stats['hi_value']:.2f}",
                ci_text
            ]
            for col, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                self.dvh_stats_table.setItem(row, col, item)
        self._resize_dvh_stats_table_height()

    def show_error_message(self, title: str, text: str):
        """Displays a critical error message box."""
        QMessageBox.critical(self, title, text)
