import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal, QSortFilterProxyModel
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

import styles


class ParametersDialog(QDialog):
    """A dialog for editing application parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Parameters")

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.jaws_resolution_spinbox = QDoubleSpinBox()
        self.jaws_resolution_spinbox.setRange(1.0, 100.0)

        self.leaf_width_spinbox = QDoubleSpinBox()
        self.leaf_width_spinbox.setRange(1.0, 100.0)

        form_layout.addRow("Jaws Resolution:", self.jaws_resolution_spinbox)
        form_layout.addRow("Leaf Width:", self.leaf_width_spinbox)

        main_layout.addLayout(form_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def get_values(self) -> dict:
        """Returns the current values from the dialog's widgets."""
        return {
            "jaws_resolution": self.jaws_resolution_spinbox.value(),
            "leaf_width": self.leaf_width_spinbox.value(),
        }


class ExportDialog(QDialog):
    """A dialog for setting resampling parameters before export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resample & Export Options")

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.spacing_x_spinbox = QDoubleSpinBox()
        self.spacing_y_spinbox = QDoubleSpinBox()
        self.spacing_z_spinbox = QDoubleSpinBox()
        for spinbox in [self.spacing_x_spinbox, self.spacing_y_spinbox, self.spacing_z_spinbox]:
            spinbox.setRange(0.1, 10.0)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(2)

        spacing_container = self._create_horizontal_container(
            [
                QLabel("X:"),
                self.spacing_x_spinbox,
                QLabel("Y:"),
                self.spacing_y_spinbox,
                QLabel("Z:"),
                self.spacing_z_spinbox,
            ]
        )
        form_layout.addRow("New Spacing (mm):", spacing_container)

        self.shape_x_spinbox = QSpinBox()
        self.shape_y_spinbox = QSpinBox()
        self.shape_z_spinbox = QSpinBox()
        for spinbox in [self.shape_x_spinbox, self.shape_y_spinbox, self.shape_z_spinbox]:
            spinbox.setRange(0, 4096)
            spinbox.setToolTip("Target dimension. 0 = auto.")

        shape_container = self._create_horizontal_container(
            [
                QLabel("X:"),
                self.shape_x_spinbox,
                QLabel("Y:"),
                self.shape_y_spinbox,
                QLabel("Z:"),
                self.shape_z_spinbox,
            ]
        )
        form_layout.addRow("New Shape (voxels):", shape_container)

        main_layout.addLayout(form_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _create_horizontal_container(self, widgets: list) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        for widget in widgets:
            layout.addWidget(widget)
        return container

    def set_values(self, spacing: tuple, shape: tuple):
        """Sets the initial values of the spinboxes."""
        self.spacing_x_spinbox.setValue(spacing[0])
        self.spacing_y_spinbox.setValue(spacing[1])
        self.spacing_z_spinbox.setValue(spacing[2])
        self.shape_x_spinbox.setValue(shape[0])
        self.shape_y_spinbox.setValue(shape[1])
        self.shape_z_spinbox.setValue(shape[2])

    def get_values(self) -> tuple:
        """Returns the current values as (spacing_tuple, shape_tuple)."""
        spacing = (
            self.spacing_x_spinbox.value(),
            self.spacing_y_spinbox.value(),
            self.spacing_z_spinbox.value(),
        )
        shape = (
            self.shape_x_spinbox.value(),
            self.shape_y_spinbox.value(),
            self.shape_z_spinbox.value(),
        )
        return spacing, shape


class LoadingDialog(QDialog):
    """A modal dialog that shows progress and allows cancelling the load."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setMinimumSize(420, 170)
        self.setModal(True)
        self._cancel_requested = False

        layout = QVBoxLayout(self)

        title = QLabel("Loading Patient Data...")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("LoadingTitle")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)

        self.message_label = QLabel("Initializing...")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._request_cancel)
        button_row.addWidget(self.cancel_button)

        layout.addWidget(title)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.message_label)
        layout.addLayout(button_row)

    def reset_state(self):
        """Resets the dialog to its default state before a new loading run."""
        self._cancel_requested = False
        self.progress_bar.setValue(0)
        self.message_label.setText("Initializing...")
        self.cancel_button.setEnabled(True)

    def _request_cancel(self):
        """Sends one cancel request and disables the cancel button."""
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.message_label.setText("Cancelling...")
        self.cancel_requested.emit()

    def set_cancel_pending(self):
        """Marks the dialog as cancellation pending."""
        self._request_cancel()

    def update_progress(self, percentage: int, message: str):
        """Updates the progress bar and message if cancellation is not pending."""
        if self._cancel_requested:
            return
        self.progress_bar.setValue(percentage)
        self.message_label.setText(message)


class CompatibilityCheckDialog(QDialog):
    """Shows a pre-load compatibility summary for the selected folder."""

    def __init__(self, audit: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Folder Compatibility Check")
        self.setMinimumSize(760, 420)

        layout = QVBoxLayout(self)
        title = QLabel("Pre-load Compatibility Check")
        title.setObjectName("LoadingTitle")
        layout.addWidget(title)

        chain_count = int(audit.get("complete_dataset_count", 0))
        readable = int(audit.get("readable_dicom_files", 0))
        total = int(audit.get("total_files", 0))
        summary_text = (
            f"Readable DICOM files: {readable} / {total}. "
            f"Linkable CT->RTSTRUCT->RTPLAN->RTDOSE datasets found: {chain_count}."
        )
        self.summary_label = QLabel(summary_text)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Check", "Status", "Details"])
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.results_table, 1)

        self._populate_checks(audit)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _add_result_row(self, check: str, status: str, details: str):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        for col, text in enumerate([check, status, details]):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(row, col, item)

    def _populate_checks(self, audit: dict):
        modality_counts = audit.get("modality_counts", {})
        required_modalities = audit.get(
            "required_modalities", ["CT", "RTSTRUCT", "RTPLAN", "RTDOSE"]
        )
        missing = set(audit.get("missing_modalities", []))

        self._add_result_row(
            "Readable DICOM files",
            "PASS" if audit.get("readable_dicom_files", 0) > 0 else "FAIL",
            f"{audit.get('readable_dicom_files', 0)} / {audit.get('total_files', 0)}",
        )

        for modality in required_modalities:
            count = int(modality_counts.get(modality, 0))
            status = "PASS" if modality not in missing and count > 0 else "FAIL"
            self._add_result_row(f"Contains modality: {modality}", status, f"{count} file(s)")

        chain_count = int(audit.get("complete_dataset_count", 0))
        chain_status = "PASS" if chain_count > 0 else "FAIL"
        self._add_result_row(
            "Linkable CT->RTSTRUCT->RTPLAN->RTDOSE chain",
            chain_status,
            f"{chain_count} complete dataset(s)",
        )


class TreeFilterProxyModel(QSortFilterProxyModel):
    """Recursive filter for tree content across all visible columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setRecursiveFilteringEnabled(True)

    def filterAcceptsRow(self, source_row: int, source_parent):
        pattern = self.filterRegularExpression().pattern()
        if not pattern:
            return True

        model = self.sourceModel()
        if model is None:
            return True

        column_count = model.columnCount(source_parent)
        for col in range(column_count):
            index = model.index(source_row, col, source_parent)
            text = str(model.data(index) or "")
            if self.filterRegularExpression().match(text).hasMatch():
                return True

        branch_root_index = model.index(source_row, 0, source_parent)
        for child_row in range(model.rowCount(branch_root_index)):
            if self.filterAcceptsRow(child_row, branch_root_index):
                return True

        return False


class SelectionTreeViewDialog(QDialog):
    """Displays all available datasets and returns one complete treatment chain."""

    def __init__(self, organized_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Full Treatment Dataset")
        self.setMinimumSize(760, 560)
        self.selected_paths = None

        layout = QVBoxLayout(self)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by type, label, or details...")
        layout.addWidget(self.filter_input)

        self.tree_view = QTreeView()
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setUniformRowHeights(True)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setExpandsOnDoubleClick(True)
        layout.addWidget(self.tree_view, 1)

        self.selection_summary_label = QLabel("Selected: none")
        self.selection_summary_label.setWordWrap(True)
        layout.addWidget(self.selection_summary_label)

        hint_label = QLabel(
            "Tip: select a specific RTDOSE row for an unambiguous chain. "
            "Parent rows are accepted only when they lead to exactly one RTDOSE."
        )
        hint_label.setObjectName("HintLabel")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Type", "Label", "Details"])
        self._populate_tree_model_columns(organized_data, self.model.invisibleRootItem())

        self.proxy_model = TreeFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.model)
        self.tree_view.setModel(self.proxy_model)
        self.tree_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree_view.expandToDepth(1)

        header = self.tree_view.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.filter_input.textChanged.connect(self._on_filter_text_changed)
        self.tree_view.selectionModel().selectionChanged.connect(self._update_selection_summary)
        self.tree_view.doubleClicked.connect(self._on_double_clicked)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_selection_summary()

    def _populate_tree_model(self, data_dict, parent_item):
        """Backward-compatible wrapper around the column-based tree builder."""
        self._populate_tree_model_columns(data_dict, parent_item)

    def _populate_tree_model_columns(self, data_dict, parent_item):
        """Populates a three-column tree model with type, label, and details."""
        for ct_uid, ct_data in data_dict.items():
            ct_desc = ct_data["info"].get("SeriesDescription", "N/A")
            ct_type_item = QStandardItem("CT")
            ct_label_item = QStandardItem(ct_desc)
            ct_details_item = QStandardItem(f"{len(ct_data['ct_files'])} slices")
            for item in (ct_type_item, ct_label_item, ct_details_item):
                item.setEditable(False)
            ct_type_item.setData(ct_uid, Qt.ItemDataRole.UserRole)
            parent_item.appendRow([ct_type_item, ct_label_item, ct_details_item])

            for struct_path, struct_data in ct_data["rtstructs"].items():
                struct_label = struct_data["info"].get("StructureSetLabel", "N/A")
                struct_type_item = QStandardItem("RTSTRUCT")
                struct_label_item = QStandardItem(struct_label)
                struct_details_item = QStandardItem(f"{len(struct_data['rtplans'])} plan(s)")
                for item in (struct_type_item, struct_label_item, struct_details_item):
                    item.setEditable(False)
                struct_type_item.setData(struct_path, Qt.ItemDataRole.UserRole)
                ct_type_item.appendRow([struct_type_item, struct_label_item, struct_details_item])

                for plan_path, plan_data in struct_data["rtplans"].items():
                    plan_label = plan_data["info"].get("RTPlanLabel", "N/A")
                    plan_type_item = QStandardItem("RTPLAN")
                    plan_label_item = QStandardItem(plan_label)
                    plan_details_item = QStandardItem(f"{len(plan_data['rtdoses'])} dose set(s)")
                    for item in (plan_type_item, plan_label_item, plan_details_item):
                        item.setEditable(False)
                    plan_type_item.setData(plan_path, Qt.ItemDataRole.UserRole)
                    struct_type_item.appendRow([plan_type_item, plan_label_item, plan_details_item])

                    for dose_path, dose_data in plan_data["rtdoses"].items():
                        dose_type = dose_data["info"].get("DoseSummationType", "N/A")
                        dose_type_item = QStandardItem("RTDOSE")
                        dose_label_item = QStandardItem(dose_type)
                        dose_details_item = QStandardItem(os.path.basename(dose_path))
                        for item in (dose_type_item, dose_label_item, dose_details_item):
                            item.setEditable(False)
                        dose_type_item.setData(dose_path, Qt.ItemDataRole.UserRole)
                        plan_type_item.appendRow([dose_type_item, dose_label_item, dose_details_item])

    def _on_filter_text_changed(self, text: str):
        self.proxy_model.setFilterFixedString(text.strip())
        if text.strip():
            self.tree_view.expandAll()
        else:
            self.tree_view.collapseAll()
            self.tree_view.expandToDepth(1)

    def _on_double_clicked(self, _index):
        """Accepts immediately when the selected branch resolves unambiguously."""
        if self._resolve_selection() is not None:
            self.accept()

    def _get_selected_source_item(self) -> Optional[QStandardItem]:
        selection_model = self.tree_view.selectionModel()
        if selection_model is None:
            return None
        selected_rows = selection_model.selectedRows(0)
        if not selected_rows:
            return None
        source_index = self.proxy_model.mapToSource(selected_rows[0])
        return self.model.itemFromIndex(source_index)

    def _collect_descendant_dose_items(self, start_item: Optional[QStandardItem]) -> List[QStandardItem]:
        """Collects all RTDOSE descendants reachable from a selected branch."""
        if start_item is None:
            return []
        if start_item.text() == "RTDOSE":
            return [start_item]

        dose_items: List[QStandardItem] = []
        stack: List[QStandardItem] = [start_item]
        while stack:
            item = stack.pop()
            for row in range(item.rowCount()):
                child = item.child(row, 0)
                if child is None:
                    continue
                if child.text() == "RTDOSE":
                    dose_items.append(child)
                else:
                    stack.append(child)
        return dose_items

    def _resolve_selection(self) -> Optional[Dict]:
        """Resolves the current tree selection into one complete chain if unambiguous."""
        selected_item = self._get_selected_source_item()
        dose_items = self._collect_descendant_dose_items(selected_item)
        if len(dose_items) != 1:
            return None
        return self._build_selection_from_dose_item(dose_items[0])

    def _build_selection_from_dose_item(self, dose_item: Optional[QStandardItem]) -> Optional[Dict]:
        if dose_item is None:
            return None
        plan_item = dose_item.parent()
        struct_item = plan_item.parent() if plan_item else None
        ct_item = struct_item.parent() if struct_item else None
        if not all([plan_item, struct_item, ct_item]):
            return None
        return {
            "ct_series_uid": ct_item.data(Qt.ItemDataRole.UserRole),
            "rtstruct_path": struct_item.data(Qt.ItemDataRole.UserRole),
            "rtplan_path": plan_item.data(Qt.ItemDataRole.UserRole),
            "dose_path": dose_item.data(Qt.ItemDataRole.UserRole),
        }

    def _update_selection_summary(self):
        selected_item = self._get_selected_source_item()
        if selected_item is None:
            self.selection_summary_label.setText(
                "Selected: none (choose a branch that leads to RTDOSE)."
            )
            return

        dose_items = self._collect_descendant_dose_items(selected_item)
        if len(dose_items) == 0:
            self.selection_summary_label.setText(
                "Selected branch does not contain RTDOSE. Choose a branch that leads to RTDOSE."
            )
            return
        if len(dose_items) > 1:
            self.selection_summary_label.setText(
                f"Selected branch is ambiguous: {len(dose_items)} RTDOSE options found. "
                "Select a specific RTDOSE row."
            )
            return

        selection = self._build_selection_from_dose_item(dose_items[0])
        if not selection:
            self.selection_summary_label.setText(
                "Selected: none (unable to resolve a complete dataset chain)."
            )
            return

        summary = (
            f"Selected chain: CT [{selection['ct_series_uid']}] -> "
            f"RTSTRUCT [{os.path.basename(selection['rtstruct_path'])}] -> "
            f"RTPLAN [{os.path.basename(selection['rtplan_path'])}] -> "
            f"RTDOSE [{os.path.basename(selection['dose_path'])}]"
        )
        self.selection_summary_label.setText(summary)

    def accept(self):
        """Validates selection and stores one complete chain."""
        selection = self._resolve_selection()
        if not selection:
            selected_item = self._get_selected_source_item()
            dose_count = len(self._collect_descendant_dose_items(selected_item))
            if dose_count > 1:
                message = (
                    "The selected branch contains multiple RTDOSE datasets. "
                    "Please select one specific RTDOSE row."
                )
            else:
                message = (
                    "The selected branch does not contain a complete RTDOSE chain. "
                    "Please select a branch with CT, RTSTRUCT, RTPLAN, and RTDOSE."
                )
            QMessageBox.warning(self, "Incomplete Selection", message)
            return

        self.selected_paths = selection
        super().accept()

    def get_selection(self) -> Optional[Dict]:
        """Returns the dictionary of selected paths."""
        return self.selected_paths


class DVHPlotWidget(QWidget):
    """A themed widget for displaying an interactive DVH plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark_theme = True
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor=styles.COLOR_WIDGET_BACKGROUND)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.annot = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.set_initial_state(draw=False)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.draw()

    def _theme_tokens(self) -> Dict[str, str]:
        """Returns the active color tokens for the current DVH theme."""
        if self.is_dark_theme:
            return {
                "figure_bg": styles.COLOR_WIDGET_BACKGROUND,
                "axes_bg": styles.COLOR_BACKGROUND,
                "text": styles.COLOR_TEXT,
                "border": styles.COLOR_WIDGET_BORDER,
                "accent": styles.COLOR_PRIMARY,
                "legend_bg": styles.COLOR_WIDGET_BACKGROUND,
                "legend_text": styles.COLOR_TEXT,
            }
        return {
            "figure_bg": styles.LIGHT_COLOR_WIDGET_BACKGROUND,
            "axes_bg": styles.LIGHT_COLOR_BACKGROUND,
            "text": styles.LIGHT_COLOR_TEXT,
            "border": styles.LIGHT_COLOR_WIDGET_BORDER,
            "accent": styles.LIGHT_COLOR_PRIMARY,
            "legend_bg": styles.LIGHT_COLOR_WIDGET_BACKGROUND,
            "legend_text": styles.LIGHT_COLOR_TEXT,
        }

    def set_initial_state(self, draw: bool = True):
        """Clears the plot and resets it using the current theme colors."""
        tokens = self._theme_tokens()
        self.axes.cla()
        self.figure.set_facecolor(tokens["figure_bg"])
        self.axes.set_facecolor(tokens["axes_bg"])
        self.axes.tick_params(axis="x", colors=tokens["text"])
        self.axes.tick_params(axis="y", colors=tokens["text"])
        for spine in self.axes.spines.values():
            spine.set_edgecolor(tokens["border"])

        self.annot = self.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(-50, 20),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round",
                fc=tokens["axes_bg"],
                ec=tokens["accent"],
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
                color=tokens["accent"],
            ),
            color=tokens["text"],
            fontweight="bold",
        )
        self.annot.set_visible(False)

        self.axes.set_title("Dose-Volume Histogram", color=tokens["text"], weight="bold")
        self.axes.set_xlabel("Dose (Gy)", color=tokens["text"])
        self.axes.set_ylabel("Volume (%)", color=tokens["text"])
        self.axes.grid(True, color=tokens["border"], linestyle="--")
        self.axes.set_ylim(0, 101)
        self.figure.tight_layout(pad=1.5)
        self.figure.subplots_adjust(top=0.85, bottom=0.1)
        if draw:
            self.canvas.draw()

    def plot_multiple_dvhs(self, dvh_results: list, use_percentage: bool, prescription_dose: float, unit: str):
        """Plots multiple DVH curves on the same canvas."""
        tokens = self._theme_tokens()
        self.set_initial_state(draw=False)

        xlabel = f"Dose ({unit})"
        if use_percentage and prescription_dose > 0:
            xlabel = "Dose (%)"
        self.axes.set_xlabel(xlabel, color=tokens["text"])

        if not dvh_results:
            self.canvas.draw()
            return

        for result in dvh_results:
            dvh_data = result["dvh_data"]
            structure_name = result["name"]
            color = result["color"]

            dose_levels = np.asarray(dvh_data["dose_levels"], dtype=float)
            volumes = np.asarray(dvh_data["dvh_percentages"], dtype=float)
            if use_percentage and prescription_dose > 0:
                dose_levels = (dose_levels / prescription_dose) * 100.0

            if dose_levels.size == 0 or volumes.size == 0:
                continue
            self.axes.plot(dose_levels, volumes, color=color, label=structure_name, linewidth=2)

        self.axes.set_xlim(left=0)
        legend = self.axes.legend(facecolor=tokens["legend_bg"], edgecolor=tokens["border"])
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(tokens["legend_text"])
        self.canvas.draw()

    def _closest_curve_point(self, event) -> Tuple[Optional[object], int, float]:
        """Returns the closest plotted point in display coordinates."""
        min_dist_px = float("inf")
        closest_line = None
        closest_point_idx = -1

        for line in self.axes.lines:
            x_data, y_data = line.get_data()
            if len(x_data) == 0:
                continue
            if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
                continue

            points = np.column_stack((x_data, y_data))
            display_points = self.axes.transData.transform(points)
            deltas = display_points - np.array([event.x, event.y], dtype=float)
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            idx = int(np.argmin(distances))
            dist = float(distances[idx])
            if dist < min_dist_px:
                min_dist_px = dist
                closest_line = line
                closest_point_idx = idx

        return closest_line, closest_point_idx, min_dist_px

    def _on_hover(self, event):
        """Shows a tooltip-like annotation for the curve point nearest the cursor."""
        if event.inaxes != self.axes or self.annot is None:
            if self.annot is not None and self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()
            return

        closest_line, closest_point_idx, min_dist_px = self._closest_curve_point(event)
        if closest_line is None or closest_point_idx < 0 or min_dist_px > 20.0:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()
            return

        x_data, y_data = closest_line.get_data()
        x = float(x_data[closest_point_idx])
        y = float(y_data[closest_point_idx])
        self.annot.xy = (x, y)

        structure_name = closest_line.get_label()
        xlabel = self.axes.get_xlabel()
        if "%" in xlabel:
            dose_text = f"Dose: {x:.1f} %"
        else:
            unit = xlabel.split("(")[-1].split(")")[0] if "(" in xlabel and ")" in xlabel else "Gy"
            dose_text = f"Dose: {x:.2f} {unit}"

        self.annot.set_text(
            f"Structure: {structure_name}\n{dose_text}\nVolume: {y:.1f}%"
        )
        self.annot.set_visible(True)
        self.canvas.draw_idle()

    def save_plot(self, filepath: str):
        """Saves the current figure to a file."""
        self.figure.savefig(
            filepath,
            dpi=300,
            facecolor=self.figure.get_facecolor(),
            bbox_inches="tight",
        )

    def set_theme(self, dark: bool):
        """Updates the plot theme and restyles the current plot in place."""
        self.is_dark_theme = dark
        tokens = self._theme_tokens()

        self.figure.set_facecolor(tokens["figure_bg"])
        self.axes.set_facecolor(tokens["axes_bg"])
        self.axes.tick_params(axis="x", colors=tokens["text"])
        self.axes.tick_params(axis="y", colors=tokens["text"])
        for spine in self.axes.spines.values():
            spine.set_edgecolor(tokens["border"])

        self.axes.title.set_color(tokens["text"])
        self.axes.xaxis.label.set_color(tokens["text"])
        self.axes.yaxis.label.set_color(tokens["text"])
        self.axes.grid(True, color=tokens["border"], linestyle="--")

        if self.annot is not None:
            self.annot.get_bbox_patch().set_facecolor(tokens["axes_bg"])
            self.annot.get_bbox_patch().set_edgecolor(tokens["accent"])
            self.annot.arrow_patch.set_color(tokens["accent"])
            self.annot.set_color(tokens["text"])

        legend = self.axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(tokens["legend_bg"])
            legend.get_frame().set_edgecolor(tokens["border"])
            for text in legend.get_texts():
                text.set_color(tokens["legend_text"])

        self.canvas.draw()
