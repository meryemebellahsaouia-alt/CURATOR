# controller.py

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QApplication, QDialog

from model import DICOMDataModel, LoadCancelledError
from view import MainWindow
import config
from vtk_controller import VTKController
from vtk_controller_3d import VTK3DController
from dicom_organizer import organize_radiotherapy_dicom
from widgets import SelectionTreeViewDialog, ExportDialog, CompatibilityCheckDialog


class Worker(QObject):
    """Runs one patient-load operation on a dedicated background thread."""
    finished = pyqtSignal(bool, str, bool)
    progress = pyqtSignal(int, str)

    def __init__(self, paths: Dict[str, str], mask_backend: str = "legacy"):
        super().__init__()
        self.paths = dict(paths)
        self.mask_backend = str(mask_backend or "legacy")
        self.loaded_model: Optional[DICOMDataModel] = None
        self._cancel_requested = False

    def cancel(self):
        """Requests cancellation of the running load operation."""
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self):
        """Executes the data loading process and emits a finished signal."""
        try:
            if self._is_cancelled():
                raise LoadCancelledError("Loading cancelled by user.")

            local_model = DICOMDataModel()
            try:
                local_model.set_structure_mask_backend(self.mask_backend)
            except ValueError:
                logging.warning(
                    "Invalid structure mask backend %r requested for worker. Falling back to legacy.",
                    self.mask_backend,
                )
                local_model.set_structure_mask_backend("legacy")

            local_model.load_data(
                **self.paths,
                progress_callback=self.progress.emit,
                is_cancelled=self._is_cancelled,
            )
            if self._is_cancelled():
                raise LoadCancelledError("Loading cancelled by user.")

            self.loaded_model = local_model
            self.finished.emit(True, "", False)
        except LoadCancelledError:
            self.loaded_model = None
            self.finished.emit(False, "Loading cancelled by user.", True)
        except Exception as e:
            self.loaded_model = None
            error_message = f"Failed to load data: {e}"
            logging.error(error_message, exc_info=True)
            self.finished.emit(False, str(e), False)


class DiagnosticsBridge(QObject):
    """Qt signal bridge for log lines that need to reach the UI thread."""
    log_line = pyqtSignal(str, str)


class UILogHandler(logging.Handler):
    """Logging handler that forwards records to the DiagnosticsBridge."""

    def __init__(self, bridge: DiagnosticsBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
            self.bridge.log_line.emit(record.levelname, message)
        except Exception:
            return


class AppController:
    """The main controller for the application, connecting the view and model."""
    
    # --- 1. Initialization & Core Logic ---

    def __init__(self, model: DICOMDataModel, view: MainWindow):
        self.model = model
        self.view = view
        
        self.vtk_controller = VTKController(self.view.vtkWidget)
        self.vtk_3d_controller = VTK3DController(self.view.vtk_3d_Widget.GetRenderWindow())
        
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None
        self.plotted_dvh_results: list = []
        self._last_dvh_signature: Optional[Tuple[Tuple[int, ...], bool]] = None
        self._last_dose_slider_value: Optional[int] = None
        self._last_machine_state: Optional[Tuple[int, int]] = None
        self.machine_beam_indices: List[int] = []
        
        self.last_export_spacing = (1.0, 1.0, 1.0)
        self.last_export_shape = (0, 0, 0)
        self.window_presets = {
            "Soft Tissue": (config.DEFAULT_CT_WINDOW, config.DEFAULT_CT_LEVEL),
            "Lung": (1500, -600),
            "Bone": (2000, 350),
        }
        self.machine_play_timer = QTimer(self.view)
        self.machine_play_timer.setSingleShot(False)
        self.machine_play_timer.timeout.connect(self._on_machine_playback_tick)
        self.machine_lock_beam = True
        self.diagnostics_bridge = DiagnosticsBridge()
        self.ui_log_handler = UILogHandler(self.diagnostics_bridge)
        self._configure_diagnostics_logging()
        
        self.view.vtkWidget.GetRenderWindow().GetInteractor().Initialize()
        self.view.vtk_3d_Widget.GetRenderWindow().GetInteractor().Initialize()
        self._connect_signals()

    def _is_loading(self) -> bool:
        """Returns True while a background patient load is active."""
        return bool(
            self.worker_thread
            and self.worker_thread.isRunning()
            and self.worker is not None
        )

    def _set_loading_ui_state(self, is_loading: bool):
        """Keeps the UI in a consistent enabled/disabled state during loads."""
        self.view.load_action.setEnabled(not is_loading)
        if not is_loading:
            return

        self.view.set_controls_enabled(False)
        self.view.set_slice_controls_enabled(False)
        self.view.set_machine_play_controls_enabled(False)
        self.view.set_dvh_export_enabled(False)
        self.view.machine_beam_slider.setEnabled(False)
        self.view.machine_cp_slider.setEnabled(False)
        self.view.machine_lock_beam_checkbox.setEnabled(False)

    def _connect_signals(self):
        """Connects signals from the view to the controller's slots."""
        # File/Export actions
        self.view.load_files_requested.connect(self.select_and_load_data)
        self.view.export_ct_requested.connect(self._on_export_ct)
        self.view.export_dose_requested.connect(self._on_export_dose)
        self.view.resample_and_export_mask_requested.connect(self._on_resample_export_mask)
        
        # DVH actions
        self.view.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.view.export_dvh_data_requested.connect(self._on_export_dvh_data) 
        self.view.export_dvh_plot_requested.connect(self._on_export_dvh_plot)
        self.view.export_dvh_stats_requested.connect(self._on_export_dvh_stats)
        
        #View actions
        self.view.export_screenshot_viewer_requested.connect(self._on_export_screenshot_3d)
        
        # Viewer interactions
        self.view.structure_visibility_changed.connect(self._on_structure_visibility_changed)
        self.view.dose_slider_changed.connect(self._on_dose_slider_changed)
        self.view.units_toggled.connect(self._on_units_toggled)
        self.view.machine_beam_slider_changed.connect(self._on_machine_beam_changed)
        self.view.machine_cp_slider_changed.connect(self._on_machine_cp_changed)
        self.view.machine_visibility_changed.connect(self._on_machine_visibility_changed)
        self.view.slice_slider_changed.connect(self._on_slice_slider_changed)
        self.view.window_preset_selected.connect(self._on_window_preset_selected)
        self.view.reset_slice_view_requested.connect(self._on_reset_slice_view_requested)
        self.view.machine_play_toggled.connect(self._on_machine_play_toggled)
        self.view.machine_stop_requested.connect(self._on_machine_stop_requested)
        self.view.machine_playback_speed_changed.connect(self._on_machine_playback_speed_changed)
        self.view.machine_lock_beam_toggled.connect(self._on_machine_lock_beam_toggled)
        self.view.structure_palette_toggled.connect(self._on_structure_palette_toggled)
        self.view.diagnostics_cleared_requested.connect(self._on_diagnostics_cleared_requested)
        self.view.loading_dialog.cancel_requested.connect(self._on_loading_cancel_requested)
        self.diagnostics_bridge.log_line.connect(self.view.append_diagnostic)

    def _configure_diagnostics_logging(self):
        """Installs one UI logging handler so diagnostics can be shown in-app."""
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if isinstance(handler, UILogHandler):
                root_logger.removeHandler(handler)
        self.ui_log_handler.setLevel(logging.INFO)
        self.ui_log_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        root_logger.addHandler(self.ui_log_handler)

    @staticmethod
    def _normalize_dicom_cs(value) -> str:
        """Normalizes DICOM CS-like string values for stable comparisons."""
        return str(value or "").strip().upper()

    def _is_treatment_beam(self, beam) -> bool:
        """Returns True for treatment beams (or unknown delivery type)."""
        delivery_type = self._normalize_dicom_cs(getattr(beam, "TreatmentDeliveryType", ""))
        return delivery_type in ("", "TREATMENT")

    def _is_dynamic_beam(self, beam) -> bool:
        """Detects dynamic delivery beams for 3D machine playback."""
        beam_type = self._normalize_dicom_cs(getattr(beam, "BeamType", ""))
        if beam_type == "DYNAMIC":
            return True
        if beam_type == "STATIC":
            return False

        control_points = getattr(beam, "ControlPointSequence", [])
        if len(control_points) <= 2:
            return False

        # Fallback when BeamType is missing/unreliable:
        # treat as dynamic when gantry actually rotates through the sequence.
        for cp in control_points:
            rotation_dir = self._normalize_dicom_cs(getattr(cp, "GantryRotationDirection", ""))
            if rotation_dir not in ("", "NONE"):
                return True

        angles = []
        for cp in control_points:
            angle = getattr(cp, "GantryAngle", None)
            if isinstance(angle, (int, float)):
                angles.append(float(angle))
        if len(angles) >= 2 and abs(max(angles) - min(angles)) > 0.1:
            return True

        return False

    def _build_machine_beam_indices(self) -> List[int]:
        """Builds slider->plan beam mapping: dynamic treatment -> treatment -> all."""
        if not self.model or not self.model.rt_plan:
            return []
        beams = getattr(self.model.rt_plan, "BeamSequence", [])
        treatment_indices = [i for i, beam in enumerate(beams) if self._is_treatment_beam(beam)]
        dynamic_treatment_indices = [i for i in treatment_indices if self._is_dynamic_beam(beams[i])]
        if dynamic_treatment_indices:
            return dynamic_treatment_indices
        if treatment_indices:
            return treatment_indices
        return list(range(len(beams)))
        
    # --- 2. Data Loading Workflow ---

    def select_and_load_data(self):
        """
        Handles selecting a folder and using a tree view to select the
        exact dataset to load, then starts the data loading worker.
        """
        if self._is_loading():
            self.view.update_status_label("A patient load is already in progress.")
            return

        patient_folder = QFileDialog.getExistingDirectory(self.view, "Select Patient Folder")
        if not patient_folder:
            return

        # 1. Organize the folder to find all complete datasets
        organized_data, audit = organize_radiotherapy_dicom(patient_folder, return_audit=True)
        compatibility_dialog = CompatibilityCheckDialog(audit, self.view)
        if compatibility_dialog.exec() != QDialog.DialogCode.Accepted:
            self.view.update_status_label("Load cancelled.")
            return

        if not organized_data:
            missing_modalities = audit.get('missing_modalities', [])
            missing_text = ", ".join(missing_modalities) if missing_modalities else "N/A"
            self.view.show_error_message(
                "No Data Found",
                "No complete CT/RTSTRUCT/RTPLAN/RTDOSE datasets were found.\n"
                f"Missing required modalities: {missing_text}"
            )
            return
            
        paths = self._find_single_path(organized_data)
        if not paths:
            dialog = SelectionTreeViewDialog(organized_data, self.view)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                paths = dialog.get_selection()
            else:
                self.view.update_status_label("Load cancelled.")
                return
        if not paths: # This can happen if the user clicks OK without a selection
            self.view.update_status_label("Load cancelled.")
            return
            
        # Add the top-level folder path to the dictionary
        paths['patient_folder'] = patient_folder

        # 4. Start the worker thread with the determined paths
        self.worker_thread = QThread()
        self.worker = Worker(paths, mask_backend=self.model.structure_mask_backend)

        self.worker.moveToThread(self.worker_thread)
        self._set_loading_ui_state(True)
        self.view.show_loading_dialog()
        self.worker.progress.connect(self.view.update_loading_progress)
        self.worker.finished.connect(self._on_loading_finished)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def _on_loading_cancel_requested(self):
        """Handles user cancel requests from the loading dialog."""
        if self.worker:
            self.worker.cancel()
            self.view.update_status_label("Cancelling load...")

    def _on_loading_finished(self, success: bool, error_msg: str, cancelled: bool):
        """Handles the completion of the data loading and triggers visualization setup."""
        loaded_model = self.worker.loaded_model if self.worker is not None else None

        self._last_dvh_signature = None
        self._last_dose_slider_value = None
        self._last_machine_state = None
        self.machine_beam_indices = []
        self._stop_machine_playback(reset_to_start=False)

        try:
            if success:
                if loaded_model is None:
                    raise RuntimeError("Background load finished without a valid data model.")

                self.model = loaded_model
                self.view.update_loading_progress(95, "Drawing 3D View...")
                QApplication.processEvents()
                self._setup_visualization()
                self.view.update_status_label("Data loaded successfully.")
            elif cancelled:
                self.view.update_status_label("Load cancelled.")
            else:
                self.view.update_status_label("Error loading data.")
                self.view.show_error_message("Data Loading Error", error_msg)
        except Exception as e:
            logging.error("Failed during post-load visualization setup: %s", e, exc_info=True)
            self.view.update_status_label("Error initializing visualization.")
            self.view.show_error_message("Visualization Error", str(e))
            self.plotted_dvh_results = []
            self._last_dvh_signature = None
            self._reset_machine_controls()
        finally:
            self.view.hide_loading_dialog()
            self._set_loading_ui_state(False)
            self.worker = None
            self.worker_thread = None
            
    def _find_single_path(self, organized_data: dict) -> Optional[dict]:
        """
        If there is only one unambiguous path from CT to Dose, return it.
        Otherwise, return None.
        """
        try:
            # Check for a single CT Series
            if len(organized_data) != 1: return None
            ct_uid, ct_data = next(iter(organized_data.items()))

            # Check for a single Structure Set
            if len(ct_data['rtstructs']) != 1: return None
            struct_path, struct_data = next(iter(ct_data['rtstructs'].items()))

            # Check for a single Plan
            if len(struct_data['rtplans']) != 1: return None
            plan_path, plan_data = next(iter(struct_data['rtplans'].items()))
            
            # Check for a single Dose
            if len(plan_data['rtdoses']) != 1: return None
            dose_path, _ = next(iter(plan_data['rtdoses'].items()))
            
            # If we get here, the path is unambiguous. Return it.
            return {
                'ct_series_uid': ct_uid,
                'rtstruct_path': struct_path,
                'rtplan_path': plan_path,
                'dose_path': dose_path
            }
        except (StopIteration, KeyError):
            return None

    # --- 3. Post-Load UI Setup ---

    def _setup_visualization(self):
        """Configures the UI with the newly loaded model data."""
        self._last_dvh_signature = None
        self.plotted_dvh_results = []
        self.vtk_controller.setup_scene(self.model)
        self.vtk_3d_controller.setup_scene(self.model)
        self.model.apply_structure_palette(self.view.colorblind_palette_action.isChecked())
        self.view.populate_structures_legend(self.model.structures)

        prescription_dose = self.model.prescription_dose or 0.0
        self.view.update_prescription_label(f"Prescription Dose: {prescription_dose:.2f} {self.model.dose_unit}")
        self.view.set_dvh_export_enabled(False)
        
        if self.model.max_dose_as_percent_of_rx is not None:
            max_slider_val = int(np.ceil(self.model.max_dose_as_percent_of_rx))
            self.view.update_dose_slider_range(max_slider_val)
        else:
            max_slider_val = 100

        style = self.vtk_controller.create_interactor_style(self.model, self.view)
        self.view.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.vtk_controller.reset_camera()
        
        # Set the initial window/level directly from the config file
        self.vtk_controller.update_ct_window_level(config.DEFAULT_CT_WINDOW, config.DEFAULT_CT_LEVEL)
    
        style.update_slice_view()
        self.view.set_slice_controls_enabled(True)
        self.view.window_preset_combo.blockSignals(True)
        self.view.window_preset_combo.setCurrentText("Auto")
        self.view.window_preset_combo.blockSignals(False)
        self._on_window_preset_selected("Auto")
        
        beam_sequence = getattr(self.model.rt_plan, "BeamSequence", []) if self.model and self.model.rt_plan else []
        if beam_sequence:
            self.machine_beam_indices = self._build_machine_beam_indices()
            num_beams = len(self.machine_beam_indices)
            self.view.machine_beam_slider.setMinimum(0)
            self.view.machine_beam_slider.setMaximum(num_beams - 1)
            self.view.machine_beam_slider.setValue(0)
            self.view.machine_beam_slider.setEnabled(True)
            self.machine_lock_beam = self.view.machine_lock_beam_checkbox.isChecked()
            self.view.set_machine_play_controls_enabled(True)
            self._on_machine_beam_changed(0)
            total_plan_beams = len(beam_sequence)
            if num_beams < total_plan_beams:
                self.view.update_status_label(
                    f"3D controls filtered to relevant beams: {num_beams}/{total_plan_beams}"
                )
        else:
            self._reset_machine_controls()
        
        self.view.set_controls_enabled(True)
        self._last_dose_slider_value = None
        initial_dose_slider = int(
            np.ceil((config.DOSE_INITIAL_THRESHOLD_PERCENT / 100.0) * max_slider_val)
        )
        if max_slider_val > 0:
            initial_dose_slider = max(1, initial_dose_slider)
        initial_dose_slider = int(np.clip(initial_dose_slider, 0, max_slider_val))
        self.view.dose_slider.setValue(initial_dose_slider)
        self._update_machine_control_enablement()
        
    # --- 4. Signal Handlers (Slots) ---

    def _on_clipping_slider_changed(self, value: int):
        """Updates the position of the 3D clipping plane."""
        if not self.model:
            return

        # Get the visible bounds of the entire scene
        bounds = self.vtk_3d_controller.renderer.ComputeVisiblePropBounds()
        
        # Get the Y-axis bounds instead of the X-axis
        y_min, y_max = bounds[2], bounds[3]

        # Map the slider's 0-100% range to the scene's Y-axis range
        position = y_min + (y_max - y_min) * (value / 100.0)

        plane = self.vtk_3d_controller.clipping_plane
        
        # Set the plane's position (origin) to move along the Y-axis
        plane.SetOrigin(0, position, 0)
        # Set the direction it faces (normal) to point along the Y-axis
        plane.SetNormal(0, 1, 0)

        # Re-render the scene to show the change
        self.vtk_3d_controller.render_window.Render()
    
                    
    def _on_tab_changed(self, index: int):
        """Updates the status bar and triggers actions when the user switches tabs."""
        if self._is_loading():
            if index == 3:
                self.view.update_status_label("Viewing Diagnostics")
            return
        if index == 0: # Axial Viewer
            self.view.update_status_label(self.view.last_slice_status)
        elif index == 1:
            self.view.update_status_label("Viewing 3D View")
            self._on_structure_visibility_changed()
        elif index == 2: # DVH Analysis
            self.view.update_status_label("Viewing DVH Plot")
            self._calculate_and_plot_dvhs()
        elif index == 3:
            self.view.update_status_label("Viewing Diagnostics")
        self.view.set_dvh_export_enabled(index == 2 and bool(self.plotted_dvh_results))

    def _calculate_and_plot_dvhs(self, force: bool = False):
        """
        Calculates DVH for all checked structures and tells the view to plot them
        and populate the stats table.
        """
        if not self.model or not self.model.structures:
            return

        rois_to_plot = [num for num, cb in self.view.structure_checkboxes.items() if cb.isChecked()]
        
        if not rois_to_plot:
            self.view.dvh_plot_widget.plot_multiple_dvhs([], False, 0.0, "")
            self.view.populate_dvh_stats_table([]) # Clear the table
            self.view.update_status_label("Select structures to plot their DVH.")
            self.plotted_dvh_results = []
            self.view.set_dvh_export_enabled(False)
            self._last_dvh_signature = None
            return

        dvh_signature = (tuple(sorted(rois_to_plot)), self.view.unit_toggle_checkbox.isChecked())
        if (not force and self.plotted_dvh_results and
                self._last_dvh_signature == dvh_signature):
            return

        self.view.update_status_label(f"Calculating DVH for {len(rois_to_plot)} structure(s)...")
        QApplication.processEvents()

        dvh_results = []
        for roi_number in rois_to_plot:
            dvh_data = self.model.calculate_dvh(roi_number)
    
            ci_value = self.model.calculate_ci(roi_number) # Calculate CI for each structure
            
            if dvh_data:
                dvh_results.append({
                    'dvh_data': dvh_data,
                    'ci_value': ci_value, 
                    'name': self.model.structures[roi_number]['name'],
                    'color': self.model.structures[roi_number]['color']
                })
        
        self.view.dvh_plot_widget.plot_multiple_dvhs(
            dvh_results,
            self.view.unit_toggle_checkbox.isChecked(),
            self.model.prescription_dose or 0.0,
            self.model.dose_unit
        )

        self.view.populate_dvh_stats_table(dvh_results)
        self.plotted_dvh_results = dvh_results
        self.view.set_dvh_export_enabled(bool(dvh_results) and self.view.tab_widget.currentIndex() == 2)
        self._last_dvh_signature = dvh_signature
        self.view.update_status_label("DVH calculation complete.")
        
    def _on_dose_slider_changed(self, slider_value: int, force: bool = False):
        """Updates the dose visualization threshold based on the slider value."""
        if self._is_loading():
            return
        if not force and self._last_dose_slider_value == slider_value:
            return
        if self.model.dose_max_in_gy is None or self.model.max_dose_as_percent_of_rx is None: return

        self._last_dose_slider_value = slider_value
        max_dose_gy = self.model.dose_max_in_gy
        max_slider_value = int(np.ceil(self.model.max_dose_as_percent_of_rx))
        if max_slider_value == 0: return
            
        proportion = slider_value / max_slider_value
        threshold_dose_gy = proportion * max_dose_gy
        relative_noise_floor_gy = (
            float(config.DOSE_DISPLAY_NOISE_FLOOR_PERCENT_OF_MAX) / 100.0
        ) * float(max_dose_gy)
        effective_threshold_gy = max(
            float(threshold_dose_gy),
            float(config.DOSE_DISPLAY_NOISE_FLOOR_GY),
            relative_noise_floor_gy,
        )
        self.vtk_controller.update_dose_visibility(effective_threshold_gy)

        is_percentage_mode = self.view.unit_toggle_checkbox.isChecked()
        prescription_dose = self.model.prescription_dose or 0.0
        
        if is_percentage_mode and prescription_dose > 0:
            display_percent = (effective_threshold_gy / prescription_dose) * 100.0
            self.view.update_dose_display_label(f"Min: {display_percent:.1f}%")
        else:
            self.view.update_dose_display_label(f"Min: {effective_threshold_gy:.2f} {self.model.dose_unit}")
            
        self.view.vtkWidget.GetRenderWindow().Render()

    def _on_structure_visibility_changed(self):
        """
        Updates contour visibility in the active view (2D or 3D)
        based on the state of the structure checkboxes.
        """
        if self._is_loading():
            return
        active_tab_index = self.view.tab_widget.currentIndex()

        # If the "Axial Viewer" (2D) tab is active
        if active_tab_index == 0:
            if self.vtk_controller.interactor_style:
                self.vtk_controller.interactor_style.update_contours()
        
        # If the "3D Viewer" tab is active
        elif active_tab_index == 1:
            for roi_number, checkbox in self.view.structure_checkboxes.items():
                self.vtk_3d_controller.set_structure_visibility(
                    roi_number, checkbox.isChecked(), render=False
                )
            self.vtk_3d_controller.render_window.Render()
        
        # If the DVH tab is active, checking/unchecking should update the plot
        elif active_tab_index == 2:
            self._calculate_and_plot_dvhs(force=True)

    def _on_units_toggled(self, is_percentage: bool):
        """Updates the dose legend and slider label when the display unit is toggled."""
        if self._is_loading():
            return
        self.vtk_controller.update_dose_legend_units(is_percentage)
        self._on_dose_slider_changed(self.view.dose_slider.value(), force=True)
        if self.view.tab_widget.currentIndex() == 2:
            self._calculate_and_plot_dvhs(force=True)
            
    def _on_machine_visibility_changed(self, object_name: str, is_visible: bool):
        """Tells the 3D controller to update object visibility."""
        if self._is_loading():
            return
        if self.vtk_3d_controller:
            self.vtk_3d_controller.set_visibility(object_name, is_visible)

    def _on_slice_slider_changed(self, slice_index: int):
        """Moves the 2D viewer to the selected slice."""
        if self._is_loading():
            return
        style = self.vtk_controller.interactor_style
        if style:
            style.set_slice(slice_index)

    def _on_window_preset_selected(self, preset_name: str):
        """Applies one predefined CT window/level preset."""
        if self._is_loading():
            return
        if preset_name == "Auto":
            window, level = self._compute_auto_window_level()
        else:
            window, level = self.window_presets.get(
                preset_name, (config.DEFAULT_CT_WINDOW, config.DEFAULT_CT_LEVEL)
            )
        self.vtk_controller.update_ct_window_level(window, level)
        self.view.vtkWidget.GetRenderWindow().Render()

    def _compute_auto_window_level(self) -> Tuple[int, int]:
        """Computes a robust CT window/level from volume statistics."""
        if self.model is None or self.model.masked_ct_array is None:
            return config.DEFAULT_CT_WINDOW, config.DEFAULT_CT_LEVEL

        ct_array = self.model.masked_ct_array
        finite = ct_array[np.isfinite(ct_array)]
        if finite.size == 0:
            return config.DEFAULT_CT_WINDOW, config.DEFAULT_CT_LEVEL

        # Focus on likely in-body voxels first; fall back progressively.
        body_like = finite[finite > -700]
        if body_like.size < 1000:
            body_like = finite[finite > -950]
        if body_like.size < 1000:
            body_like = finite

        low_hu, high_hu = np.percentile(body_like, [2.0, 98.0])
        window = int(np.clip(high_hu - low_hu, 250, 2200))
        level = int(np.clip((high_hu + low_hu) / 2.0, -800, 800))
        return window, level

    def _on_reset_slice_view_requested(self):
        """Resets the 2D camera while keeping current slice state synchronized."""
        if self._is_loading():
            return
        self.vtk_controller.reset_camera()
        if self.vtk_controller.interactor_style:
            self.vtk_controller.interactor_style.update_slice_view()

    def _on_structure_palette_toggled(self, enabled: bool):
        """Updates ROI colors to either default colors or a colorblind-safe palette."""
        if not self.model:
            return
        self._last_dvh_signature = None
        self.model.apply_structure_palette(enabled)
        self.view.populate_structures_legend(self.model.structures)
        self._on_structure_visibility_changed()
        if self.view.tab_widget.currentIndex() == 2:
            self._calculate_and_plot_dvhs(force=True)
        self.view.vtkWidget.GetRenderWindow().Render()
        self.view.vtk_3d_Widget.GetRenderWindow().Render()

    def _on_diagnostics_cleared_requested(self):
        """Updates status after user clears diagnostics output."""
        self.view.update_status_label("Diagnostics cleared.")
             
    def _handle_single_file_export(self, model_method_name: str, dialog_title: str, 
                                  default_filename_template: str, file_filter: str, 
                                  new_spacing: tuple, target_shape: tuple):
        """A generic handler for exporting a single resampled file (CT or Dose)."""
        spacing_str = f"{new_spacing[0]}x{new_spacing[1]}x{new_spacing[2]}"
        default_filename = default_filename_template.format(spacing=spacing_str)
        
        filepath, _ = QFileDialog.getSaveFileName(self.view, dialog_title, default_filename, file_filter)
        if not filepath: return
        
        try:
            self.view.update_status_label(f"Processing and exporting {os.path.basename(filepath)}...")
            QApplication.processEvents()
            
            model_method = getattr(self.model, model_method_name)
            model_method(new_spacing, target_shape, filepath)
            
            self.view.update_status_label(f"Successfully saved to {os.path.basename(filepath)}")
        except Exception as e:
            logging.error(f"Failed during export: {e}", exc_info=True)
            self.view.show_error_message("Export Error", f"Could not save the file: {e}")


    def _on_export_ct(self):
        """Handles the request to resample and export the CT image."""
        if not self.model or not self.model.ct_image:
            self.view.show_error_message("Export Error", "No CT data is available to export.")
            return

        dialog = ExportDialog(self.view)
        dialog.set_values(self.last_export_spacing, self.last_export_shape)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_spacing, target_shape = dialog.get_values()
            self.last_export_spacing, self.last_export_shape = new_spacing, target_shape
            self._handle_single_file_export(
                "resample_and_export_ct", "Save Resampled CT", "CT_resampled.nii.gz",
                "NIfTI Files (*.nii.gz *.nii)", new_spacing, target_shape
            )
            
    def _on_export_dose(self):
        """Handles the request to resample and export the dose grid."""
        if not self.model or not self.model.dose_path:
            self.view.show_error_message("Export Error", "No dose data to export.")
            return

        dialog = ExportDialog(self.view)
        dialog.set_values(self.last_export_spacing, self.last_export_shape)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_spacing, target_shape = dialog.get_values()
            self.last_export_spacing, self.last_export_shape = new_spacing, target_shape
            self._handle_single_file_export(
                "resample_and_export_dose", "Save Resampled Dose", "dose_resampled.nii.gz",
                "NIfTI Files (*.nii.gz *.nii)", new_spacing, target_shape
            )
            
    def _on_resample_export_mask(self):
        """Handles exporting multiple checked structure masks."""
        if not self.model or not self.model.structures:
            self.view.show_error_message("Export Error", "No model data loaded.")
            return

        rois_to_export = [num for num, cb in self.view.structure_checkboxes.items() if cb.isChecked()]
        if not rois_to_export:
            self.view.show_error_message("No Structures Selected", "Please check at least one structure to export.")
            return

        dialog = ExportDialog(self.view)
        dialog.set_values(self.last_export_spacing, self.last_export_shape)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_spacing, target_shape = dialog.get_values()
        self.last_export_spacing, self.last_export_shape = new_spacing, target_shape
        output_folder = QFileDialog.getExistingDirectory(self.view, "Select Folder to Save Resampled Masks")
        if not output_folder:
            return

        try:
            self.view.update_status_label(f"Exporting {len(rois_to_export)} mask(s)...")
            QApplication.processEvents()

            for roi_number in rois_to_export:
                name = self.model.structures[roi_number]['name']
                safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
                filename = f"{safe_name}_resampled_{new_spacing[0]}x{new_spacing[1]}x{new_spacing[2]}.nii.gz"
                filepath = os.path.join(output_folder, filename)
                self.model.resample_and_export_mask(roi_number, new_spacing, target_shape, filepath)

            self.view.update_status_label(f"Successfully exported {len(rois_to_export)} mask(s).")
        except Exception as e:
            logging.error(f"Failed during mask export: {e}", exc_info=True)
            self.view.show_error_message("Export Error", f"An error occurred during export: {e}")
    
    def _on_export_screenshot_3d(self):
        """Handles the request to save a screenshot of the 3D view."""
        if not self.model or not self.model.ct_image:
            self.view.show_error_message("Error", "No data is loaded to take a screenshot of.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self.view,
            "Save 3D Screenshot",
            "3D_View_Screenshot.png",
            "PNG Images (*.png)"
        )

        if filepath:
            try:
                self.vtk_3d_controller.save_screenshot(filepath)
                self.view.update_status_label(f"Screenshot saved to {os.path.basename(filepath)}")
            except Exception as e:
                logging.error(f"Failed to save screenshot: {e}", exc_info=True)
                self.view.show_error_message("Screenshot Error", f"Could not save the image: {e}")
                
    def _on_export_dvh_stats(self):
        """Handles exporting the DVH statistics table to a CSV file."""
        if self.view.dvh_stats_table.rowCount() == 0:
            self.view.show_error_message("No Data", "There is no DVH statistics data to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self.view, "Save DVH Stats", "dvh_stats.csv", "CSV Files (*.csv)"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write headers
                headers = [self.view.dvh_stats_table.horizontalHeaderItem(c).text() 
                           for c in range(self.view.dvh_stats_table.columnCount())]
                writer.writerow(headers)

                # Write data rows
                for r in range(self.view.dvh_stats_table.rowCount()):
                    row_data = [self.view.dvh_stats_table.item(r, c).text() 
                                for c in range(self.view.dvh_stats_table.columnCount())]
                    writer.writerow(row_data)
            
            self.view.update_status_label(f"DVH stats exported to {os.path.basename(filepath)}")

        except Exception as e:
            logging.error(f"Failed to export DVH stats: {e}", exc_info=True)
            self.view.show_error_message("Export Error", f"Could not save the CSV file: {e}")

    def _on_export_dvh_data(self):
        """Handles exporting data for multiple DVHs to a single CSV file."""
        if not self.plotted_dvh_results: return

        default_filename = "DVH_Comparison_Data.csv"
        filepath, _ = QFileDialog.getSaveFileName(self.view, "Save DVH Data", default_filename, "CSV Files (*.csv)")
        if not filepath: return

        try:
            # Normalize each DVH onto one shared dose axis so export works even when
            # structures have different histogram lengths/max doses.
            curves = []
            max_dose = 0.0
            max_points = 0
            for result in self.plotted_dvh_results:
                dvh_data = result.get('dvh_data', {})
                dose_levels = np.asarray(dvh_data.get('dose_levels', []), dtype=float)
                dvh_percentages = np.asarray(dvh_data.get('dvh_percentages', []), dtype=float)
                if dose_levels.size == 0 or dvh_percentages.size == 0:
                    continue

                n_points = int(min(dose_levels.size, dvh_percentages.size))
                if n_points <= 1:
                    continue

                dose_levels = dose_levels[:n_points]
                dvh_percentages = dvh_percentages[:n_points]
                sort_idx = np.argsort(dose_levels)
                dose_levels = dose_levels[sort_idx]
                dvh_percentages = dvh_percentages[sort_idx]

                curves.append((result['name'], dose_levels, dvh_percentages))
                max_dose = max(max_dose, float(dose_levels[-1]))
                max_points = max(max_points, n_points)

            if not curves:
                self.view.show_error_message("Export Error", "No valid DVH curve data to export.")
                return

            sample_count = max(2, max_points)
            common_dose_levels = np.linspace(0.0, max_dose, sample_count)
            headers = [f"Dose ({self.model.dose_unit})"] + [f"{name} Volume (%)" for name, _, _ in curves]

            resampled_volume_columns = []
            for _, dose_levels, dvh_percentages in curves:
                resampled = np.interp(
                    common_dose_levels,
                    dose_levels,
                    dvh_percentages,
                    left=float(dvh_percentages[0]),
                    right=float(dvh_percentages[-1]),
                )
                resampled_volume_columns.append(resampled)

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for i in range(sample_count):
                    row = [f"{common_dose_levels[i]:.4f}"]
                    for vol_col in resampled_volume_columns:
                        row.append(f"{vol_col[i]:.4f}")
                    writer.writerow(row)

            self.view.update_status_label(f"DVH data exported to {os.path.basename(filepath)}")
        except Exception as e:
            self.view.show_error_message("Export Error", f"Failed to save CSV file: {e}")

    def _on_export_dvh_plot(self):
        """Handles exporting the DVH comparison plot as a PNG image."""
        if not self.plotted_dvh_results: return

        default_filename = "DVH_Comparison_Plot.png"
        filepath, _ = QFileDialog.getSaveFileName(self.view, "Save DVH Plot", default_filename, "PNG Images (*.png)")
        if not filepath: return

        try:
            self.view.dvh_plot_widget.save_plot(filepath)
            self.view.update_status_label(f"DVH plot saved to {os.path.basename(filepath)}")
        except Exception as e:
            self.view.show_error_message("Export Error", f"Failed to save PNG file: {e}")
            
    def _on_machine_beam_changed(self, beam_index: int):
        """Handles changes from the 3D view's beam slider."""
        if not self.model or not self.model.rt_plan:
            return
        beams = getattr(self.model.rt_plan, "BeamSequence", [])
        if not beams:
            self._reset_machine_controls()
            return

        if not self.machine_beam_indices:
            self.machine_beam_indices = self._build_machine_beam_indices()
        if not self.machine_beam_indices:
            self._reset_machine_controls()
            return

        visible_count = len(self.machine_beam_indices)
        beam_index = int(np.clip(beam_index, 0, visible_count - 1))
        plan_beam_index = self.machine_beam_indices[beam_index]
        self.view.machine_beam_label.setText(f"Beam: {beam_index + 1} / {visible_count}")

        num_points = len(getattr(beams[plan_beam_index], "ControlPointSequence", []))
        self.view.machine_cp_slider.setMinimum(0)
        self.view.machine_cp_slider.blockSignals(True)
        self.view.machine_cp_slider.setMaximum(max(0, num_points - 1))
        self.view.machine_cp_slider.setValue(0)
        self.view.machine_cp_slider.blockSignals(False)
        self.view.machine_cp_slider.setEnabled(num_points > 0 and not self.machine_play_timer.isActive())
        self._on_machine_cp_changed(0)
        self._update_machine_control_enablement()

    def _on_machine_cp_changed(self, cp_index: int):
        """Handles changes from the 3D view's control point slider."""
        if not self.model or not self.model.rt_plan:
            return

        ui_beam_index = self.view.machine_beam_slider.value()
        beams = getattr(self.model.rt_plan, "BeamSequence", [])
        if not beams:
            self._reset_machine_controls()
            return
        if not self.machine_beam_indices:
            self.machine_beam_indices = self._build_machine_beam_indices()
        if not self.machine_beam_indices:
            self._reset_machine_controls()
            return
        ui_beam_index = int(np.clip(ui_beam_index, 0, len(self.machine_beam_indices) - 1))
        beam_index = self.machine_beam_indices[ui_beam_index]

        num_points = len(getattr(beams[beam_index], "ControlPointSequence", []))
        if num_points <= 0:
            self.view.machine_cp_label.setText("Control Point: N/A")
            self.view.gantry_angle_label.setText("Gantry Angle: N/A")
            self._last_machine_state = None
            self.view.machine_cp_slider.setEnabled(False)
            self._update_machine_control_enablement()
            return

        cp_index = int(np.clip(cp_index, 0, num_points - 1))
        if cp_index != self.view.machine_cp_slider.value():
            self.view.machine_cp_slider.blockSignals(True)
            self.view.machine_cp_slider.setValue(cp_index)
            self.view.machine_cp_slider.blockSignals(False)

        machine_state = (beam_index, cp_index)
        if machine_state == self._last_machine_state:
            self._update_machine_control_enablement()
            return

        self.view.machine_cp_label.setText(f"Control Point: {cp_index + 1} / {num_points}")
        try:
            control_point = beams[beam_index].ControlPointSequence[cp_index]
            gantry_angle = getattr(control_point, "GantryAngle", "N/A")
            if isinstance(gantry_angle, (int, float)):
                self.view.gantry_angle_label.setText(f"Gantry Angle: {gantry_angle:.1f} deg")
            else:
                self.view.gantry_angle_label.setText("Gantry Angle: N/A")
            self.vtk_3d_controller.update_machine_actors(beam_index, cp_index)
            self._last_machine_state = machine_state
        except (AttributeError, IndexError) as e:
            logging.warning(f"Could not update machine for beam {beam_index}, cp {cp_index}: {e}")
            self.view.gantry_angle_label.setText("Gantry Angle: N/A")
        self._update_machine_control_enablement()

    def _on_machine_play_toggled(self, is_playing: bool):
        """Starts or pauses machine playback timer."""
        if is_playing:
            if not self._has_playable_machine_data():
                self.view.update_machine_play_state(False)
                return
            if self.machine_lock_beam and self.model and self.model.rt_plan:
                beams = getattr(self.model.rt_plan, "BeamSequence", [])
                if beams and self.machine_beam_indices:
                    ui_beam_index = int(np.clip(self.view.machine_beam_slider.value(), 0, len(self.machine_beam_indices) - 1))
                    beam_index = self.machine_beam_indices[ui_beam_index]
                    current_points = len(getattr(beams[beam_index], "ControlPointSequence", []))
                    if current_points == 0:
                        self.view.update_machine_play_state(False)
                        self.view.update_status_label("Selected beam has no control points.")
                        return
            interval_ms = self._playback_interval_ms(self.view.playback_speed_slider.value())
            self.machine_play_timer.start(interval_ms)
            self.view.update_status_label("Machine playback running.")
        else:
            self.machine_play_timer.stop()
            self.view.update_status_label("Machine playback paused.")
        self._update_machine_control_enablement()

    def _on_machine_stop_requested(self):
        """Stops playback and rewinds the current beam to control point zero."""
        self._stop_machine_playback(reset_to_start=True)
        self.view.update_status_label("Machine playback stopped.")

    def _on_machine_playback_speed_changed(self, speed_value: int):
        """Applies the new playback speed if playback is active."""
        if self.machine_play_timer.isActive():
            self.machine_play_timer.setInterval(self._playback_interval_ms(speed_value))

    def _on_machine_lock_beam_toggled(self, is_locked: bool):
        """Locks playback traversal to the current beam or allows beam cycling."""
        self.machine_lock_beam = is_locked
        self._update_machine_control_enablement()

    def _on_machine_playback_tick(self):
        """Advances machine playback by one control point (or beam when unlocked)."""
        if not self.model or not self.model.rt_plan:
            self._stop_machine_playback(reset_to_start=False)
            return

        beams = getattr(self.model.rt_plan, "BeamSequence", [])
        if not beams:
            self._stop_machine_playback(reset_to_start=False)
            return
        if not self.machine_beam_indices:
            self.machine_beam_indices = self._build_machine_beam_indices()
        num_visible_beams = len(self.machine_beam_indices)
        if num_visible_beams == 0:
            self._stop_machine_playback(reset_to_start=False)
            return

        ui_beam_index = int(np.clip(self.view.machine_beam_slider.value(), 0, num_visible_beams - 1))
        beam_index = self.machine_beam_indices[ui_beam_index]
        cp_index = int(self.view.machine_cp_slider.value())
        current_points = len(getattr(beams[beam_index], "ControlPointSequence", []))

        if current_points <= 0 and self.machine_lock_beam:
            self._stop_machine_playback(reset_to_start=False)
            return

        if current_points > 0 and cp_index + 1 < current_points:
            self._set_machine_state(ui_beam_index, cp_index + 1)
            return

        if self.machine_lock_beam:
            self._set_machine_state(ui_beam_index, 0)
            return

        for offset in range(1, num_visible_beams + 1):
            next_ui_beam = (ui_beam_index + offset) % num_visible_beams
            next_plan_beam = self.machine_beam_indices[next_ui_beam]
            next_points = len(getattr(beams[next_plan_beam], "ControlPointSequence", []))
            if next_points > 0:
                self._set_machine_state(next_ui_beam, 0)
                return

        self._stop_machine_playback(reset_to_start=False)

    def _playback_interval_ms(self, speed_value: int) -> int:
        """Converts speed slider values (1..10) to timer intervals in ms."""
        speed = int(np.clip(speed_value, 1, 10))
        return int(np.interp(speed, [1, 10], [950, 120]))

    def _has_playable_machine_data(self) -> bool:
        """Returns True when at least one beam has control points."""
        if not self.model or not self.model.rt_plan:
            return False
        beams = getattr(self.model.rt_plan, "BeamSequence", [])
        if not self.machine_beam_indices:
            self.machine_beam_indices = self._build_machine_beam_indices()
        if not self.machine_beam_indices:
            return False
        return any(
            len(getattr(beams[beam_index], "ControlPointSequence", [])) > 0
            for beam_index in self.machine_beam_indices
            if 0 <= beam_index < len(beams)
        )

    def _stop_machine_playback(self, reset_to_start: bool):
        """Stops playback and optionally rewinds the current beam."""
        self.machine_play_timer.stop()
        self.view.update_machine_play_state(False)
        if reset_to_start and self.model and self.model.rt_plan:
            self._set_machine_state(self.view.machine_beam_slider.value(), 0)
        self._update_machine_control_enablement()

    def _set_machine_state(self, beam_index: int, cp_index: int):
        """Updates beam/cp slider state safely without recursive UI signal storms."""
        if not self.model or not self.model.rt_plan:
            return
        beams = getattr(self.model.rt_plan, "BeamSequence", [])
        if not beams:
            return
        if not self.machine_beam_indices:
            self.machine_beam_indices = self._build_machine_beam_indices()
        if not self.machine_beam_indices:
            return
        beam_index = int(np.clip(beam_index, 0, len(self.machine_beam_indices) - 1))

        beam_changed = beam_index != self.view.machine_beam_slider.value()
        if beam_changed:
            self.view.machine_beam_slider.blockSignals(True)
            self.view.machine_beam_slider.setValue(beam_index)
            self.view.machine_beam_slider.blockSignals(False)
            self._on_machine_beam_changed(beam_index)

        plan_beam_index = self.machine_beam_indices[beam_index]
        num_points = len(getattr(beams[plan_beam_index], "ControlPointSequence", []))
        cp_index = 0 if num_points == 0 else int(np.clip(cp_index, 0, num_points - 1))
        if cp_index != self.view.machine_cp_slider.value() or beam_changed:
            self.view.machine_cp_slider.blockSignals(True)
            self.view.machine_cp_slider.setValue(cp_index)
            self.view.machine_cp_slider.blockSignals(False)
            self._on_machine_cp_changed(cp_index)

    def _update_machine_control_enablement(self):
        """Keeps 3D machine controls in a coherent enabled/disabled state."""
        has_plan = bool(self.model and self.model.rt_plan and getattr(self.model.rt_plan, "BeamSequence", []))
        if has_plan and not self.machine_beam_indices:
            self.machine_beam_indices = self._build_machine_beam_indices()
        has_machine = bool(has_plan and self.machine_beam_indices)
        can_play = has_machine and self._has_playable_machine_data()
        is_playing = self.machine_play_timer.isActive()
        current_cp_count = 0
        if has_machine:
            ui_beam_index = int(np.clip(
                self.view.machine_beam_slider.value(), 0, len(self.machine_beam_indices) - 1
            ))
            beam_index = self.machine_beam_indices[ui_beam_index]
            current_cp_count = len(getattr(self.model.rt_plan.BeamSequence[beam_index], "ControlPointSequence", []))

        self.view.set_machine_play_controls_enabled(can_play)
        self.view.machine_beam_slider.setEnabled(has_machine and (not is_playing or not self.machine_lock_beam))
        self.view.machine_cp_slider.setEnabled(current_cp_count > 0 and not is_playing)
        self.view.machine_lock_beam_checkbox.setEnabled(has_machine)
        self.view.update_machine_play_state(is_playing)

    def _reset_machine_controls(self):
        """Resets machine controls when no RTPLAN/beam data is available."""
        self.machine_play_timer.stop()
        self._last_machine_state = None
        self.machine_beam_indices = []
        self.view.update_machine_play_state(False)
        self.view.machine_beam_label.setText("Beam: N/A")
        self.view.machine_cp_label.setText("Control Point: N/A")
        self.view.gantry_angle_label.setText("Gantry Angle: N/A")
        self.view.machine_beam_slider.setEnabled(False)
        self.view.machine_cp_slider.setEnabled(False)
        self.view.machine_beam_slider.setMinimum(0)
        self.view.machine_beam_slider.setMaximum(0)
        self.view.machine_cp_slider.setMinimum(0)
        self.view.machine_cp_slider.setMaximum(0)
        self.view.machine_beam_slider.setValue(0)
        self.view.machine_cp_slider.setValue(0)
        self.view.set_machine_play_controls_enabled(False)
