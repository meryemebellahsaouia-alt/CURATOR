# vtk_controller.py
"""
Handles all VTK-related logic for visualization and interaction.

The VTKController class manages the VTK scene, including CT and dose actors,
color lookup tables, and the scalar bar. The UnifiedViewerInteractorStyle
class defines custom user interaction logic for the VTK render window, such
as scrolling through slices with the mouse wheel.
"""
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import config

if TYPE_CHECKING:
    from model import DICOMDataModel
    from view import MainWindow  

class VTKController:
    """Manages the VTK rendering scene and its components."""

    def __init__(self, vtk_widget: QVTKRenderWindowInteractor):
        self.renderer = vtk.vtkRenderer()
        vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        self.renderer.SetBackground(config.COLORS.GetColor3d(config.BACKGROUND_COLOR_BOTTOM))
        self.renderer.SetBackground2(config.COLORS.GetColor3d(config.BACKGROUND_COLOR_TOP))
        self.renderer.SetGradientBackground(True)
        
        self.model: Optional['DICOMDataModel'] = None
        self.interactor_style: Optional['UnifiedViewerInteractorStyle'] = None

        self.ct_actor: Optional[vtk.vtkImageActor] = None
        self.dose_actor: Optional[vtk.vtkImageActor] = None
        self.scalar_bar: Optional[vtk.vtkScalarBarActor] = None
        
        self.dose_lut: Optional[vtk.vtkLookupTable] = None
        self.lut_for_legend_gy: Optional[vtk.vtkLookupTable] = None
        self.lut_for_legend_pct: Optional[vtk.vtkLookupTable] = None
        self.original_lut_array: Optional[np.ndarray] = None
        
        self.dose_color_mapper: Optional[vtk.vtkImageMapToColors] = None

    def _render_if_possible(self) -> None:
        """Requests a render only when the render window is still valid."""
        try:
            render_window = self.renderer.GetRenderWindow()
            if render_window is not None:
                render_window.Render()
        except Exception:
            # Rendering can fail transiently during shutdown; keep this non-fatal.
            return

    # --- 1. Public Methods ---

    def setup_scene(self, model: 'DICOMDataModel'):
        """Initializes the entire VTK scene from the data model."""
        if model.ct_image is None or model.dose_image is None:
            raise ValueError("CT and dose images must be loaded before VTK scene setup.")
        if model.masked_ct_array is None or model.masked_dose_array is None:
            raise ValueError("Masked CT and dose arrays must be prepared before VTK scene setup.")

        self.model = model
        ct_array_int16 = model.masked_ct_array.astype(np.int16, copy=False)
        ct_image_vtk = self._numpy_to_vtk(ct_array_int16, model.ct_image)
        dose_image_vtk = self._numpy_to_vtk(model.masked_dose_array.astype(np.float32, copy=False), model.dose_image)

        self._create_luts(model.dose_max_in_gy or 0.0, model.prescription_dose or 0.0)
        self._create_ct_actor(ct_image_vtk)
        self._create_dose_actor(dose_image_vtk)
        self._create_scalar_bar()

        self.renderer.RemoveAllViewProps()
        if self.ct_actor:
            self.renderer.AddViewProp(self.ct_actor)
        if self.dose_actor:
            self.renderer.AddViewProp(self.dose_actor)
        if self.scalar_bar:
            self.renderer.AddViewProp(self.scalar_bar)

        for slice_actors in model.contours_by_slice.values():
            for actor in slice_actors:
                self.renderer.AddViewProp(actor)

    def create_interactor_style(self, model: 'DICOMDataModel', view: 'MainWindow') -> 'UnifiedViewerInteractorStyle':
        """Creates and returns a custom interactor style."""
        self.interactor_style = UnifiedViewerInteractorStyle(model, view, self)
        return self.interactor_style

    def update_dose_visibility(self, threshold_dose_gy: float):
        """Updates dose visibility by modifying the LUT alpha channel only once."""
        if not self.dose_lut or self.original_lut_array is None or self.model is None:
            return
        if self.model.dose_max_in_gy is None:
            return

        max_dose_gy = float(self.model.dose_max_in_gy)
        num_colors = int(self.dose_lut.GetNumberOfTableValues())
        modified_lut = self.original_lut_array.copy()
        threshold_idx = self._dose_threshold_to_lut_index(float(threshold_dose_gy), max_dose_gy, num_colors)

        modified_lut[:, 3] = 0
        if threshold_idx < num_colors:
            modified_lut[threshold_idx:num_colors, 3] = self._dose_alpha_value()
        modified_lut[0, 3] = 0  # Ensure zero dose is always transparent.

        vtk_array = numpy_support.numpy_to_vtk(modified_lut, deep=True)
        self.dose_lut.SetTable(vtk_array)
        self.dose_lut.Build()

        if self.dose_color_mapper:
            self.dose_color_mapper.Modified()

    def update_dose_legend_units(self, is_percentage: bool):
        """Switches the scalar bar between Gy and %."""
        if not self.scalar_bar: return
        
        if is_percentage:
            self.scalar_bar.SetLookupTable(self.lut_for_legend_pct)
            self.scalar_bar.SetTitle("Dose (%)")
        else:
            self.scalar_bar.SetLookupTable(self.lut_for_legend_gy)
            self.scalar_bar.SetTitle(f"Dose ({self.model.dose_unit})")
        
        self.scalar_bar.GetLabelTextProperty().SetFontSize(10)
        self.scalar_bar.GetTitleTextProperty().SetFontSize(12)
        self._render_if_possible()

    def reset_camera(self):
        """
        Fits the camera to visible patient/model bounds with a stable axial
        orientation and viewport-aware scale.
        """
        if not self.ct_actor:
            return

        bounds = self.renderer.ComputeVisiblePropBounds()
        if (
            not bounds
            or len(bounds) != 6
            or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
            or bounds[2] >= bounds[3]
            or bounds[4] >= bounds[5]
        ):
            bounds = self.ct_actor.GetInput().GetBounds()
        if (
            not bounds
            or len(bounds) != 6
            or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
            or bounds[2] >= bounds[3]
            or bounds[4] >= bounds[5]
        ):
            return

        center_x = (bounds[0] + bounds[1]) / 2.0
        center_y = (bounds[2] + bounds[3]) / 2.0
        center_z = (bounds[4] + bounds[5]) / 2.0

        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(center_x, center_y, center_z)
        camera.SetViewUp(0, -1, 0)

        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        depth = bounds[5] - bounds[4]

        window_width, window_height = self.renderer.GetRenderWindow().GetSize()
        aspect = max(float(window_width) / max(float(window_height), 1.0), 1e-3)
        fit_scale = max(height / 2.0, (width / 2.0) / aspect)
        camera.SetParallelScale(max(fit_scale * 1.08, 1.0))

        distance = max(depth * 3.0, 500.0)
        camera.SetPosition(center_x, center_y, center_z - distance)

        self.renderer.ResetCameraClippingRange()
        
    def update_ct_window_level(self, window: int, level: int):
        """Sets the color window and level for the CT actor."""
        if self.ct_actor:
            self.ct_actor.GetProperty().SetColorWindow(window)
            self.ct_actor.GetProperty().SetColorLevel(level)

    # --- 2. Private Helper Methods ---

    def _create_lookup_table(self, table_range: Tuple[float, float], num_values: int = 256,
                             hue_range: tuple = (0.667, 0.0)) -> vtk.vtkLookupTable:
        """Factory method to create and configure a VTK lookup table."""
        low, high = float(table_range[0]), float(table_range[1])
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            high = max(low + 1.0, 1.0)

        lut = vtk.vtkLookupTable()
        lut.SetTableRange(low, high)
        lut.SetHueRange(hue_range)
        lut.SetNumberOfTableValues(int(max(2, num_values)))
        lut.Build()
        return lut

    def _create_luts(self, dose_max: float, prescription_dose: float):
        """Creates all necessary lookup tables for dose visualization."""
        dose_max = float(dose_max) if np.isfinite(dose_max) else 0.0
        dose_max = max(dose_max, 0.0)
        self.dose_lut = self._create_lookup_table(table_range=(0.0, dose_max))
        vtk_array = self.dose_lut.GetTable()
        self.original_lut_array = numpy_support.vtk_to_numpy(vtk_array).copy()
        self.original_lut_array[:, 3] = self._dose_alpha_value()
        self.original_lut_array[0, 3] = 0
        self.dose_lut.SetTable(numpy_support.numpy_to_vtk(self.original_lut_array, deep=True))
        self.dose_lut.Build()

        self.lut_for_legend_gy = self._create_lookup_table(table_range=(0.0, dose_max))

        max_dose_pct = (dose_max / prescription_dose) * 100.0 if prescription_dose > 0 else 100.0
        self.lut_for_legend_pct = self._create_lookup_table(table_range=(0.0, max_dose_pct))

    def _create_ct_actor(self, ct_image_vtk: vtk.vtkImageData):
        """Creates the vtkImageActor for the CT data."""
        self.ct_actor = vtk.vtkImageActor()
        self.ct_actor.GetMapper().SetInputData(ct_image_vtk)
        self.ct_actor.GetProperty().SetColorWindow(config.DEFAULT_CT_WINDOW)
        self.ct_actor.GetProperty().SetColorLevel(config.DEFAULT_CT_LEVEL)

    def _create_dose_actor(self, dose_image_vtk: vtk.vtkImageData):
        """Creates the vtkImageActor for the dose data."""
        self.dose_color_mapper = vtk.vtkImageMapToColors()
        self.dose_color_mapper.SetInputData(dose_image_vtk)
        self.dose_color_mapper.SetLookupTable(self.dose_lut)
        self.dose_color_mapper.PassAlphaToOutputOn()
        self.dose_color_mapper.Update()

        self.dose_actor = vtk.vtkImageActor()
        self.dose_actor.GetMapper().SetInputConnection(self.dose_color_mapper.GetOutputPort())
        # Opacity is handled through the LUT alpha channel to avoid double attenuation.
        self.dose_actor.GetProperty().SetOpacity(1.0)
        self.dose_actor.SetPickable(False)

    def _create_scalar_bar(self):
        """Creates and configures the scalar bar actor."""
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut_for_legend_gy)
        dose_unit = getattr(self.model, "dose_unit", "Gy") if self.model is not None else "Gy"
        self.scalar_bar.SetTitle(f"Dose ({dose_unit})")
        self.scalar_bar.SetNumberOfLabels(6)
        self.scalar_bar.SetUnconstrainedFontSize(True)
        self.scalar_bar.GetLabelTextProperty().SetFontSize(10)
        self.scalar_bar.GetTitleTextProperty().SetFontSize(10)
        self.scalar_bar.SetWidth(0.08)
        self.scalar_bar.SetHeight(0.75)
        self.scalar_bar.SetPosition(0.9, 0.14)

    def _numpy_to_vtk(self, numpy_array: np.ndarray, ref_sitk_image: sitk.Image) -> vtk.vtkImageData:
        """Converts a NumPy array to vtkImageData, copying spatial metadata."""
        vtk_data = numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True)
        
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(ref_sitk_image.GetSize())
        vtk_image.SetOrigin(ref_sitk_image.GetOrigin())
        vtk_image.SetSpacing(ref_sitk_image.GetSpacing())
        vtk_image.GetPointData().SetScalars(vtk_data)
        
        direction = np.array(ref_sitk_image.GetDirection()).reshape(3, 3)
        vtk_matrix = vtk.vtkMatrix3x3()
        for i in range(3):
            for j in range(3):
                vtk_matrix.SetElement(i, j, direction[i, j])
        vtk_image.SetDirectionMatrix(vtk_matrix)
        
        return vtk_image

    @staticmethod
    def _dose_alpha_value() -> int:
        """Returns the configured global dose alpha in 8-bit LUT units."""
        return int(np.clip(round(float(config.DOSE_OPACITY) * 255.0), 0, 255))

    @staticmethod
    def _dose_threshold_to_lut_index(threshold_dose_gy: float, max_dose_gy: float, num_colors: int) -> int:
        """Maps a physical dose threshold to a LUT table index safely."""
        if num_colors <= 0:
            return 0
        if max_dose_gy <= 0:
            return num_colors
        clamped_threshold = float(np.clip(threshold_dose_gy, 0.0, max_dose_gy))
        return int(np.clip(round((clamped_threshold / max_dose_gy) * (num_colors - 1)), 0, num_colors - 1))


class UnifiedViewerInteractorStyle(vtk.vtkInteractorStyleImage):
    """Custom interactor style for 2D slice viewing and navigation."""

    def __init__(self, model: 'DICOMDataModel', view: 'MainWindow', vtk_controller: VTKController):
        super().__init__()
        self.model = model
        self.view = view
        self.vtk_controller = vtk_controller
        
        self._min_slice = 0
        self._max_slice = model.ct_image.GetSize()[2] - 1
        self._slice = self._initial_slice_index()
        
        
        self.AddObserver("MouseWheelForwardEvent", self._on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._on_mouse_wheel_backward)
        self.AddObserver("KeyPressEvent", self._on_key_press)

    def _initial_slice_index(self) -> int:
        """Picks a stable, information-rich startup slice for the 2D viewer."""
        if self._max_slice < 0:
            return 0

        ct_array = getattr(self.model, "masked_ct_array", None)
        if isinstance(ct_array, np.ndarray) and ct_array.ndim == 3 and ct_array.shape[0] > 0:
            occupancy = np.mean(ct_array > -500, axis=(1, 2))
            best = int(np.argmax(occupancy))
            if occupancy[best] > 0.005:
                return int(np.clip(best, self._min_slice, self._max_slice))

        return int((self._min_slice + self._max_slice) // 2)

    # --- Public Methods ---

    def update_slice_view(self):
        """Updates the displayed 2D slice for all image actors."""
        if not self.vtk_controller.ct_actor:
            return

        self._slice = int(np.clip(self._slice, self._min_slice, self._max_slice))
        extent = self.vtk_controller.ct_actor.GetInput().GetExtent()
        slice_extent = (extent[0], extent[1], extent[2], extent[3], self._slice, self._slice)

        self.vtk_controller.ct_actor.SetDisplayExtent(slice_extent)
        if self.vtk_controller.dose_actor:
            self.vtk_controller.dose_actor.SetDisplayExtent(slice_extent)

        interactor = self.GetInteractor()
        if interactor is not None and interactor.GetRenderWindow() is not None:
            renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
            if renderer is not None:
                renderer.ResetCameraClippingRange()
        self.update_contours()
        self._update_status_bar()

    def update_contours(self):
        """Updates the visibility of ROI contours based on the current slice and view checkboxes."""
        all_actors = (actor for roi in self.model.structures.values() for actor in roi['actors'])
        for actor in all_actors:
            actor.SetVisibility(False)

        actors_on_this_slice = self.model.contours_by_slice.get(self._slice, [])
        for actor in actors_on_this_slice:
            roi_number = self.model.actor_to_roi_map.get(id(actor))
            if roi_number is not None:
                checkbox = self.view.structure_checkboxes.get(roi_number)
                if checkbox and checkbox.isChecked():
                    actor.SetVisibility(True)

        interactor = self.GetInteractor()
        if interactor is not None and interactor.GetRenderWindow() is not None:
            interactor.GetRenderWindow().Render()

    def set_slice(self, slice_index: int):
        """Sets an absolute slice index, clamped to the valid range."""
        clamped = int(np.clip(slice_index, self._min_slice, self._max_slice))
        if clamped == self._slice:
            return
        self._slice = clamped
        self.update_slice_view()

    def get_slice(self) -> int:
        """Returns the current zero-based slice index."""
        return int(self._slice)

    def get_slice_bounds(self) -> tuple:
        """Returns (min_slice, max_slice) for the current image."""
        return int(self._min_slice), int(self._max_slice)

    # --- Private Event Handlers & Helpers ---

    def _move_slice(self, direction: int):
        """Moves the slice index by a given direction and updates the view."""
        new_slice = np.clip(self._slice + direction, self._min_slice, self._max_slice)
        if new_slice != self._slice:
            self._slice = new_slice
            self.update_slice_view()

    def _update_status_bar(self):
        """Updates the slice number in the main window's status bar."""
        status_text = f"Slice: {self._slice + 1} / {self._max_slice + 1}"
        self.view.last_slice_status = status_text
        if hasattr(self.view, "update_slice_navigation"):
            self.view.update_slice_navigation(self._slice, self._max_slice)
        if self.view.tab_widget.currentIndex() == 0:
            self.view.update_status_label(status_text)

    def _on_mouse_wheel_forward(self, obj, event):
        self._move_slice(1)

    def _on_mouse_wheel_backward(self, obj, event):
        self._move_slice(-1)

    def _on_key_press(self, obj, event):
        """Handles key press events for slice navigation."""
        key = self.GetInteractor().GetKeySym()
        if key in ("Right", "Up", "k", "K", "Prior"):
            self._move_slice(1)
        elif key in ("Left", "Down", "j", "J", "Next"):
            self._move_slice(-1)
        
        self.OnKeyPress() 
        
