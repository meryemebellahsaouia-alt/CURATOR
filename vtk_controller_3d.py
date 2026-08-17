from typing import TYPE_CHECKING, Dict, Optional, Tuple
import logging

import numpy as np
import SimpleITK as sitk
import vtk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkQuad
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkWindowToImageFilter
from vtkmodules.vtkIOImage import vtkPNGWriter

import config

if TYPE_CHECKING:
    from model import DICOMDataModel


MACHINE_ELEMENT_THICKNESS_MM = 15.0
MACHINE_OUTER_EXTENT_MM = 200.0


def _numpy_to_vtk(numpy_array: np.ndarray, ref_sitk_image: sitk.Image) -> vtk.vtkImageData:
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


def find_device_in_sequence(sequence: Sequence, device_type: str) -> Optional[Dataset]:
    """Finds a beam-limiting device by type in a DICOM sequence."""
    for device in sequence or []:
        if getattr(device, "RTBeamLimitingDeviceType", None) == device_type:
            return device
    return None


class VTK3DController:
    def __init__(self, render_window):
        self.render_window = render_window
        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)

        self.renderer.SetBackground(config.COLORS.GetColor3d(config.BACKGROUND_COLOR_BOTTOM))
        self.renderer.SetBackground2(config.COLORS.GetColor3d(config.BACKGROUND_COLOR_TOP))
        self.renderer.SetGradientBackground(True)

        interactor = self.render_window.GetInteractor()
        if interactor is not None:
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.model: Optional["DICOMDataModel"] = None
        self.jaws_actor: Optional[vtkActor] = None
        self.mlc_actor: Optional[vtkActor] = None
        self.ct_volume: Optional[vtk.vtkVolume] = None
        self.machine_actor_cache: Dict[Tuple[int, int], Tuple[Optional[vtkActor], Optional[vtkActor]]] = {}
        self.visibility_states = {"MLC": True, "Jaws": True, "CT": True}

        axes = vtk.vtkAxesActor()
        axes_transform = vtkTransform()
        axes_transform.Scale(1, 1, -1)
        axes.SetUserTransform(axes_transform)
        axes.SetYAxisLabelText("Z")
        axes.SetZAxisLabelText("Y")

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        if interactor is not None:
            self.orientation_widget.SetInteractor(interactor)
        self.orientation_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()

        self.coord_transform = vtkTransform()
        self.coord_transform.RotateX(180)

    def setup_scene(self, model: "DICOMDataModel"):
        """Initializes the 3D VTK scene; structure meshes are added lazily."""
        self.model = model
        self.renderer.RemoveAllViewProps()
        self.machine_actor_cache.clear()
        self.jaws_actor = None
        self.mlc_actor = None

        self.ct_volume = self._create_ct_volume()
        if self.ct_volume:
            self.ct_volume.SetUserTransform(self.coord_transform)
            self.ct_volume.SetVisibility(self.visibility_states.get("CT", True))
            self.renderer.AddVolume(self.ct_volume)

        for info in self.model.structures.values():
            actor_3d = info.get("actor_3d")
            if actor_3d:
                actor_3d.SetUserTransform(self.coord_transform)
                self.renderer.AddActor(actor_3d)

        if self.model.rt_plan and getattr(self.model.rt_plan, "BeamSequence", None):
            self.update_machine_actors(beam_index=0, cp_index=0)
        else:
            self._safe_render()
        self.reset_camera()

    def update_machine_actors(self, beam_index: int, cp_index: int):
        """Removes old machine actors and adds new ones for the selected state."""
        if not self.model or not self.model.rt_plan:
            return

        if self.jaws_actor and self.renderer.HasViewProp(self.jaws_actor):
            self.renderer.RemoveActor(self.jaws_actor)
        if self.mlc_actor and self.renderer.HasViewProp(self.mlc_actor):
            self.renderer.RemoveActor(self.mlc_actor)
        self.jaws_actor = None
        self.mlc_actor = None

        try:
            beams = getattr(self.model.rt_plan, "BeamSequence", [])
            if not beams:
                self._safe_render()
                return

            beam_index = int(np.clip(beam_index, 0, len(beams) - 1))
            beam = beams[beam_index]
            control_points = getattr(beam, "ControlPointSequence", [])
            if not control_points:
                self._safe_render()
                return

            cp_index = int(np.clip(cp_index, 0, len(control_points) - 1))
            cache_key = (beam_index, cp_index)
            cached = self.machine_actor_cache.get(cache_key)
            if cached is None:
                control_point = control_points[cp_index]
                jaws_actor = self._create_jaws_actor(control_point, beam, cp_index)
                mlc_actor = self._create_mlc_actor(control_point, beam, cp_index)
                self.machine_actor_cache[cache_key] = (jaws_actor, mlc_actor)
            else:
                jaws_actor, mlc_actor = cached

            self.jaws_actor = jaws_actor
            if self.jaws_actor:
                self.jaws_actor.SetVisibility(self.visibility_states.get("Jaws", True))
                self.renderer.AddActor(self.jaws_actor)

            self.mlc_actor = mlc_actor
            if self.mlc_actor:
                self.mlc_actor.SetVisibility(self.visibility_states.get("MLC", True))
                self.renderer.AddActor(self.mlc_actor)

        except Exception as exc:
            logging.warning(
                "Could not display machine for beam %s, cp %s: %s",
                beam_index,
                cp_index,
                exc,
            )

        self._safe_render()

    def set_structure_visibility(self, roi_number: int, is_visible: bool, render: bool = True):
        """Sets 3D structure visibility, lazily creating the actor when needed."""
        if not self.model or roi_number not in self.model.structures:
            return

        actor_3d = self.model.structures[roi_number].get("actor_3d")
        if actor_3d is None and is_visible:
            actor_3d = self.model.get_or_create_3d_structure_actor(roi_number)
            if actor_3d:
                actor_3d.SetUserTransform(self.coord_transform)
                self.renderer.AddActor(actor_3d)

        if actor_3d and not self.renderer.HasViewProp(actor_3d):
            actor_3d.SetUserTransform(self.coord_transform)
            self.renderer.AddActor(actor_3d)

        if actor_3d:
            actor_3d.SetVisibility(is_visible)
            if render:
                self._safe_render()

    def set_visibility(self, object_name: str, is_visible: bool):
        """Sets the visibility for a specific object in the 3D scene."""
        self.visibility_states[object_name] = is_visible
        actor_to_toggle = None
        if object_name == "MLC":
            actor_to_toggle = self.mlc_actor
        elif object_name == "CT":
            actor_to_toggle = self.ct_volume
        elif object_name == "Jaws":
            actor_to_toggle = self.jaws_actor

        if actor_to_toggle:
            actor_to_toggle.SetVisibility(is_visible)
            self._safe_render()

    def save_screenshot(self, filepath: str):
        """Saves the current render window view to a PNG file."""
        self._safe_render()
        window_to_image_filter = vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetInputBufferTypeToRGB()
        window_to_image_filter.ReadFrontBufferOff()
        window_to_image_filter.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(filepath)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def reset_camera(self):
        """Points the camera at the center of the scene."""
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._safe_render()

    def _safe_render(self):
        """Renders the window unless the render pipeline is already torn down."""
        try:
            if self.render_window is not None:
                self.render_window.Render()
        except Exception:
            return

    def _create_ct_volume(self) -> Optional[vtk.vtkVolume]:
        """Creates a VTK volume for the CT data from the model."""
        if self.model is None or self.model.ct_image is None:
            return None

        ct_array = sitk.GetArrayFromImage(self.model.ct_image).astype(np.int16)
        vtk_image_data = _numpy_to_vtk(ct_array, self.model.ct_image)

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_image_data)

        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()

        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(-500, 0.0)
        opacity_func.AddPoint(100, 0.2)
        opacity_func.AddPoint(1000, 0.8)
        prop.SetScalarOpacity(opacity_func)

        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(-500, 0.2, 0.2, 0.2)
        color_func.AddRGBPoint(100, 0.8, 0.8, 0.8)
        color_func.AddRGBPoint(1000, 1.0, 1.0, 1.0)
        prop.SetColor(color_func)

        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)
        return volume

    def _get_control_point_device(self, beam: Dataset, cp_index: int, device_type: str) -> Optional[Dataset]:
        """Returns the most recent control-point device definition for a given type."""
        control_points = getattr(beam, "ControlPointSequence", [])
        if not control_points:
            return None
        cp_index = int(np.clip(cp_index, 0, len(control_points) - 1))
        for idx in range(cp_index, -1, -1):
            cp = control_points[idx]
            sequence = getattr(cp, "BeamLimitingDevicePositionSequence", None)
            if not sequence:
                continue
            device = find_device_in_sequence(sequence, device_type)
            if device is not None:
                return device
        return None

    def _get_control_point_angle(self, beam: Dataset, cp_index: int, attribute_name: str, default: float = 0.0) -> float:
        """Returns a control-point angle using backward fallback if the current CP omits it."""
        control_points = getattr(beam, "ControlPointSequence", [])
        if not control_points:
            return float(default)
        cp_index = int(np.clip(cp_index, 0, len(control_points) - 1))
        for idx in range(cp_index, -1, -1):
            value = getattr(control_points[idx], attribute_name, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    break
        return float(default)

    def _get_source_to_device_distance(self, beam: Dataset, device_type: str) -> Optional[float]:
        """Returns the geometric source-to-device distance for one beam-limiting device."""
        device_sequence = getattr(beam, "BeamLimitingDeviceSequence", None)
        if not device_sequence:
            return None
        device = find_device_in_sequence(device_sequence, device_type)
        if device is None:
            return None
        value = getattr(device, "SourceToBeamLimitingDeviceDistance", None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_jaw_source_to_device_distance(self, beam: Dataset) -> Optional[float]:
        """Resolves a usable jaw-placement distance with tolerant fallbacks.

        Preferred order:
        1. Jaw X/Y device distances.
        2. MLC device distance.
        3. Any available beam-limiting device distance.

        This preserves jaw rendering for plans where jaw devices omit
        SourceToBeamLimitingDeviceDistance but another beam-limiting device
        still provides a valid geometric placement distance.
        """
        preferred = [
            self._get_source_to_device_distance(beam, config.JAWS_X_DEVICE_TYPE),
            self._get_source_to_device_distance(beam, config.JAWS_Y_DEVICE_TYPE),
        ]
        jaw_distances = [float(d) for d in preferred if d is not None]
        if jaw_distances:
            return max(jaw_distances)

        mlc_distance = self._get_source_to_device_distance(beam, config.MLC_DEVICE_TYPE)
        if mlc_distance is not None:
            logging.info(
                "Jaw device distance missing; falling back to MLC device distance for jaw placement."
            )
            return float(mlc_distance)

        device_sequence = getattr(beam, "BeamLimitingDeviceSequence", None) or []
        fallback_distances = []
        for device in device_sequence:
            value = getattr(device, "SourceToBeamLimitingDeviceDistance", None)
            if value is None:
                continue
            try:
                fallback_distances.append(float(value))
            except (TypeError, ValueError):
                continue
        if fallback_distances:
            logging.info(
                "Jaw and MLC device distances missing; falling back to another beam-limiting device distance for jaw placement."
            )
            return max(fallback_distances)

        return None

    def _create_jaws_actor(self, control_point: Dataset, beam: Dataset, cp_index: int) -> Optional[vtkActor]:
        """Creates a VTK actor representing the treatment jaws."""
        x_jaws_device = self._get_control_point_device(beam, cp_index, config.JAWS_X_DEVICE_TYPE)
        y_jaws_device = self._get_control_point_device(beam, cp_index, config.JAWS_Y_DEVICE_TYPE)
        if not x_jaws_device or not y_jaws_device:
            logging.debug("Jaws not found for beam-limiting devices at control point %s.", cp_index)
            return None

        try:
            x_field = [float(v) for v in x_jaws_device.LeafJawPositions]
            y_field = [float(v) for v in y_jaws_device.LeafJawPositions]
        except Exception:
            logging.warning("Invalid jaw positions at control point %s.", cp_index)
            return None
        if len(x_field) < 2 or len(y_field) < 2:
            return None

        points = vtkPoints()
        cells = vtkCellArray()
        thickness = MACHINE_ELEMENT_THICKNESS_MM
        extent = MACHINE_OUTER_EXTENT_MM

        self._create_box(points, cells, -extent, extent, -thickness, thickness, y_field[1], extent)
        self._create_box(points, cells, -extent, extent, -thickness, thickness, -extent, y_field[0])
        self._create_box(points, cells, -extent, x_field[0], -thickness, thickness, y_field[0], y_field[1])
        self._create_box(points, cells, x_field[1], extent, -thickness, thickness, y_field[0], y_field[1])

        jaws_polydata = vtkPolyData()
        jaws_polydata.SetPoints(points)
        jaws_polydata.SetPolys(cells)

        gantry_angle = 360.0 - self._get_control_point_angle(beam, cp_index, "GantryAngle", default=0.0)
        collimator_angle = 360.0 - self._get_control_point_angle(beam, cp_index, "BeamLimitingDeviceAngle", default=0.0)
        try:
            sad = float(getattr(beam, "SourceAxisDistance", 0.0))
        except (TypeError, ValueError):
            sad = 0.0

        source_to_device_dist = self._resolve_jaw_source_to_device_distance(beam)
        if source_to_device_dist is None:
            logging.warning(
                "Cannot determine jaw device distance from jaw, MLC, or other beam-limiting devices. Actor will not be drawn."
            )
            return None

        device_position = sad - max(source_to_device_dist - (2.0 * thickness), 0.0)

        transform = vtkTransform()
        transform.RotateZ(gantry_angle)
        transform.Translate(0.0, device_position, 0.0)
        transform.RotateY(collimator_angle)

        transform_filter = vtkTransformFilter()
        transform_filter.SetInputData(jaws_polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(config.COLORS.GetColor3d(config.JAWS_COLOR))
        return actor

    def _create_mlc_actor(self, control_point: Dataset, beam: Dataset, cp_index: int) -> Optional[vtkActor]:
        """Creates a VTK actor representing the MLC."""
        mlc_device = find_device_in_sequence(getattr(beam, "BeamLimitingDeviceSequence", []), config.MLC_DEVICE_TYPE)
        mlc_positions_device = self._get_control_point_device(beam, cp_index, config.MLC_DEVICE_TYPE)
        if not mlc_device or not mlc_positions_device:
            logging.debug("MLC data not found for control point %s.", cp_index)
            return None

        try:
            leaf_boundaries = [float(v) for v in mlc_device.LeafPositionBoundaries]
            leaf_positions = [float(v) for v in mlc_positions_device.LeafJawPositions]
            num_leaf_pairs = int(mlc_device.NumberOfLeafJawPairs)
        except Exception:
            logging.warning("Invalid MLC geometry at control point %s.", cp_index)
            return None

        if len(leaf_boundaries) < num_leaf_pairs + 1 or len(leaf_positions) < (2 * num_leaf_pairs):
            logging.warning("Incomplete MLC geometry at control point %s.", cp_index)
            return None

        points = vtkPoints()
        cells = vtkCellArray()
        thickness = MACHINE_ELEMENT_THICKNESS_MM
        extent_x = MACHINE_OUTER_EXTENT_MM

        for i in range(num_leaf_pairs):
            y_min = leaf_boundaries[i]
            y_max = leaf_boundaries[i + 1]
            x_min_a = -extent_x
            x_max_a = leaf_positions[i]
            self._create_box(points, cells, x_min_a, x_max_a, -thickness, thickness, y_min, y_max)

            x_min_b = leaf_positions[i + num_leaf_pairs]
            x_max_b = extent_x
            self._create_box(points, cells, x_min_b, x_max_b, -thickness, thickness, y_min, y_max)

        mlc_polydata = vtkPolyData()
        mlc_polydata.SetPoints(points)
        mlc_polydata.SetPolys(cells)

        gantry_angle = 360.0 - self._get_control_point_angle(beam, cp_index, "GantryAngle", default=0.0)
        collimator_angle = 360.0 - self._get_control_point_angle(beam, cp_index, "BeamLimitingDeviceAngle", default=0.0)
        try:
            sad = float(getattr(beam, "SourceAxisDistance", 0.0))
        except (TypeError, ValueError):
            sad = 0.0

        source_to_device_dist = self._get_source_to_device_distance(beam, config.MLC_DEVICE_TYPE)
        if source_to_device_dist is None:
            logging.warning(
                "Device '%s' is missing SourceToBeamLimitingDeviceDistance. Cannot draw MLCs.",
                getattr(mlc_device, "RTBeamLimitingDeviceType", config.MLC_DEVICE_TYPE),
            )
            return None

        device_position = sad - source_to_device_dist

        transform = vtkTransform()
        transform.RotateZ(gantry_angle)
        transform.Translate(0.0, device_position, 0.0)
        transform.RotateY(collimator_angle)

        transform_filter = vtkTransformFilter()
        transform_filter.SetInputData(mlc_polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(config.COLORS.GetColor3d(config.MLC_COLOR))
        return actor

    def _create_box(
        self,
        points: vtkPoints,
        cells: vtkCellArray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ):
        """Creates one 3D box and adds it to the provided VTK point/cell buffers."""
        base_id = points.GetNumberOfPoints()

        points.InsertNextPoint(x_min, y_min, z_min)
        points.InsertNextPoint(x_max, y_min, z_min)
        points.InsertNextPoint(x_max, y_max, z_min)
        points.InsertNextPoint(x_min, y_max, z_min)
        points.InsertNextPoint(x_min, y_min, z_max)
        points.InsertNextPoint(x_max, y_min, z_max)
        points.InsertNextPoint(x_max, y_max, z_max)
        points.InsertNextPoint(x_min, y_max, z_max)

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ]

        for face in faces:
            quad = vtkQuad()
            for i, point_index in enumerate(face):
                quad.GetPointIds().SetId(i, base_id + point_index)
            cells.InsertNextCell(quad)
