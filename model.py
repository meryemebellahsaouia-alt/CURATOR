# model.py
"""
Contains the DICOMDataModel class, which is responsible for all data handling.

This includes loading DICOM files (CT, RTSTRUCT, RTDOSE, RTPLAN), processing
the data (resampling, parsing structures, creating masks), and providing
methods to access this data (e.g., for DVH calculation).
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from vtkmodules.util import numpy_support
import numpy as np
import pydicom
import SimpleITK as sitk
import vtk
from skimage.draw import polygon

from config import CONTOUR_LINE_WIDTH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LoadCancelledError(Exception):
    """Raised when a user-initiated cancel request interrupts data loading."""


class DICOMDataModel:
    """Manages loading and processing of all DICOM data."""

    _MASK_BACKEND_LEGACY = "legacy"
    _MASK_BACKEND_VTK = "vtk"
    _VALID_MASK_BACKENDS = {_MASK_BACKEND_LEGACY, _MASK_BACKEND_VTK}
    _UNION_BRIDGE_MAX_GAP_SLICES = 2
    _UNION_BRIDGE_MIN_IOU = 0.85
    _UNION_BRIDGE_MAX_AREA_RATIO = 1.40
    _UNION_BRIDGE_DISJOINT_MAX_IOU = 0.01
    _UNION_BRIDGE_DISJOINT_MIN_CENTROID_SHIFT_MM = 40.0
    _UNION_BRIDGE_DISJOINT_MIN_GAP_MM = 30.0
    _SIGNED_DISTANCE_DISJOINT_MAX_IOU = 0.05
    _SIGNED_DISTANCE_DISJOINT_MIN_CENTROID_SHIFT_MM = 15.0
    _SIGNED_DISTANCE_DISJOINT_MIN_GAP_MM = 15.0
    _SIGNED_DISTANCE_TOPOLOGY_MISMATCH_MIN_GAP_MM = 8.0

    def __init__(self):
        self.ct_image: Optional[sitk.Image] = None
        self.rt_struct: Optional[pydicom.FileDataset] = None
        self.dose_image: Optional[sitk.Image] = None
        self.rt_plan: Optional[pydicom.FileDataset] = None
        
        self.dose_path: Optional[str] = None
        self.prescription_dose: Optional[float] = None
        self.dose_unit: str = "Gy"
        
        self.structures: Dict[int, Dict] = {}
        self.body_roi_number: int = -1
        
        # Data for visualization and calculation
        self.contours_by_slice: Dict[int, List[vtk.vtkActor]] = defaultdict(list)
        self.actor_to_roi_map: Dict[int, int] = {}
        self.masked_ct_array: Optional[np.ndarray] = None
        self.masked_dose_array: Optional[np.ndarray] = None
        self.dose_max_in_gy: Optional[float] = None
        self.max_dose_as_percent_of_rx: Optional[float] = None
        # A dictionary to store calculated DVH results to avoid re-computation
        self.dvh_cache: Dict[int, Dict] = {}
        self.ci_cache: Dict[int, float] = {}
        self.structure_mask_cache: Dict[int, np.ndarray] = {}
        self.structure_mask_backend: str = self._MASK_BACKEND_LEGACY
        self.patient_folder: Optional[str] = None
        self._colorblind_palette_enabled: bool = False
        self._colorblind_palette: List[Tuple[float, float, float]] = [
            (0.0, 0.45, 0.70),   # blue
            (0.90, 0.62, 0.00),  # orange
            (0.00, 0.60, 0.50),  # bluish green
            (0.80, 0.47, 0.65),  # reddish purple
            (0.95, 0.90, 0.25),  # yellow
            (0.35, 0.70, 0.90),  # sky blue
            (0.80, 0.60, 0.70),  # pink
            (0.70, 0.70, 0.70),  # grey
        ]

    def set_structure_mask_backend(self, backend: str) -> None:
        """Selects the contour-to-mask rasterization backend."""
        backend_name = str(backend or "").strip().lower()
        if backend_name not in self._VALID_MASK_BACKENDS:
            raise ValueError(
                f"Invalid mask backend '{backend}'. "
                f"Expected one of: {sorted(self._VALID_MASK_BACKENDS)}"
            )
        if backend_name == self.structure_mask_backend:
            return
        self.structure_mask_backend = backend_name
        self.structure_mask_cache.clear()

    def _reset_loaded_state(self) -> None:
        """Clears all patient-specific runtime state before a new load."""
        self.ct_image = None
        self.rt_struct = None
        self.dose_image = None
        self.rt_plan = None
        self.dose_path = None
        self.prescription_dose = None
        self.dose_unit = "Gy"
        self.structures.clear()
        self.body_roi_number = -1
        self.contours_by_slice.clear()
        self.actor_to_roi_map.clear()
        self.masked_ct_array = None
        self.masked_dose_array = None
        self.dose_max_in_gy = None
        self.max_dose_as_percent_of_rx = None
        self.dvh_cache.clear()
        self.ci_cache.clear()
        self.structure_mask_cache.clear()
        self.patient_folder = None

    def _dose_is_absolute_gy(self) -> bool:
        """Returns True when dose values represent absolute Gy dose."""
        return str(self.dose_unit or "").strip().upper() == "GY"

    # --- 1. Public API Methods ---
    
    def load_data(self, patient_folder: str, ct_series_uid: str, rtstruct_path: str,
                  dose_path: str, rtplan_path: str, progress_callback: callable = None,
                  is_cancelled: Optional[Callable[[], bool]] = None):
        """
        Loads and processes all required DICOM files.
        Raises specific exceptions on failure.
        """
        self._reset_loaded_state()
        self.patient_folder = patient_folder
        self.dose_path = dose_path
        total_steps = 8
        def report_progress(step_number, message):
            logging.info(message)
            if progress_callback:
                percentage = int((step_number / total_steps) * 100)
                progress_callback(percentage, message)

        def check_cancelled(context_message: str):
            if is_cancelled and is_cancelled():
                raise LoadCancelledError(context_message)
                
        try:
            check_cancelled("Loading cancelled before CT loading.")
            report_progress(1, "Loading CT image series...")
            self.ct_image = self._load_ct_image(patient_folder, ct_series_uid)

            check_cancelled("Loading cancelled before RT Structure Set loading.")
            report_progress(2, "Loading RT Structure Set...")
            self.rt_struct = pydicom.dcmread(rtstruct_path)

            check_cancelled("Loading cancelled before RT Plan loading.")
            report_progress(3, "Loading RT Plan...")
            self.rt_plan = pydicom.dcmread(rtplan_path)
            
            check_cancelled("Loading cancelled before RT Dose metadata read.")
            report_progress(4, "Reading RT Dose metadata...")
            self._update_dose_unit_from_file(dose_path)

            check_cancelled("Loading cancelled before dose resampling.")
            report_progress(5, "Loading and resampling RT Dose to CT grid...")
            self.dose_image = self._load_and_resample_dose_to_ct(self.ct_image, dose_path)
            self.prescription_dose = self._get_prescription_dose()
            
            check_cancelled("Loading cancelled before structure processing.")
            report_progress(6, "Processing RT Structures...")
            self._process_structures(is_cancelled=is_cancelled)
            
            check_cancelled("Loading cancelled before body mask processing.")
            report_progress(7, "Applying body mask and calculating stats...")
            self._apply_body_mask_to_images()
            
            check_cancelled("Loading cancelled before finalization.")
            report_progress(8, "All data successfully loaded and processed.")

        except LoadCancelledError:
            logging.info("Data loading cancelled by user.")
            raise
        except FileNotFoundError as e:
            logging.error(f"A required file was not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to load or process data: {e}", exc_info=True)
            raise

    def calculate_dvh(self, roi_number: int) -> Optional[Dict]:
        """Calculates the Dose-Volume Histogram for a given ROI."""
        
        if roi_number in self.dvh_cache:
            logging.info(f"Using cached DVH for ROI {roi_number}...")
            return self.dvh_cache[roi_number]
        if self.masked_dose_array is None:
            return None

        logging.info(f"Calculating DVH for ROI {roi_number}...")
        structure_mask = self.create_structure_mask(roi_number)
        dose_values = self.masked_dose_array[structure_mask]

        if dose_values.size == 0:
            logging.warning(f"No dose voxels found for ROI {roi_number}. Cannot calculate DVH.")
            return None

        min_dose = float(np.min(dose_values))
        mean_dose = float(np.mean(dose_values))
        max_dose = float(np.max(dose_values))
        
        num_bins = int(max_dose * 100) + 2
        bins = np.linspace(0, max_dose * 1.05, num_bins)
        
        hist, bin_edges = np.histogram(dose_values, bins=bins)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        
        dvh_percentages = (cumulative_hist / dose_values.size) * 100.0
        dose_levels = bin_edges[:-1]
        
        volume_interp = dvh_percentages[::-1]
        dose_interp = dose_levels[::-1]

        d98 = np.interp(98.0, volume_interp, dose_interp)
        d50 = np.interp(50.0, volume_interp, dose_interp)
        d2 = np.interp(2.0, volume_interp, dose_interp)
        d95 = np.interp(95.0, volume_interp, dose_interp)

        v20 = np.interp(20.0, dose_levels, dvh_percentages) if max_dose >= 20.0 else 0.0
        v30 = np.interp(30.0, dose_levels, dvh_percentages) if max_dose >= 30.0 else 0.0
        if self.prescription_dose and self.prescription_dose > 0:
            rx_dose = float(self.prescription_dose)
            v100_rx = np.interp(rx_dose, dose_levels, dvh_percentages) if max_dose >= rx_dose else 0.0
        else:
            v100_rx = 0.0

        hi_value = (d2 - d98) / d50 if d50 > 0 else 0.0
        
        result = {'dose_levels': dose_levels, 'dvh_percentages': dvh_percentages,
                  'mean_dose': mean_dose, 'min_dose': min_dose, 'max_dose': max_dose,
                  'hi_value': hi_value, 'd95': float(d95), 'd50': float(d50),
                  'd2': float(d2), 'd98': float(d98), 'v20': float(v20),
                  'v30': float(v30), 'v100_rx': float(v100_rx)}
        self.dvh_cache[roi_number] = result
        
        return result

    def calculate_ci(self, roi_number: int) -> Optional[float]:
        """Calculates the Paddick Conformity Index for a given ROI."""
        if roi_number in self.ci_cache:
            return self.ci_cache[roi_number]
        if self.masked_dose_array is None or self.ct_image is None:
            logging.warning("Dose or CT data not ready. Cannot calculate CI.")
            return None
        if self.prescription_dose is None or self.prescription_dose == 0:
            logging.warning("Prescription dose not set. Cannot calculate CI.")
            return None
        if not self._dose_is_absolute_gy():
            logging.warning("CI calculation requires absolute dose in Gy. Current dose units: %s", self.dose_unit)
            return None

        spacing = self.ct_image.GetSpacing()
        voxel_volume_cc = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

        tv_mask = self.create_structure_mask(roi_number)
        tv = np.sum(tv_mask) * voxel_volume_cc
        if tv == 0:
            self.ci_cache[roi_number] = 0.0
            return 0.0

        piv_mask = self.masked_dose_array >= self.prescription_dose
        piv = np.sum(piv_mask) * voxel_volume_cc
        if piv == 0:
            self.ci_cache[roi_number] = 0.0
            return 0.0
            
        tv_piv = np.sum(tv_mask & piv_mask) * voxel_volume_cc
        ci_value = (tv_piv**2) / (tv * piv)
        self.ci_cache[roi_number] = float(ci_value)
        return float(ci_value)

    # --- 2. Export Methods ---

    def resample_and_export_ct(self, new_spacing: tuple, target_shape: tuple, filepath: str):
        """Resamples and resizes the CT image, then saves it as a NIfTI file."""
        if not self.ct_image:
            raise ValueError("CT image is not available for resampling.")

        logging.info(f"Resampling CT to spacing {new_spacing}...")
        resampled_ct = self._resample_image(
            self.ct_image, new_spacing, sitk.sitkBSpline, default_pixel_value=-1024
        )
        
        final_image = self._resize_image(resampled_ct, target_shape)
        
        logging.info(f"Saving final CT to {filepath}...")
        sitk.WriteImage(sitk.Cast(final_image, sitk.sitkInt16), filepath)
        logging.info("Final CT saved successfully.")

    def resample_and_export_dose(self, new_spacing: tuple, target_shape: tuple, filepath: str):
        """Loads original dose, resamples and resizes it, then saves it as NIfTI."""
        if not self.dose_path or not self.ct_image:
            raise ValueError("Original dose path or CT image is not available.")

        logging.info(f"Loading original dose from {self.dose_path}...")
        rt_dose = sitk.ReadImage(self.dose_path)
        dose_grid_scaling = float(rt_dose.GetMetaData('3004|000e'))
        dose_in_gy = sitk.Cast(rt_dose, sitk.sitkFloat32) * dose_grid_scaling
        
        logging.info(f"Resampling dose to spacing {new_spacing}...")
        resampled_dose = self._resample_image(
            dose_in_gy, new_spacing, sitk.sitkLinear, default_pixel_value=0.0
        )

        final_image = self._resize_image(resampled_dose, target_shape)

        logging.info(f"Saving final dose to {filepath}...")
        sitk.WriteImage(final_image, filepath)
        logging.info("Final dose saved successfully.")

    def resample_and_export_mask(self, roi_number: int, new_spacing: tuple, target_shape: tuple, filepath: str):
        """Creates a mask, resamples and resizes it, then saves it as NIfTI."""
        if self.ct_image is None:
            raise ValueError("CT image is not loaded. Cannot resample mask.")

        logging.info(f"Generating original mask for ROI {roi_number}...")
        mask_numpy = self.create_structure_mask(roi_number)
        original_mask_sitk = sitk.GetImageFromArray(mask_numpy.astype(np.uint8))
        original_mask_sitk.CopyInformation(self.ct_image)

        logging.info(f"Resampling mask to spacing {new_spacing}...")
        resampled_mask = self._resample_image(
            original_mask_sitk, new_spacing, sitk.sitkNearestNeighbor, default_pixel_value=0
        )
        
        final_image = self._resize_image(resampled_mask, target_shape)
        
        logging.info(f"Saving final mask to {filepath}...")
        sitk.WriteImage(final_image, filepath)
        logging.info("Final mask saved successfully.")

    # --- 3. Core Utility Methods ---

    def create_structure_mask(self, roi_number: int) -> np.ndarray:
        """Creates a 3D boolean numpy mask for a given ROI number from contours."""
        if roi_number in self.structure_mask_cache:
            return self.structure_mask_cache[roi_number]

        if self.ct_image is None or self.rt_struct is None:
            empty_shape = self.ct_image.GetSize()[::-1] if self.ct_image is not None else (0, 0, 0)
            empty_mask = np.zeros(empty_shape, dtype=bool)
            self.structure_mask_cache[roi_number] = empty_mask
            return empty_mask

        contour_sequence = self._get_roi_contour_sequence(roi_number)
        if not contour_sequence:
            empty_mask = np.zeros(self.ct_image.GetSize()[::-1], dtype=bool)
            self.structure_mask_cache[roi_number] = empty_mask
            return empty_mask

        backend = str(self.structure_mask_backend or self._MASK_BACKEND_LEGACY).lower()
        if backend not in self._VALID_MASK_BACKENDS:
            backend = self._MASK_BACKEND_LEGACY

        if backend == self._MASK_BACKEND_VTK:
            try:
                mask_3d = self._create_structure_mask_vtk(contour_sequence)
            except Exception as exc:
                logging.warning(
                    "VTK contour rasterization failed for ROI %s (%s). Falling back to legacy rasterization.",
                    roi_number,
                    exc,
                )
                mask_3d = self._create_structure_mask_legacy(contour_sequence)
        else:
            mask_3d = self._create_structure_mask_legacy(contour_sequence)

        self.structure_mask_cache[roi_number] = mask_3d
        return mask_3d

    def _get_roi_contour_sequence(self, roi_number: int):
        """Returns the RTSTRUCT contour sequence for a specific ROI."""
        for roi_contour in getattr(self.rt_struct, "ROIContourSequence", []) or []:
            if getattr(roi_contour, "ReferencedROINumber", None) != roi_number:
                continue
            return getattr(roi_contour, "ContourSequence", None)
        return None

    def _create_structure_mask_legacy(self, contour_sequence: Sequence) -> np.ndarray:
        """Legacy numpy/skimage contour rasterization with sparse-slice interpolation."""
        image_size = self.ct_image.GetSize()
        width = int(image_size[0])
        height = int(image_size[1])
        depth = int(image_size[2])
        mask_3d = np.zeros((depth, height, width), dtype=bool)

        # Build contours per slice first, then apply parity fill (XOR) to support holes.
        contours_by_slice: Dict[int, List[np.ndarray]] = defaultdict(list)
        for contour in contour_sequence:
            contour_data_raw = getattr(contour, "ContourData", None)
            if not contour_data_raw or len(contour_data_raw) % 3 != 0:
                continue

            contour_data = np.asarray(contour_data_raw, dtype=float).reshape(-1, 3)
            continuous_points: List[Tuple[float, float, float]] = []
            for point in contour_data:
                try:
                    cidx = self.ct_image.TransformPhysicalPointToContinuousIndex(
                        (float(point[0]), float(point[1]), float(point[2]))
                    )
                except Exception:
                    continue
                continuous_points.append((float(cidx[0]), float(cidx[1]), float(cidx[2])))

            if len(continuous_points) < 3:
                continue

            points_arr = np.asarray(continuous_points, dtype=float)
            slice_index = int(round(float(np.mean(points_arr[:, 2]))))
            if slice_index < 0 or slice_index >= depth:
                continue

            rows = points_arr[:, 1]
            cols = points_arr[:, 0]
            unique_points = np.unique(np.round(np.column_stack((rows, cols)), decimals=3), axis=0)
            if unique_points.shape[0] < 3:
                continue
            contours_by_slice[slice_index].append(np.column_stack((rows, cols)))

        for slice_index, slice_contours in contours_by_slice.items():
            slice_mask = np.zeros((height, width), dtype=bool)
            seen_polygons = set()
            for contour_points in slice_contours:
                try:
                    fill_rows, fill_cols = polygon(
                        contour_points[:, 0],
                        contour_points[:, 1],
                        (height, width),
                    )
                except Exception:
                    continue
                if fill_rows.size == 0 or fill_cols.size == 0:
                    continue

                min_row = int(np.min(fill_rows))
                max_row = int(np.max(fill_rows))
                min_col = int(np.min(fill_cols))
                max_col = int(np.max(fill_cols))

                local_mask = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=bool)
                local_mask[fill_rows - min_row, fill_cols - min_col] = True
                poly_key = (min_row, max_row, min_col, max_col, hash(local_mask.tobytes()))
                if poly_key in seen_polygons:
                    continue
                seen_polygons.add(poly_key)

                slice_mask[min_row : max_row + 1, min_col : max_col + 1] ^= local_mask

            mask_3d[slice_index] = slice_mask

        non_empty_slices = sorted(idx for idx in contours_by_slice.keys() if np.any(mask_3d[idx]))
        self._interpolate_sparse_mask_slices(mask_3d, non_empty_slices)
        return mask_3d

    def _create_structure_mask_vtk(self, contour_sequence: Sequence) -> np.ndarray:
        """VTK stencil-based contour rasterization in CT continuous-index space."""
        image_size = self.ct_image.GetSize()
        width = int(image_size[0])
        height = int(image_size[1])
        depth = int(image_size[2])

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        has_valid_contour = False

        for contour in contour_sequence:
            contour_data_raw = getattr(contour, "ContourData", None)
            if not contour_data_raw or len(contour_data_raw) % 3 != 0:
                continue

            contour_data = np.asarray(contour_data_raw, dtype=float).reshape(-1, 3)
            if contour_data.shape[0] < 3:
                continue

            contour_points: List[Tuple[float, float, float]] = []
            for point in contour_data:
                try:
                    cidx = self.ct_image.TransformPhysicalPointToContinuousIndex(
                        (float(point[0]), float(point[1]), float(point[2]))
                    )
                except Exception:
                    continue
                contour_points.append((float(cidx[0]), float(cidx[1]), float(cidx[2])))

            if len(contour_points) < 3:
                continue

            if np.linalg.norm(np.asarray(contour_points[0]) - np.asarray(contour_points[-1])) <= 1e-3:
                contour_points = contour_points[:-1]
            if len(contour_points) < 3:
                continue

            poly_line = vtk.vtkPolyLine()
            poly_line.GetPointIds().SetNumberOfIds(len(contour_points) + 1)
            for i, contour_point in enumerate(contour_points):
                point_id = points.InsertNextPoint(contour_point)
                poly_line.GetPointIds().SetId(i, point_id)
            poly_line.GetPointIds().SetId(len(contour_points), poly_line.GetPointIds().GetId(0))
            lines.InsertNextCell(poly_line)
            has_valid_contour = True

        if not has_valid_contour:
            return np.zeros((depth, height, width), dtype=bool)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(width, height, depth)
        image_data.SetOrigin(0.0, 0.0, 0.0)
        image_data.SetSpacing(1.0, 1.0, 1.0)
        image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        image_scalars = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
        image_scalars.fill(1)

        poly_to_stencil = vtk.vtkPolyDataToImageStencil()
        poly_to_stencil.SetInputData(poly_data)
        poly_to_stencil.SetOutputWholeExtent(image_data.GetExtent())
        poly_to_stencil.SetTolerance(0.0)
        poly_to_stencil.Update()

        image_stencil = vtk.vtkImageStencil()
        image_stencil.SetInputData(image_data)
        image_stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
        image_stencil.ReverseStencilOff()
        image_stencil.SetBackgroundValue(0)
        image_stencil.Update()

        output_scalars = image_stencil.GetOutput().GetPointData().GetScalars()
        if output_scalars is None:
            return np.zeros((depth, height, width), dtype=bool)

        mask_3d = numpy_support.vtk_to_numpy(output_scalars).reshape((depth, height, width)).astype(bool)
        non_empty_slices = [idx for idx in range(depth) if np.any(mask_3d[idx])]
        self._interpolate_sparse_mask_slices(mask_3d, non_empty_slices)
        return mask_3d

    def _interpolate_sparse_mask_slices(self, mask_3d: np.ndarray, non_empty_slices: Sequence[int]) -> None:
        """
        Fills gaps between contoured slices.

        Default behavior uses signed-distance blending, with a conservative
        union-bridge fallback only for nearly identical adjacent slices.
        An additional signed-distance discontinuity guard suppresses
        interpolation when two anchor slices are too sparse or too topologically
        inconsistent to support a single continuous volume.
        """
        if len(non_empty_slices) < 2:
            return

        single_gap_mode = len(non_empty_slices) == 2
        signed_distance_cache: Dict[int, np.ndarray] = {}
        component_count_cache: Dict[int, int] = {}
        pair_metrics_cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}

        def get_signed_distance(slice_idx: int) -> np.ndarray:
            if slice_idx in signed_distance_cache:
                return signed_distance_cache[slice_idx]
            binary_image = sitk.GetImageFromArray(mask_3d[slice_idx].astype(np.uint8))
            distance_image = sitk.SignedMaurerDistanceMap(
                binary_image,
                insideIsPositive=False,
                squaredDistance=False,
                useImageSpacing=False,
            )
            signed = sitk.GetArrayFromImage(distance_image).astype(np.float32)
            signed_distance_cache[slice_idx] = signed
            return signed

        def get_component_count(slice_idx: int) -> int:
            if slice_idx in component_count_cache:
                return component_count_cache[slice_idx]
            slice_mask = mask_3d[slice_idx]
            if not np.any(slice_mask):
                component_count_cache[slice_idx] = 0
                return 0
            binary_image = sitk.GetImageFromArray(slice_mask.astype(np.uint8))
            connected = sitk.ConnectedComponent(binary_image)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(connected)
            count = int(len(stats.GetLabels()))
            component_count_cache[slice_idx] = count
            return count

        def get_pair_metrics(z0: int, z1: int, gap: int) -> Dict[str, float]:
            cache_key = (int(z0), int(z1), int(gap))
            if cache_key in pair_metrics_cache:
                return pair_metrics_cache[cache_key]

            mask0 = mask_3d[z0]
            mask1 = mask_3d[z1]
            area0 = int(np.sum(mask0))
            area1 = int(np.sum(mask1))
            union = int(np.sum(mask0 | mask1))
            intersection = int(np.sum(mask0 & mask1))
            iou = float(intersection) / float(union) if union > 0 else 0.0
            area_ratio = (
                float(max(area0, area1)) / float(min(area0, area1))
                if area0 > 0 and area1 > 0 else float('inf')
            )

            row0, col0 = np.where(mask0)
            row1, col1 = np.where(mask1)
            sx, sy, sz = self.ct_image.GetSpacing()
            gap_mm = abs(float(gap) * float(sz))
            if row0.size > 0 and row1.size > 0:
                center0 = np.array([float(np.mean(col0)), float(np.mean(row0))], dtype=float)
                center1 = np.array([float(np.mean(col1)), float(np.mean(row1))], dtype=float)
                delta_index = center1 - center0
                shift_mm = float(
                    np.linalg.norm(np.array([delta_index[0] * float(sx), delta_index[1] * float(sy)]))
                )
            else:
                shift_mm = 0.0

            metrics = {
                'area0': float(area0),
                'area1': float(area1),
                'iou': float(iou),
                'area_ratio': float(area_ratio),
                'gap_mm': float(gap_mm),
                'shift_mm': float(shift_mm),
                'component_count0': float(get_component_count(z0)),
                'component_count1': float(get_component_count(z1)),
            }
            pair_metrics_cache[cache_key] = metrics
            return metrics

        def should_union_bridge(z0: int, z1: int, gap: int) -> bool:
            metrics = get_pair_metrics(z0, z1, gap)
            area0 = int(metrics['area0'])
            area1 = int(metrics['area1'])
            if area0 <= 0 or area1 <= 0:
                return False

            iou = float(metrics['iou'])
            shift_mm = float(metrics['shift_mm'])
            gap_mm = float(metrics['gap_mm'])

            # For exactly two contoured slices, many clinical RTSTRUCTs imply a
            # slab-like region between anchors; union bridge preserves that intent.
            if single_gap_mode:
                if (
                    gap_mm >= self._UNION_BRIDGE_DISJOINT_MIN_GAP_MM
                    and iou <= self._UNION_BRIDGE_DISJOINT_MAX_IOU
                    and shift_mm >= self._UNION_BRIDGE_DISJOINT_MIN_CENTROID_SHIFT_MM
                ):
                    # Large-z-gap disjoint anchors with big lateral shift are likely
                    # separate islands; avoid forced slab bridging.
                    return False
                return gap >= 3

            # Otherwise, union bridge is safe only for very similar, near-adjacent endpoints.
            if gap > self._UNION_BRIDGE_MAX_GAP_SLICES:
                return False

            if float(metrics['area_ratio']) > self._UNION_BRIDGE_MAX_AREA_RATIO:
                return False

            return iou >= self._UNION_BRIDGE_MIN_IOU

        def should_skip_signed_distance(z0: int, z1: int, gap: int) -> bool:
            metrics = get_pair_metrics(z0, z1, gap)
            too_sparse_for_signed_distance = (
                float(metrics['gap_mm']) >= self._SIGNED_DISTANCE_DISJOINT_MIN_GAP_MM
                and float(metrics['iou']) <= self._SIGNED_DISTANCE_DISJOINT_MAX_IOU
                and float(metrics['shift_mm']) >= self._SIGNED_DISTANCE_DISJOINT_MIN_CENTROID_SHIFT_MM
            )
            topology_mismatch = (
                int(metrics['component_count0']) != int(metrics['component_count1'])
                and float(metrics['gap_mm']) >= self._SIGNED_DISTANCE_TOPOLOGY_MISMATCH_MIN_GAP_MM
            )
            return bool(too_sparse_for_signed_distance or topology_mismatch)

        for z0, z1 in zip(non_empty_slices[:-1], non_empty_slices[1:]):
            gap = int(z1 - z0)
            if gap <= 1:
                continue
            if should_union_bridge(z0, z1, gap):
                bridge_mask = np.logical_or(mask_3d[z0], mask_3d[z1])
                for z in range(z0 + 1, z1):
                    mask_3d[z] = bridge_mask
                continue
            if should_skip_signed_distance(z0, z1, gap):
                logging.info(
                    "Skipping signed-distance interpolation between slices %s and %s due to sparse/discontinuous anchors.",
                    z0,
                    z1,
                )
                continue
            d0 = get_signed_distance(z0)
            d1 = get_signed_distance(z1)
            for z in range(z0 + 1, z1):
                t = float(z - z0) / float(gap)
                blended = ((1.0 - t) * d0) + (t * d1)
                mask_3d[z] = blended <= 0.0

    def create_body_mask(self) -> np.ndarray:
        """Creates a 3D mask for the body contour."""
        if self.body_roi_number != -1:
            return self.create_structure_mask(self.body_roi_number)
        # If no body found, return a mask allowing everything
        return np.ones(self.ct_image.GetSize()[::-1], dtype=bool)

    def apply_structure_palette(self, enabled: bool):
        """Applies or restores a colorblind-safe palette to all loaded structures."""
        self._colorblind_palette_enabled = enabled
        if not self.structures:
            return

        for index, roi_number in enumerate(sorted(self.structures.keys())):
            info = self.structures[roi_number]
            current_color = info.get('color', [1.0, 1.0, 1.0])
            if 'original_color' not in info:
                info['original_color'] = list(current_color)

            if enabled:
                new_color = list(self._colorblind_palette[index % len(self._colorblind_palette)])
            else:
                new_color = list(info.get('original_color', current_color))

            info['color'] = new_color
            for actor in info.get('actors', []):
                actor.GetProperty().SetColor(new_color)
            actor_3d = info.get('actor_3d')
            if actor_3d:
                actor_3d.GetProperty().SetColor(new_color)

    def get_or_create_3d_structure_actor(self, roi_number: int) -> Optional[vtk.vtkActor]:
        """Lazily creates and caches one 3D actor for a structure ROI."""
        info = self.structures.get(roi_number)
        if not info:
            return None

        actor_3d = info.get('actor_3d')
        if actor_3d is None:
            actor_3d = self._create_3d_structure_actor(roi_number)
            info['actor_3d'] = actor_3d
        return actor_3d
    
    # --- 4. Internal Private Helper Methods ---

    def _create_3d_structure_actor(self, roi_number: int) -> vtk.vtkActor:
        """Creates a 3D surface actor from a structure's 3D mask."""
        
        color = self.structures[roi_number].get('color', [1.0, 1.0, 1.0])
        # 1. Get the 3D boolean mask for the ROI
        mask_numpy = self.create_structure_mask(roi_number)
        
        # 2. Convert the NumPy mask to a VTK image
        mask_vtk_image = vtk.vtkImageData()
        mask_vtk_image.SetDimensions(self.ct_image.GetSize())
        mask_vtk_image.SetOrigin(self.ct_image.GetOrigin())
        mask_vtk_image.SetSpacing(self.ct_image.GetSpacing())
        direction = np.array(self.ct_image.GetDirection()).reshape(3, 3)
        vtk_matrix = vtk.vtkMatrix3x3()
        for i in range(3):
            for j in range(3):
                vtk_matrix.SetElement(i, j, direction[i, j])
        mask_vtk_image.SetDirectionMatrix(vtk_matrix)
        
        # Convert boolean array to VTK-compatible unsigned char array (0s and 1s)
        vtk_array = numpy_support.numpy_to_vtk(mask_numpy.ravel().astype(np.uint8), deep=True)
        mask_vtk_image.GetPointData().SetScalars(vtk_array)

        # 3. Use vtkDiscreteMarchingCubes to extract the surface
        marcher = vtk.vtkDiscreteMarchingCubes()
        marcher.SetInputData(mask_vtk_image)
        marcher.GenerateValues(1, 1, 1) # Extract the surface for the value '1'
        marcher.Update()
        
        # 4. Smooth the resulting mesh to make it look less blocky
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marcher.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()

        # 5. Create the actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.4) # Make it semi-transparent
        actor.SetVisibility(False) # Invisible by default
        return actor

    def _resample_image(self, image: sitk.Image, new_spacing: tuple, interpolator, default_pixel_value) -> sitk.Image:
        """Resamples a SimpleITK image to a new spacing."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(orig_sz * (orig_sp / new_sp)))
            for orig_sz, orig_sp, new_sp in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_pixel_value)
        
        return resampler.Execute(image)

    def _resize_image(self, image: sitk.Image, target_shape: tuple) -> sitk.Image:
        """Crops or pads a SimpleITK image to a target shape (if target_shape is not (0,0,0))."""
        if target_shape == (0, 0, 0):
            return image

        resampled_size = image.GetSize()
        crop_pad_amount = [rs - ts for rs, ts in zip(resampled_size, target_shape)]
        
        is_padding_needed = any(val < 0 for val in crop_pad_amount)
        is_cropping_needed = any(val > 0 for val in crop_pad_amount)

        final_image = image

        if is_padding_needed:
            lower_bound = [abs(min(0, val)) // 2 for val in crop_pad_amount]
            upper_bound = [abs(min(0, val)) - lb for lb, val in zip(lower_bound, crop_pad_amount)]
            padder = sitk.ConstantPadImageFilter()
            padder.SetPadLowerBound(lower_bound)
            padder.SetPadUpperBound(upper_bound)
            padder.SetConstant(0)
            final_image = padder.Execute(final_image)
            logging.info(f"Padded image to shape {final_image.GetSize()}")

        if is_cropping_needed:
            crop_lower = [max(0, val) // 2 for val in crop_pad_amount]
            crop_upper = [max(0, val) - cl for cl, val in zip(crop_lower, crop_pad_amount)]
            
            cropper = sitk.CropImageFilter()
            cropper.SetLowerBoundaryCropSize(crop_lower)
            cropper.SetUpperBoundaryCropSize(crop_upper)
            final_image = cropper.Execute(final_image)
            logging.info(f"Cropped image to shape {final_image.GetSize()}")

        return final_image
    
    def _load_ct_image(self, folder_path: str, series_uid: str) -> sitk.Image:
        """
        Scans a directory, finds the specific series UID, and lets SimpleITK
        determine the correct anatomical sorting of the files for that series.
        """
        reader = sitk.ImageSeriesReader()
        
        # Get all series UIDs found in the folder and its subfolders
        all_series_uids = reader.GetGDCMSeriesIDs(folder_path)
        
        if not all_series_uids:
            raise FileNotFoundError(f"No DICOM series were found in '{folder_path}'.")
            
        if series_uid not in all_series_uids:
            raise FileNotFoundError(f"The selected CT Series UID '{series_uid}' was not found in the directory.")

        # Get the SimpleITK-sorted list of files for ONLY the specific series UID
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_uid)
        
        reader.SetFileNames(dicom_names)
        return reader.Execute()

    def _update_dose_unit_from_file(self, dose_path: str):
        """Reads dose metadata to determine units, defaulting to Gy."""
        logging.info("Reading RT Dose metadata for units...")
        self.dose_unit = "Gy"
        try:
            dose_dcm = pydicom.dcmread(dose_path, stop_before_pixels=True)
            dose_units = str(getattr(dose_dcm, 'DoseUnits', '') or '').strip().upper()
            if dose_units == 'GY':
                self.dose_unit = 'Gy'
            elif dose_units:
                self.dose_unit = dose_units
        except Exception as e:
            logging.warning(f"Could not read dose units, defaulting to Gy. Error: {e}")

    def _read_dose_grid_scaling(self, dose_filepath: str, rt_dose: Optional[sitk.Image] = None) -> float:
        """Reads DoseGridScaling robustly from SimpleITK metadata or pydicom fallback."""
        if rt_dose is not None:
            try:
                if rt_dose.HasMetaDataKey('3004|000e'):
                    return float(rt_dose.GetMetaData('3004|000e'))
            except Exception:
                pass

        dose_dcm = pydicom.dcmread(dose_filepath, stop_before_pixels=True)
        scaling = getattr(dose_dcm, 'DoseGridScaling', None)
        if scaling is None:
            raise ValueError('RTDOSE is missing DoseGridScaling (3004,000E).')
        return float(scaling)
    
    def _load_and_resample_dose_to_ct(self, ct_image: sitk.Image, dose_filepath: str) -> sitk.Image:
        """Loads an RTDOSE file and resamples it to match the CT grid."""
        rt_dose = sitk.ReadImage(dose_filepath)
        dose_grid_scaling = self._read_dose_grid_scaling(dose_filepath, rt_dose=rt_dose)
        dose_image = sitk.Cast(rt_dose, sitk.sitkFloat32) * dose_grid_scaling

        if not self._dose_is_absolute_gy():
            logging.warning(
                "RTDOSE DoseUnits=%s. Values will be loaded and displayed, but Gy-based metrics may be invalid.",
                self.dose_unit,
            )

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_image)
        # Linear interpolation reduces low-dose spread artifacts in the display.
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampled_dose_image = resampler.Execute(dose_image)
        clamped_dose_image = sitk.Clamp(resampled_dose_image, lowerBound=0.0)
        return clamped_dose_image

    def _get_prescription_dose(self) -> float:
        """Extracts the prescription dose from the RTPLAN file."""
        if not self.rt_plan: return 0.0
        try:
            # Use getattr to safely access the sequence
            dose_ref_seq = getattr(self.rt_plan, 'DoseReferenceSequence', [])
            for dose_ref in dose_ref_seq:
                if hasattr(dose_ref, 'TargetPrescriptionDose'):
                    return float(dose_ref.TargetPrescriptionDose)
        except Exception as e:
            logging.warning(f"Could not read TargetPrescriptionDose from RT Plan. Defaulting to 0. Error: {e}")
        
        return 0.0

    def _process_structures(self, is_cancelled: Optional[Callable[[], bool]] = None):
        """Finds the body contour and parses all structures for visualization."""
        logging.info("Processing RT Structures...")
        if is_cancelled and is_cancelled():
            raise LoadCancelledError("Loading cancelled during structure processing.")
        self.body_roi_number = self._find_enveloping_roi_number()
        self._parse_rtstruct_for_vtk(is_cancelled=is_cancelled)

    def _apply_body_mask_to_images(self):
        """Applies the body mask to the CT and Dose arrays for calculations."""
        logging.info("Applying body mask to CT and Dose...")
        ct_array = sitk.GetArrayFromImage(self.ct_image)
        dose_array = sitk.GetArrayFromImage(self.dose_image)
        body_mask_3d = self.create_body_mask()

        # Some studies have no valid external contour, which can leave dose visible
        # across the full CT extent. Fall back to a CT-derived body mask in that case.
        mask_fill_ratio = float(np.mean(body_mask_3d)) if body_mask_3d.size else 1.0
        if self.body_roi_number == -1 or mask_fill_ratio > 0.97:
            logging.info(
                "Using CT-intensity body mask fallback "
                f"(roi={self.body_roi_number}, fill_ratio={mask_fill_ratio:.3f})."
            )
            derived_mask = self._create_ct_intensity_body_mask(ct_array)
            if derived_mask is not None:
                body_mask_3d = derived_mask

        dose_array[~body_mask_3d] = 0

        # Keep CT unmasked for 2D rendering robustness across edge-case contours.
        # (3D already renders from original CT image directly.)
        self.masked_ct_array = ct_array
        self.masked_dose_array = dose_array
        self.dose_max_in_gy = float(np.max(self.masked_dose_array))
        
        if self.prescription_dose and self.prescription_dose > 0:
            self.max_dose_as_percent_of_rx = (self.dose_max_in_gy / self.prescription_dose) * 100.0
        else:
            self.max_dose_as_percent_of_rx = 100.0 if self.dose_max_in_gy else 0.0

    def _create_ct_intensity_body_mask(self, ct_array: np.ndarray) -> Optional[np.ndarray]:
        """Builds a fallback body mask from CT intensities and largest component."""
        if ct_array.size == 0:
            return None

        threshold_hu = -500
        binary_mask = ct_array > threshold_hu
        if not np.any(binary_mask):
            return None

        binary_image = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
        connected = sitk.ConnectedComponent(binary_image)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(connected)
        labels = list(stats.GetLabels())
        if not labels:
            return binary_mask

        largest_label = max(labels, key=lambda label: stats.GetNumberOfPixels(label))
        largest = sitk.Equal(connected, int(largest_label))
        # Small XY dilation recovers the skin boundary without leaking between slices.
        largest = sitk.BinaryDilate(largest, [1, 1, 0], sitk.sitkBall)
        return sitk.GetArrayFromImage(largest).astype(bool)

    def _find_enveloping_roi_number(self) -> int:
        """Identifies body contour quickly (name-first, area fallback)."""
        roi_contour_sequence = getattr(self.rt_struct, "ROIContourSequence", [])
        if not roi_contour_sequence:
            logging.warning("Could not determine enveloping body contour: no ROIContourSequence.")
            return -1

        contours_by_roi = {}
        for roi_contour in roi_contour_sequence:
            roi_number = getattr(roi_contour, "ReferencedROINumber", None)
            contour_sequence = getattr(roi_contour, "ContourSequence", None)
            if roi_number is None or not contour_sequence:
                continue
            contours_by_roi[roi_number] = roi_contour
        if not contours_by_roi:
            logging.warning("Could not determine enveloping body contour: no valid contour sequence.")
            return -1

        roi_names = {}
        for roi in getattr(self.rt_struct, "StructureSetROISequence", []):
            roi_number = getattr(roi, "ROINumber", None)
            if roi_number is None:
                continue
            roi_names[roi_number] = str(getattr(roi, "ROIName", "")).strip()

        # Fast path: typical external-body labels (including French naming).
        keyword_order = (
            "body",
            "external",
            "contour externe",
            "externe",
            "patient",
            "skin",
            "outline",
            "surface",
            "outer",
            "corps",
        )
        best_named = None
        best_named_score = -1.0
        for roi_number, roi_contour in contours_by_roi.items():
            name = roi_names.get(roi_number, "").lower()
            if not name:
                continue
            score = -1.0
            for idx, keyword in enumerate(keyword_order):
                if name == keyword:
                    score = max(score, 1000.0 - idx)
                elif keyword in name:
                    score = max(score, 500.0 - idx)
            if score < 0:
                continue

            # Break ties with contour point count (cheap to compute).
            point_count = 0
            for contour in getattr(roi_contour, "ContourSequence", []):
                point_count += int(getattr(contour, "NumberOfContourPoints", 0) or 0)
            combined_score = score + (point_count * 1e-4)

            if combined_score > best_named_score:
                best_named_score = combined_score
                best_named = roi_number

        if best_named is not None:
            logging.info(
                f"Identified body contour by name: ROINumber = {best_named} "
                f"({roi_names.get(best_named, 'N/A')})"
            )
            return best_named

        # Fallback: largest accumulated XY bounding-box area across contour slices.
        largest_area, enveloping_roi_number = -1.0, -1
        for roi_number, roi_contour in contours_by_roi.items():
            current_roi_area = 0.0
            for contour in getattr(roi_contour, "ContourSequence", []):
                contour_data = getattr(contour, "ContourData", None)
                if not contour_data or len(contour_data) < 6:
                    continue
                coords = np.asarray(contour_data, dtype=np.float32)
                xs = coords[0::3]
                ys = coords[1::3]
                if xs.size == 0 or ys.size == 0:
                    continue
                current_roi_area += float((xs.max() - xs.min()) * (ys.max() - ys.min()))

            if current_roi_area > largest_area:
                largest_area = current_roi_area
                enveloping_roi_number = roi_number

        if enveloping_roi_number != -1:
            logging.info(
                f"Identified body contour by area: ROINumber = {enveloping_roi_number} "
                f"({roi_names.get(enveloping_roi_number, 'N/A')})"
            )
        else:
            logging.warning("Could not determine enveloping body contour.")

        return enveloping_roi_number

    def _contour_slice_index_from_points(self, contour_data: np.ndarray) -> Optional[int]:
        """Maps one contour to the nearest CT slice using real contour coordinates."""
        if self.ct_image is None or contour_data.size == 0:
            return None

        continuous_indices = []
        for point in contour_data:
            try:
                cidx = self.ct_image.TransformPhysicalPointToContinuousIndex(
                    (float(point[0]), float(point[1]), float(point[2]))
                )
                continuous_indices.append(float(cidx[2]))
            except Exception:
                continue

        if not continuous_indices:
            return None

        slice_index = int(round(float(np.mean(continuous_indices))))
        depth = int(self.ct_image.GetSize()[2])
        if 0 <= slice_index < depth:
            return slice_index
        return None


    def _parse_rtstruct_for_vtk(self, is_cancelled: Optional[Callable[[], bool]] = None):
        """
        Parses the RTSTRUCT file to create VTK actors for each contour,
        handling missing or malformed data gracefully.
        3D structure actors are created lazily on first 3D visibility request.
        """
        self.structures.clear()
        self.contours_by_slice.clear()
        self.actor_to_roi_map.clear()
        self.structure_mask_cache.clear()

        # Safely get the main sequences, defaulting to an empty list if they don't exist
        structure_set_sequence = getattr(self.rt_struct, 'StructureSetROISequence', [])
        roi_contour_sequence = getattr(self.rt_struct, 'ROIContourSequence', [])

        # Build the name mapping safely
        roi_names = {}
        for roi in structure_set_sequence:
            roi_number = getattr(roi, 'ROINumber', None)
            roi_name = getattr(roi, 'ROIName', f"ROI {roi_number}")
            if roi_number is not None:
                roi_names[roi_number] = roi_name

        # Process each ROI within its own try-except block
        for roi_contour in roi_contour_sequence:
            try:
                if is_cancelled and is_cancelled():
                    raise LoadCancelledError("Loading cancelled during contour parsing.")
                roi_number = roi_contour.ReferencedROINumber
                name = roi_names.get(roi_number, f"ROI {roi_number}")
                color_list = getattr(roi_contour, 'ROIDisplayColor', [255, 255, 255])
                vtk_color = [float(c) / 255.0 for c in color_list]
                
                self.structures[roi_number] = {
                    'name': name,
                    'actors': [],
                    'color': vtk_color,
                    'actor_3d': None,
                }
                
                # Skip if there's no contour data for this ROI
                if 'ContourSequence' not in roi_contour:
                    continue

                for contour in roi_contour.ContourSequence:
                    if is_cancelled and is_cancelled():
                        raise LoadCancelledError("Loading cancelled during contour parsing.")
                    # Check for contour data existence and validity
                    if 'ContourData' not in contour or not contour.ContourData:
                        continue
                    
                    contour_data_raw = contour.ContourData
                    if len(contour_data_raw) % 3 != 0:
                        logging.warning(f"Skipping malformed contour for ROI '{name}' (length not a multiple of 3).")
                        continue

                    contour_data = np.array(contour_data_raw, dtype=float).reshape(-1, 3)
                    slice_index = self._contour_slice_index_from_points(contour_data)
                    if slice_index is None:
                        continue

                    actor = self._create_vtk_actor_from_contour(contour_data, vtk_color)
                    self.contours_by_slice[slice_index].append(actor)
                    self.structures[roi_number]['actors'].append(actor)
                    self.actor_to_roi_map[id(actor)] = roi_number
                
            except LoadCancelledError:
                raise
            except Exception as e:
                # If anything goes wrong with one ROI, log it and continue to the next
                roi_num_str = getattr(roi_contour, 'ReferencedROINumber', 'Unknown')
                logging.warning(f"Could not process ROI {roi_num_str}. Error: {e}")
                continue

        if self._colorblind_palette_enabled:
            self.apply_structure_palette(True)

    def _create_vtk_actor_from_contour(self, contour_data: np.ndarray, color: list) -> vtk.vtkActor:
        """Creates a single VTK actor for a given contour slice."""
        points = vtk.vtkPoints()
        for point in contour_data:
            points.InsertNextPoint(point)
        
        poly_line = vtk.vtkPolyLine()
        num_points = len(contour_data)
        poly_line.GetPointIds().SetNumberOfIds(num_points + 1)
        for i in range(num_points):
            poly_line.GetPointIds().SetId(i, i)
        poly_line.GetPointIds().SetId(num_points, 0)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(poly_line)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(CONTOUR_LINE_WIDTH)
        actor.SetPickable(False)
        actor.SetVisibility(False)
        return actor
