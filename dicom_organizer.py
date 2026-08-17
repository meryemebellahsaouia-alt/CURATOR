"""
Scan and organize radiotherapy DICOM files into linked treatment datasets.

The returned structure is a nested hierarchy:
    CT Series -> RTSTRUCT -> RTPLAN -> RTDOSE

This module is intentionally conservative about linkage extraction but tolerant
of routine dataset variability (missing InstanceNumber, multiple reference
items, incomplete chains, etc.).
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pydicom
from pydicom.dataset import Dataset


def get_safe_dicom_uid(dicom_file: Dataset, tag_sequence: Sequence[Tuple[int, int, Optional[int]]]):
    """Safely navigates a sequence of DICOM tags to extract one UID value.

    This helper is kept for backward compatibility. For new code, prefer the
    dedicated reference-extraction helpers below, which support multiple items.
    """
    try:
        dataset = dicom_file
        for group, element, index in tag_sequence[:-1]:
            data_element = dataset[group, element]
            value = data_element.value
            if index is None:
                dataset = value
            else:
                dataset = value[index]
        final_group, final_element, _ = tag_sequence[-1]
        return dataset[final_group, final_element].value
    except Exception:
        return None


def _as_str(value, default: str = "N/A") -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else default


def _as_uid(value) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _iter_sequence_items(dataset: Dataset, attribute_name: str) -> Iterable[Dataset]:
    sequence = getattr(dataset, attribute_name, None)
    if not sequence:
        return ()
    return tuple(item for item in sequence if isinstance(item, Dataset))


def _extract_rtstruct_referenced_series_uids(dataset: Dataset) -> Set[str]:
    """Extract all CT SeriesInstanceUIDs referenced by an RTSTRUCT."""
    result: Set[str] = set()
    for ref_for in _iter_sequence_items(dataset, "ReferencedFrameOfReferenceSequence"):
        for ref_study in _iter_sequence_items(ref_for, "RTReferencedStudySequence"):
            for ref_series in _iter_sequence_items(ref_study, "RTReferencedSeriesSequence"):
                uid = _as_uid(getattr(ref_series, "SeriesInstanceUID", None))
                if uid:
                    result.add(uid)
    return result


def _extract_plan_referenced_rtstruct_uids(dataset: Dataset) -> Set[str]:
    """Extract all RTSTRUCT SOPInstanceUIDs referenced by an RTPLAN."""
    result: Set[str] = set()
    for ref_struct in _iter_sequence_items(dataset, "ReferencedStructureSetSequence"):
        uid = _as_uid(getattr(ref_struct, "ReferencedSOPInstanceUID", None))
        if uid:
            result.add(uid)
    return result


def _extract_dose_referenced_rtplan_uids(dataset: Dataset) -> Set[str]:
    """Extract all RTPLAN SOPInstanceUIDs referenced by an RTDOSE."""
    result: Set[str] = set()
    for ref_plan in _iter_sequence_items(dataset, "ReferencedRTPlanSequence"):
        uid = _as_uid(getattr(ref_plan, "ReferencedSOPInstanceUID", None))
        if uid:
            result.add(uid)
    return result


def _build_ct_sort_key(dataset: Dataset, file_path: str) -> Tuple[int, float, int, str]:
    """Return a stable CT file sort key.

    Preference order:
    1. ImagePositionPatient z-coordinate when available.
    2. InstanceNumber when available.
    3. File path as deterministic fallback.

    The organizer only needs a stable display/load list. Final anatomical order
    is still determined later by the DICOM reader.
    """
    ipp = getattr(dataset, "ImagePositionPatient", None)
    z_pos = None
    if ipp is not None:
        try:
            if len(ipp) >= 3:
                z_pos = _as_float(ipp[2])
        except Exception:
            z_pos = None

    instance_number = _as_int(getattr(dataset, "InstanceNumber", None))

    if z_pos is not None and instance_number is not None:
        return (0, float(z_pos), int(instance_number), file_path)
    if z_pos is not None:
        return (0, float(z_pos), 0, file_path)
    if instance_number is not None:
        return (1, 0.0, int(instance_number), file_path)
    return (2, 0.0, 0, file_path)


def organize_radiotherapy_dicom(directory: str, return_audit: bool = False):
    """
    Scans and organizes DICOM files into a nested hierarchy:
        CT -> [RTSTRUCTs] -> [RTPLANs] -> [RTDOSEs]

    If return_audit is True, returns ``(organized_data, audit_dict)``.
    """
    logging.info("Scanning directory: %s", directory)

    ct_series: Dict[str, List[dict]] = defaultdict(list)
    ct_info: Dict[str, dict] = {}

    rtstructs_by_ct_series: Dict[str, List[str]] = defaultdict(list)
    rtstruct_sops: Dict[str, str] = {}
    rtstruct_info: Dict[str, dict] = {}
    rtstruct_ref_series: Dict[str, Set[str]] = {}

    rtplans_by_struct_sop: Dict[str, List[str]] = defaultdict(list)
    rtplan_sops: Dict[str, str] = {}
    rtplan_info: Dict[str, dict] = {}
    rtplan_ref_structs: Dict[str, Set[str]] = {}

    rtdoses_by_plan_sop: Dict[str, List[str]] = defaultdict(list)
    rtdose_info: Dict[str, dict] = {}
    rtdose_ref_plans: Dict[str, Set[str]] = {}

    audit = {
        "directory": directory,
        "total_files": 0,
        "readable_dicom_files": 0,
        "skipped_files": 0,
        "modality_counts": Counter(),
        "required_modalities": ["CT", "RTSTRUCT", "RTPLAN", "RTDOSE"],
        "missing_modalities": [],
        "ct_series_count": 0,
        "rtstruct_count": 0,
        "rtplan_count": 0,
        "rtdose_count": 0,
        "complete_dataset_count": 0,
        "is_linkable": False,
        "unlinked_rtstruct_count": 0,
        "unlinked_rtplan_count": 0,
        "unlinked_rtdose_count": 0,
    }

    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            audit["total_files"] += 1
            try:
                dcm = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            except Exception as exc:
                audit["skipped_files"] += 1
                logging.debug("Skipping file %s due to read error: %s", file_path, exc)
                continue

            audit["readable_dicom_files"] += 1
            modality = _as_uid(getattr(dcm, "Modality", None))
            if modality:
                audit["modality_counts"][modality] += 1
            sop_uid = _as_uid(getattr(dcm, "SOPInstanceUID", None))

            if modality == "CT":
                series_uid = _as_uid(getattr(dcm, "SeriesInstanceUID", None))
                if not series_uid:
                    continue
                ct_series[series_uid].append(
                    {
                        "path": file_path,
                        "sort_key": _build_ct_sort_key(dcm, file_path),
                    }
                )
                if series_uid not in ct_info:
                    ct_info[series_uid] = {
                        "StudyDate": _as_str(getattr(dcm, "StudyDate", None)),
                        "SeriesDescription": _as_str(getattr(dcm, "SeriesDescription", None)),
                        "FrameOfReferenceUID": _as_uid(getattr(dcm, "FrameOfReferenceUID", None)),
                    }

            elif modality == "RTSTRUCT":
                referenced_series_uids = _extract_rtstruct_referenced_series_uids(dcm)
                if not referenced_series_uids or not sop_uid:
                    continue
                rtstruct_sops[file_path] = sop_uid
                rtstruct_ref_series[file_path] = set(referenced_series_uids)
                rtstruct_info[file_path] = {
                    "StructureSetLabel": _as_str(getattr(dcm, "StructureSetLabel", None)),
                    "FrameOfReferenceUID": _as_uid(getattr(dcm, "FrameOfReferenceUID", None)),
                    "ReferencedSeriesCount": len(referenced_series_uids),
                }
                for ref_uid in referenced_series_uids:
                    rtstructs_by_ct_series[ref_uid].append(file_path)

            elif modality == "RTPLAN":
                referenced_struct_uids = _extract_plan_referenced_rtstruct_uids(dcm)
                if not referenced_struct_uids or not sop_uid:
                    continue
                rtplan_sops[file_path] = sop_uid
                rtplan_ref_structs[file_path] = set(referenced_struct_uids)
                rtplan_info[file_path] = {
                    "RTPlanLabel": _as_str(getattr(dcm, "RTPlanLabel", None)),
                    "ReferencedStructureSetCount": len(referenced_struct_uids),
                }
                for ref_uid in referenced_struct_uids:
                    rtplans_by_struct_sop[ref_uid].append(file_path)

            elif modality == "RTDOSE":
                referenced_plan_uids = _extract_dose_referenced_rtplan_uids(dcm)
                if not referenced_plan_uids:
                    continue
                rtdose_ref_plans[file_path] = set(referenced_plan_uids)
                rtdose_info[file_path] = {
                    "DoseSummationType": _as_str(getattr(dcm, "DoseSummationType", None)),
                    "DoseUnits": _as_str(getattr(dcm, "DoseUnits", None)),
                    "ReferencedPlanCount": len(referenced_plan_uids),
                }
                for ref_uid in referenced_plan_uids:
                    rtdoses_by_plan_sop[ref_uid].append(file_path)

    organized_data = {}
    linked_rtstruct_paths: Set[str] = set()
    linked_rtplan_paths: Set[str] = set()
    linked_rtdose_paths: Set[str] = set()

    for ct_uid, file_entries in ct_series.items():
        sorted_entries = sorted(file_entries, key=lambda item: item["sort_key"])
        final_ct_paths = [entry["path"] for entry in sorted_entries]

        ct_entry = {
            "ct_files": final_ct_paths,
            "info": ct_info.get(ct_uid, {}),
            "rtstructs": {},
        }

        for struct_path in sorted(set(rtstructs_by_ct_series.get(ct_uid, []))):
            struct_sop = rtstruct_sops.get(struct_path)
            if not struct_sop:
                continue

            struct_entry = {
                "info": rtstruct_info.get(struct_path, {}),
                "rtplans": {},
            }

            for plan_path in sorted(set(rtplans_by_struct_sop.get(struct_sop, []))):
                plan_sop = rtplan_sops.get(plan_path)
                if not plan_sop:
                    continue

                dose_files = sorted(set(rtdoses_by_plan_sop.get(plan_sop, [])))
                if not dose_files:
                    continue

                doses_with_info = {
                    dose_path: {"info": rtdose_info.get(dose_path, {})}
                    for dose_path in dose_files
                }
                struct_entry["rtplans"][plan_path] = {
                    "info": rtplan_info.get(plan_path, {}),
                    "rtdoses": doses_with_info,
                }
                linked_rtplan_paths.add(plan_path)
                linked_rtdose_paths.update(dose_files)

            if struct_entry["rtplans"]:
                ct_entry["rtstructs"][struct_path] = struct_entry
                linked_rtstruct_paths.add(struct_path)

        if ct_entry["rtstructs"]:
            organized_data[ct_uid] = ct_entry

    audit["ct_series_count"] = len(ct_series)
    audit["rtstruct_count"] = len(rtstruct_sops)
    audit["rtplan_count"] = len(rtplan_sops)
    audit["rtdose_count"] = len(rtdose_info)
    audit["complete_dataset_count"] = len(organized_data)
    audit["is_linkable"] = len(organized_data) > 0
    audit["unlinked_rtstruct_count"] = max(0, len(rtstruct_sops) - len(linked_rtstruct_paths))
    audit["unlinked_rtplan_count"] = max(0, len(rtplan_sops) - len(linked_rtplan_paths))
    audit["unlinked_rtdose_count"] = max(0, len(rtdose_info) - len(linked_rtdose_paths))

    required = set(audit["required_modalities"])
    present_required = {m for m in required if audit["modality_counts"].get(m, 0) > 0}
    audit["missing_modalities"] = sorted(required - present_required)
    audit["modality_counts"] = dict(audit["modality_counts"])

    logging.info(
        "Found %d complete dataset(s); CT series=%d, RTSTRUCT=%d, RTPLAN=%d, RTDOSE=%d.",
        audit["complete_dataset_count"],
        audit["ct_series_count"],
        audit["rtstruct_count"],
        audit["rtplan_count"],
        audit["rtdose_count"],
    )

    if return_audit:
        return organized_data, audit
    return organized_data
