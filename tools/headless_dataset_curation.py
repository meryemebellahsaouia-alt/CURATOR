"""
Headless batch curation pipeline for RT DICOM to standardized NIfTI exports.

This script uses the project interfaces:
  - dicom_organizer.organize_radiotherapy_dicom
  - model.DICOMDataModel

Pipeline stages:
  1) Integrity check (find complete CT -> RTSTRUCT -> RTPLAN -> RTDOSE chain)
  2) Headless load with DICOMDataModel
  3) Export standardized CT, dose, and selected structure masks to NIfTI
  4) Save manifest + summary for thesis/reporting

Usage example:
  python tools/headless_dataset_curation.py ^
    --raw-data-root "C:\\Research\\Raw_DICOM_Database" ^
    --output-root "C:\\Research\\Curated_NIfTI_Dataset" ^
    --spacing 1,1,1 --shape 128,128,64 ^
    --structures PTV Rectum Bladder Femur_L Femur_R
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

THIS_FILE = Path(__file__).resolve()
REPO_ROOT_CANDIDATES = [THIS_FILE.parent, *THIS_FILE.parents]
for candidate in REPO_ROOT_CANDIDATES:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dicom_organizer import organize_radiotherapy_dicom  # noqa: E402
from model import DICOMDataModel  # noqa: E402


LOG = logging.getLogger("headless_dataset_curation")
DEFAULT_STRUCTURES = ["PTV", "Rectum", "Bladder", "Femur_L", "Femur_R"]
MANIFEST_HEADERS = [
    "ID",
    "Status",
    "Reason",
    "ChainCandidates",
    "SelectedDoseSummationType",
    "DoseUnit",
    "PrescriptionDoseGy",
    "CTSeriesUID",
    "RTSTRUCT",
    "RTPLAN",
    "RTDOSE",
    "RequestedStructures",
    "MatchedStructures",
    "MissingRequestedStructures",
    "MasksExported",
    "ExportedMaskNames",
    "OutputFolder",
]


@dataclass(frozen=True)
class DatasetPaths:
    patient_folder: str
    ct_series_uid: str
    rtstruct_path: str
    rtplan_path: str
    dose_path: str
    dose_summation_type: str = "N/A"


@dataclass(frozen=True)
class CaseSelection:
    selected: Optional[DatasetPaths]
    candidate_count: int


def _parse_tuple_arg(value: str, length: int, cast_type, label: str) -> Tuple:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(
            f"{label} must have exactly {length} comma-separated values (got: {value})."
        )
    try:
        return tuple(cast_type(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} contains invalid numeric values: {value}"
        ) from exc


def _collect_target_dirs(root: Path, include_root: bool) -> List[Path]:
    targets: List[Path] = []
    if include_root:
        targets.append(root)
    targets.extend(sorted(child for child in root.iterdir() if child.is_dir()))
    return targets


def _safe_nested_get(mapping: dict, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


_SUMMATION_PRIORITY = {
    "PLAN": 0,
    "BEAM": 1,
    "FRACTION": 2,
    "MULTI_PLAN": 3,
    "BRACHY": 4,
    "CONTROL_POINT": 5,
    "N/A": 99,
    "": 99,
}


def _dose_sort_key(dose_path: str, dose_payload: dict) -> Tuple[int, str]:
    summation = str(_safe_nested_get(dose_payload, "info", "DoseSummationType", default="N/A") or "N/A").upper()
    return (_SUMMATION_PRIORITY.get(summation, 50), str(dose_path))


def _iter_complete_chains(patient_folder: Path) -> List[DatasetPaths]:
    """Returns all deterministic complete CT->RTSTRUCT->RTPLAN->RTDOSE chains."""
    organized = organize_radiotherapy_dicom(str(patient_folder))
    if not organized:
        return []

    chains: List[DatasetPaths] = []
    for ct_uid in sorted(organized.keys()):
        ct_data = organized[ct_uid]
        for struct_path in sorted(ct_data.get("rtstructs", {}).keys()):
            struct_data = ct_data["rtstructs"][struct_path]
            for plan_path in sorted(struct_data.get("rtplans", {}).keys()):
                plan_data = struct_data["rtplans"][plan_path]
                sorted_doses = sorted(
                    plan_data.get("rtdoses", {}).items(),
                    key=lambda item: _dose_sort_key(item[0], item[1]),
                )
                for dose_path, dose_payload in sorted_doses:
                    dose_summation_type = str(
                        _safe_nested_get(dose_payload, "info", "DoseSummationType", default="N/A") or "N/A"
                    )
                    chains.append(
                        DatasetPaths(
                            patient_folder=str(patient_folder),
                            ct_series_uid=str(ct_uid),
                            rtstruct_path=str(struct_path),
                            rtplan_path=str(plan_path),
                            dose_path=str(dose_path),
                            dose_summation_type=dose_summation_type,
                        )
                    )
    return chains


def _select_preferred_chain(patient_folder: Path) -> CaseSelection:
    """Selects one deterministic preferred chain while tracking candidate count."""
    chains = _iter_complete_chains(patient_folder)
    if not chains:
        return CaseSelection(selected=None, candidate_count=0)

    chains = sorted(
        chains,
        key=lambda c: (
            _SUMMATION_PRIORITY.get(str(c.dose_summation_type).upper(), 50),
            c.ct_series_uid,
            c.rtstruct_path,
            c.rtplan_path,
            c.dose_path,
        ),
    )
    return CaseSelection(selected=chains[0], candidate_count=len(chains))


def _sanitize_name(name: str) -> str:
    normalized = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in str(name).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized or "ROI"


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _setup_logger(output_root: Path, log_level: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = output_root / "curation_log.txt"

    LOG.handlers.clear()
    LOG.propagate = False
    LOG.setLevel(getattr(logging, log_level))

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level))
    LOG.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(getattr(logging, log_level))
    LOG.addHandler(stream_handler)

    return log_file


def _normalize_structure_token(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _pattern_aliases(pattern: str) -> List[str]:
    text = str(pattern).strip()
    if not text:
        return []
    normalized = _normalize_structure_token(text)
    aliases = {normalized}

    if normalized in {"femurl", "leftfemur", "femurleft", "lfemur"}:
        aliases.update({"femurl", "leftfemur", "femurleft", "lfemur", "femoralheadl", "leftfemoralhead"})
    elif normalized in {"femurr", "rightfemur", "femurright", "rfemur"}:
        aliases.update({"femurr", "rightfemur", "femurright", "rfemur", "femoralheadr", "rightfemoralhead"})
    elif normalized == "bladder":
        aliases.update({"bladder", "vesica", "urinarybladder"})
    elif normalized == "rectum":
        aliases.update({"rectum", "rectal"})
    elif normalized == "ptv":
        aliases.update({"ptv", "ptvboost", "ptvhigh", "ptvlow"})

    return sorted(a for a in aliases if a)


def _match_structure_name(roi_name: str, target_patterns: Sequence[str]) -> bool:
    roi_token = _normalize_structure_token(roi_name)
    if not roi_token:
        return False
    for pattern in target_patterns:
        for alias in _pattern_aliases(pattern):
            if alias and alias in roi_token:
                return True
    return False


def _build_mask_export_list(
    model: DICOMDataModel,
    structure_patterns: Sequence[str],
) -> List[Tuple[int, str]]:
    """Returns list of (roi_number, roi_name) to export using normalized matching."""
    targets = [p.strip() for p in structure_patterns if str(p).strip()]
    if not targets:
        return []

    selected: List[Tuple[int, str]] = []
    for roi_number, info in sorted(model.structures.items(), key=lambda item: int(item[0])):
        roi_name = str(info.get("name", ""))
        if _match_structure_name(roi_name, targets):
            selected.append((int(roi_number), roi_name))
    return selected


def _summarize_requested_structure_matches(
    requested: Sequence[str],
    exported_names: Sequence[str],
) -> Tuple[str, str, str]:
    requested_clean = [str(s).strip() for s in requested if str(s).strip()]
    exported_clean = [str(s).strip() for s in exported_names if str(s).strip()]
    matched_requested: List[str] = []
    missing_requested: List[str] = []

    for requested_name in requested_clean:
        if any(_match_structure_name(exported_name, [requested_name]) for exported_name in exported_clean):
            matched_requested.append(requested_name)
        else:
            missing_requested.append(requested_name)

    return (
        ";".join(requested_clean),
        ";".join(matched_requested),
        ";".join(missing_requested),
    )


def _base_manifest_row(patient_id: str, candidate_count: int = 0) -> Dict[str, object]:
    return {
        "ID": patient_id,
        "Status": "",
        "Reason": "",
        "ChainCandidates": candidate_count,
        "SelectedDoseSummationType": "",
        "DoseUnit": "",
        "PrescriptionDoseGy": "",
        "CTSeriesUID": "",
        "RTSTRUCT": "",
        "RTPLAN": "",
        "RTDOSE": "",
        "RequestedStructures": "",
        "MatchedStructures": "",
        "MissingRequestedStructures": "",
        "MasksExported": 0,
        "ExportedMaskNames": "",
        "OutputFolder": "",
    }


def _safe_unlink_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return


def _export_case(
    model: DICOMDataModel,
    patient_id: str,
    save_dir: Path,
    target_spacing: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
    structures_to_export: Sequence[str],
) -> Tuple[List[str], Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)

    ct_path = save_dir / f"{patient_id}_CT.nii.gz"
    dose_path = save_dir / f"{patient_id}_Dose.nii.gz"

    model.resample_and_export_ct(target_spacing, target_shape, str(ct_path))
    model.resample_and_export_dose(target_spacing, target_shape, str(dose_path))

    selected_masks = _build_mask_export_list(model, structures_to_export)
    exported_names: List[str] = []
    for roi_number, roi_name in selected_masks:
        safe_name = _sanitize_name(roi_name)
        mask_path = save_dir / f"{patient_id}_Mask_{safe_name}.nii.gz"
        model.resample_and_export_mask(
            roi_number=roi_number,
            new_spacing=target_spacing,
            target_shape=target_shape,
            filepath=str(mask_path),
        )
        exported_names.append(roi_name)

    return exported_names, ct_path, dose_path


def run_curation_pipeline(
    raw_data_root: Path,
    output_root: Path,
    target_spacing: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
    structures_to_export: Sequence[str],
    include_root: bool = False,
) -> int:
    start_time = time.time()

    target_dirs = _collect_target_dirs(raw_data_root, include_root=include_root)
    if not target_dirs:
        LOG.error("No patient folders found in: %s", raw_data_root)
        return 2

    LOG.info("INIT: Found %d patient directories in %s", len(target_dirs), raw_data_root)
    LOG.info("TARGET SPACING: %s mm", target_spacing)
    LOG.info("TARGET SHAPE: %s voxels", target_shape)
    LOG.info("REQUESTED STRUCTURES: %s", ", ".join(structures_to_export) or "<none>")

    stats = {
        "Total": len(target_dirs),
        "Success": 0,
        "Fail_Integrity": 0,
        "Fail_Load": 0,
        "Fail_Export": 0,
    }
    manifest_rows: List[Dict[str, object]] = []

    for patient_dir in target_dirs:
        patient_id = patient_dir.name
        LOG.info("PROCESSING: %s", patient_id)

        selection = _select_preferred_chain(patient_dir)
        chain = selection.selected
        base_row = _base_manifest_row(patient_id, selection.candidate_count)

        if not chain:
            LOG.warning(
                "REJECTED %s: Incomplete chain (missing/invalid CT->RTSTRUCT->RTPLAN->RTDOSE link).",
                patient_id,
            )
            stats["Fail_Integrity"] += 1
            base_row.update(
                {
                    "Status": "REJECTED_INTEGRITY",
                    "Reason": "No complete linked RT chain",
                }
            )
            manifest_rows.append(base_row)
            continue

        if selection.candidate_count > 1:
            LOG.info(
                "INFO %s: %d complete chain(s) found. Selecting preferred chain with dose summation '%s'.",
                patient_id,
                selection.candidate_count,
                chain.dose_summation_type,
            )

        model = DICOMDataModel()
        requested_str, matched_str, missing_str = _summarize_requested_structure_matches(structures_to_export, [])
        try:
            model.load_data(
                patient_folder=chain.patient_folder,
                ct_series_uid=chain.ct_series_uid,
                rtstruct_path=chain.rtstruct_path,
                dose_path=chain.dose_path,
                rtplan_path=chain.rtplan_path,
            )
        except Exception as exc:
            LOG.error("ERROR %s: Load failed - %s", patient_id, exc)
            stats["Fail_Load"] += 1
            base_row.update(
                {
                    "Status": "ERROR_LOAD",
                    "Reason": str(exc),
                    "SelectedDoseSummationType": chain.dose_summation_type,
                    "CTSeriesUID": chain.ct_series_uid,
                    "RTSTRUCT": chain.rtstruct_path,
                    "RTPLAN": chain.rtplan_path,
                    "RTDOSE": chain.dose_path,
                    "RequestedStructures": requested_str,
                    "MatchedStructures": matched_str,
                    "MissingRequestedStructures": missing_str,
                }
            )
            manifest_rows.append(base_row)
            continue

        save_dir = output_root / patient_id
        exported_names: List[str] = []
        ct_path = save_dir / f"{patient_id}_CT.nii.gz"
        dose_path = save_dir / f"{patient_id}_Dose.nii.gz"

        try:
            exported_names, ct_path, dose_path = _export_case(
                model=model,
                patient_id=patient_id,
                save_dir=save_dir,
                target_spacing=target_spacing,
                target_shape=target_shape,
                structures_to_export=structures_to_export,
            )
            requested_str, matched_str, missing_str = _summarize_requested_structure_matches(
                structures_to_export,
                exported_names,
            )

            if missing_str:
                LOG.warning("WARN %s: Requested structure patterns not found: %s", patient_id, missing_str)

            stats["Success"] += 1
            LOG.info(
                "SUCCESS %s: Exported CT, Dose, and %d mask(s).",
                patient_id,
                len(exported_names),
            )
            base_row.update(
                {
                    "Status": "SUCCESS",
                    "Reason": "",
                    "SelectedDoseSummationType": chain.dose_summation_type,
                    "DoseUnit": str(model.dose_unit or ""),
                    "PrescriptionDoseGy": float(model.prescription_dose or 0.0),
                    "CTSeriesUID": chain.ct_series_uid,
                    "RTSTRUCT": chain.rtstruct_path,
                    "RTPLAN": chain.rtplan_path,
                    "RTDOSE": chain.dose_path,
                    "RequestedStructures": requested_str,
                    "MatchedStructures": matched_str,
                    "MissingRequestedStructures": missing_str,
                    "MasksExported": len(exported_names),
                    "ExportedMaskNames": ";".join(_sanitize_name(name) for name in exported_names),
                    "OutputFolder": str(save_dir),
                }
            )
            manifest_rows.append(base_row)

        except Exception as exc:
            LOG.error("ERROR %s: Export failed - %s", patient_id, exc)
            stats["Fail_Export"] += 1
            requested_str, matched_str, missing_str = _summarize_requested_structure_matches(
                structures_to_export,
                exported_names,
            )
            base_row.update(
                {
                    "Status": "ERROR_EXPORT",
                    "Reason": str(exc),
                    "SelectedDoseSummationType": chain.dose_summation_type,
                    "DoseUnit": str(model.dose_unit or ""),
                    "PrescriptionDoseGy": float(model.prescription_dose or 0.0),
                    "CTSeriesUID": chain.ct_series_uid,
                    "RTSTRUCT": chain.rtstruct_path,
                    "RTPLAN": chain.rtplan_path,
                    "RTDOSE": chain.dose_path,
                    "RequestedStructures": requested_str,
                    "MatchedStructures": matched_str,
                    "MissingRequestedStructures": missing_str,
                    "MasksExported": len(exported_names),
                    "ExportedMaskNames": ";".join(_sanitize_name(name) for name in exported_names),
                    "OutputFolder": str(save_dir),
                }
            )
            manifest_rows.append(base_row)
            # Remove potentially partial primary outputs if export failed midway.
            _safe_unlink_if_exists(ct_path)
            _safe_unlink_if_exists(dose_path)

    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    rejected = stats["Fail_Integrity"] + stats["Fail_Load"] + stats["Fail_Export"]

    LOG.info("=" * 50)
    LOG.info("BATCH COMPLETE in %s", elapsed)
    LOG.info("Total Patients: %d", stats["Total"])
    LOG.info("Successful:     %d", stats["Success"])
    LOG.info("Rejected/Errors:%d", rejected)
    LOG.info(
        "Breakdown -> Integrity: %d | Load: %d | Export: %d",
        stats["Fail_Integrity"],
        stats["Fail_Load"],
        stats["Fail_Export"],
    )
    LOG.info("=" * 50)

    manifest_path = output_root / "dataset_manifest.csv"
    _write_csv(manifest_path, manifest_rows, MANIFEST_HEADERS)

    summary_rows = [
        {"Metric": "Total", "Value": stats["Total"]},
        {"Metric": "Success", "Value": stats["Success"]},
        {"Metric": "Fail_Integrity", "Value": stats["Fail_Integrity"]},
        {"Metric": "Fail_Load", "Value": stats["Fail_Load"]},
        {"Metric": "Fail_Export", "Value": stats["Fail_Export"]},
        {"Metric": "Rejected_Or_Error_Total", "Value": rejected},
        {"Metric": "Elapsed", "Value": elapsed},
        {"Metric": "TargetSpacing", "Value": "x".join(str(v) for v in target_spacing)},
        {"Metric": "TargetShape", "Value": "x".join(str(v) for v in target_shape)},
        {"Metric": "RequestedStructures", "Value": ";".join(structures_to_export)},
    ]
    summary_path = output_root / "curation_summary.csv"
    _write_csv(summary_path, summary_rows, headers=["Metric", "Value"])

    print(f"Done. Check {(output_root / 'curation_log.txt')} for details.")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    return 0 if rejected == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless RT DICOM curation pipeline to standardized NIfTI outputs."
    )
    parser.add_argument(
        "--raw-data-root",
        required=True,
        help="Root folder containing patient DICOM folders.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output folder for curated NIfTI files and reports.",
    )
    parser.add_argument(
        "--spacing",
        default="1,1,1",
        help="Target voxel spacing as x,y,z in mm (default: 1,1,1).",
    )
    parser.add_argument(
        "--shape",
        default="128,128,64",
        help="Target tensor shape as x,y,z (default: 128,128,64).",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=DEFAULT_STRUCTURES,
        help="ROI name patterns for mask export (case-insensitive normalized contains match).",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Also treat raw-data-root itself as one patient folder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_data_root = Path(args.raw_data_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if not raw_data_root.exists() or not raw_data_root.is_dir():
        print(f"Invalid --raw-data-root: {raw_data_root}")
        return 2

    target_spacing = _parse_tuple_arg(args.spacing, length=3, cast_type=float, label="--spacing")
    target_shape = _parse_tuple_arg(args.shape, length=3, cast_type=int, label="--shape")
    if any(v <= 0 for v in target_spacing):
        print(f"Invalid --spacing values (must be > 0): {target_spacing}")
        return 2
    if any(v <= 0 for v in target_shape):
        print(f"Invalid --shape values (must be > 0): {target_shape}")
        return 2

    log_path = _setup_logger(output_root, log_level=args.log_level)
    LOG.info("Log file: %s", log_path)

    return run_curation_pipeline(
        raw_data_root=raw_data_root,
        output_root=output_root,
        target_spacing=target_spacing,
        target_shape=target_shape,
        structures_to_export=[s for s in args.structures if str(s).strip()],
        include_root=bool(args.include_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
