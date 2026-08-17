"""
Central configuration for the DICOM Viewer application.

This module intentionally keeps a flat, import-friendly API so the rest of the
application can continue to access settings as ``config.SOME_NAME``.

A few goals guide this file:
- keep visualization defaults in one place,
- document whether a setting affects display only or core analysis,
- provide small validation helpers for values that are easy to misconfigure.
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

from vtkmodules.vtkCommonColor import vtkNamedColors


# --- Logging Configuration -------------------------------------------------
LOG_LEVEL: Final[int] = logging.INFO
LOG_FORMAT: Final[str] = "%(asctime)s - %(levelname)s - %(message)s"


# --- CT / Dose Visualization Defaults -------------------------------------
# Standard soft-tissue defaults for initial 2D CT display.
DEFAULT_CT_WINDOW: Final[int] = 400
DEFAULT_CT_LEVEL: Final[int] = 40

# Optional preset map for future reuse by the UI/controller.
WINDOW_PRESETS: Final[dict[str, tuple[int, int]]] = {
    "Soft Tissue": (DEFAULT_CT_WINDOW, DEFAULT_CT_LEVEL),
    "Lung": (1500, -600),
    "Bone": (2000, 350),
}

# Display-only opacity for dose overlays. This does not affect DVH/CI.
DOSE_OPACITY: Final[float] = 0.5

# Initial dose threshold shown in the UI as a percentage of the available
# slider range, not directly as a percentage of prescription.
DOSE_INITIAL_THRESHOLD_PERCENT: Final[float] = 5.0

# Display-only low-dose suppression to reduce interpolation wash/noise.
DOSE_DISPLAY_NOISE_FLOOR_GY: Final[float] = 0.05
DOSE_DISPLAY_NOISE_FLOOR_PERCENT_OF_MAX: Final[float] = 2.0

CONTOUR_LINE_WIDTH: Final[float] = 2.5

# Legacy flat background tuple retained for backward compatibility with older
# rendering code. VTK expects values in the range [0, 1].
BACKGROUND_COLOR: Final[Tuple[float, float, float]] = (0.1, 0.1, 0.1)
BACKGROUND_COLOR_TOP: Final[str] = "DarkSlateGray"
BACKGROUND_COLOR_BOTTOM: Final[str] = "SlateGray"


# --- 3D Machine Visualization ---------------------------------------------
# DICOM RT beam limiting device identifiers used by the 3D machine view.
MLC_DEVICE_TYPE: Final[str] = "MLCX"
JAWS_X_DEVICE_TYPE: Final[str] = "ASYMX"
JAWS_Y_DEVICE_TYPE: Final[str] = "ASYMY"

JAWS_COLOR: Final[str] = "Tomato"
MLC_COLOR: Final[str] = "ForestGreen"

# Schematic geometry defaults used by the 3D beam-device visualization.
# These are display parameters, not vendor-accurate mechanical dimensions.
MACHINE_GEOMETRY_OUTER_EXTENT_MM: Final[float] = 200.0
JAW_THICKNESS_MM: Final[float] = 15.0
LEAF_THICKNESS_MM: Final[float] = 15.0


# --- Shared VTK Color Registry --------------------------------------------
COLORS: Final[vtkNamedColors] = vtkNamedColors()


# --- Validation / Normalization Helpers -----------------------------------
def clamp_unit_interval(value: float, default: float = 1.0) -> float:
    """Returns ``value`` clamped to the inclusive range [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(1.0, numeric))


def positive_float(value: float, default: float) -> float:
    """Returns a strictly positive float, falling back to ``default``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if numeric > 0.0 else float(default)


# Normalized values exposed under the legacy names used across the app.
DOSE_OPACITY = clamp_unit_interval(DOSE_OPACITY, default=0.5)
DOSE_INITIAL_THRESHOLD_PERCENT = positive_float(DOSE_INITIAL_THRESHOLD_PERCENT, default=5.0)
DOSE_DISPLAY_NOISE_FLOOR_GY = max(0.0, float(DOSE_DISPLAY_NOISE_FLOOR_GY))
DOSE_DISPLAY_NOISE_FLOOR_PERCENT_OF_MAX = max(0.0, float(DOSE_DISPLAY_NOISE_FLOOR_PERCENT_OF_MAX))
CONTOUR_LINE_WIDTH = positive_float(CONTOUR_LINE_WIDTH, default=2.5)
JAW_THICKNESS_MM = positive_float(JAW_THICKNESS_MM, default=15.0)
LEAF_THICKNESS_MM = positive_float(LEAF_THICKNESS_MM, default=15.0)
MACHINE_GEOMETRY_OUTER_EXTENT_MM = positive_float(MACHINE_GEOMETRY_OUTER_EXTENT_MM, default=200.0)
