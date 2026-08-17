"""
Application entry point for CURATOR.

This module bootstraps the Qt application, configures logging and locale,
creates the MVC components, and starts the main event loop.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from PyQt6.QtCore import QLocale
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

import config
from controller import AppController
from model import DICOMDataModel
from view import MainWindow

_DEFAULT_MASK_BACKEND = "legacy"


def _configure_logging() -> None:
    """Configures root logging once for the application process."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        force=True,
    )


def _create_application(argv: list[str]) -> QApplication:
    """Creates and configures the QApplication instance."""
    app = QApplication(argv)
    app.setApplicationName("CURATOR")
    app.setApplicationDisplayName("CURATOR")
    
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CURATOR_icon.png")
    app.setWindowIcon(QIcon(icon_path))
 
    # Ensure numeric parsing/formatting stays stable regardless of system locale.
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    return app


def _resolve_mask_backend() -> str:
    """Reads and validates the requested contour rasterization backend."""
    backend = os.environ.get("DICOM_MASK_BACKEND", _DEFAULT_MASK_BACKEND)
    backend = str(backend or "").strip().lower()
    return backend or _DEFAULT_MASK_BACKEND


def _build_model() -> DICOMDataModel:
    """Creates the data model and applies the requested mask backend."""
    model = DICOMDataModel()
    requested_backend = _resolve_mask_backend()
    try:
        model.set_structure_mask_backend(requested_backend)
    except ValueError:
        logging.warning(
            "Invalid DICOM_MASK_BACKEND=%r. Falling back to %r.",
            requested_backend,
            _DEFAULT_MASK_BACKEND,
        )
        model.set_structure_mask_backend(_DEFAULT_MASK_BACKEND)
    return model


def _show_startup_error(message: str, parent: Optional[MainWindow] = None) -> None:
    """Displays a fatal startup error dialog when the UI is available."""
    try:
        QMessageBox.critical(parent, "Startup Error", message)
    except Exception:
        # Last-resort fallback for environments where message boxes fail.
        logging.error("Startup Error: %s", message)


def main() -> int:
    """Initializes and runs CURATOR"""
    _configure_logging()
    logging.info("Application starting...")

    app = _create_application(sys.argv)
    view: Optional[MainWindow] = None

    try:
        model = _build_model()
        view = MainWindow()

        controller = AppController(model=model, view=view)
        view.controller = controller
        view.apply_theme(dark=True)
        view.show()

        logging.info("Application initialized successfully.")
        return app.exec()

    except Exception as exc:
        logging.exception("Fatal startup error: %s", exc)
        _show_startup_error(str(exc), parent=view)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
