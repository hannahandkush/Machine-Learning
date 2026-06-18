"""Swin-YNet adapter — wraps the local `bacdm_predict` model package.

Isolates the two awkward facts about this model in one place:

1. It is a *git-ignored, local-only* package under `models/` (see .gitignore),
   imported by adding its directory to ``sys.path`` rather than via a normal
   package import.
2. Its public API (`load_model`, `predict_before_after_chips`) lives in a module
   literally named ``predict`` with sibling modules ``AAA_Configs`` and ``data``
   — generic names we keep contained here.

The shared pipeline talks to this model only through the adapter interface
documented in ``inference/adapters/__init__.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAME = "swin_ynet"
# AAA_Configs.CLASS_NAMES = {0: 'Background', 1: 'Cuts', 2: 'Fires'} → burned = Fires.
BURNED_CLASS = 2


def _add_package(package_dir: str | Path) -> None:
    """Put the bacdm_predict directory first on ``sys.path`` (idempotent)."""
    package_dir = str(Path(package_dir).resolve())
    if not Path(package_dir, "predict.py").exists():
        raise FileNotFoundError(
            f"Model package not found at {package_dir!r}. The models/ folder is "
            "git-ignored and kept local — make sure it is present on this machine."
        )
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)


def load(weights_path: str | Path, package_dir: str | Path, device):
    """Import the model package and load the checkpoint onto ``device``.

    Returns ``(handle, model)`` where ``handle`` is the imported ``predict``
    module (carrying ``predict_before_after_chips``).
    """
    _add_package(package_dir)
    import predict as bacdm  # noqa: E402  (path injected above)

    model = bacdm.load_model(str(weights_path), device=device)
    return bacdm, model


def predict(handle, before: np.ndarray, after: np.ndarray, model, device) -> np.ndarray:
    """Run one batch. before/after: (B,256,256,C) uint16 → labels (B,256,256) uint8."""
    return handle.predict_before_after_chips(before, after, model, device=device)


def class_names(package_dir: str | Path) -> dict:
    """Return AAA_Configs.CLASS_NAMES (e.g. {0:'Background',1:'Cuts',2:'Fires'})."""
    _add_package(package_dir)
    import AAA_Configs  # noqa: E402

    return dict(AAA_Configs.CLASS_NAMES)
