"""EfficientNet-B2 adapter, wraps the local `efficienT_b2_2classes` model package.

Isolates two awkward facts about this model in one place (same spirit as
`swin_ynet.py`'s docstring):

1. It is a *git-ignored, local-only* package under `models/` (see .gitignore),
   imported by adding its directory to ``sys.path`` rather than via a normal
   package import. Its own modules do bare `import configs`, `from model
   import build_model`, `from dataset import chip_to_tensor`, so the package
   directory must be first on `sys.path` for those bare imports to resolve.
2. It was trained on bands in descending-wavelength order
   (B12,B11,B8A,B08,B07,B06,B05,B04,B03,B02), not the cube's native
   ascending `band_names` order. `chips.read_chip()` always returns chips in
   the cube's native column order, so this adapter reorders channels before
   calling the model. The order is taken from `configs.py`'s own comment
   ("band order is B12,B11,...,B2"), cross-checked against its
   `DISPLAY_BANDS = (3, 8, 9)` ("approx NIR/Red/Green": index 3 -> B8A,
   index 8 -> B4, index 9 -> B3, which matches). The mapping onto the cube's
   actual column positions is computed from the cube's own `band_names`
   attribute (read once, lazily) rather than hardcoded indices, so a
   mismatched cube fails loudly instead of silently mispredicting.

   This band order is the primary-source comment in the model's own config,
   not an independent confirmation from Manuel, worth a final check before
   the report is finalised, but solid enough to run on.

The shared pipeline talks to this model only through the adapter interface
documented in `inference/adapters/__init__.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAME = "efficientnet_b2"
# models/efficienT_b2_2classes/configs.py: CLASS_NAMES = {0: "Background", 1:
# "Burned"} -> burned = "Burned".
BURNED_CLASS = 1

# Band order this model was trained on (descending wavelength). See module
# docstring point 2 for the derivation.
_MODEL_BAND_ORDER = ["B12", "B11", "B8A", "B08", "B07", "B06", "B05", "B04", "B03", "B02"]

_band_reorder_idx = None  # computed once, lazily, from the cube's band_names (see predict())


def _normalize_band_name(b: str) -> str:
    """'B2' / 'B02' / 'b2' -> 'B02'. Sentinel-2 band codes use inconsistent zero-padding
    across tools, so comparisons are done on the normalized form."""
    b = b.strip().upper()
    if b.startswith("B") and b[1:].isdigit():
        return f"B{int(b[1:]):02d}"
    return b


def _add_package(package_dir: str | Path) -> None:
    """Put the model package directory first on `sys.path` (idempotent)."""
    package_dir = str(Path(package_dir).resolve())
    if not Path(package_dir, "predict.py").exists():
        raise FileNotFoundError(
            f"Model package not found at {package_dir!r}. The models/ folder is "
            "git-ignored and kept local; make sure it is present on this machine."
        )
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)


def _reorder_idx(cube_band_names: list) -> list:
    """Map this model's expected band order onto the cube's native column order."""
    lookup = {_normalize_band_name(b): i for i, b in enumerate(cube_band_names)}
    missing = [b for b in _MODEL_BAND_ORDER if _normalize_band_name(b) not in lookup]
    if missing:
        raise ValueError(
            f"_MODEL_BAND_ORDER entries {missing} have no match in the cube's own "
            f"band_names ({cube_band_names}), even after normalizing zero-padding "
            "(e.g. 'B2' vs 'B02'). Do not guess further, check with Manuel."
        )
    return [lookup[_normalize_band_name(b)] for b in _MODEL_BAND_ORDER]


def load(weights_path: str | Path, package_dir: str | Path, device):
    """Build the model and load the checkpoint onto `device`.

    Returns `(handle, model)` where `handle` is the imported `predict` module
    (carrying `predict_chips`).
    """
    _add_package(package_dir)
    import torch
    from model import build_model  # noqa: E402  (path injected above)

    import predict as effnet  # noqa: E402

    model = build_model(encoder_weights=None).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return effnet, model


def predict(handle, before: np.ndarray, after: np.ndarray, model, device) -> np.ndarray:
    """Run one batch. before/after: (B,256,256,C) uint16 -> labels (B,256,256) uint8.

    The adapter interface (see `inference/adapters/__init__.py`) doesn't pass
    cube metadata into `predict()`, so the band order is resolved lazily here
    on first call, via `utils.config.load_config()` + `inference.chips.open_meta()`
    (the same cube every run targets), and cached for the rest of the run.
    """
    global _band_reorder_idx
    if _band_reorder_idx is None:
        from utils.config import load_config
        from inference import chips

        cfg = load_config()
        meta = chips.open_meta(cfg.hdf5_path, cfg.xys_sentinel)
        _band_reorder_idx = _reorder_idx(meta.band_names)
        print(f"[efficientnet_b2] band reorder (cube -> model): {_band_reorder_idx}")

    before = before[..., _band_reorder_idx]
    after = after[..., _band_reorder_idx]
    # predict_chips wants a *list* of (H,W,C) uint16 arrays, not a stacked batch.
    return handle.predict_chips(list(before), list(after), model, device=device)
