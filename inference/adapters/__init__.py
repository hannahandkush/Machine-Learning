"""Per-model inference adapters.

The rest of the pipeline (``chips``, ``scene_select``, ``mosaic``, ``run``) is
model-agnostic. Each model that follows the before/after change-detection
contract gets ONE adapter module here, exposing a uniform interface:

    load(weights_path, package_dir, device) -> (handle, model)
    predict(handle, before, after, model, device) -> np.ndarray (B, H, W) uint8
    BURNED_CLASS : int   # class index meaning "burned" for this model
    NAME : str

To add a model: drop a new module in this folder, implement that interface, and
register it in ``ADAPTERS`` below. Select it at runtime via config ``model.kind``
(or the ``--model-kind``/``--weights``/``--package-dir`` CLI overrides in
``run.py``). See ``swin_ynet.py`` for the reference implementation and
``efficientnet.py`` for a second one (also handles a model-specific band
reorder, see its module docstring).
"""
from __future__ import annotations

import importlib

# name (config model.kind) -> module path
ADAPTERS = {
    "swin_ynet": "inference.adapters.swin_ynet",
    "efficientnet_b2": "inference.adapters.efficientnet",
}


def get_adapter(kind: str):
    """Return the adapter module for ``kind`` (raises with the valid options)."""
    if kind not in ADAPTERS:
        raise ValueError(
            f"Unknown model kind {kind!r}. Available: {sorted(ADAPTERS)}"
        )
    return importlib.import_module(ADAPTERS[kind])


def pick_device(prefer: str | None = None):
    """Select a torch device: explicit override, else cuda → mps → cpu.

    Shared across all adapters — device choice is not model-specific.
    """
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
