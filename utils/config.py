"""Project-wide configuration loader.

Centralises path handling so notebooks and scripts don't hardcode
machine-specific absolute paths (e.g. `/Users/<name>/Documents/...`).

Usage (from a notebook in `notebooks/`):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))   # repo root on sys.path

    from utils.config import load_config
    cfg = load_config()

    HDF5_PATH = cfg.hdf5_path
    OUT_DIR   = cfg.out_dir
    TILE_CRS  = cfg.tile_crs
    ...

Override any value for your machine by copying `config.yaml` to
`config.local.yaml` (git-ignored) and editing it — `load_config` prefers
the local file when present, so you never need to touch tracked files or
hardcode personal paths in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_FILENAMES = ("config.local.yaml", "config.yaml")


def _find_repo_root(start: Path) -> Path:
    """Walk upwards from `start` until a config file (or .git) is found."""
    for candidate in (start, *start.parents):
        if any((candidate / name).exists() for name in _CONFIG_FILENAMES):
            return candidate
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root (no config.yaml or .git found "
        f"above {start}). Run notebooks from within the cloned repository."
    )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    data_root: Path
    tile_id: str
    hdf5_path: Path
    out_dir: Path
    pixel_size_m: float
    tile_crs: str
    buffer_m: float
    nodata_val: int
    xys_sentinel: int


def load_config(start: Path | str | None = None) -> Config:
    """Load configuration, resolving every path relative to the repo root.

    Parameters
    ----------
    start:
        Where to begin searching for the repo root. Defaults to the
        current working directory (so it works whether you run notebooks
        from `notebooks/` or the repo root).
    """
    start_path = Path(start) if start is not None else Path.cwd()
    repo_root = _find_repo_root(start_path.resolve())

    config_path = next(
        (repo_root / name for name in _CONFIG_FILENAMES if (repo_root / name).exists()),
        None,
    )
    if config_path is None:  # pragma: no cover - guarded by _find_repo_root
        raise FileNotFoundError(f"No config file found in {repo_root}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    data_root = (repo_root / raw["data"]["root"]).resolve()
    tile_id = raw["data"]["tile_id"]
    # `hdf5_filename` is resolved relative to data_root, so it may include a
    # subdirectory (e.g. "hdf5/T29TPG.h5"). Kept local only — see .gitignore.
    hdf5_path = (data_root / raw["data"]["hdf5_filename"]).resolve()

    out_dir = (repo_root / raw["output"]["figures_dir"] / tile_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    constants = raw.get("constants", {})

    return Config(
        repo_root=repo_root,
        data_root=data_root,
        tile_id=tile_id,
        hdf5_path=hdf5_path,
        out_dir=out_dir,
        pixel_size_m=float(constants.get("pixel_size_m", 10.0)),
        tile_crs=constants.get("tile_crs", "EPSG:32629"),
        buffer_m=float(constants.get("buffer_m", 5000)),
        nodata_val=int(constants.get("nodata_val", 65535)),
        xys_sentinel=int(constants.get("xys_sentinel", -9999)),
    )
