"""Read and reconstruct dense 256x256x10 chips from the HDF5 cube.

The cube stores pixels on a flat axis in 65536-pixel blocks — one block per
256x256 chip. Each pixel carries its own UTM coordinate (`xs_new`/`ys_new`,
sentinel for padding), and each chip block carries its position on the tile's
256-chip grid (`chip_x_bin`/`chip_y_bin`). We scatter a chip's valid pixels into
a dense grid using those coordinates; padding / missing pixels stay NODATA.

Reconstruction math is verified to place all 65536 pixels of a full chip with
zero collisions (see the build-time validation).
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from rasterio.transform import from_origin

PIXELS_PER_CHIP = 65536  # 256 * 256


@dataclass(frozen=True)
class CubeMeta:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    res: float
    chip_size: int
    n_bands: int
    nodata: int
    sentinel: int
    band_names: list

    @property
    def width(self) -> int:
        return int(round((self.max_x - self.min_x) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.max_y - self.min_y) / self.res))

    @property
    def transform(self):
        return from_origin(self.min_x, self.max_y, self.res, self.res)


def open_meta(hdf5_path, sentinel: int) -> CubeMeta:
    with h5py.File(hdf5_path, "r") as f:
        a = f.attrs
        band_names = [b.decode() if isinstance(b, bytes) else b for b in a["band_names"]]
        return CubeMeta(
            min_x=float(a["bounds_left"]),
            max_x=float(a["bounds_right"]),
            min_y=float(a["bounds_bottom"]),
            max_y=float(a["bounds_top"]),
            res=float(a["pixel_res"]),
            chip_size=int(a["chip_size"]),
            n_bands=int(f["values"].shape[1]),
            nodata=int(a["nodata_val"]),
            sentinel=int(sentinel),
            band_names=band_names,
        )


def chip_ids_with_data(hdf5_path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        cpc = f["chip_pixel_count"][:]
    return np.nonzero(cpc > 0)[0]


def load_chip_bins(hdf5_path):
    """Return (chip_x_bin, chip_y_bin) arrays, indexed by chip id."""
    with h5py.File(hdf5_path, "r") as f:
        return f["chip_x_bin"][:], f["chip_y_bin"][:]


def read_chip(f: h5py.File, t_idx: int, chip_id: int, meta: CubeMeta, cxb, cyb):
    """Reconstruct one dense chip for acquisition ``t_idx``.

    Returns
    -------
    chip      : (H, W, n_bands) uint16 — NODATA where missing/padding
    footprint : (H, W) bool — pixels that belong to the tile (have valid coords)
    observed  : (H, W) bool — footprint pixels with real (non-NODATA) data here
    """
    lo = chip_id * PIXELS_PER_CHIP
    hi = lo + PIXELS_PER_CHIP
    vals = f["values"][t_idx, :, lo:hi]      # (n_bands, 65536) uint16
    xs = f["xs_new"][lo:hi]
    ys = f["ys_new"][lo:hi]

    cs = meta.chip_size
    res = meta.res
    chip_left = meta.min_x + int(cxb[chip_id]) * cs * res
    chip_top = meta.max_y - int(cyb[chip_id]) * cs * res

    valid_coord = (xs != meta.sentinel) & (ys != meta.sentinel)
    col = np.round((xs[valid_coord] - chip_left) / res).astype(np.int64)
    row = np.round((chip_top - ys[valid_coord]) / res).astype(np.int64)
    in_bounds = (col >= 0) & (col < cs) & (row >= 0) & (row < cs)
    col = col[in_bounds]
    row = row[in_bounds]
    src = np.flatnonzero(valid_coord)[in_bounds]   # indices into the flat pixel axis

    chip = np.full((cs, cs, meta.n_bands), meta.nodata, dtype=np.uint16)
    chip[row, col, :] = vals[:, src].T             # (n_sel, n_bands)

    footprint = np.zeros((cs, cs), dtype=bool)
    footprint[row, col] = True
    observed = footprint & (chip[:, :, 0] != meta.nodata)
    return chip, footprint, observed
