"""Assemble per-chip predictions into tile-wide georeferenced rasters.

Three single-band GeoTIFFs are written on the tile grid (10 m, EPSG:32629),
aligned pixel-for-pixel with `data/processed/icnf_burned_labels_t29tpg_2025.tif`:

  * pred     — 3-class label (0=Background, 1=Cuts, 2=Fires), nodata 255
  * burned   — binary burned mask (Fires only), nodata 255
  * observed — 1 where both before & after scenes had data, 0 where in-footprint
               but unobserved, nodata 255 outside the footprint

The whole tile is accumulated in memory (3 x ~69 MB uint8 arrays) and each
GeoTIFF is written in a single pass. We deliberately do NOT stream scattered
windowed writes into a compressed+tiled GeoTIFF: GDAL cannot rewrite an
already-flushed compressed block, so out-of-spatial-order chip writes get
dropped. A single sequential write avoids that entirely and keeps the output
small (LZW) and resumable via periodic full saves.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio

NODATA = 255
KEYS = ("pred", "burned", "observed")


def build_blocks(labels, footprint, observed_both, burned_class):
    """Turn one chip's model output + masks into the three uint8 blocks.

    Pixels are trusted only where observed in *both* scenes; everything else is
    NODATA in the pred/burned blocks. The observed block additionally records
    in-footprint-but-unobserved pixels as 0.
    """
    cs = labels.shape[0]
    pred = np.full((cs, cs), NODATA, dtype=np.uint8)
    burned = np.full((cs, cs), NODATA, dtype=np.uint8)
    obs = np.full((cs, cs), NODATA, dtype=np.uint8)

    pred[observed_both] = labels[observed_both].astype(np.uint8)
    burned[observed_both] = (labels[observed_both] == burned_class).astype(np.uint8)
    obs[footprint] = observed_both[footprint].astype(np.uint8)
    return pred, burned, obs


def _write(path, arr, transform, crs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


class TileMosaic:
    """In-memory tile accumulator for the three output bands."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.bands = {k: np.full((height, width), NODATA, dtype=np.uint8) for k in KEYS}

    @classmethod
    def load(cls, paths, width, height):
        """Reconstruct accumulator state from existing rasters (for --resume)."""
        m = cls(width, height)
        for k in KEYS:
            if os.path.exists(paths[k]):
                with rasterio.open(paths[k]) as r:
                    m.bands[k] = r.read(1)
        return m

    def place(self, chip_id, cxb, cyb, blocks, cs):
        """Drop one chip's (pred, burned, observed) blocks at its grid position."""
        r0 = int(cyb[chip_id]) * cs
        c0 = int(cxb[chip_id]) * cs
        h = min(cs, self.height - r0)
        w = min(cs, self.width - c0)
        if h <= 0 or w <= 0:
            return
        for k, block in zip(KEYS, blocks):
            self.bands[k][r0:r0 + h, c0:c0 + w] = block[:h, :w]

    def save(self, paths, transform, crs):
        for k in KEYS:
            _write(paths[k], self.bands[k], transform, crs)

    def burned_pixels(self) -> int:
        return int((self.bands["burned"] == 1).sum())
