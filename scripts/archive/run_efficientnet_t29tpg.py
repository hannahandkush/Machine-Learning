"""
run_efficientnet_t29tpg.py

Deploys the EfficientNet-B2 U-Net change-detection model (Manuel's weights,
models/efficienT_b2_2classes/best_model.pth) over the evaluable T29TPG fire
events, following the before/after pair selection strategy fixed in
documents/evaluation_protocol.md (Sections 4 and 5).

Pipeline
--------
1.  Read scene metadata directly from the HDF5 cube and classify every
    acquisition as full_clear / partial_clear / edge_pass / clouded
    (protocol Section 5.1).
2.  Pick the shared July 1-25 baseline (June fallback) used as the "before"
    image for every fire (Section 5.2).
3.  Load ICNF fire polygons, clip to the T29TPG footprint, keep events
    above the 65 ha evaluable threshold (Section 4.2).
4.  For each evaluable fire, pick the earliest usable scene within 30 days
    of DH_Fim as the "after" image (Section 5.3). Fires with no usable
    after scene are skipped, matching the protocol.
5.  Reconstruct dense (256, 256, 10) uint16 chips for every grid cell that
    intersects the fire's bounding box, for both the before and after
    dates, run them through predict_chips(), and stitch the per-chip
    predictions into one tile-aligned GeoTIFF.
6.  Write a per-event summary CSV (dates used, scene labels, predicted
    burned area) alongside the raster.

IMPORTANT: one thing worth a final check before trusting the output
---------------------------------------------------------------------
(a) Band order. The HDF5 stores bands in the order given by its own
    `band_names` attribute, which need not match the order the model was
    trained on. H5_BAND_ORDER below is set to B12,B11,B8A,B8,B7,B6,B5,B4,B3,B2,
    taken directly from a comment in models/efficienT_b2_2classes/configs.py
    (DISPLAY_BANDS = (3, 8, 9), documented there as "approx NIR/Red/Green",
    which only lines up with that order: index 3 = B8A, index 8 = B4,
    index 9 = B3). This is the model's own documented order, not a guess,
    but it is still worth confirming directly with Manuel before a graded
    submission. load_h5_meta() checks every entry against the file's own
    band_names (tolerant of zero-padding, e.g. "B2" vs "B02") and raises
    immediately if any band name doesn't match, rather than silently
    running on misaligned channels.
(b) Chip reconstruction from the HDF5's flattened/chunked pixel layout
    (the "values" dataset is (T, B, N), not a dense raster) is a
    derivation from inspecting notebooks/data_exploration.ipynb, not a
    documented format. Run with --selftest first: it reconstructs the
    single best-covered chip and checks the valid-pixel count against the
    HDF5's own `chip_pixel_count` array before any model weights are
    touched.

Usage
-----
    python scripts/run_efficientnet_t29tpg.py --selftest
    python scripts/run_efficientnet_t29tpg.py --limit-events 2   # dry run on 2 fires
    python scripts/run_efficientnet_t29tpg.py                    # full run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "efficienT_b2_2classes"))

from utils.config import load_config          # noqa: E402
import configs as C                            # noqa: E402  (model's own configs.py)
# predict_chips (torch / segmentation_models_pytorch / scipy / skimage) is imported
# lazily inside run(), after the --selftest early-return, so --selftest works on a
# machine that only has h5py/geopandas/rasterio and not the full model environment.

# --- (a) band order ---
# models/efficienT_b2_2classes/configs.py line 60 documents this directly:
# "1-based band indices used for RGB display; band order is B12,B11,...,B2",
# with DISPLAY_BANDS = (3, 8, 9) commented "approx NIR / Red / Green". Checking
# that comment against the order below: index 3 -> B8A (narrow NIR), index 8
# -> B4 (Red), index 9 -> B3 (Green). That matches exactly, so this is the
# model's own documented training order, not a guess, and is the standard
# 10-band Sentinel-2 change-detection set (B1/B9/B10 dropped) in descending
# wavelength. Still worth a final sanity check against Manuel directly before
# a graded submission, but this is enough to proceed under the deadline.
H5_BAND_ORDER: list[str] | None = ["B12", "B11", "B8A", "B08", "B07", "B06", "B05", "B04", "B03", "B02"]

USABLE_LABELS = {"full_clear", "partial_clear"}
JULY_START, JULY_END = "2025-07-01", "2025-07-25"
JUNE_START, JUNE_END = "2025-06-01", "2025-06-30"
AFTER_WINDOW_DAYS = 30
MIN_COVERAGE_FRACTION = 0.10
XYS_SENTINEL = -9999          # matches config.yaml constants.xys_sentinel


# ───────────────────────────── HDF5 metadata ──────────────────────────────────

def _normalize_band_name(b: str) -> str:
    """Canonicalize a Sentinel-2 band label so 'B2' and 'B02' compare equal.
    H5_BAND_ORDER above is transcribed from a code comment, not the HDF5
    file itself, so its zero-padding convention isn't guaranteed to match
    band_names exactly. Comparing on this normalized form means a padding
    difference can't cause a spurious "band not found" failure, or worse,
    silently match the wrong band."""
    b = b.strip().upper()
    if b.startswith("B") and b[1:].isdigit():
        return f"B{int(b[1:]):02d}"
    return b


@dataclass
class H5Meta:
    band_names: list
    chip_size: int
    pixel_res: float
    bounds_left: float
    bounds_right: float
    bounds_bottom: float
    bounds_top: float
    n_timestamps: int
    n_bands: int
    n_pixels: int
    chunk_pixel_len: int
    chip_x_bin: np.ndarray
    chip_y_bin: np.ndarray
    chip_pixel_count: np.ndarray
    chunk_lookup: dict          # (chip_x_bin, chip_y_bin) -> chunk id
    band_lookup: dict           # normalized band name -> index into band_names
    nodata_val: int


def load_h5_meta(h5_path: Path, nodata_val: int) -> H5Meta:
    with h5py.File(h5_path, "r") as f:
        a = dict(f.attrs)
        band_names = [b.decode() if isinstance(b, bytes) else b for b in a["band_names"]]
        chip_size = int(a["chip_size"])
        pixel_res = float(a["pixel_res"])
        bounds_left, bounds_right = float(a["bounds_left"]), float(a["bounds_right"])
        bounds_bottom, bounds_top = float(a["bounds_bottom"]), float(a["bounds_top"])

        values_shape = f["values"].shape
        chunk_pixel_len = f["values"].chunks[2]
        n_timestamps, n_bands, n_pixels = values_shape

        chip_x_bin = f["chip_x_bin"][:]
        chip_y_bin = f["chip_y_bin"][:]
        chip_pixel_count = f["chip_pixel_count"][:]

    band_lookup = {_normalize_band_name(b): i for i, b in enumerate(band_names)}

    if H5_BAND_ORDER is not None:
        if len(H5_BAND_ORDER) != len(band_names):
            raise ValueError(
                f"H5_BAND_ORDER has {len(H5_BAND_ORDER)} entries but the file has "
                f"{len(band_names)} bands ({band_names})."
            )
        missing = [b for b in H5_BAND_ORDER if _normalize_band_name(b) not in band_lookup]
        if missing:
            raise ValueError(
                f"H5_BAND_ORDER entries {missing} have no match in the file's own "
                f"band_names ({band_names}), even after normalizing zero-padding "
                f"(e.g. 'B2' vs 'B02'). The band order transcribed from configs.py "
                f"does not line up with this file; do not guess further, check with Manuel."
            )
    if n_bands != C.NUM_BANDS:
        raise ValueError(
            f"HDF5 has {n_bands} bands but models/efficienT_b2_2classes/configs.py "
            f"expects NUM_BANDS={C.NUM_BANDS}. Check this isn't the wrong file/model pairing."
        )
    if chip_size != C.CHIP_SIZE:
        raise ValueError(
            f"HDF5 chip_size={chip_size} does not match model CHIP_SIZE={C.CHIP_SIZE}."
        )

    # `reconstruct_chip` relies on a strict 1:1 mapping between chunks along the
    # pixel axis of `values` and entries in chip_x_bin/chip_y_bin (chunk ci <->
    # slice [ci*chunk_pixel_len : (ci+1)*chunk_pixel_len]). This was inferred from
    # notebooks/data_exploration.ipynb, not documented, so it is checked explicitly
    # rather than assumed: a mismatch here would silently slice the wrong pixels.
    import math
    expected_n_chunks = math.ceil(n_pixels / chunk_pixel_len)
    if len(chip_x_bin) != expected_n_chunks:
        raise ValueError(
            f"chip_x_bin has {len(chip_x_bin)} entries but values' pixel axis "
            f"({n_pixels} pixels / {chunk_pixel_len} per chunk) implies "
            f"{expected_n_chunks} chunks. The chunk<->chip 1:1 assumption this "
            f"script relies on does not hold for this file."
        )

    chunk_lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(chip_x_bin, chip_y_bin))}
    if len(chunk_lookup) != len(chip_x_bin):
        raise ValueError(
            "Duplicate (chip_x_bin, chip_y_bin) pairs found: the chunk<->chip "
            "mapping is not 1:1, reconstruct_chip() would silently use the wrong chunk."
        )

    return H5Meta(
        band_names=band_names, chip_size=chip_size, pixel_res=pixel_res,
        bounds_left=bounds_left, bounds_right=bounds_right,
        bounds_bottom=bounds_bottom, bounds_top=bounds_top,
        n_timestamps=n_timestamps, n_bands=n_bands, n_pixels=n_pixels,
        chunk_pixel_len=chunk_pixel_len, chip_x_bin=chip_x_bin, chip_y_bin=chip_y_bin,
        chip_pixel_count=chip_pixel_count, chunk_lookup=chunk_lookup,
        band_lookup=band_lookup, nodata_val=nodata_val,
    )


def classify_scenes(h5_path: Path) -> pd.DataFrame:
    """Protocol Section 5.1, adapted to the dataset names confirmed by
    notebooks/data_exploration.ipynb (the protocol draft used placeholder
    names, `timestamps` and string dates, written before the file was
    accessible; the real dataset is `original_timestamps`, epoch ms)."""
    with h5py.File(h5_path, "r") as f:
        timestamps = pd.to_datetime(f["original_timestamps"][:], unit="ms")
        df_scenes = pd.DataFrame({
            # Position along the HDF5 time axis. values[t_idx] only means the right
            # acquisition if t_idx is this, the file's own order, not the row's
            # position after the date-sort two lines below. Kept as an explicit
            # column so it survives the sort instead of being silently lost.
            "t_idx": np.arange(len(timestamps)),
            "date": timestamps,
            "cloud_pct": f["cloud_cover_pt"][:],
            "total_px": f["pixel_count_pt"][:],
            "orbit_px": f["count_orbit_pixels_pt"][:],
        })

    df_scenes["swath_coverage"] = df_scenes["orbit_px"] / df_scenes["total_px"]

    def classify(row):
        if row.swath_coverage >= 0.95 and row.cloud_pct <= 10:
            return "full_clear"
        if row.swath_coverage < 0.50:
            return "edge_pass"
        if row.swath_coverage >= 0.50 and row.cloud_pct <= 20:
            return "partial_clear"
        return "clouded"

    df_scenes["label"] = df_scenes.apply(classify, axis=1)
    return df_scenes.sort_values("date").reset_index(drop=True)


def select_baseline(df_scenes: pd.DataFrame) -> pd.Series:
    """Protocol Section 5.2. Returns the full scene row (date, label, t_idx, ...)
    rather than a (date, label) tuple, so the caller reads t_idx straight off it
    instead of re-deriving it by matching on date (see note on t_idx above)."""
    july = df_scenes[
        (df_scenes["date"] >= JULY_START) & (df_scenes["date"] <= JULY_END)
        & (df_scenes["label"].isin(USABLE_LABELS))
    ].sort_values("cloud_pct")
    if not july.empty:
        return july.iloc[0]

    june = df_scenes[
        (df_scenes["date"] >= JUNE_START) & (df_scenes["date"] <= JUNE_END)
        & (df_scenes["label"].isin(USABLE_LABELS))
    ].sort_values("cloud_pct")
    if june.empty:
        raise RuntimeError("No usable scene in July 1-25 or all of June. Cannot pick a baseline.")
    return june.iloc[0]


def select_after_scene(fire_end: pd.Timestamp, df_scenes: pd.DataFrame):
    """Protocol Section 5.3. Returns None if no usable scene within 30 days."""
    candidates = df_scenes[
        (df_scenes["date"] >= fire_end)
        & (df_scenes["date"] <= fire_end + pd.Timedelta(days=AFTER_WINDOW_DAYS))
        & (df_scenes["label"].isin(USABLE_LABELS))
    ].sort_values("date")
    return None if candidates.empty else candidates.iloc[0]


# ───────────────────────────── fire events ────────────────────────────────────

def evaluable_size_threshold_ha(meta: H5Meta) -> float:
    """Protocol Section 4.2: a fire only counts as evaluable if it covers at
    least MIN_COVERAGE_FRACTION of one chip's footprint. Factored out so this
    figure is computed in exactly one place (it is also used for the log line
    in run(), and two independent copies of the same arithmetic is how the
    two quietly drift out of sync after a future edit)."""
    chip_area_ha = (meta.chip_size * meta.pixel_res) ** 2 / 1e4
    return chip_area_ha * MIN_COVERAGE_FRACTION


def load_evaluable_fires(meta: H5Meta, tile_crs: str) -> gpd.GeoDataFrame:
    """Protocol Section 4.2, against the raw ICNF shapefile (has geometry,
    unlike data/processed/icnf_fires_t29tpg_2025.csv which drops it).
    Note: ESRI .shp/.dbf field names are truncated to 10 characters, so any
    new column referenced here must be checked against the .dbf, not assumed
    from a longer name elsewhere in the project."""
    fires = gpd.read_file(REPO_ROOT / "data/shapefiles/ground_truth_ICNF/ardida_2025.shp")
    fires = fires.to_crs(tile_crs)

    tile_box = box(meta.bounds_left, meta.bounds_bottom, meta.bounds_right, meta.bounds_top)
    fires_in_tile = fires[fires.intersects(tile_box)].copy()
    fires_in_tile["DH_Fim"] = pd.to_datetime(fires_in_tile["DH_Fim"])
    fires_in_tile["DH_Inicio"] = pd.to_datetime(fires_in_tile["DH_Inicio"])

    size_threshold_ha = evaluable_size_threshold_ha(meta)

    large_fires = (
        fires_in_tile[fires_in_tile["AreaHaSIG"] > size_threshold_ha]
        .sort_values("AreaHaSIG", ascending=False)
        .reset_index(drop=True)
    )
    return large_fires


def chip_indices_for_bounds(meta: H5Meta, minx, miny, maxx, maxy):
    """Map a bounding box (tile CRS) to the set of (chip_x_bin, chip_y_bin)
    cells it touches, restricted to chips that actually exist in the file."""
    cell_m = meta.chip_size * meta.pixel_res
    col_start = int(np.floor((minx - meta.bounds_left) / cell_m))
    col_end = int(np.floor((maxx - meta.bounds_left) / cell_m))
    row_start = int(np.floor((meta.bounds_top - maxy) / cell_m))
    row_end = int(np.floor((meta.bounds_top - miny) / cell_m))

    cells = []
    for cy in range(row_start, row_end + 1):
        for cx in range(col_start, col_end + 1):
            if (cx, cy) in meta.chunk_lookup:
                cells.append((cx, cy))
    return cells


# ───────────────────────────── chip reconstruction ─────────────────────────────

def reconstruct_chip(h5_path: Path, meta: H5Meta, cx: int, cy: int, t_idx: int) -> np.ndarray:
    """Rebuild a dense (chip_size, chip_size, n_bands) uint16 array for chip
    (cx, cy) at timestamp index t_idx from the HDF5's flattened, chunked
    pixel-list storage. See module docstring point (b): verify with
    --selftest before relying on this for a real run."""
    ci = meta.chunk_lookup[(cx, cy)]
    lo, hi = ci * meta.chunk_pixel_len, (ci + 1) * meta.chunk_pixel_len

    with h5py.File(h5_path, "r") as f:
        sx = f["xs_new"][lo:hi]
        sy = f["ys_new"][lo:hi]
        v = f["values"][t_idx, :, lo:hi]            # (n_bands, chunk_pixel_len)

    valid = (sx != XYS_SENTINEL) & (sy != XYS_SENTINEL)
    chip_xmin = meta.bounds_left + cx * meta.chip_size * meta.pixel_res
    chip_ymax = meta.bounds_top - cy * meta.chip_size * meta.pixel_res

    col = np.floor((sx - chip_xmin) / meta.pixel_res).astype(np.int64)
    row = np.floor((chip_ymax - sy) / meta.pixel_res).astype(np.int64)
    inb = valid & (row >= 0) & (row < meta.chip_size) & (col >= 0) & (col < meta.chip_size)

    dense = np.full((meta.chip_size, meta.chip_size, meta.n_bands), meta.nodata_val, dtype=np.uint16)
    dense[row[inb], col[inb], :] = v[:, inb].T.astype(np.uint16)

    if H5_BAND_ORDER is not None:
        order_idx = [meta.band_lookup[_normalize_band_name(b)] for b in H5_BAND_ORDER]
        dense = dense[:, :, order_idx]

    return dense


def selftest(h5_path: Path, meta: H5Meta, baseline_t_idx: int):
    """Reconstruct the best-covered chip and check it against the file's
    own bookkeeping before touching the model at all.

    Two independent checks, because they catch different failure modes:
    (1) a pixel COUNT check, confirming the chunk<->chip mapping and the
        sentinel-value filter select the right number of pixels;
    (2) a coordinate-BBOX check, confirming those pixels' real-world (sx, sy)
        coordinates actually fall inside the bbox reconstruct_chip() computes
        for this chip. (1) alone cannot catch a flipped row/col axis: an
        orientation bug (e.g. chip_y_bin increasing the opposite way to what
        chip_ymax assumes) would still pass the count check while silently
        producing vertically or horizontally mirrored chips, which would
        misregister every prediction without ever raising an error."""
    ci = int(np.argmax(meta.chip_pixel_count))
    cx, cy = int(meta.chip_x_bin[ci]), int(meta.chip_y_bin[ci])
    lo, hi = ci * meta.chunk_pixel_len, (ci + 1) * meta.chunk_pixel_len

    with h5py.File(h5_path, "r") as f:
        sx = f["xs_new"][lo:hi]
        sy = f["ys_new"][lo:hi]
    valid = (sx != XYS_SENTINEL) & (sy != XYS_SENTINEL)

    print(f"[selftest] chunk {ci} -> chip ({cx},{cy})")
    print(f"[selftest] valid xy slots counted directly : {int(valid.sum())}")
    print(f"[selftest] chip_pixel_count[{ci}] from file : {int(meta.chip_pixel_count[ci])}")
    if int(valid.sum()) != int(meta.chip_pixel_count[ci]):
        print("[selftest] MISMATCH: the chunk<->chip mapping or sentinel value "
              "assumption in reconstruct_chip() is wrong. Do not trust predictions "
              "until this matches exactly.")
    else:
        print("[selftest] OK: valid-pixel count matches the file's own record.")

    # Orientation/scale check: do the chip's own pixel coordinates actually
    # land inside the bbox reconstruct_chip() computes for it from
    # (cx, cy, bounds_left, bounds_top)? This is the check that (1) cannot do.
    chip_xmin = meta.bounds_left + cx * meta.chip_size * meta.pixel_res
    chip_xmax = chip_xmin + meta.chip_size * meta.pixel_res
    chip_ymax = meta.bounds_top - cy * meta.chip_size * meta.pixel_res
    chip_ymin = chip_ymax - meta.chip_size * meta.pixel_res
    sx_v, sy_v = sx[valid], sy[valid]
    in_bbox = (
        (sx_v >= chip_xmin) & (sx_v <= chip_xmax) & (sy_v >= chip_ymin) & (sy_v <= chip_ymax)
    )
    frac_in_bbox = in_bbox.mean() if len(sx_v) else float("nan")
    print(f"[selftest] valid pixels falling inside the computed chip bbox: "
          f"{frac_in_bbox*100:.1f}% (expect ~100%)")
    if frac_in_bbox < 0.99:
        print("[selftest] MISMATCH: pixel coordinates do not line up with the computed "
              "chip bbox. The chip_x_bin/chip_y_bin orientation or scale assumption in "
              "reconstruct_chip() is wrong (e.g. row/col axis flipped). Do not trust "
              "predictions until this is ~100%.")
    else:
        print("[selftest] OK: pixel coordinates line up with the computed chip bbox.")

    dense = reconstruct_chip(h5_path, meta, cx, cy, baseline_t_idx)
    frac_nodata = (dense[:, :, 0] == meta.nodata_val).mean()
    print(f"[selftest] reconstructed chip at baseline date: "
          f"{frac_nodata*100:.1f}% nodata pixels (band 0)")


# ───────────────────────────── main pipeline ───────────────────────────────────

def run(args):
    cfg = load_config()
    h5_path = cfg.hdf5_path
    print(f"HDF5: {h5_path}")
    meta = load_h5_meta(h5_path, nodata_val=cfg.nodata_val)
    print(f"Bands: {meta.band_names}  |  chip {meta.chip_size}px @ {meta.pixel_res}m  "
          f"|  {meta.n_timestamps} timestamps  |  {len(meta.chunk_lookup)} chips in file")

    df_scenes = classify_scenes(h5_path)
    print(df_scenes["label"].value_counts())

    baseline_row = select_baseline(df_scenes)
    baseline_date, baseline_label = baseline_row["date"], baseline_row["label"]
    # t_idx comes straight off the row (set before the date-sort in classify_scenes),
    # not from looking up df_scenes.index against this date. df_scenes is sorted and
    # reset_index(drop=True) by classify_scenes, so its positional index is the rank
    # of a scene by date, not its position along the HDF5 time axis; values[t_idx]
    # would silently read the wrong acquisition's bands whenever the file's own
    # acquisition order is not already date-sorted.
    baseline_t_idx = int(baseline_row["t_idx"])
    print(f"Baseline ('before'): {baseline_date.date()} ({baseline_label}), t_idx={baseline_t_idx}")

    if args.selftest:
        selftest(h5_path, meta, baseline_t_idx)
        return

    if H5_BAND_ORDER is None:
        raise RuntimeError(
            "H5_BAND_ORDER is not set (see module docstring, point (a)). "
            "Running inference with an unverified band order would silently "
            "feed the model misaligned channels. Confirm the order Manuel's "
            "training TIFFs used and set the constant before continuing."
        )

    from predict import predict_chips  # noqa: E402 (deferred, see import block above)

    fires = load_evaluable_fires(meta, tile_crs=cfg.tile_crs)
    print(f"Evaluable fires (>{evaluable_size_threshold_ha(meta):.0f} ha): {len(fires)}")
    if args.limit_events:
        fires = fires.iloc[: args.limit_events]

    out_dir = REPO_ROOT / "outputs" / "predictions" / cfg.tile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tile-aligned canvas for stitched predictions; 255 = not evaluated.
    width = int(round((meta.bounds_right - meta.bounds_left) / meta.pixel_res))
    height = int(round((meta.bounds_top - meta.bounds_bottom) / meta.pixel_res))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    transform = from_origin(meta.bounds_left, meta.bounds_top, meta.pixel_res, meta.pixel_res)

    rows = []
    for _, fire in fires.iterrows():
        after = select_after_scene(fire["DH_Fim"], df_scenes)
        if after is None:
            print(f"[SKIP] {fire['AreaHaSIG']:.0f} ha in {fire.get('PI_Conc', '?')} "
                  f"- no usable after scene within {AFTER_WINDOW_DAYS} days")
            rows.append({"fire_id_sgif": fire.get("Cod_SGIF"), "area_ha": fire["AreaHaSIG"],
                         "status": "skipped_no_after_scene"})
            continue

        after_t_idx = int(after["t_idx"])  # same reasoning as baseline_t_idx above
        minx, miny, maxx, maxy = fire.geometry.bounds
        cells = chip_indices_for_bounds(meta, minx, miny, maxx, maxy)
        if not cells:
            print(f"[SKIP] {fire['AreaHaSIG']:.0f} ha - bounding box matched no stored chips")
            rows.append({"fire_id_sgif": fire.get("Cod_SGIF"), "area_ha": fire["AreaHaSIG"],
                         "status": "skipped_no_chip_overlap"})
            continue

        before_batch, after_batch, positions = [], [], []
        for cx, cy in cells:
            before_batch.append(reconstruct_chip(h5_path, meta, cx, cy, baseline_t_idx))
            after_batch.append(reconstruct_chip(h5_path, meta, cx, cy, after_t_idx))
            positions.append((cx, cy))

        # predict_chips() postprocesses each chip independently (morphological
        # closing + small-component removal, see predict.py). A burn scar that
        # straddles two chips can therefore end up split into two components
        # that are each individually small enough to be removed near the seam,
        # even though the scar is contiguous on the ground. Worth flagging in
        # the write-up for the larger ICNF events, which are the ones most
        # likely to span more than one chip.
        preds = predict_chips(before_batch, after_batch,
                               REPO_ROOT / "models/efficienT_b2_2classes/best_model.pth")

        burned_px = 0
        for (cx, cy), pred in zip(positions, preds):
            r0, c0 = cy * meta.chip_size, cx * meta.chip_size
            # Defensive: cells came from meta.chunk_lookup so this should never
            # trigger, but a negative r0/c0 would otherwise wrap around to the
            # far edge of canvas under numpy slicing instead of raising, and
            # silently corrupt an unrelated part of the tile.
            if r0 < 0 or c0 < 0 or r0 >= height or c0 >= width:
                print(f"[WARN] chip ({cx},{cy}) maps outside the canvas (r0={r0}, c0={c0}); skipping")
                continue
            r1 = min(r0 + meta.chip_size, height)
            c1 = min(c0 + meta.chip_size, width)
            canvas[r0:r1, c0:c1] = pred[: r1 - r0, : c1 - c0]
            burned_px += int((pred == 1).sum())

        burned_ha = burned_px * (meta.pixel_res ** 2) / 1e4
        print(f"[OK] {fire['AreaHaSIG']:.0f} ha ICNF in {fire.get('PI_Conc', '?')} "
              f"-> after {after['date'].date()} ({after['label']}), "
              f"{len(cells)} chips, predicted {burned_ha:.1f} ha burned")

        rows.append({
            "fire_id_sgif": fire.get("Cod_SGIF"), "municipality": fire.get("PI_Conc"),
            "area_ha_icnf": fire["AreaHaSIG"], "before_date": baseline_date.date(),
            "before_label": baseline_label, "after_date": after["date"].date(),
            "after_label": after["label"], "n_chips": len(cells),
            "predicted_burned_ha": burned_ha, "status": "ok",
        })

    out_tif = out_dir / "efficientnet_b2_predictions.tif"
    with rasterio.open(
        out_tif, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs=cfg.tile_crs, transform=transform, nodata=255, compress="lzw",
    ) as dst:
        dst.write(canvas, 1)
    print(f"Wrote {out_tif}")

    summary_csv = out_dir / "efficientnet_b2_event_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true",
                    help="Verify chip reconstruction against the file's own bookkeeping, then exit.")
    p.add_argument("--limit-events", type=int, default=None,
                    help="Only process the first N evaluable fires (for a quick test run).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
