"""Shared config, cached data helpers, and the focal-zone inspector for the
app.py / pages/2_Run_New_Configuration.py multipage app.

Everything here is imported by both pages so a prediction raster is read and
reprojected at most once per session (st.cache_data is keyed per function, not
per page, so splitting the UI into multiple files does not duplicate this
work or fragment the cache). Page-specific UI flow (widgets, st.tabs/radio
selection, the run-launch button) stays in the page files; only logic that is
genuinely shared lives here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components
from rasterio.features import rasterize
from rasterio.transform import array_bounds, rowcol
from rasterio.warp import (Resampling, calculate_default_transform, reproject,
                           transform as warp_transform_points, transform_bounds)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.config import load_config
from utils.sentinel_hub import fetch_truecolor, CDSEAuthError

cfg = load_config()
PRED = cfg.predictions_dir
ICNF = REPO / "data/shapefiles/ground_truth_ICNF/ardida_2025.shp"
PT_BOUND = REPO / "data/shapefiles/boundary_files/portugal_continental_32629.gpkg"
PIX_HA = 0.01

MODELS = {"EfficientNet-B2": "efficientnet_b2", "Swin-YNet": "swin_ynet"}
MODEL_LABELS = {v: k for k, v in MODELS.items()}
EFF_WEIGHTS = REPO / "models/efficienT_b2_2classes/best_model.pth"
EFF_PKG = REPO / "models/efficienT_b2_2classes"
OVERLAPS = [0, 25, 50, 75]
GT_RASTER = REPO / "data/processed/icnf_burned_labels_t29tpg_2025.tif"

# TP/FP/FN error-map encoding and display colours (TN and unobserved stay
# transparent: only disagreements with ground truth, and correct detections,
# are drawn). Colours follow a colourblind-safe qualitative palette.
ERR_TN, ERR_TP, ERR_FP, ERR_FN, ERR_NODATA = 0, 1, 2, 3, 255
ERROR_STYLE = {
    ERR_TP: ("True positive", (117, 107, 177, 255)),
    ERR_FP: ("False positive", (227, 74, 51, 255)),
    ERR_FN: ("False negative", (49, 130, 189, 255)),
}
# Usable scenes (full_clear + partial_clear) with cloud cover %, from the cube's
# scene screening. Each entry is (date, cloud %, coverage); partial scenes carry
# more cloud and are flagged in the date dropdowns.
SCENES = [
    ("2025-01-13", 15, "partial"), ("2025-01-18", 3, "full"),
    ("2025-03-29", 0, "full"),     ("2025-04-23", 0, "full"),
    ("2025-04-28", 1, "full"),     ("2025-05-23", 0, "full"),
    ("2025-05-28", 3, "full"),     ("2025-06-02", 14, "partial"),
    ("2025-06-07", 2, "full"),     ("2025-06-17", 0, "full"),
    ("2025-06-22", 0, "full"),     ("2025-06-27", 0, "full"),
    ("2025-07-02", 1, "full"),     ("2025-07-07", 1, "full"),
    ("2025-07-14", 14, "partial"), ("2025-07-22", 7, "full"),
    ("2025-07-24", 0, "full"),     ("2025-07-27", 0, "full"),
    ("2025-08-01", 0, "full"),     ("2025-08-11", 10, "full"),
    ("2025-08-16", 6, "full"),     ("2025-08-21", 10, "full"),
    ("2025-08-23", 0, "full"),     ("2025-08-26", 1, "full"),
    ("2025-09-05", 0, "full"),     ("2025-09-15", 6, "full"),
    ("2025-09-25", 0, "full"),     ("2025-09-30", 0, "full"),
    ("2025-10-02", 0, "full"),     ("2025-10-05", 0, "full"),
    ("2025-10-10", 0, "full"),     ("2025-10-12", 17, "partial"),
    ("2025-10-15", 1, "full"),     ("2025-11-21", 16, "partial"),
    ("2025-12-29", 18, "partial"),
]
SCENE_INFO = {d: (c, cov) for d, c, cov in SCENES}
USABLE_DATES = [d for d, _, _ in SCENES]


def _date_label(d):
    """Dropdown label: the date plus its cloud cover, flagging partial coverage."""
    cloud, cov = SCENE_INFO[d]
    tag = "" if cov == "full" else ", partial"
    return f"{d}  ({cloud:.0f}% cloud{tag})"


def run_paths(model: str, overlap: int, before: str, after: str) -> dict:
    """The filenames a run of this exact configuration would produce in
    outputs/predictions/, keyed by kind. `inference.run_overlap` writes all of
    these together on success, so checking any one of them for existence is
    equivalent; `burned` is used because it's the file Page 1's run browser
    also globs for, so "this configuration already exists" means the same
    thing on both pages."""
    B, A = before.replace("-", ""), after.replace("-", "")
    tag = f"T29TPG_{model}_ov{overlap:02d}_{B}_{A}"
    return {
        "tag": tag,
        "votes": PRED / f"{tag}_votes.tif",
        "burned": PRED / f"{tag}_burned.tif",
        "observed": PRED / f"{tag}_observed.tif",
        "manifest": PRED / f"{tag}_manifest.json",
    }


# ---------- data helpers (cached so each raster is read and reprojected only once) ----------
# These load the prediction rasters and reference vectors and turn them into the
# lat/lon overlays the map draws. Shared by both pages.
@st.cache_data(show_spinner=False, max_entries=3)
def native_votes(path_str: str) -> np.ndarray:
    """Full-resolution vote-fraction raster (uint8 0..100, 255 = not observed)."""
    with rasterio.open(path_str) as src:
        return src.read(1)


@st.cache_data(show_spinner="Reprojecting for display...", max_entries=4)
def votes_4326(path_str: str, max_w: int = 1600):
    """Reproject the vote raster to lat/lon (downsampled). Returns (array, (s,w,n,e))."""
    with rasterio.open(path_str) as src:
        t0, w0, h0 = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        dst_w = min(max_w, w0)
        dst_h = int(h0 * dst_w / w0)
        t, w_, h_ = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds,
            dst_width=dst_w, dst_height=dst_h)
        dst = np.full((h_, w_), 255, np.uint8)
        reproject(rasterio.band(src, 1), dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=t, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, dst_nodata=255)
    left, bottom, right, top = array_bounds(h_, w_, t)
    return dst, (bottom, left, top, right)


@st.cache_data(show_spinner=False, max_entries=8)
def ground_truth(before: str, after: str, bbox: tuple):
    """ICNF fire events active within the [before, after] window, clipped to the
    tile, as lat/lon GeoJSON plus a count.

    A fire counts only if it both started on or after the before date AND ended on
    or before the after date, so the overlay shows only fires active within the
    selected period (the same containment rule the notebooks use for scoring)."""
    w, s, e, n = bbox
    f = gpd.read_file(ICNF).to_crs("EPSG:4326").cx[w:e, s:n]
    start = pd.to_datetime(f["DH_Inicio"], errors="coerce")
    end = pd.to_datetime(f["DH_Fim"], errors="coerce").fillna(start)
    bt, at = pd.Timestamp(before), pd.Timestamp(after)
    sel = f[start.notna() & (start >= bt) & (end <= at)]
    # keep geometry only: the date columns are pandas Timestamps, which folium
    # cannot JSON-serialize when it renders the layer.
    return sel[[sel.geometry.name]].__geo_interface__, int(len(sel))


@st.cache_data(show_spinner=False, max_entries=8)
def portugal_boundary(bbox: tuple):
    """Continental Portugal outline near the tile, as lat/lon GeoJSON."""
    w, s, e, n = bbox
    g = gpd.read_file(PT_BOUND).to_crs("EPSG:4326").cx[w:e, s:n]
    return g[[g.geometry.name]].__geo_interface__


def burned_rgba(votes: np.ndarray, thr: int) -> np.ndarray:
    """Red where the vote fraction meets the strictness, transparent elsewhere."""
    rgba = np.zeros(votes.shape + (4,), np.uint8)
    rgba[(votes != 255) & (votes >= thr)] = (215, 25, 28, 255)
    return rgba


@st.cache_data(show_spinner=False, max_entries=3)
def inside_portugal(ref_path: str) -> np.ndarray:
    """Boolean mask, True where a pixel of `ref_path`'s grid falls inside
    continental Portugal. Tile T29TPG extends north into Spain, which ICNF
    (Portugal-only) never maps, so a fire detected there has no ground truth to
    be scored against and must not be counted as a false positive (or drawn on
    the error map) the way it would be if the gt raster's all-zero-by-default
    Spain side were treated as confirmed "not burned". Mirrors the inside_pt
    construction in notebooks/min_fire_size.ipynb, overlap_experiment.ipynb,
    burned_area_date_comparison.ipynb, and false_positive_review.ipynb."""
    with rasterio.open(ref_path) as r:
        transform, shape, crs = r.transform, r.shape, r.crs
    pt = gpd.read_file(PT_BOUND).to_crs(crs)
    return rasterize([(g, 1) for g in pt.geometry], out_shape=shape,
                      transform=transform, fill=0, dtype=np.uint8).astype(bool)


@st.cache_data(show_spinner=False, max_entries=3)
def ground_truth_raster(before: str, after: str, ref_path: str) -> np.ndarray:
    """Date-windowed ICNF ground truth, rasterised onto the grid of `ref_path`.

    Mirrors the construction of data/processed/icnf_burned_labels_t29tpg_2025.tif
    (1-pixel/10 m interior erosion before rasterisation, see its README and
    evaluation_protocol.md S3.2), but recomputed per before/after window so the
    error map stays correct for any date pair, not only the full-season run the
    static raster was built for. Returns a uint8 array (1 = burned, 0 = not)
    on the same grid as `ref_path`."""
    with rasterio.open(ref_path) as r:
        transform, shape, crs = r.transform, r.shape, r.crs
    f = gpd.read_file(ICNF).to_crs(crs)
    start = pd.to_datetime(f["DH_Inicio"], errors="coerce")
    end = pd.to_datetime(f["DH_Fim"], errors="coerce").fillna(start)
    bt, at = pd.Timestamp(before), pd.Timestamp(after)
    sel = f[start.notna() & (start >= bt) & (end <= at)]
    if sel.empty:
        return np.zeros(shape, dtype=np.uint8)
    eroded = sel.geometry.buffer(-10)
    eroded = eroded[~eroded.is_empty]
    if eroded.empty:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize([(geom, 1) for geom in eroded], out_shape=shape,
                      transform=transform, fill=0, dtype=np.uint8)


@st.cache_data(show_spinner="Computing error map...", max_entries=3)
def error_map_native(pred_kind: str, path_str: str, before: str, after: str,
                      strictness: int | None):
    """Pixel-level TP/FP/FN classification against date-windowed ICNF ground
    truth, restricted to pixels observed in both scenes and lying inside
    continental Portugal (mirrors the corrected_metrics() logic in
    notebooks/Model_comparison.ipynb, §7). The Spain-side portion of tile
    T29TPG has no ICNF ground truth, so it is excluded rather than scored as
    "not burned" by default.

    pred_kind="votes": `path_str` is a _votes.tif, re-thresholded live by
    `strictness` (no separate observed mask needed; 255 already means
    unobserved). pred_kind="burned": `path_str` is a fixed _burned.tif, scored
    against its sibling _observed.tif at whatever threshold produced it.

    Returns (category array uint8, transform, crs) at native resolution, where
    category is one of ERR_TN/ERR_TP/ERR_FP/ERR_FN/ERR_NODATA."""
    with rasterio.open(path_str) as r:
        raw = r.read(1)
        transform, crs = r.transform, r.crs
    if pred_kind == "votes":
        obs = raw != 255
        pred = obs & (raw >= strictness)
    else:
        obs_path = path_str.replace("_burned.tif", "_observed.tif")
        with rasterio.open(obs_path) as r:
            obs = r.read(1) == 1
        pred = raw == 1

    obs = obs & inside_portugal(path_str)
    gt = ground_truth_raster(before, after, path_str) == 1
    cat = np.full(raw.shape, ERR_NODATA, dtype=np.uint8)
    cat[obs & ~pred & ~gt] = ERR_TN
    cat[obs & pred & gt] = ERR_TP
    cat[obs & pred & ~gt] = ERR_FP
    cat[obs & ~pred & gt] = ERR_FN
    return cat, transform, crs


def _reproject_to_4326(arr: np.ndarray, src_transform, src_crs, nodata,
                        max_w: int = 1600):
    """Same downsample-then-reproject recipe as votes_4326, generalised to an
    in-memory array (so it also works for the computed error-map categories,
    which have no file of their own)."""
    h0, w0 = arr.shape
    bounds0 = array_bounds(h0, w0, src_transform)
    t0, w0_, h0_ = calculate_default_transform(src_crs, "EPSG:4326", w0, h0, *bounds0)
    dst_w = min(max_w, w0_)
    dst_h = int(h0_ * dst_w / w0_)
    t, w_, h_ = calculate_default_transform(src_crs, "EPSG:4326", w0, h0, *bounds0,
                                             dst_width=dst_w, dst_height=dst_h)
    dst = np.full((h_, w_), nodata, dtype=arr.dtype)
    reproject(arr, dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=t, dst_crs="EPSG:4326",
              resampling=Resampling.nearest, dst_nodata=nodata)
    left, bottom, right, top = array_bounds(h_, w_, t)
    return dst, (bottom, left, top, right)


def error_rgba(cat: np.ndarray) -> np.ndarray:
    """TP/FP/FN in their display colours; TN and unobserved stay transparent."""
    rgba = np.zeros(cat.shape + (4,), np.uint8)
    for code, (_, color) in ERROR_STYLE.items():
        rgba[cat == code] = color
    return rgba


def error_counts_ha(cat: np.ndarray) -> dict:
    """TP/FP/FN areas in hectares, computed at native resolution (i.e. before
    any display downsampling) so the numbers shown alongside the map stay exact."""
    return {label: int((cat == code).sum()) * PIX_HA for code, (label, _) in ERROR_STYLE.items()}


def error_legend_html() -> str:
    chips = "".join(
        f'<span style="display:inline-block;width:11px;height:11px;'
        f'background:rgb({c[0]},{c[1]},{c[2]});margin:0 4px 0 12px;'
        f'border-radius:2px;vertical-align:middle;"></span>{label}'
        for label, c in ERROR_STYLE.values()
    )
    return f'<span style="font-size:0.9rem;">Error map:{chips}</span>'


@st.cache_data(show_spinner=False)
def burned_area_ha(path_str: str) -> float:
    """Total burned area (majority-vote burned map) in hectares."""
    with rasterio.open(path_str) as r:
        return int((r.read(1) == 1).sum()) * PIX_HA


@st.cache_data(show_spinner=False)
def burned_thumb(path_str: str, k: int = 24):
    """Small red-on-grey thumbnail of a burned map, block-downsampled."""
    with rasterio.open(path_str) as r:
        m = r.read(1) == 1
    H, W = (m.shape[0] // k) * k, (m.shape[1] // k) * k
    red = m[:H, :W].reshape(H // k, k, W // k, k).max(axis=(1, 3))
    rgba = np.empty(red.shape + (4,), np.uint8)
    rgba[...] = (245, 245, 245, 255)
    rgba[red] = (215, 25, 28, 255)
    return rgba


# ---------- focal-zone inspector: click a point, get a 4-way verification chip ----------
# Same idea as notebooks/false_positive_review.ipynb Section 5 (Sentinel-2 before/after
# against a flagged cluster), but driven by an arbitrary click instead of a pre-ranked
# cluster, and with an OpenStreetMap panel + the error map crop added alongside.
def point_to_bbox(lat: float, lon: float, pad: float = 1500.0,
                   dst_crs: str = "EPSG:32629") -> tuple:
    """A (left, bottom, right, top) bbox in `dst_crs`, `pad` metres either side of
    a clicked lat/lon point. Same bbox convention as crop_window() and
    fetch_truecolor() in the notebook, so the same bbox drives every panel."""
    xs, ys = warp_transform_points("EPSG:4326", dst_crs, [lon], [lat])
    x, y = xs[0], ys[0]
    return (x - pad, y - pad, x + pad, y + pad)


def crop_rowcol(bbox: tuple, transform_, shape: tuple) -> tuple:
    """Row/col window into a raster of `shape` covering `bbox`, clipped to the
    raster's extent. Mirrors crop_window() in false_positive_review.ipynb."""
    left, bottom, right, top_ = bbox
    r0, c0 = rowcol(transform_, left, top_)
    r1, c1 = rowcol(transform_, right, bottom)
    r0, r1 = max(0, min(r0, r1)), min(shape[0], max(r0, r1))
    c0, c1 = max(0, min(c0, c1)), min(shape[1], max(c0, c1))
    return r0, r1, c0, c1


def render_focal_chip(lat: float, lon: float, run: dict, bbox: tuple) -> None:
    """Below a clicked point, a ~3 km focal chip shown four ways, side by side:
    Sentinel-2 before/after true colour, OpenStreetMap (with the chip footprint
    drawn for cross-reference), and the selected run's error map crop."""
    # Fixed pixel size for every panel below. st.image() is column-fluid and
    # preserves the source array's aspect ratio (square in, square out), but
    # components.html()'s iframe has a height fixed in Python at render time
    # that does NOT track the column's actual rendered width — wide screens
    # give a tall column but the same fixed iframe height, so the OSM panel
    # reads as a flat rectangle next to genuinely square before/after chips.
    # Pinning all four panels to the same explicit width=height square is the
    # one sizing model that's guaranteed consistent across screen sizes.
    CHIP_PX = 280
    cols = st.columns(4)
    for col, label, date in zip(cols[:2], ("Before", "After"), (run["before"], run["after"])):
        with col:
            st.caption(f"{label} ({date} ± 5 d)")
            try:
                st.image(fetch_truecolor(bbox, date, bbox_crs="EPSG:32629"), width=CHIP_PX)
            except CDSEAuthError:
                st.info("Needs CDSE credentials — see utils/sentinel_hub.py.")
            except Exception as e:  # noqa: BLE001 - network/CDSE outages must not crash the page
                st.info(f"Could not reach CDSE ({type(e).__name__}). Try again shortly.")

    with cols[2]:
        st.caption("OpenStreetMap")
        osm = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap",
                         zoom_control=False)
        folium.Marker([lat, lon]).add_to(osm)
        lo0, la0, lo1, la1 = transform_bounds("EPSG:32629", "EPSG:4326", *bbox)
        folium.Rectangle([[la0, lo0], [la1, lo1]], color="#e34a33", weight=2,
                         fill=False).add_to(osm)
        components.html(osm._repr_html_(), height=CHIP_PX, width=CHIP_PX)

    with cols[3]:
        st.caption("Error map (TP / FP / FN)")
        obs_path = run["path"].replace("_burned.tif", "_observed.tif")
        if not Path(obs_path).exists():
            st.info("No sidecar _observed.tif for this run; cannot score it.")
        else:
            cat, transform_, _ = error_map_native("burned", run["path"], run["before"],
                                                   run["after"], None)
            r0, r1, c0, c1 = crop_rowcol(bbox, transform_, cat.shape)
            crop = cat[r0:r1, c0:c1]
            if crop.size == 0:
                st.info("Click point falls outside this run's tile.")
            else:
                st.image(error_rgba(crop), width=CHIP_PX)
    st.markdown(error_legend_html(), unsafe_allow_html=True)
