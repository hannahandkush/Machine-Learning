"""Local burned-area viewer for tile T29TPG.

Pick a model, scene pair, overlap and voting strictness, and see the burned-area
map on an interactive basemap. Cache-first: if a run for the chosen configuration
already exists in outputs/predictions/, it is loaded instantly and the strictness
slider re-thresholds it live; otherwise you can launch the run from the app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import MiniMap
from folium.raster_layers import ImageOverlay
from branca.element import MacroElement
from jinja2 import Template
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from utils.config import load_config

cfg = load_config()
PRED = cfg.predictions_dir
ICNF = REPO / "data/shapefiles/ground_truth_ICNF/ardida_2025.shp"
PT_BOUND = REPO / "data/shapefiles/boundary_files/portugal_continental_32629.gpkg"
PIX_HA = 0.01

MODELS = {"EfficientNet-B2": "efficientnet_b2", "Swin-YNet": "swin_ynet"}
EFF_WEIGHTS = REPO / "models/efficienT_b2_2classes/best_model.pth"
EFF_PKG = REPO / "models/efficienT_b2_2classes"
OVERLAPS = [0, 25, 50, 75]
USABLE_DATES = [
    "2025-01-18", "2025-03-29", "2025-04-23", "2025-04-28", "2025-05-23",
    "2025-05-28", "2025-06-07", "2025-06-17", "2025-06-22", "2025-06-27",
    "2025-07-02", "2025-07-07", "2025-07-22", "2025-07-24", "2025-07-27",
    "2025-08-01", "2025-08-11", "2025-08-16", "2025-08-21", "2025-08-23",
    "2025-08-26", "2025-09-05", "2025-09-15", "2025-09-25", "2025-09-30",
    "2025-10-02", "2025-10-05", "2025-10-10", "2025-10-15",
]


# ---------- data helpers (cached) ----------
@st.cache_data(show_spinner=False)
def native_votes(path_str: str) -> np.ndarray:
    """Full-resolution vote-fraction raster (uint8 0..100, 255 = not observed)."""
    with rasterio.open(path_str) as src:
        return src.read(1)


@st.cache_data(show_spinner="Reprojecting for display...")
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


class CenterButton(MacroElement):
    """A Leaflet control button that fits the map to `bounds` ([[s, w], [n, e]])
    when clicked, used to recenter on the detected burned areas."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var bounds = {{ this.bounds }};
            var ctrl = L.control({position: 'topleft'});
            ctrl.onAdd = function(map) {
                var div = L.DomUtil.create('div', 'leaflet-bar');
                var a = L.DomUtil.create('a', '', div);
                a.innerHTML = '\\uD83D\\uDD25';
                a.title = 'Center on detected burned areas';
                a.href = '#';
                a.style.cssText = 'width:26px;height:26px;line-height:26px;text-align:center;font-size:15px;';
                L.DomEvent.on(a, 'click', function(e) {
                    L.DomEvent.stop(e);
                    {{ this._parent.get_name() }}.fitBounds(bounds);
                });
                return div;
            };
            ctrl.addTo({{ this._parent.get_name() }});
        })();
        {% endmacro %}
    """)

    def __init__(self, bounds):
        super().__init__()
        self._name = "CenterButton"
        # coerce to plain Python floats: numpy 2.x repr (np.float64(...)) is not
        # valid JavaScript and would break every script added after this one.
        (s, w), (n, e) = bounds
        self.bounds = [[float(s), float(w)], [float(n), float(e)]]


# ---------- UI ----------
st.set_page_config(page_title="Burned-area viewer — T29TPG", layout="wide")

# tighten the page so the map fits without scrolling, and make the progress bar green
st.markdown(
    "<style>"
    ".block-container {padding-top: 4rem; padding-bottom: 0rem;}"
    ".stProgress > div > div > div > div,"
    "[data-testid='stProgressBar'] > div > div,"
    "div[role='progressbar'] > div {background-color: #21a366 !important;}"
    "</style>",
    unsafe_allow_html=True,
)
st.markdown("##### Burned-area model viewer — Sentinel-2 tile T29TPG")

sb = st.sidebar
sb.header("Configuration")
model = MODELS[sb.selectbox("Model", list(MODELS))]
overlap = sb.selectbox("Window overlap (%)", OVERLAPS, index=2)
before = sb.selectbox("Before date", USABLE_DATES, index=USABLE_DATES.index("2025-07-07"))
after = sb.selectbox("After date", USABLE_DATES, index=USABLE_DATES.index("2025-10-15"))
strictness = sb.slider("Voting strictness (% of windows)", 50, 100, 75, 5,
                       help="Re-thresholds the existing run live; never triggers a new run.")
opacity = sb.slider("Overlay opacity", 0.0, 1.0, 0.75, 0.05)
show_gt = sb.checkbox("Show ICNF ground truth", value=False)

if model == "efficientnet_b2":
    sb.caption("EfficientNet band order is provisional (pending confirmation with Manuel).")

B, A = before.replace("-", ""), after.replace("-", "")
votes_path = PRED / f"T29TPG_{model}_ov{overlap:02d}_{B}_{A}_votes.tif"

if votes_path.exists():
    v_native = native_votes(str(votes_path))
    burned_ha = int(((v_native != 255) & (v_native >= strictness)).sum()) * PIX_HA
    arr, (s, w, n, e) = votes_4326(str(votes_path))

    st.markdown(f"**{burned_ha:,.0f} ha** burned &nbsp;·&nbsp; vote ≥ {strictness}% "
                f"&nbsp;·&nbsp; {before} → {after}")

    fmap = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=9,
                      tiles="CartoDB positron")
    ImageOverlay(burned_rgba(arr, strictness), bounds=[[s, w], [n, e]],
                 opacity=opacity, name="Burned").add_to(fmap)
    bm = (arr != 255) & (arr >= strictness)        # burned pixels in the display array
    if bm.any():
        H_, W_ = arr.shape
        rr = np.where(bm.any(axis=1))[0]; cc = np.where(bm.any(axis=0))[0]
        bn = n - rr[0] / H_ * (n - s); bs = n - (rr[-1] + 1) / H_ * (n - s)
        bw = w + cc[0] / W_ * (e - w); be = w + (cc[-1] + 1) / W_ * (e - w)
        CenterButton([[bs, bw], [bn, be]]).add_to(fmap)
    folium.GeoJson(portugal_boundary((w, s, e, n)), name="Portugal boundary",
                   style_function=lambda _: {"color": "#444444", "weight": 1.5,
                                             "fillOpacity": 0.0}).add_to(fmap)
    n_gt = None
    if show_gt:
        gj, n_gt = ground_truth(before, after, (w, s, e, n))
        folium.GeoJson(gj, name="ICNF ground truth",
                       style_function=lambda _: {"color": "#2c7bb6", "weight": 1,
                                                 "fillOpacity": 0.0}).add_to(fmap)
    MiniMap(tile_layer="OpenStreetMap", position="bottomright", width=190, height=140,
            zoom_level_offset=-5, toggle_display=True).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    components.html(fmap._repr_html_(), height=580)
    cap = ("Map is shown at reduced resolution; the burned-area number is from the "
           "full-resolution raster.")
    if n_gt is not None:
        cap = (f"Ground truth: {n_gt} ICNF fire events active within {before} to "
               f"{after}. ") + cap
    st.caption(cap)
else:
    st.warning(f"No saved run for {model} at {overlap}% overlap, {before} → {after}.")
    st.write("Running the model produces this configuration (minutes, depending on overlap). "
             "The voting strictness does not need a run, it is applied to an existing one.")
    if st.button("Run this configuration now"):
        tag = f"T29TPG_{model}_ov{overlap:02d}_{B}_{A}"
        prog_file = Path(tempfile.gettempdir()) / f"prog_{tag}.json"
        log_file = Path(tempfile.gettempdir()) / f"run_{tag}.log"
        prog_file.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "inference.run_overlap",
               "--overlap", str(overlap / 100), "--device", "cpu",
               "--before-date", before, "--after-date", after,
               "--model-kind", model, "--progress-file", str(prog_file)]
        if model == "efficientnet_b2":
            cmd += ["--weights", str(EFF_WEIGHTS), "--package-dir", str(EFF_PKG)]

        bar = st.progress(0.0, text="Starting the run...")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=lf, text=True)
            while proc.poll() is None:
                try:
                    d = json.loads(prog_file.read_text())
                    total = d.get("total", 0)
                    if total > 0:
                        frac = min(d["done"] / total, 1.0)
                        bar.progress(frac, text=f"{d['phase']}: batch {d['done']}/{total} ({frac*100:.0f}%)")
                    else:
                        bar.progress(0.0, text=d.get("phase", "Working..."))
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    pass
                time.sleep(1.0)
        if proc.returncode == 0:
            bar.progress(1.0, text="Done")
            st.success("Run complete. Move any control to view the new map.")
        else:
            st.error("Run failed.")
            st.code(log_file.read_text()[-2000:] if log_file.exists() else "")
