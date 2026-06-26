"""Page 2 of the burned-area viewer — launch a new model/overlap/date-pair run.

Picks a configuration (model, window overlap, before/after scene pair). If that
exact configuration already has a finished run sitting in outputs/predictions/,
re-running is skipped — the underlying model is frozen, not fine-tuned per run,
so the same inputs always produce the same prediction — and the existing result
is shown directly below instead. Otherwise a button launches
`inference.run_overlap` as a subprocess, streaming a progress bar from the
runner's progress file, and once it finishes the result is shown the same way.

This mirrors the original single-tab "Map viewer" (see README): config picker,
voting-strictness slider, and the resulting map all live together in one place,
so a finished run becomes visible immediately rather than requiring a hop to
Page 1. Page 1 ("View a processed output") stays the no-slider browse/compare
view across every run, exactly as the original "Processed outputs" tab was.

Page config, global CSS, and the app-wide title are set once by app.py (the
entry point that hands off to this page via st.navigation), not here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import folium
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import MiniMap
from folium.raster_layers import ImageOverlay

from viewer_common import (
    EFF_PKG, EFF_WEIGHTS, MODEL_LABELS, MODELS, REPO, USABLE_DATES,
    _date_label, burned_rgba, portugal_boundary, run_paths, votes_4326,
    votes_burned_area_ha,
)


def _render_run_map(paths: dict) -> None:
    """Voting-strictness slider + live-rethresholded map for one run's votes
    raster. No model re-run: moving the slider only re-thresholds the already
    written `_votes.tif` (see burned_rgba / votes_burned_area_ha)."""
    thr = st.sidebar.slider(
        "Voting strictness (% of overlapping windows that must agree)", 0, 100, 50, 5,
        key=f"strictness_{paths['tag']}",
        help="Re-thresholds this run's vote-fraction raster instantly — no model "
             "run. Higher = fewer false positives, at some cost to recall (see "
             "README headline results).")
    op = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.75, 0.05, key=f"opacity_{paths['tag']}")

    st.metric("Burned area at this strictness", f"{votes_burned_area_ha(str(paths['votes']), thr):,.0f} ha")

    votes, (s, w, n, e) = votes_4326(str(paths["votes"]))
    rgba = burned_rgba(votes, thr)
    fmap = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=9, tiles="CartoDB positron")
    ImageOverlay(rgba, bounds=[[s, w], [n, e]], opacity=op, name="Burned").add_to(fmap)
    folium.GeoJson(portugal_boundary((w, s, e, n)), name="Portugal boundary",
                   style_function=lambda _: {"color": "#444444", "weight": 1.5,
                                             "fillOpacity": 0.0}).add_to(fmap)
    MiniMap(tile_layer="OpenStreetMap", position="bottomright", width=190, height=140,
            zoom_level_offset=-5, toggle_display=True).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    components.html(fmap._repr_html_(), height=520)
    st.page_link("pages/1_View_a_processed_output.py",
                 label="Go to Browse Results for the error map and focal-zone inspector",
                 icon="📊")


st.markdown("##### Run a Model")
st.caption("Pick a model, window overlap, and before/after scene pair in the sidebar. "
           "If this exact configuration hasn't been run yet, a button below launches "
           "it; either way, the result map is shown below and the voting-strictness "
           "slider lives in the sidebar.")

st.sidebar.markdown("##### Controls")
use_swin = st.sidebar.toggle("Use Swin-YNet", value=False,
                     help="Off = EfficientNet-B2 (better precision at every overlap "
                          "tested, see README headline results). On = Swin-YNet.")
model = MODELS["Swin-YNet" if use_swin else "EfficientNet-B2"]
st.sidebar.caption(f"Model: **{'Swin-YNet' if use_swin else 'EfficientNet-B2'}**")

overlap = st.sidebar.slider(
    "Window overlap (%)", 0, 75, 50, 5,
    help="Sliding-window overlap fraction passed to inference.run_overlap "
         "(0–75%, the range its argparse accepts). Higher overlap means more "
         "windows per pixel — slower, but a stronger majority-vote filter "
         "against scattered false positives.")

before = st.sidebar.selectbox("Before date", USABLE_DATES, index=USABLE_DATES.index("2025-07-07"),
                              format_func=_date_label)
after = st.sidebar.selectbox("After date", USABLE_DATES, index=USABLE_DATES.index("2025-10-15"),
                             format_func=_date_label)

paths = run_paths(model, overlap, before, after)
have_run = paths["votes"].exists()

if have_run:
    st.info(
        f"A run for **{MODEL_LABELS.get(model, model)}**, {overlap}% overlap, {before} → {after} "
        f"already exists in `outputs/predictions/` — shown below. Re-running would spend "
        f"several minutes reproducing an identical result, since neither model is "
        f"fine-tuned per run."
    )
else:
    st.write("Running the model produces this configuration (minutes, depending on overlap).")
    if st.button("Run this configuration now"):
        tag = paths["tag"]
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
            st.success("Run complete — rendering the result below.")
            have_run = True
        else:
            st.error("Run failed.")
            st.code(log_file.read_text()[-2000:] if log_file.exists() else "")

if have_run:
    st.markdown("---")
    _render_run_map(paths)
