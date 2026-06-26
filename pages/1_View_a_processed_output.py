"""Page 1 of the burned-area viewer for tile T29TPG — browse processed outputs.

Browse every model/overlap/date-pair run that already exists in
outputs/predictions/, switch between the "Burned area" and "Error map
(TP/FP/FN)" overlays, and use the focal-zone inspector to spot-check
individual points against Sentinel-2 imagery and OpenStreetMap.

To launch a new configuration that doesn't exist yet, use the "Run new
configuration" page in the sidebar. Page config, global CSS, and the app-wide
title are set once by app.py (the entry point that hands off to this page via
st.navigation), not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import MiniMap
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds

from utils.viewer_common import (
    ERR_NODATA, MODEL_LABELS, PIX_HA, PRED,
    _reproject_to_4326, burned_area_ha, burned_rgba, error_counts_ha,
    error_legend_html, error_map_native, error_rgba, ground_truth,
    portugal_boundary, point_to_bbox, render_focal_chip, votes_4326,
    votes_burned_area_ha,
)

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:  # pragma: no cover - degrades gracefully, see environment.yml
    HAS_ST_FOLIUM = False

st.markdown("##### View a processed output")

# each finished run leaves a *_burned.tif plus a manifest with its timing; gather them
runs = []
for bp in sorted(PRED.glob("T29TPG_*_burned.tif")):
    parts = bp.name[:-len("_burned.tif")].split("_")
    if len(parts) < 5 or not parts[-3].startswith("ov"):
        continue
    mk = "_".join(parts[1:-3]); ov = int(parts[-3][2:])
    b8, a8 = parts[-2], parts[-1]
    bd = f"{b8[:4]}-{b8[4:6]}-{b8[6:]}"; ad = f"{a8[:4]}-{a8[4:6]}-{a8[6:]}"
    mp = bp.with_name(bp.name.replace("_burned.tif", "_manifest.json"))
    cost = json.loads(mp.read_text()) if mp.exists() else {}
    runs.append({
        "model": MODEL_LABELS.get(mk, mk), "overlap %": ov, "before": bd, "after": ad,
        "burned ha": round(burned_area_ha(str(bp))),
        "time min": round(cost["elapsed_s"] / 60, 1) if cost.get("elapsed_s") else None,
        "peak GB": cost.get("peak_rss_gb"), "windows": cost.get("n_windows"),
        "path": str(bp), "votes_path": str(bp).replace("_burned.tif", "_votes.tif"),
    })
if not runs:
    st.info("No generated outputs in outputs/predictions/ yet. Use the "
            "**Run new configuration** page (sidebar) to create one.")
else:
    # Cascading filters, narrowest to widest: model -> overlap -> date pair. Each
    # control's options are derived from what actually exists in runs (not the
    # full catalogue Page 2 offers), so you can only ever land on a real run.
    use_swin2 = st.toggle("Use Swin-YNet", value=False, key="proc_swin",
                         help="Off = EfficientNet-B2. Filters the runs below to this model.")
    model_label2 = "Swin-YNet" if use_swin2 else "EfficientNet-B2"
    st.caption(f"Model: **{model_label2}**")

    model_runs = [r for r in runs if r["model"] == model_label2]
    if not model_runs:
        st.warning(f"No existing runs for {model_label2}. Use the **Run new "
                   f"configuration** page (sidebar) to create one.")
        st.stop()

    avail_overlaps = sorted({r["overlap %"] for r in model_runs})
    if len(avail_overlaps) == 1:
        overlap2 = avail_overlaps[0]
        st.caption(f"Window overlap: **{overlap2}%** (only one available for this model)")
    else:
        overlap2 = st.select_slider(
            "Window overlap (%)", options=avail_overlaps,
            value=avail_overlaps[len(avail_overlaps) // 2],
            key=f"proc_overlap_{model_label2}",
            help="Only overlaps with an existing run for this model are offered.")

    overlap_runs = [r for r in model_runs if r["overlap %"] == overlap2]
    date_labels = [f"{r['before']} → {r['after']}" for r in overlap_runs]
    date_choice = st.selectbox("Before → after dates", date_labels,
                               key=f"proc_dates_{model_label2}_{overlap2}")
    run = overlap_runs[date_labels.index(date_choice)]
    run_label = f"{run['model']} | {run['overlap %']}% overlap | {date_choice}"
    has_votes2 = Path(run["votes_path"]).exists()

    # Overlap is baked into the run at generation time (it's part of the model
    # call), so it's fixed and shown in the selectbox label above. Voting
    # strictness is the opposite: inference/run_overlap.py never bakes a
    # strictness into a run, it always re-threshold-able from the saved
    # per-pixel vote fraction (_votes.tif) — so it's a live slider here, not a
    # run attribute, and "burned ha" below updates as you move it. 50% is the
    # default and matches the *_burned.tif majority-vote file used as the
    # fallback when an older run has no _votes.tif sidecar.
    thr2 = st.slider(
        "Voting strictness (% of overlapping windows that must agree)", 0, 100, 50, 5,
        key="proc_strictness", disabled=not has_votes2,
        help=("Re-thresholds this run's vote-fraction raster instantly — no model run."
              if has_votes2 else
              "This run has no _votes.tif sidecar (older run); showing the fixed "
              "50% majority-vote map instead."))

    burned_ha_live = (round(votes_burned_area_ha(run["votes_path"], thr2))
                       if has_votes2 else run["burned ha"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Burned area", f"{burned_ha_live:,} ha")
    c2.metric("Run time", f"{run['time min']} min" if run["time min"] else "n/a")
    c3.metric("Windows", f"{run['windows']:,}" if run["windows"] else "n/a")
    op2 = st.slider("Overlay opacity", 0.0, 1.0, 0.75, 0.05, key="proc_op")
    mode2 = st.radio("Map layer", ["Burned area", "Error map (TP / FP / FN)"],
                     key="proc_mode", horizontal=True,
                     help="Error map scores the burned prediction against the "
                          "date-windowed ICNF ground truth, restricted to pixels "
                          "observed in both scenes.")
    is_error2 = mode2 != "Burned area"
    gt2 = st.checkbox("Show ICNF ground truth", value=False, key="proc_gt",
                      disabled=is_error2)

    if is_error2:
        if has_votes2:
            cat2, transform2, crs2 = error_map_native("votes", run["votes_path"],
                                                      run["before"], run["after"], thr2)
            counts2 = error_counts_ha(cat2)
            arr, (s, w, n, e) = _reproject_to_4326(cat2, transform2, crs2, ERR_NODATA)
            rgba, overlay_name2 = error_rgba(arr), "Error map"
            st.markdown(
                f"**{counts2['True positive']:,.0f} ha** TP &nbsp;·&nbsp; "
                f"**{counts2['False positive']:,.0f} ha** FP &nbsp;·&nbsp; "
                f"**{counts2['False negative']:,.0f} ha** FN &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"at {thr2}% strictness &nbsp;&nbsp;|&nbsp;&nbsp; "
                + error_legend_html(), unsafe_allow_html=True)
        else:
            obs_path2 = run["path"].replace("_burned.tif", "_observed.tif")
            if Path(obs_path2).exists():
                cat2, transform2, crs2 = error_map_native("burned", run["path"],
                                                          run["before"], run["after"], None)
                counts2 = error_counts_ha(cat2)
                arr, (s, w, n, e) = _reproject_to_4326(cat2, transform2, crs2, ERR_NODATA)
                rgba, overlay_name2 = error_rgba(arr), "Error map"
                st.markdown(
                    f"**{counts2['True positive']:,.0f} ha** TP &nbsp;·&nbsp; "
                    f"**{counts2['False positive']:,.0f} ha** FP &nbsp;·&nbsp; "
                    f"**{counts2['False negative']:,.0f} ha** FN &nbsp;&nbsp;|&nbsp;&nbsp; "
                    + error_legend_html(), unsafe_allow_html=True)
            else:
                st.warning(f"No sidecar _observed.tif for this run; cannot score it. "
                           f"Expected {Path(obs_path2).name}.")
                arr, (s, w, n, e) = votes_4326(run["path"])
                rgba, overlay_name2 = np.zeros(arr.shape + (4,), np.uint8), "Burned"
    elif has_votes2:
        arr, (s, w, n, e) = votes_4326(run["votes_path"])  # live-rethresholded vote fraction
        rgba = burned_rgba(arr, thr2)
        overlay_name2 = "Burned"
    else:
        arr, (s, w, n, e) = votes_4326(run["path"])      # reproject the fixed burned raster
        rgba = np.zeros(arr.shape + (4,), np.uint8)
        rgba[arr == 1] = (215, 25, 28, 255)              # red where burned
        overlay_name2 = "Burned"

    fmap2 = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=9, tiles="CartoDB positron")
    ImageOverlay(rgba, bounds=[[s, w], [n, e]], opacity=op2, name=overlay_name2).add_to(fmap2)
    folium.GeoJson(portugal_boundary((w, s, e, n)), name="Portugal boundary",
                   style_function=lambda _: {"color": "#444444", "weight": 1.5,
                                             "fillOpacity": 0.0}).add_to(fmap2)
    if gt2 and not is_error2:
        gj2, _ = ground_truth(run["before"], run["after"], (w, s, e, n))
        folium.GeoJson(gj2, name="ICNF ground truth",
                       style_function=lambda _: {"color": "#2c7bb6", "weight": 1,
                                                 "fillOpacity": 0.0}).add_to(fmap2)
    MiniMap(tile_layer="OpenStreetMap", position="bottomright", width=190, height=140,
            zoom_level_offset=-5, toggle_display=True).add_to(fmap2)
    folium.LayerControl().add_to(fmap2)
    components.html(fmap2._repr_html_(), height=560)
    with st.expander("All processed outputs (table)"):
        st.dataframe(pd.DataFrame(runs).drop(columns="path").sort_values(["model", "overlap %"]),
                     use_container_width=True, hide_index=True)

    # ---------- focal-zone inspector: click the map above, verify the spot below ----------
    st.markdown("---")
    st.markdown("##### Focal zone inspector")
    st.caption("Click a point on the map below for a ~3 km chip at that location: "
               "Sentinel-2 before/after, OpenStreetMap, and this run's error map, "
               "side by side. Mirrors the manual checks in "
               "notebooks/false_positive_review.ipynb Section 5.")

    if not HAS_ST_FOLIUM:
        st.warning("Click-to-inspect needs the `streamlit-folium` package "
                   "(added to environment.yml — re-run `conda env update` to pick it up).")
    else:
        key_state = "inspect_point"
        bbox_sel = None
        if key_state in st.session_state:
            lat_sel, lon_sel = st.session_state[key_state]
            bbox_sel = point_to_bbox(lat_sel, lon_sel)

        inspect_map = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=9,
                                 tiles="CartoDB positron")
        ImageOverlay(rgba, bounds=[[s, w], [n, e]], opacity=0.55,
                    name=overlay_name2).add_to(inspect_map)
        if bbox_sel is not None:
            folium.Marker([lat_sel, lon_sel], icon=folium.Icon(color="red")).add_to(inspect_map)
            lo0, la0, lo1, la1 = transform_bounds("EPSG:32629", "EPSG:4326", *bbox_sel)
            folium.Rectangle([[la0, lo0], [la1, lo1]], color="#e34a33", weight=2,
                             fill=False).add_to(inspect_map)
        click = st_folium(inspect_map, height=480, key="inspector_map",
                          returned_objects=["last_clicked"])

        # A click changes st_folium's return value, which triggers a Streamlit
        # rerun on its own, but inspect_map (marker + rectangle) was already
        # built further up using the *old* session_state value before this
        # block runs, so the just-rendered map would still show the previous
        # point. st.rerun() here throws away the rest of this pass and starts
        # a fresh one immediately, so inspect_map gets rebuilt from the
        # now-updated session_state before anything is shown — the marker
        # lands on the clicked point instead of trailing by one click. The
        # state-equality guard prevents this from looping: on the rerun it
        # triggers, session_state already matches new_pt, so the condition is
        # False and nothing fires again.
        if click and click.get("last_clicked"):
            new_pt = (click["last_clicked"]["lat"], click["last_clicked"]["lng"])
            if st.session_state.get(key_state) != new_pt:
                st.session_state[key_state] = new_pt
                st.rerun()

        if key_state in st.session_state:
            lat_sel, lon_sel = st.session_state[key_state]
            bbox_sel = point_to_bbox(lat_sel, lon_sel)
            cap_col, clear_col = st.columns([5, 1])
            cap_col.caption(f"Selected: {lat_sel:.4f}°N, {lon_sel:.4f}°W &nbsp;·&nbsp; "
                           f"run: {run_label}", unsafe_allow_html=True)
            if clear_col.button("Clear", key="inspect_clear"):
                # the button click itself already reruns the script; the
                # del just needs to happen before that rerun re-reads state.
                del st.session_state[key_state]
            else:
                render_focal_chip(lat_sel, lon_sel, run, bbox_sel,
                                  strictness=thr2 if has_votes2 else None)
        else:
            st.caption("No point selected yet.")
