# Burned-area viewer app

A two-page Streamlit app for exploring the project's burned-area predictions
for Sentinel-2 tile T29TPG: run or browse a model/overlap/date configuration,
compare it against ICNF ground truth, and spot-check individual points with a
focal-zone inspector. See the repository root [README](../README.md) for the
project overview, data, and results; this file covers the app itself.

## Key widgets

**Run a Model** (`pages/2_Run_New_Configuration.py`), sidebar:

- `st.toggle` — model (EfficientNet-B2 / Swin-YNet)
- `st.slider` — window overlap, 0–75%
- `st.selectbox` ×2 — before/after dates
- `st.button` — launch the run as a subprocess
- `st.progress` — live progress bar, polled from the runner's progress file
- `st.slider` ×2 — voting strictness and overlay opacity, shown once a run exists (re-thresholds the vote-fraction raster instantly, no model re-run)
- `st.page_link` — jump to "View a processed output" for the error map and inspector

**View a processed output** (`pages/1_View_a_processed_output.py`), sidebar:

- `st.toggle` → `st.select_slider` → `st.selectbox` — cascading model / overlap / date-pair filters, each one's options limited to what actually exists for the choice above it, so you can't land on a combination with no run
- `st.slider` — voting strictness (disabled if the run has no `_votes.tif` sidecar)
- `st.slider` — overlay opacity
- `st.radio` — map layer: burned area vs. error map (TP/FP/FN)
- `st.checkbox` — ICNF ground truth overlay (disabled in error-map mode)
- `st.expander` + `st.dataframe` — full runs table
- two `st_folium` maps — the main overlay map, and the focal-zone inspector map (its `last_clicked` return value drives the chip comparison below it)
- `st.button` — clear the inspector's selected point

The folium layers underneath (`ImageOverlay`, `GeoJson` boundary, `MiniMap`,
`LayerControl`) aren't Streamlit widgets — they're rendered HTML embedded via
`st_folium`/`components.html`.

## Credits

The focal-zone inspector reviews Sentinel-2 imagery against the model's
predictions using:

- **Sentinel-2 imagery** — [Copernicus Data Space Ecosystem (CDSE) Process API](https://documentation.dataspace.copernicus.eu/APIs.html), fetched live for the before/after true-colour chips.
- **Basemap and reference features** — [OpenStreetMap](https://www.openstreetmap.org).
- **Error maps (TP/FP/FN)** — produced by this project's own model runs, scoring predictions against the ICNF ground truth, to evaluate outputs alongside the imagery above.

## How the app was developed

The app began as a focused adaptation of an app Danilo had built for a
previous project with AI assistance — a first draft that rendered outputs
from existing model runs and let you set up a new one. Hannah developed it
into the current two-page structure, following Streamlit's [multipage-app
guide](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app)
with AI assistance, adding the error-map overlay and the focal-zone
inspector. Hannah and Danilo then both worked on the UI, improving the
configuration-selection experience and adding the project logo. Development
was iterative rather than linear — for example, a `st.session_state`/rerun
timing bug left the focal-zone inspector's click marker trailing one click
behind the actual click; this was debugged with Claude.

See the root README's "Use of AI" section for the full declaration of how AI
was used across the project, not just the app.

## Contributions

Sourced from the actual `git log` for commits that touch the app, in order:

| Date | Author | Change |
|---|---|---|
| 2026-06-21 | Danilo | First version of `app.py`: a local Streamlit demo of the model's predictions, with a progress bar for long-running steps and a "zoom to burned area" button |
| 2026-06-21 | Danilo | Added `streamlit`/`folium` to `environment.yml`; documented how to run the app in the README |
| 2026-06-24 | Danilo | Restructured into a tabbed layout — kept the interactive map viewer and added a view-only "Processed outputs" tab to browse already-computed runs (per-run stats, summary table); commented the code |
| 2026-06-25 | Hannah | Added the TP/FP/FN error map, the `sentinel_hub.py` helper for live Sentinel-2 chip fetches, and the focal-zone inspector (click a point to compare Sentinel-2 before/after, OpenStreetMap, and the error map); wrote the false-positive review notebook it mirrors |
| 2026-06-25 | Hannah | Improved the focal-zone inspector's responsiveness |
| 2026-06-26 | Danilo | Decluttered `app.py` (removed a redundant ICNF-burned-area checkbox); added cloud-cover % next to the date picker |
| 2026-06-26 | Hannah | Split the single-page app into the current two-page structure: "View a processed output" and "Run a new configuration" |
| 2026-06-26 | Hannah | Added the voting-strictness slider to both pages — re-thresholds the vote-fraction raster live, no model re-run |
| 2026-06-26 | Hannah | Fixed a `st.session_state`/rerun timing bug that left the focal-zone inspector's click marker trailing one click behind |
| 2026-06-26 | Danilo | Reorganised the app under `app/` (entry point, pages, and helpers as siblings) and added the ULisboa/ISA logo |
| 2026-06-26 | Hannah & Danilo | Final repo cleanup before submission |

Net split: Danilo built the original app and its data-selection/display features
(progress bar, tabbed browsing, cloud-cover filter, the `app/` restructure, the
logo); Hannah added the evaluation-facing features (error map, focal-zone
inspector, voting-strictness control) and the two-page split, plus the
session-state bug fix.
