"""Entry point for the burned-area viewer (T29TPG) multipage app.

Run with:  streamlit run app/app.py   (from the repository root)

Thin router: it sets the page config once (must happen before anything else),
defines the global CSS and title, makes sure the repo root and this app/ folder
are both on sys.path (so the pages can import the shared utils.config package
and the sibling viewer_common / sentinel_hub modules), and wires up the two
pages under a "Pages" section in the sidebar via st.navigation.

The real page content lives in pages/1_View_a_processed_output.py (browse
existing runs, error map, focal-zone inspector) and
pages/2_Run_New_Configuration.py (launch a new model run).
"""
import sys
from pathlib import Path

# this app/ folder (for the sibling viewer_common and sentinel_hub modules) and
# the repo root (for the shared utils.config package) both need to be importable.
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (str(_REPO_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

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
# ULisboa / ISA logo (horizontal) shown as a banner above the title, on every page
st.image(str(_APP_DIR / "Logo_ULisboa_ISA_horizontal_color.png"), width=400)
st.markdown("## Burned-area model viewer — Sentinel-2 tile T29TPG")

# the two pages grouped under a "Pages" header in the sidebar
pg = st.navigation({
    "Pages": [
        st.Page("pages/1_View_a_processed_output.py",
                title="Browse Results", default=True),
        st.Page("pages/2_Run_New_Configuration.py",
                title="Run a Model"),
    ]
})
pg.run()
