"""Entry point for the burned-area viewer (T29TPG) multipage app.

Run with: streamlit run app.py

This file is now a thin router: page config (must be set exactly once, before
anything else) and the global CSS/title live here, and st.navigation wires up
the two actual pages with explicit sidebar labels. Streamlit's classic
pages/-folder auto-discovery derives a page's nav label from its filename —
the entry script would show up as the literal "App" — so st.navigation's
explicit `title=` is used instead to get "View a processed output" and "Run
new configuration" in the sidebar.

The real page content lives in pages/1_View_a_processed_output.py (browse
existing runs, error map, focal-zone inspector) and
pages/2_Run_New_Configuration.py (launch a new model run).
"""
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
st.markdown("##### Burned-area model viewer — Sentinel-2 tile T29TPG")

pg = st.navigation([
    st.Page("pages/1_View_a_processed_output.py",
            title="View a processed output", default=True),
    st.Page("pages/2_Run_New_Configuration.py",
            title="Run new configuration"),
])
pg.run()
