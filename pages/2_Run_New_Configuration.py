"""Page 2 of the burned-area viewer — launch a new model/overlap/date-pair run.

Picks a configuration (model, window overlap, before/after scene pair) and
launches `inference.run_overlap` as a subprocess, streaming a progress bar from
the runner's progress file. If the chosen configuration already has a finished
run sitting in outputs/predictions/, running it again is blocked — Page 1
already shows it, and a re-run is wasted compute for an identical output (the
underlying model is frozen, not fine-tuned per run, so the same inputs always
produce the same prediction).

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

import streamlit as st

from utils.viewer_common import (
    EFF_PKG, EFF_WEIGHTS, MODELS, OVERLAPS, REPO, USABLE_DATES,
    _date_label, run_paths,
)

st.markdown("##### Run a new configuration")
st.caption("Pick a model, window overlap, and before/after scene pair. If this "
           "exact configuration hasn't been run yet, a button below launches it.")

col1, col2 = st.columns(2)
with col1:
    model = MODELS[st.selectbox("Model", list(MODELS))]
    overlap = st.selectbox("Window overlap (%)", OVERLAPS, index=2)
with col2:
    before = st.selectbox("Before date", USABLE_DATES, index=USABLE_DATES.index("2025-07-07"),
                          format_func=_date_label)
    after = st.selectbox("After date", USABLE_DATES, index=USABLE_DATES.index("2025-10-15"),
                         format_func=_date_label)

paths = run_paths(model, overlap, before, after)

if paths["burned"].exists():
    st.warning(
        f"A run for **{model}**, {overlap}% overlap, {before} → {after} already "
        f"exists in `outputs/predictions/`. Re-running would spend several "
        f"minutes reproducing an identical result, since neither model is "
        f"fine-tuned per run — the same inputs always score the same way."
    )
    st.page_link("pages/1_View_a_processed_output.py",
                 label="Go to Processed outputs to view it", icon="📊")
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
            st.success("Run complete.")
            st.page_link("pages/1_View_a_processed_output.py",
                 label="Go to Processed outputs to view it", icon="📊")
        else:
            st.error("Run failed.")
            st.code(log_file.read_text()[-2000:] if log_file.exists() else "")
