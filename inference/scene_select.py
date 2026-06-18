"""Per-scene quality screening and before/after pair selection.

Mirrors the scene-classification logic in `notebooks/hdf5_data_exploration.ipynb`
(cell 20) and `documents/evaluation_protocol.md` §5.1, reading the per-timestamp
metadata stored in the HDF5 cube.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

def _classify(row) -> str:
    if row.swath >= 0.95 and row.cloud_pct <= 10:
        return "full_clear"
    if row.swath < 0.50:
        return "edge_pass"
    if row.swath >= 0.50 and row.cloud_pct <= 20:
        return "partial_clear"
    return "clouded"


def load_scene_table(hdf5_path) -> pd.DataFrame:
    """Return one row per acquisition with date, cloud %, swath and category."""
    with h5py.File(hdf5_path, "r") as f:
        dates = pd.to_datetime(f["original_timestamps"][:], unit="ms")
        cloud = f["cloud_cover_pt"][:].astype(float)
        total = f["pixel_count_pt"][:].astype(float)
        orbit = f["count_orbit_pixels_pt"][:].astype(float)
        clear = f["clear_pixel_count_pt"][:].astype(float)

    df = pd.DataFrame(
        {
            "t_idx": np.arange(len(dates)),
            "date": dates,
            "cloud_pct": cloud,
            "total_px": total,
            "orbit_px": orbit,
            "clear_px": clear,
        }
    )
    df["swath"] = df["orbit_px"] / df["total_px"]
    df["category"] = df.apply(_classify, axis=1)
    return df.sort_values("date").reset_index(drop=True)


def _candidates(df, start, end, usable):
    return df[
        (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
        & (df["category"].isin(usable))
    ].copy()


def _pick(df, start, end, usable, role):
    """Best scene in [start, end] restricted to ``usable`` categories.

    role='before' → cleanest scene, latest as tie-break (a tight pre-fire baseline).
    role='after'  → latest scene, cleanest as tie-break (bracket as many fires as
                    possible; among full-coverage scenes cloud is already low).
    """
    win = _candidates(df, start, end, usable)
    if win.empty:
        return None
    if role == "before":
        win = win.sort_values(["cloud_pct", "date"], ascending=[True, False])
    else:  # after
        win = win.sort_values(["date", "cloud_pct"], ascending=[False, True])
    return win.iloc[0]


def select_before_after(df: pd.DataFrame, before_window, after_window, usable=("full_clear",)):
    """Pick one before (pre-season baseline) and one after (post-fire) scene.

    Only scenes whose ``category`` is in ``usable`` are eligible — by default
    ``full_clear`` only (full swath coverage AND low cloud), so edge passes and
    cloudy/partial scenes are never used. Raises with the available dates if a
    window has no eligible scene.
    """
    before = _pick(df, before_window[0], before_window[1], usable, role="before")
    if before is None:
        avail = _candidates(df, before_window[0], before_window[1], df["category"].unique())
        raise ValueError(
            f"No {list(usable)} before scene in {tuple(before_window)}. "
            f"Categories present: {avail['category'].value_counts().to_dict()}"
        )
    after = _pick(df, after_window[0], after_window[1], usable, role="after")
    if after is None:
        avail = _candidates(df, after_window[0], after_window[1], df["category"].unique())
        raise ValueError(
            f"No {list(usable)} after scene in {tuple(after_window)}. "
            f"Categories present: {avail['category'].value_counts().to_dict()}"
        )
    return before, after


def resolve_by_date(df: pd.DataFrame, date_str: str):
    """Return the scene row whose acquisition date matches ``date_str`` (YYYY-MM-DD)."""
    target = pd.Timestamp(date_str).normalize()
    hit = df[df["date"].dt.normalize() == target]
    if hit.empty:
        raise ValueError(
            f"No acquisition on {date_str}. Available dates: "
            + ", ".join(d.date().isoformat() for d in df["date"])
        )
    return hit.iloc[0]
