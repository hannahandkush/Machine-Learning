"""Sliding-window (overlap) inference experiment.

Standard inference (`inference/run.py`) uses the cube's fixed, non-overlapping
256x256 chips. This mode rebuilds the full before/after image, slides a 256x256
window across it at a chosen overlap, runs the model on each window, and merges
the overlapping predictions by majority vote (a pixel is burned if most of the
windows covering it say burned).

It mainly affects edge/seam quality; it does not fix burn-vs-browning confusion.

Usage
-----
    python -m inference.run_overlap --overlap 0.5 --device cpu
    python -m inference.run_overlap --overlap 0.75 \
        --before-date 2025-07-07 --after-date 2025-10-15
    python -m inference.run_overlap --overlap 0.25 --max-windows 16   # smoke test

Outputs use an `ov<pct>` tag, e.g. T29TPG_swin_ynet_ov50_20250707_20251015_*.tif
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.config import load_config                       # noqa: E402
from inference import chips, mosaic, scene_select          # noqa: E402
from inference.adapters import get_adapter, pick_device    # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

CHIP = 256


def reconstruct_tile(f, t_idx, meta, cxb, cyb, chip_ids):
    """Rebuild the full dense (H, W, bands) image for one acquisition."""
    H, W, B = meta.height, meta.width, meta.n_bands
    tile = np.full((H, W, B), meta.nodata, dtype=np.uint16)
    footprint = np.zeros((H, W), dtype=bool)
    observed = np.zeros((H, W), dtype=bool)
    cs = meta.chip_size
    for cid in chip_ids:
        chip, fp, obs = chips.read_chip(f, t_idx, int(cid), meta, cxb, cyb)
        r0 = int(cyb[cid]) * cs
        c0 = int(cxb[cid]) * cs
        h = min(cs, H - r0)
        w = min(cs, W - c0)
        if h <= 0 or w <= 0:
            continue
        tile[r0:r0 + h, c0:c0 + w, :] = chip[:h, :w, :]
        footprint[r0:r0 + h, c0:c0 + w] = fp[:h, :w]
        observed[r0:r0 + h, c0:c0 + w] = obs[:h, :w]
    return tile, footprint, observed


def window_starts(n, step, win=CHIP):
    """Top-left positions so 256-windows tile [0, n) and the last edge is covered."""
    if n <= win:
        return [0]
    xs = list(range(0, n - win + 1, step))
    if xs[-1] != n - win:
        xs.append(n - win)
    return xs


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlap", type=float, default=0.0,
                   help="window overlap fraction, 0.0 to 0.75 (0 = no overlap).")
    p.add_argument("--before-date")
    p.add_argument("--after-date")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device")
    p.add_argument("--max-windows", type=int, help="process only the first N windows (smoke test).")
    p.add_argument("--out")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = load_config()
    ov = float(args.overlap)
    if not 0.0 <= ov <= 0.75:
        sys.exit("--overlap must be between 0.0 and 0.75")
    step = max(1, int(round(CHIP * (1.0 - ov))))
    t0 = time.time()

    meta = chips.open_meta(cfg.hdf5_path, cfg.xys_sentinel)
    scenes = scene_select.load_scene_table(cfg.hdf5_path)
    if args.before_date or args.after_date:
        if not (args.before_date and args.after_date):
            sys.exit("provide both --before-date and --after-date, or neither")
        before = scene_select.resolve_by_date(scenes, args.before_date)
        after = scene_select.resolve_by_date(scenes, args.after_date)
    else:
        before, after = scene_select.select_before_after(
            scenes, cfg.before_window, cfg.after_window, usable=cfg.usable_categories)
    b_date = before["date"].date().isoformat()
    a_date = after["date"].date().isoformat()
    print(f"[scene] before {b_date} ({before['category']}), after {a_date} ({after['category']})")
    print(f"[overlap] {ov * 100:.0f}%  ->  step {step} px")

    cxb, cyb = chips.load_chip_bins(cfg.hdf5_path)
    ids = chips.chip_ids_with_data(cfg.hdf5_path)

    adapter = get_adapter(cfg.model_kind)
    device = pick_device(args.device)
    print(f"[model] {adapter.NAME} on {device}")
    handle, model = adapter.load(cfg.weights_path, cfg.package_dir, device)
    bs = args.batch_size or cfg.batch_size
    n_class = int(adapter.BURNED_CLASS) + 1   # classes are 0..BURNED_CLASS

    print("[build] reconstructing full before/after images ...")
    with h5py.File(cfg.hdf5_path, "r") as f:
        before_tile, footprint, obs_b = reconstruct_tile(f, int(before["t_idx"]), meta, cxb, cyb, ids)
        after_tile, _, obs_a = reconstruct_tile(f, int(after["t_idx"]), meta, cxb, cyb, ids)
    obs_both = obs_b & obs_a
    del obs_b, obs_a

    H, W = meta.height, meta.width
    class_votes = np.zeros((n_class, H, W), dtype=np.uint8)   # votes per class
    total_votes = np.zeros((H, W), dtype=np.uint8)            # windows covering each pixel

    rows = window_starts(H, step)
    cols = window_starts(W, step)
    positions = [(r, c) for r in rows for c in cols
                 if obs_both[r:r + CHIP, c:c + CHIP].any()]
    if args.max_windows:
        positions = positions[:args.max_windows]
    print(f"[windows] {len(positions)} windows (empty ones pruned from {len(rows) * len(cols)} grid)")

    for s in tqdm(range(0, len(positions), bs), desc="batches", unit="batch"):
        batch = positions[s:s + bs]
        bw = np.stack([before_tile[r:r + CHIP, c:c + CHIP, :] for r, c in batch])
        aw = np.stack([after_tile[r:r + CHIP, c:c + CHIP, :] for r, c in batch])
        labels = adapter.predict(handle, bw, aw, model, device)
        for (r, c), lab in zip(batch, labels):
            m = obs_both[r:r + CHIP, c:c + CHIP]
            tv = total_votes[r:r + CHIP, c:c + CHIP]
            tv[m] += 1
            for k in range(n_class):
                cv = class_votes[k, r:r + CHIP, c:c + CHIP]
                cv[m & (lab == k)] += 1

    # Resolve votes -> maps
    voted = total_votes > 0
    pred = np.full((H, W), 255, dtype=np.uint8)
    pred[voted] = class_votes[:, voted].argmax(axis=0).astype(np.uint8)
    bc = int(adapter.BURNED_CLASS)
    burned = np.full((H, W), 255, dtype=np.uint8)
    burned[voted] = (class_votes[bc][voted].astype(int) * 2 > total_votes[voted].astype(int)).astype(np.uint8)
    observed = np.full((H, W), 255, dtype=np.uint8)
    observed[footprint] = obs_both[footprint].astype(np.uint8)

    out_dir = Path(args.out) if args.out else cfg.predictions_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"T29TPG_{adapter.NAME}_ov{int(round(ov * 100)):02d}_"
            f"{b_date.replace('-', '')}_{a_date.replace('-', '')}")
    paths = {k: out_dir / f"{stem}_{k}.tif" for k in ("pred", "burned", "observed")}
    for k, arr in (("pred", pred), ("burned", burned), ("observed", observed)):
        mosaic._write(paths[k], arr, meta.transform, cfg.tile_crs)
    (out_dir / f"{stem}_manifest.json").write_text(json.dumps({
        "tile": "T29TPG", "model": adapter.NAME,
        "overlap": ov, "step_px": step,
        "before_date": b_date, "after_date": a_date,
        "n_windows": len(positions), "device": str(device),
    }, indent=2))

    nb = int((burned == 1).sum())
    print(f"[done] overlap {ov * 100:.0f}% in {time.time() - t0:.0f}s | "
          f"burned {nb:,} px (~{nb / 100:,.0f} ha)")
    print(f"[done] wrote {paths['burned']}")


if __name__ == "__main__":
    main()
