"""Tile-wide burned-area inference over the T29TPG Sentinel-2 cube.

Selects one before/after scene pair, runs the fine-tuned Swin-YNet model over
every chip with data, and writes three georeferenced rasters (pred / burned /
observed). See the project plan for the design.

Usage
-----
    python -m inference.run                      # auto-select scenes, full tile
    python -m inference.run --max-chips 20        # quick smoke run
    python -m inference.run --before-date 2025-07-22 --after-date 2025-11-21
    python -m inference.run --batch-size 16 --device cpu
    python -m inference.run --resume              # continue an interrupted run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Repo root on sys.path so `utils` and `inference` import when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.config import load_config              # noqa: E402
from inference import chips, mosaic, scene_select  # noqa: E402
from inference.adapters import get_adapter, pick_device  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):
        return it


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before-date", help="Override before scene (YYYY-MM-DD).")
    p.add_argument("--after-date", help="Override after scene (YYYY-MM-DD).")
    p.add_argument("--batch-size", type=int, help="Chips per model batch.")
    p.add_argument("--max-chips", type=int, help="Process only the first N chips (smoke run).")
    p.add_argument("--model-kind", help="Override config model.kind (e.g. efficientnet_b2), to run "
                                        "a different adapter without editing config.yaml. Pass "
                                        "--weights/--package-dir alongside this, since those still "
                                        "default to whatever model.kind in config.yaml points at.")
    p.add_argument("--weights", help="Override the model weights .pth (must match the configured "
                                     "model package). Default: config model.weights_path.")
    p.add_argument("--package-dir", help="Override the model package directory. Default: config "
                                         "model.package_dir.")
    p.add_argument("--device", help="Force a torch device (cuda/mps/cpu).")
    p.add_argument("--out", help="Output directory (default: config inference.predictions_dir).")
    p.add_argument("--resume", action="store_true", help="Skip chips already in the manifest.")
    return p.parse_args()


def _check_grid_alignment(meta, repo_root):
    """Soft-check that the output grid matches the ICNF label raster."""
    ref = repo_root / "data/processed/icnf_burned_labels_t29tpg_2025.tif"
    if not ref.exists():
        return
    import rasterio

    with rasterio.open(ref) as r:
        ok = (r.width, r.height) == (meta.width, meta.height) and \
            np.allclose(np.array(r.transform)[:6], np.array(meta.transform)[:6])
    flag = "OK" if ok else "MISMATCH"
    print(f"[grid] vs ICNF label raster: {flag}  "
          f"(pred {meta.width}x{meta.height}, ref {r.width}x{r.height})")
    if not ok:
        print("[grid] WARNING: output will not co-register with the ICNF labels.")


def main():
    args = _parse_args()
    cfg = load_config()
    t_start = time.time()

    meta = chips.open_meta(cfg.hdf5_path, cfg.xys_sentinel)
    print(f"[cube] {cfg.hdf5_path}")
    print(f"[cube] bands ({meta.n_bands}): {meta.band_names}")
    assert meta.n_bands == 10, f"expected 10 bands, got {meta.n_bands}"
    _check_grid_alignment(meta, cfg.repo_root)

    # ── Scene selection ─────────────────────────────────────────────────────────
    scenes = scene_select.load_scene_table(cfg.hdf5_path)
    if args.before_date or args.after_date:
        if not (args.before_date and args.after_date):
            sys.exit("Provide both --before-date and --after-date, or neither.")
        before = scene_select.resolve_by_date(scenes, args.before_date)
        after = scene_select.resolve_by_date(scenes, args.after_date)
    else:
        before, after = scene_select.select_before_after(
            scenes, cfg.before_window, cfg.after_window, usable=cfg.usable_categories
        )
    b_date = before["date"].date().isoformat()
    a_date = after["date"].date().isoformat()
    print(f"[scene] eligible categories: {list(cfg.usable_categories)}")
    print(f"[scene] before: {b_date}  ({before['category']}, cloud {before['cloud_pct']:.0f}%, swath {before['swath']:.2f})")
    print(f"[scene] after : {a_date}  ({after['category']}, cloud {after['cloud_pct']:.0f}%, swath {after['swath']:.2f})")

    # ── Outputs ─────────────────────────────────────────────────────────────────
    model_kind = args.model_kind or cfg.model_kind

    out_dir = Path(args.out) if args.out else cfg.predictions_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"T29TPG_{model_kind}_{b_date.replace('-', '')}_{a_date.replace('-', '')}"
    paths = {k: out_dir / f"{stem}_{k}.tif" for k in ("pred", "burned", "observed")}
    manifest_path = out_dir / f"{stem}_manifest.json"

    # ── Chip list (+ resume) ────────────────────────────────────────────────────
    ids = chips.chip_ids_with_data(cfg.hdf5_path)
    if args.max_chips:
        ids = ids[: args.max_chips]
    done = set()
    if args.resume and manifest_path.exists():
        done = set(json.loads(manifest_path.read_text()).get("completed_chip_ids", []))
        print(f"[resume] {len(done)} chips already done; {len(ids) - len(done)} remaining")
    todo = [int(c) for c in ids if int(c) not in done]

    cxb, cyb = chips.load_chip_bins(cfg.hdf5_path)

    # ── Tile accumulator ────────────────────────────────────────────────────────
    # Accumulate the whole tile in memory and write each GeoTIFF in one pass —
    # scattered windowed writes to a compressed tiled GeoTIFF silently drop data.
    if args.resume and all(p.exists() for p in paths.values()):
        tile = mosaic.TileMosaic.load(paths, meta.width, meta.height)
    else:
        tile = mosaic.TileMosaic(meta.width, meta.height)

    def save_outputs():
        tile.save(paths, meta.transform, cfg.tile_crs)

    # ── Model ───────────────────────────────────────────────────────────────────
    adapter = get_adapter(model_kind)
    weights_path = Path(args.weights) if args.weights else cfg.weights_path
    package_dir = Path(args.package_dir) if args.package_dir else cfg.package_dir
    if not weights_path.exists():
        sys.exit(f"Weights not found: {weights_path}")
    device = pick_device(args.device)
    print(f"[model] kind: {adapter.NAME}  (burned class = {adapter.BURNED_CLASS})")
    print(f"[model] device: {device}")
    print(f"[model] weights: {weights_path}")
    handle, model = adapter.load(weights_path, package_dir, device)
    batch_size = args.batch_size or cfg.batch_size

    completed = list(done)
    first_batch = True
    SAVE_EVERY = 20   # flush full rasters every N batches so --resume has state

    with h5py.File(cfg.hdf5_path, "r") as f:
        for bi, start in enumerate(tqdm(range(0, len(todo), batch_size), desc="batches", unit="batch")):
            batch_ids = todo[start:start + batch_size]
            before_arr, after_arr, masks = [], [], []
            for cid in batch_ids:
                cb, fp, ob_b = chips.read_chip(f, int(before["t_idx"]), cid, meta, cxb, cyb)
                ca, _, ob_a = chips.read_chip(f, int(after["t_idx"]), cid, meta, cxb, cyb)
                before_arr.append(cb)
                after_arr.append(ca)
                masks.append((fp, ob_b & ob_a))
            before_arr = np.stack(before_arr)
            after_arr = np.stack(after_arr)

            try:
                labels = adapter.predict(handle, before_arr, after_arr, model, device)
            except RuntimeError as exc:
                if first_batch and device.type != "cpu":
                    print(f"[model] {device} failed ({exc}); falling back to CPU.")
                    import torch
                    device = torch.device("cpu")
                    handle, model = adapter.load(weights_path, package_dir, device)
                    labels = adapter.predict(handle, before_arr, after_arr, model, device)
                else:
                    raise
            first_batch = False

            for cid, lab, (fp, ob_both) in zip(batch_ids, labels, masks):
                blocks = mosaic.build_blocks(lab, fp, ob_both, adapter.BURNED_CLASS)
                tile.place(cid, cxb, cyb, blocks, meta.chip_size)
                completed.append(int(cid))

            if (bi + 1) % SAVE_EVERY == 0:
                save_outputs()

            manifest_path.write_text(json.dumps({
                "tile": "T29TPG",
                "model": adapter.NAME,
                "weights": str(weights_path),
                "before_date": b_date,
                "after_date": a_date,
                "before_category": before["category"],
                "after_category": after["category"],
                "batch_size": batch_size,
                "device": str(device),
                "burned_class": adapter.BURNED_CLASS,
                "n_chips_total": len(ids),
                "n_completed": len(completed),
                "completed_chip_ids": sorted(completed),
                "outputs": {k: str(v) for k, v in paths.items()},
            }, indent=2))

    save_outputs()

    n_burned = tile.burned_pixels()
    burned_ha = n_burned * (meta.res ** 2) / 1e4
    print(f"[done] {len(completed)}/{len(ids)} chips in {time.time() - t_start:.0f}s")
    print(f"[done] burned pixels: {n_burned:,}  (~{burned_ha:,.0f} ha)")
    print(f"[done] wrote: {paths['burned']}")


if __name__ == "__main__":
    main()
