"""Copernicus Data Space Ecosystem (CDSE) Sentinel-2 image fetcher.

Provides authenticated access to the CDSE Sentinel Hub Process API
(https://documentation.dataspace.copernicus.eu/APIs.html) for pulling small,
rendered true-colour Sentinel-2 chips on demand, by bounding box and date.
This is used for visual verification of individual false positives/negatives
(app.py's "if possible" pixel/zone viewer, and notebooks/false_positive_review.ipynb)
without downloading or processing full SAFE products or the local HDF5 cube.

Setup
-----
1. Register a free CDSE account at https://dataspace.copernicus.eu/.
2. Create an OAuth client at
   https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
   ("User settings" -> "OAuth clients" -> "Create"). The client secret is
   shown only once — copy it immediately.
3. Supply the credentials either as environment variables (preferred, never
   touches a file that could be committed)::

       export CDSE_CLIENT_ID="..."
       export CDSE_CLIENT_SECRET="..."

   or in `config.local.yaml` (git-ignored — see .gitignore) at the repo root::

       cdse:
         client_id: "..."
         client_secret: "..."

Usage
-----
    from sentinel_hub import fetch_truecolor

    # bbox in the tile's native CRS (EPSG:32629), reprojected internally.
    img = fetch_truecolor(
        bbox=(640000, 4620000, 642000, 4622000),
        bbox_crs="EPSG:32629",
        date="2025-07-07",
    )  # -> (H, W, 3) uint8 array, ready for plt.imshow

Design notes
------------
- Tokens are cached in-process and refreshed only once expired (CDSE rate-
  limits the token endpoint; tokens are valid for several minutes).
- Rendered images are cached to disk under `data/cache/sentinel2/`, keyed by
  a hash of the request parameters, so repeated notebook re-runs and the
  Streamlit app (which reruns the whole script on every interaction) do not
  re-hit the API or burn quota for the same bbox/date pair.
- `mosaicking_order="leastCC"` picks the least-cloudy scene inside the
  requested date window rather than requiring an exact acquisition date,
  since the Process API mosaics over the window rather than indexing by a
  single scene id.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from rasterio.warp import transform_bounds
except ImportError:  # pragma: no cover - rasterio is a project-wide dependency
    transform_bounds = None

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# Re-fetch the token a little before actual expiry to avoid edge-case 401s.
_TOKEN_SAFETY_MARGIN_S = 30

# True-colour evalscript (Sentinel-2 L2A, bands B04/B03/B02, simple gain).
# Kept as a module constant so cache keys are stable across calls.
EVALSCRIPT_TRUECOLOR = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "dataMask"],
    output: { bands: 4 }
  };
}
function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02, sample.dataMask];
}
"""

_repo_root_cache: Optional[Path] = None
_token_cache: dict = {"access_token": None, "expires_at": 0.0}


class CDSEAuthError(RuntimeError):
    """Raised when CDSE OAuth credentials are missing or rejected."""


def _repo_root() -> Path:
    global _repo_root_cache
    if _repo_root_cache is not None:
        return _repo_root_cache
    from utils.config import _find_repo_root  # reuse the project's own search

    _repo_root_cache = _find_repo_root(Path.cwd().resolve())
    return _repo_root_cache


def _load_credentials() -> tuple[str, str]:
    """Resolve (client_id, client_secret), env vars taking priority."""
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    local_cfg = _repo_root() / "config.local.yaml"
    if local_cfg.exists():
        with open(local_cfg, "r") as f:
            raw = yaml.safe_load(f) or {}
        cdse = raw.get("cdse", {})
        client_id = client_id or cdse.get("client_id")
        client_secret = client_secret or cdse.get("client_secret")

    if not client_id or not client_secret:
        raise CDSEAuthError(
            "No CDSE credentials found. Set CDSE_CLIENT_ID and "
            "CDSE_CLIENT_SECRET as environment variables, or add a `cdse:` "
            "block with client_id/client_secret to config.local.yaml. "
            "Register an OAuth client at "
            "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings "
            "(see app/sentinel_hub.py module docstring for details)."
        )
    return client_id, client_secret


def _get_token(force_refresh: bool = False) -> str:
    """Return a valid OAuth2 access token, fetching/refreshing as needed."""
    now = time.time()
    if not force_refresh and _token_cache["access_token"] and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    import requests

    client_id, client_secret = _load_credentials()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise CDSEAuthError(
            f"CDSE token request failed ({resp.status_code}): {resp.text[:300]}"
        )
    payload = resp.json()
    _token_cache["access_token"] = payload["access_token"]
    _token_cache["expires_at"] = now + payload.get("expires_in", 600) - _TOKEN_SAFETY_MARGIN_S
    return _token_cache["access_token"]


def _cache_dir() -> Path:
    d = _repo_root() / "data" / "cache" / "sentinel2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(**params) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _to_wgs84_bbox(bbox: tuple[float, float, float, float], bbox_crs: str) -> tuple[float, float, float, float]:
    if bbox_crs.upper() in ("EPSG:4326", "CRS84", "OGC:CRS84"):
        return tuple(bbox)
    if transform_bounds is None:
        raise RuntimeError("rasterio is required to reproject bbox to WGS84")
    left, bottom, right, top = transform_bounds(bbox_crs, "EPSG:4326", *bbox)
    return (left, bottom, right, top)


def fetch_truecolor(
    bbox: tuple[float, float, float, float],
    date: str,
    bbox_crs: str = "EPSG:32629",
    *,
    window_days: int = 5,
    width: int = 512,
    height: int = 512,
    mosaicking_order: str = "leastCC",
    use_cache: bool = True,
) -> np.ndarray:
    """Fetch a rendered true-colour Sentinel-2 L2A chip for `bbox` near `date`.

    Parameters
    ----------
    bbox:
        (left, bottom, right, top) in `bbox_crs`.
    date:
        ISO date (YYYY-MM-DD) the chip should be centred on. The Process API
        mosaics over a window around this date (`window_days` either side)
        and picks the least-cloudy scene within it, rather than requiring an
        exact acquisition timestamp.
    bbox_crs:
        CRS of `bbox`. Defaults to the project's tile CRS (EPSG:32629);
        reprojected to WGS84 internally as the Process API expects CRS84.
    window_days:
        Half-width, in days, of the date window searched around `date`.
    width, height:
        Output image size in pixels (Process API caps at 2500x2500).
    mosaicking_order:
        "leastCC" (least cloud cover, default), "mostRecent", or "leastRecent".
    use_cache:
        If True (default), read/write a disk cache under
        `data/cache/sentinel2/` keyed by the request parameters.

    Returns
    -------
    np.ndarray, shape (height, width, 3), dtype uint8, RGB.
    Pixels with no valid Sentinel-2 coverage in the window are returned black.
    """
    from datetime import datetime, timedelta

    left, bottom, right, top = _to_wgs84_bbox(bbox, bbox_crs)
    center = datetime.fromisoformat(date)
    win_from = (center - timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00Z")
    win_to = (center + timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59Z")

    key_params = dict(
        bbox=(left, bottom, right, top),
        win_from=win_from,
        win_to=win_to,
        width=width,
        height=height,
        mosaicking_order=mosaicking_order,
        evalscript=EVALSCRIPT_TRUECOLOR,
    )
    cache_path = _cache_dir() / f"{_cache_key(**key_params)}.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path)

    import requests

    body = {
        "input": {
            "bounds": {
                "bbox": [left, bottom, right, top],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": win_from, "to": win_to},
                        "mosaickingOrder": mosaicking_order,
                    },
                }
            ],
        },
        "output": {"width": width, "height": height, "responses": [
            {"identifier": "default", "format": {"type": "image/tiff"}}
        ]},
        "evalscript": EVALSCRIPT_TRUECOLOR,
    }

    token = _get_token()
    resp = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=60,
    )
    if resp.status_code == 401:
        # Token may have just expired server-side; retry once with a fresh one.
        token = _get_token(force_refresh=True)
        resp = requests.post(
            PROCESS_URL,
            headers={"Authorization": f"Bearer {token}"},
            json=body,
            timeout=60,
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"CDSE Process API request failed ({resp.status_code}): {resp.text[:300]}"
        )

    import io
    import rasterio

    with rasterio.open(io.BytesIO(resp.content)) as src:
        arr = src.read()  # (4, H, W): R, G, B, dataMask

    rgb = np.clip(arr[:3].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    mask = arr[3] > 0
    rgb[~mask] = 0

    if use_cache:
        np.save(cache_path, rgb)
    return rgb


def bbox_from_pixel_window(transform, row_off: int, col_off: int, height: int, width: int) -> tuple[float, float, float, float]:
    """Convert a rasterio pixel window into a (left, bottom, right, top) bbox
    in the raster's native CRS, given its affine `transform`.

    Useful for turning a clicked pixel/zone in a TP/FP/FN raster (app.py,
    or notebooks/false_positive_review.ipynb) into a `bbox` argument for
    `fetch_truecolor`.
    """
    left, top = transform * (col_off, row_off)
    right, bottom = transform * (col_off + width, row_off + height)
    return (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
