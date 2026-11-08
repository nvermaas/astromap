"""
astromap2.py
=================
Query a REST API that returns a list of FITS URLs, then project those images
onto an all-sky map in one of two modes:

  --mode footprints : draw the outline (footprint) of each image on the sky
  --mode mosaic     : reproject the actual pixel data and blend them

Each FITS file was produced by astrometry.net, so the FITS header contains the
WCS. The pixel data can come from the FITS itself, or from the original JPG
(the FITS WCS is applied to the JPG pixels), which preserves full colour.

Background: an all-sky image in equirectangular (CAR / plate-carrée) layout,
i.e. 360° wide and 180° tall. If your background uses the geographic
convention (RA increasing to the right), pass --bg-flip.

Requirements:
    pip install astropy reproject matplotlib numpy requests pillow

Usage:
    python astromap2.py \
        --mode footprints \
        --background sky.jpg \
        --output astromap_footprints.png


background maps:
https://svs.gsfc.nasa.gov/4851/

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

DEBUG = False

# --------------------------------------------------------------------------
# 1. Query the REST API for the list of files
# --------------------------------------------------------------------------

def do_GET_LIST_page(query, page):
    pars = {}

    if page is not None:
        pars['page'] = page

    response = requests.get(query, timeout=60, params=pars)
    json_response = response.json()
    results = json_response["results"]

    if 'next' in json_response and json_response['next']:
        # return the next page number, if available
        next = json_response["next"]
        page_number = next.split('page=')[1].split('&')[0]
    else:
        page_number = 0

    return results, page_number


def fetch_file_list(query_url: str, fits_json_path, jpg_json_path) -> list[dict]:

    """GET the API and extract (fits_url, jpg_url) pairs."""
    next_page = 1
    data = []
    while next_page != 0:
        results, next_page = do_GET_LIST_page(query=query_url,page=next_page)
        data = data + results

    out = []
    for item in data:
        record = {}
        try:
            record['name'] = item['name']
            record['fits'] = item['derived_fits']
            record['jpg'] = item['derived_raw_image']
            record['quality'] = item['quality']
            record['field_fov'] = item['field_fov']
            out.append(record)
        except Exception:
            pass

    return out


def download(url: str, dest: Path, headers: dict | None = None) -> Path:
    """Stream a URL to a local file. Cached: skips if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f'downloading {url}')
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def url_to_name(url: str) -> str:
    try:
        name = os.path.basename(url.split("?")[0]) or "file"
    except Exception as error:
        return None

    return name


# --------------------------------------------------------------------------
# 2. Load an image + its WCS
# --------------------------------------------------------------------------

def find_wcs_hdu(hdul):
    """Return the HDU carrying the astrometry.net WCS (CTYPE starts with RA)."""
    for h in hdul:
        ctype = h.header.get("CTYPE1", "")
        if isinstance(ctype, str) and ctype.startswith("RA"):
            return h
    return hdul[0]


def load_image_and_wcs(fits_url: str, jpg_url: str | None,
                       workdir: Path, headers: dict | None):
    """Return (data, wcs, rgb).

    If a JPG is available it supplies the pixel data (full colour); the WCS
    comes from the FITS header, which astrometry.net solved for that JPG's
    orientation. Otherwise the FITS pixel data is used.
    """
    fits_file = download(fits_url, workdir / (url_to_name(fits_url)),headers)

    with fits.open(fits_file) as hdul:
        hdu = find_wcs_hdu(hdul)
        wcs = WCS(hdu.header, naxis=2)
        shape = hdu.data.shape if hdu.data is not None else (0, 0)

        if jpg_url:
            jpg_file = download(jpg_url, workdir / (url_to_name(jpg_url)),headers)
            data = np.asarray(Image.open(jpg_file).convert("RGB"))
            rgb = True
        else:
            data = hdu.data
            rgb = False

    # Normalise multi-plane FITS data to (H, W, 3)
    if not rgb and data is not None and data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
        rgb = True

    if data is None:
        raise ValueError("no pixel data in FITS and no JPG supplied")
    return data, wcs, rgb


# --------------------------------------------------------------------------
# 3. All-sky target grid (equirectangular / CAR, easy to reason about)
# --------------------------------------------------------------------------

def allsky_wcs(resolution: float = 0.25) -> WCS:
    """Plate-carrée world grid covering the whole sky.

    Reference point (RA 180, Dec 0) must lie on the equator for CAR.
    Pixel (0,0) = RA 180, Dec +90 (top-left); RA increases to the LEFT.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crpix = [1.0, 1.0 + 90.0 / resolution]
    w.wcs.crval = [180.0, 0.0]
    w.wcs.cdelt = [-resolution, -resolution]
    w.wcs.cunit = ["deg", "deg"]
    w.array_shape = (int(round(180 / resolution)), int(round(360 / resolution)))
    return w


# --------------------------------------------------------------------------
# 4. Footprints mode
# --------------------------------------------------------------------------

def footprint_vertices(wcs: WCS, n_edge: int = 13) -> np.ndarray:
    """Sky positions along the image border (corners + edge points, for
    fidelity on wide fields), as an (N, 2) RA/Dec array in degrees.

    IMPORTANT: the border is taken from the FITS pixel dimensions (the WCS
    describes those), NOT from the JPG's shape. astrometry.net often writes
    the solution for a frame whose size differs from the raw JPG, and using
    the JPG shape makes footprints overshoot badly.
    """
    if wcs.pixel_shape:
        nx, ny = (int(wcs.pixel_shape[0]), int(wcs.pixel_shape[1]))
    else:
        nx, ny = int(wcs._naxis[0]), int(wcs._naxis[1])
    t = np.linspace(0, 1, n_edge)
    border = []
    for side in range(4):
        if side == 0:   # top:    left -> right
            px, py = t * (nx - 1), np.zeros_like(t)
        elif side == 1: # right:  top -> bottom
            px, py = np.full_like(t, nx - 1), t * (ny - 1)
        elif side == 2: # bottom: right -> left
            px, py = t[::-1] * (nx - 1), np.full_like(t, ny - 1)
        else:          # left:   bottom -> top
            px, py = np.zeros_like(t), t[::-1] * (ny - 1)
        border.append(np.column_stack([px, py]))
    border = np.concatenate(border)
    return wcs.all_pix2world(border, 0)  # RA, Dec in degrees


def plot_footprints(items: list[dict], workdir: Path, headers: dict | None,
                    background: str | None, bg_flip: bool,
                    output: str, resolution: float,
                    default_facecolor="#22d3ee", default_edgecolor="#67e8f9"):
    """Footprint map on a plain (non-WCS) axes.

    The CAR grid is linear in RA/Dec, so we map degrees to pixels ourselves:
        x = (180 - RA) / resolution   (RA 180 at left, increasing leftward)
        y = (90 + Dec) / resolution   (Dec +90 at top)
    This avoids all WCSAxes longitude-wrap quirks (grid/ticks only being
    drawn over part of the map when world coords run 180 -> -180).
    """
    ny, nx = int(round(180 / resolution)), int(round(360 / resolution))

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()
    ax.set_facecolor("#000814")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    if background:
        bg = np.asarray(Image.open(background).convert("RGB"))
        if bg_flip:
            bg = bg[:, ::-1, :]
        # origin='upper': row 0 of the background (Dec +90) at the top.
        ax.imshow(bg, extent=(0, nx, 0, ny), origin="upper",
                  aspect="auto", zorder=0, interpolation="bilinear")

    n_ok = 0
    for item in items:
        try:
            _, wcs, _ = load_image_and_wcs(item["fits"], item["jpg"], workdir, headers)
        except Exception as e:
            print(f" skipped ({item['name']}...)")
            continue

        corners = footprint_vertices(wcs)
        lon = corners[:, 0] % 360.0
        lat = corners[:, 1]
        x = ((180.0 - lon) / resolution) % nx
        y = (90.0 + lat) / resolution

        edgecolor = default_edgecolor
        facecolor = default_facecolor
        zorder=2

        if item['quality'] == 'great':
            edgecolor = "#FFFF80"
            facecolor = "#EDEEB0"
            zorder=4

        elif item['quality'] == 'good':
            edgecolor = "#00C000"
            facecolor = "#C0FFC0"
            zorder=3
        elif item['quality'] == 'medium':
            edgecolor = "#C0C0C0"
            facecolor = "#E6E4E4"
            zorder=2

        ax.add_patch(Polygon(np.column_stack([x, y]), closed=True,
                             facecolor=facecolor, edgecolor=edgecolor,
                             alpha=0.1, lw=1.2, zorder=zorder))

        # Loud debug footprint: comment out once the map looks right.
        if DEBUG:
            ax.plot(x.mean(), y.mean(), "x", color="red", ms=10, zorder=11)

        n_ok += 1

    # Manual coordinate grid: RA ticks every 30° (as hours), Dec every 30°.
    ra_ticks = np.arange(0, 360, 30)
    ax.set_xticks([(180.0 - r) % 360 / resolution for r in ra_ticks])
    ax.set_xticklabels([f"{int(r / 15)}h" for r in ra_ticks], color="#94a3b8")
    dec_ticks = np.arange(-90, 91, 30)
    ax.set_yticks([(90.0 + d) / resolution for d in dec_ticks])
    ax.set_yticklabels([f"{int(d)}°" for d in dec_ticks], color="#94a3b8")
    ax.grid(color="#ffffff22", lw=0.4)
    ax.set_title(f"Footprints of {n_ok}/{len(items)} images")

    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="black")

    plt.close(fig)
    print(f"Footprint map with footprints written to {output}")
    #os.startfile(output)


# --------------------------------------------------------------------------
# 5. Mosaic mode
# --------------------------------------------------------------------------

def normalise_grey(d):
    d = d.astype(float)
    lo, hi = np.nanpercentile(d, 1), np.nanpercentile(d, 99.5)
    return np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)


def build_mosaic(items: list[dict], workdir: Path, headers: dict | None,
                 background: str | None, bg_flip: bool,
                 output: str, resolution: float):
    target_wcs = allsky_wcs(resolution)
    shape_out = target_wcs.array_shape

    inputs = []  # (data, wcs)
    for n, item in enumerate(items):
        try:
            data, wcs, rgb = load_image_and_wcs(
                item["fits"], item.get("jpg"), workdir, headers)
            if not rgb:
                data = normalise_grey(data)
            inputs.append((data, wcs))
        except Exception as e:
            print(f"  [{n}] skipped ({item['fits'][:70]}...): {e}", file=sys.stderr)

    if not inputs:
        sys.exit("No usable images.")
    print(f"Reprojecting {len(inputs)} images onto the all-sky grid ...")

    def coadd(plane_list):
        out, _ = reproject_and_coadd(
            plane_list, target_wcs, shape_out=shape_out,
            reproject_function=reproject_interp,
            match_background=False, roundtrip_coordinates=False)
        return out

    if inputs[0][0].ndim == 3:  # colour images (from JPGs)
        mosaic = np.stack(
            [coadd([(d[:, :, ch], w) for d, w in inputs]) for ch in range(3)],
            axis=-1)
        valid = np.isfinite(mosaic).any(axis=-1)
        mosaic = np.nan_to_num(mosaic, nan=0.0)
        mosaic = np.clip(mosaic / max(np.nanpercentile(mosaic, 99.5), 1.0), 0, 1)
    else:
        mosaic = coadd(inputs)
        valid = np.isfinite(mosaic)
        mosaic = np.nan_to_num(mosaic, nan=0.0)

    # Composite over the background all-sky image where we have no data
    if background:
        bg = np.asarray(Image.open(background).convert("RGB"), dtype=float) / 255.0
        if bg_flip:
            bg = bg[:, ::-1, :]
        bg = np.resize(bg, (shape_out[0], shape_out[1], 3))
        if mosaic.ndim == 2:
            for ch in range(3):
                layer = np.zeros_like(bg[:, :, ch]); layer[valid] = mosaic[valid]
                bg[:, :, ch] = layer + bg[:, :, ch] * (~valid)
            mosaic = bg
        else:
            mosaic[~valid] = bg[~valid]

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()   # plain axes: mosaic row 0 = Dec +90, shown at top
    ax.set_xlim(0, shape_out[1])
    ax.set_ylim(0, shape_out[0])
    ax.imshow(mosaic, extent=(0, shape_out[1], 0, shape_out[0]),
              origin="upper", zorder=0, aspect="auto")

    ra_ticks = np.arange(0, 360, 30)
    ax.set_xticks([(180.0 - r) % 360 / resolution for r in ra_ticks])
    ax.set_xticklabels([f"{int(r / 15)}h" for r in ra_ticks], color="#94a3b8")
    dec_ticks = np.arange(-90, 91, 30)
    ax.set_yticks([(90.0 + d) / resolution for d in dec_ticks])
    ax.set_yticklabels([f"{int(d)}°" for d in dec_ticks], color="#94a3b8")
    ax.grid(color="#ffffff33", lw=0.4)
    ax.set_title(f"Mosaic of {len(inputs)} images")
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Mosaic written to {output}")


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Project FITS URLs onto an all-sky image.")
    p.add_argument("--query",
                   default="https://uilennest.net/my_astrobase/observations2/?quality__in=great,good,medium&field_fov__lte=20",
                   help="REST API URL returning a JSON list (or wrapped list)")
    p.add_argument("--fits-json-path", default="fits",
                   help="Dotted JSON path to the FITS URL in each item "
                        "(ignored if the API returns bare URL strings)")
    p.add_argument("--jpg-json-path", default=None,
                   help="Dotted JSON path to the original JPG URL in each item")
    p.add_argument("--auth-header", default=None,
                   help='Header to send, e.g. "Authorization: Bearer xyz"')
    p.add_argument("--mode", choices=("footprints", "mosaic"), default="footprints")
    p.add_argument("--background", default=None,
                   help="All-sky JPG/PNG in equirectangular (CAR) layout")
    p.add_argument("--bg-flip", action="store_true",
                   help="Flip background horizontally (RA increases right)")
    p.add_argument("--resolution", type=float, default=0.25,
                   help="Output grid resolution in deg/pixel (default 0.25)")
    p.add_argument("--cache-dir", default="./astromap_cache",
                   help="Directory for downloaded FITS/JPG files (reused between runs)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    headers = None
    if args.auth_header:
        k, _, v = args.auth_header.partition(":")
        headers = {k.strip(): v.strip()}

    print("Querying API ...")
    items = fetch_file_list(args.query, args.fits_json_path, args.jpg_json_path)

    if DEBUG:
        items = [
            {"fits": "https://woodland.uilennest.net/astrobase/data/240213005/9934006.fits",
             "jpg" : "https://woodland.uilennest.net/astrobase/data/240213005/240213005_raw.jpg"
            }
        ]

    print(f"Got {len(items)} entries.")
    if not items:
        sys.exit(1)

    workdir = Path(args.cache_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    output = args.output or f"astromap_{args.mode}.png"

    if args.mode == "footprints":
        plot_footprints(items, workdir, headers, args.background,
                        args.bg_flip, output, args.resolution)
    else:
        build_mosaic(items, workdir, headers, args.background,
                     args.bg_flip, output, args.resolution)


if __name__ == "__main__":
    main()