#!/usr/bin/env python
"""
batch_optimise_v11.py  —  Batch Flight Trajectory Optimisation -> BlueSky Scenario

Reads a CSV flight list, optimises each flight for minimum cruise fuel using
BADA3 performance data and (optionally) local ERA5 wind fields, then exports
the best results as BlueSky .scn scenario files.

Changes vs v9:
  - **Wind interpolant caching**: for ``--wind-method linear`` or ``bspline``,
    wind interpolants are pre-built **once per departure-hour bucket** and
    saved to a cache directory (``<output>/wind_cache/``).  Workers load
    the cached BSplineWind objects from disk in < 0.5 s instead of
    rebuilding each flight's interpolant from the raw DataFrame (~5–30 s).
    This eliminates:
      (1) pickling a multi-GB wind DataFrame through the process pipe;
      (2) per-flight BSplineWind construction overhead;
      (3) massive per-worker memory (workers hold only the lightweight
          CasADi Function objects, not the raw DataFrame).
  - **Reduced memory footprint**: workers no longer receive the raw
    wind DataFrame.  For ``poly`` wind method (not cacheable) the old
    temp-file approach is kept.

Changes vs v8:
  - **FLST-based initial guess**: new ``--flst-log PATH`` argument.
    When provided, the script reads a BlueSky FLST log file (compact or
    full format) and reconstructs initial guesses from the simulated
    reference flights.  For each flight:
      (1) extract the trajectory from the FLST log;
      (2) trim 120 NM from origin and 120 NM from destination
          (using ``distanceflown``) to isolate the cruise segment;
      (3) resample to ``nodes + 1`` evenly-spaced (by distance)
          waypoints;
      (4) use the trim-point coordinates as optimizer origin/destination
          (``(lat, lon)`` tuples passed to ``top.Cruise()``);
      (5) generate N cross-track variations using the same ``--n-tracks``
          and ``--max-dev`` parameters as GC-based guesses.
    Flights missing from the FLST log automatically fall back to the
    GC-based initial guess.
  - **Option A cruise-segment optimisation**: when FLST is used, the
    optimizer's origin and destination are the 120 NM trim points
    (not the airports).  This means the NLP only optimises the cruise
    phase, and boundary conditions match the actual cruise endpoints.
    Scenario export still uses ICAO codes from the flight list CSV.

Changes vs v6:
  - **120 NM trim for optimised scenario**: the cruise-start scenario now
    trims the first 120 NM of the optimised trajectory.  The aircraft is
    CREated at the trim waypoint (the first point where cumulative GC
    distance from WP0 exceeds 120 NM) with ``SETMASS = trim_mass_kg``
    (the mass at that trim point).
  - **Aligned baseline scenario**: new ``--baseline-scenario PATH``
    argument.  When provided, the script reads the baseline .scn file
    and aligns it flight-by-flight as each GC optimisation completes:
      (1) find the baseline waypoint where altitude first reaches or
          exceeds ``trim_alt_ft`` (from the optimised GC variant's
          120 NM trim point);
      (2) modify the CRE line to spawn at that altitude;
      (3) add SETMASS with ``trim_mass_kg`` (same mass as the optimised
          flight at its trim point, ensuring apples-to-apples comparison);
      (4) remove all ADDWPT lines before the baseline trim point;
      (5) write the result to ``scenario_baseline_aligned.scn``.
  - This eliminates the fuel-comparison bias where the baseline burns
    high-thrust climb fuel for the first ~180 NM while the optimised
    flight is already cruising at low fuel flow.

Changes vs v5:
  - **No-wind mode**: added ``--no-wind`` boolean flag.  When set, the
    optimiser runs WITHOUT wind data — ``opt.enable_wind()`` is never
    called and no ERA5 data is loaded.  To disable wind in BlueSky
    simulation as well, simply do not load the windecmwfup plugin.
  - ``--wind-dir`` is no longer required when ``--no-wind`` is active.
  - ``"no_wind": True/False`` is recorded in all metadata files.
  - **BADA3 synonym detection**: when ``--perf-model bada3`` is used,
    synonym detection now checks the BADA3 ``SYNONYM.NEW`` file instead
    of OpenAP's synonym database.  The log message shows which BADA3
    equivalent type is used (e.g. ``B731 uses BADA3 synonym -> T134``).

Changes vs v4:
  - **Wind method selection**: added ``--wind-method`` argument with three
    choices: ``poly`` (legacy PolyWind polynomial regression), ``linear``
    (BSplineWind with 4-D multilinear CasADi interpolation), or ``bspline``
    (BSplineWind with 4-D B-spline CasADi interpolation).
  - **BSpline parameters**: added ``--bspline-degree`` and
    ``--bspline-subsample`` arguments for fine-tuning when using the
    ``bspline`` wind method.
  - Wind method is recorded in all metadata files for reproducibility.

Changes vs v3:
  - **ProcessPoolExecutor**: switched from ThreadPoolExecutor to
    ProcessPoolExecutor for true multi-core flight-level parallelism
    (bypasses the Python GIL).
  - **Single optimizer per flight**: the optimizer is created once per
    flight and reused across all track variations, eliminating redundant
    PolyWind fitting, BADA3 loading, and NLP setup.
  - **Flight-level parallelism**: the main loop submits whole flights to
    the process pool instead of parallelising tracks within a flight.
  - **IPOPT tuning**: ``hessian_approximation`` set to ``exact`` and
    ``mu_strategy`` set to ``adaptive`` for faster convergence.
  - **Wind resolution CLI**: added ``--wind-resolution`` argument for
    future use when switching to spline-based wind representation.

Changes vs v2:
  - **Resume capability**: skips flights whose trajectories already exist in
    the output folder.  Allows safe restart after interruptions.
  - **Synonym flag**: when the optimiser uses OpenAP synonym data for an
    aircraft type, a flag is stored in the metadata so those flights can
    be excluded in downstream analysis.
  - **Failed-flight log**: flights that fail optimisation are collected in
    ``failed_flights.csv`` for review.
  - **Dual scenario output**:
      (1) ``scenario_all.scn``   — all successful flights (incl. synonym)
      (2) ``scenario_wosyn.scn`` — excludes flights that used synonym data
  - **Cruise-start scenario**: optional third scenario
      ``scenario_cruisestart.scn`` where aircraft are CREated at the
      120 NM trim waypoint with the optimiser's mass.
  - **BADA3 climb/descent params**: ``_climb_descent_params`` now reads
    MTOW from the optimiser's aircraft dict (BADA3 source) rather than
    calling ``oa.prop.aircraft()`` which may fail for synonym types.
  - **Encoding**: scenario files written as UTF-8.

Key features (inherited from v2):
  - Local ERA5 NetCDF wind at native 0.25deg (no downsampling)
  - Wind time-shifted per flight (nearest-hour rounding)
  - BADA3 performance model
  - Multi-start optimisation with N initial-guess variants (GC +/- dev NM)
  - Dynamic node count (~1 per 50 km)
  - Flight-level parallel optimisation (ProcessPoolExecutor)
  - Synthetic climb / descent phases stitched to cruise-optimised trajectory
  - CAS (not TAS) in the exported BlueSky scenario
  - Best-K selection per flight for scenario export
  - Full metadata saved for reproducibility

Usage:
    # With wind + baseline alignment:
    python batch_optimise_v7.py \\
        --csv       C:/data/input/flights.csv \\
        --output    C:/data/output \\
        --wind-dir  C:/data/NetCDF \\
        --bada-path C:/openap-top-nat/top/perf/data \\
        --baseline-scenario C:/data/baseline.scn \\
        --wind-method bspline \\
        --workers   6 \\
        --n-tracks  7 \\
        --max-dev   500 \\
        --top-k     3 \\
        --max-iter  10000 \\
        --m0        0.85

    # Without wind:
    python batch_optimise_v7.py \\
        --csv       C:/data/input/flights.csv \\
        --output    C:/data/output \\
        --bada-path C:/openap-top-nat/top/perf/data \\
        --no-wind \\
        --workers   6 \\
        --n-tracks  7 \\
        --max-dev   500 \\
        --top-k     3 \\
        --max-iter  10000 \\
        --m0        0.85

Author : Nils Ahrenhold
Date   : 2026-03-25
Method : Cruise (single-phase NLP) + synthetic climb/descent
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import glob
import json
import os
import pickle
import re
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import psutil
import xarray as xr
from scipy.interpolate import CubicSpline

import openap as oa
from openap.extra.aero import fpm, ft, kts
import top

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
R = 287.05287
P0 = 101325.0
RHO0 = 1.225
T0 = 288.15
T_STRAT = 216.65
KTS_TO_MPS = 0.514444

TRIM_DISTANCE_NM = 120  # trim the first 120 NM for aligned comparison

PRESSURE_LEVELS_HPA = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
    275, 250, 225, 200, 175, 150, 125, 100,
]

MAX_FLIGHT_HOURS = 12
BUFFER_HOURS = 2

from shapely.geometry import Polygon, Point

NAT_POLYGON = Polygon([
    (-40.00925832249236, 22.30088782734948),
    (-37.50060476539184, 17.00114422180141),
    (-25.00445813799347, 23.9967312645047),
    (-25.00277008528403, 29.98685181855245),
    (-20.00051796528021, 29.99925061493726),
    (-17.41057029269978, 31.66499222709419),
    (-17.76048752503626, 34.24826638500311),
    (-14.99046243016591, 36.49982321426099),
    (-14.99639301990188, 42.00393036461706),
    (-12.99620991319748, 43.00497554551404),
    (-12.99925002993084, 44.9998639207808),
    (-8.001291915837621, 44.99904041719872),
    (-7.998472481308046, 50.99923092356601),
    (-14.98999978329783, 51.0028555788159),
    (-15.00180192348943, 53.98332778890028),
    (-10.00327405594715, 54.5559428646509),
    (-10.00262189787093, 60.998093450498),
    (-1.317824825264324e-05, 61.00029433260888),
    (-0.01724651861566229, 63.2503471328358),
    (4.046472300275676, 63.06045468162895),
    (4.987379484947743, 63.99914230975799),
    (6.265076952139397, 65.12918788878895),
    (6.833772608291644, 65.61340851165787),
    (6.984561958489692, 65.74566596574975),
    (7.707695273207458, 66.20102780089174),
    (9.428676143219747, 67.24231528906907),
    (15.00261281712878, 69.99726890196288),
    (17.99198283732568, 70.4759127250349),
    (24.99119553169029, 71.32185552307375),
    (27.98409701190277, 71.33232591860843),
    (30.00700316541029, 70.99985084500078),
    (30.02169031582981, 82.007233348761),
    (30.10381011810755, 87.00022924654341),
    (-60.03008490708459, 87.00029074992395),
    (-60.01327280617593, 81.9972185613033),
    (-74.99861533070624, 78.00083646090961),
    (-75.99219791754545, 76.00472620160922),
    (-57.74993862590701, 64.99937131613427),
    (-60.00733730663995, 64.9983687677333),
    (-62.99285667579223, 64.00221383453234),
    (-63.00663454562922, 61.00283600756163),
    (-60.370605115987, 58.47283641663137),
    (-58.98385834101614, 56.98665515561235),
    (-54.00133035219821, 52.97956144905517),
    (-51.00784938902067, 49.01011431030255),
    (-51.00387154546799, 45.00164163584257),
    (-66.99944965577443, 41.78837240746002),
    (-67.00023380910919, 39.99775940988052),
    (-68.99621205025304, 38.49754062312058),
    (-60.00574329824699, 38.50342688670094),
    (-59.99861842829986, 27.0050329259516),
    (-40.0061046916277, 27.00089726858077),
    (-40.00925832249236, 22.30088782734948),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    mem = psutil.Process().memory_info().rss / 1024 / 1024
    pid = os.getpid()
    print(f"[{ts}] [{level:>5}] [{mem:6.0f} MB] [P{pid}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════
def clear_memory() -> None:
    log("Clearing memory ...")
    for _ in range(3):
        gc.collect()
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass
    log("Memory cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ATMOSPHERE & CAS CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════
def _vatmos(h: np.ndarray):
    T = np.maximum(T0 - 0.0065 * h, T_STRAT)
    rho_trop = RHO0 * (T / T0) ** 4.256848030018761
    dh_strat = np.maximum(0.0, h - 11_000.0)
    rho = rho_trop * np.exp(-dh_strat / 6341.552161)
    p = rho * R * T
    return p, rho, T


def _vtas2cas(tas: np.ndarray, h: np.ndarray) -> np.ndarray:
    p, rho, _ = _vatmos(h)
    qdyn = p * ((1.0 + rho * tas * tas / (7.0 * p)) ** 3.5 - 1.0)
    cas = np.sqrt(7.0 * P0 / RHO0 * ((qdyn / P0 + 1.0) ** (2.0 / 7.0) - 1.0))
    return np.where(tas < 0, -cas, cas)


def tas_kn_to_cas_kn(tas_kn: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    return _vtas2cas(tas_kn * KTS_TO_MPS, h_m) / kts


# ═══════════════════════════════════════════════════════════════════════════════
#  GREAT-CIRCLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def get_airport_coords(icao: str) -> Tuple[float, float]:
    info = oa.nav.airport(icao)
    if info is None:
        raise ValueError(f"Airport '{icao}' not found in OpenAP database")
    return float(info["lat"]), float(info["lon"])


def gc_distance_nm_coords(lat1, lon1, lat2, lon2) -> float:
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 3440.065 * 2 * np.arcsin(np.sqrt(a))


def gc_distance_nm(origin: str, destination: str) -> float:
    return gc_distance_nm_coords(*get_airport_coords(origin), *get_airport_coords(destination))


def gc_intermediate_point(lat1, lon1, lat2, lon2, fraction):
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    d = 2 * np.arcsin(np.sqrt(
        np.sin((lat2_r - lat1_r) / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin((lon2_r - lon1_r) / 2) ** 2
    ))
    if d < 1e-10:
        return lat1, lon1
    a = np.sin((1 - fraction) * d) / np.sin(d)
    b = np.sin(fraction * d) / np.sin(d)
    x = a * np.cos(lat1_r) * np.cos(lon1_r) + b * np.cos(lat2_r) * np.cos(lon2_r)
    y = a * np.cos(lat1_r) * np.sin(lon1_r) + b * np.cos(lat2_r) * np.sin(lon2_r)
    z = a * np.sin(lat1_r) + b * np.sin(lat2_r)
    return np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2))), np.degrees(np.arctan2(y, x))


def _compute_bearing(lat1, lon1, lat2, lon2) -> float:
    """Compute initial bearing from (lat1, lon1) to (lat2, lon2) in degrees [0, 360)."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return float(bearing % 360)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY TRIM  (trim first N NM for aligned comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def _find_trim_point(traj_df: pd.DataFrame,
                     trim_nm: float = TRIM_DISTANCE_NM) -> dict | None:
    """Find the first waypoint where cumulative GC distance from WP0 > *trim_nm*.

    Returns a dict with trim-point metadata, or ``None`` if the trajectory
    is shorter than *trim_nm*.

    Columns expected in *traj_df*: latitude, longitude, altitude (ft),
    mass (kg), cas (kts), heading (deg).
    """
    if len(traj_df) < 3:
        return None

    lat0 = float(traj_df["latitude"].iloc[0])
    lon0 = float(traj_df["longitude"].iloc[0])

    for i in range(1, len(traj_df)):
        cum_dist = gc_distance_nm_coords(
            lat0, lon0,
            float(traj_df["latitude"].iloc[i]),
            float(traj_df["longitude"].iloc[i]),
        )
        if cum_dist >= trim_nm:
            hdg = 90.0
            if "heading" in traj_df.columns:
                hdg = float(traj_df["heading"].iloc[i])
            elif i + 1 < len(traj_df):
                hdg = _compute_bearing(
                    float(traj_df["latitude"].iloc[i]),
                    float(traj_df["longitude"].iloc[i]),
                    float(traj_df["latitude"].iloc[i + 1]),
                    float(traj_df["longitude"].iloc[i + 1]),
                )
            return {
                "trim_index": i,
                "trim_lat": float(traj_df["latitude"].iloc[i]),
                "trim_lon": float(traj_df["longitude"].iloc[i]),
                "trim_alt_ft": float(traj_df["altitude"].iloc[i]),
                "trim_mass_kg": float(traj_df["mass"].iloc[i]),
                "trim_cas": float(traj_df["cas"].iloc[i]) if "cas" in traj_df.columns else 280.0,
                "trim_heading": hdg,
                "trim_ts": float(traj_df["ts"].iloc[i]),
                "cumulative_dist_nm": cum_dist,
            }

    # Trajectory shorter than trim distance
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PRESSURE -> HEIGHT
# ═══════════════════════════════════════════════════════════════════════════════
def pressure_to_height_m(level_hpa: float) -> float:
    p_pa = level_hpa * 100.0
    return (1.0 - (p_pa / 101325.0) ** 0.190264) * 44330.76923


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNONYM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def _detect_openap_synonym(actype: str) -> Tuple[bool, str | None]:
    """
    Check whether OpenAP uses synonym data for *actype*.

    Returns (used_synonym: bool, mapped_to: str | None).
    """
    try:
        oa.prop.aircraft(actype, use_synonym=False)
        return False, None
    except Exception:
        pass
    # Type not natively available — check if synonym resolves it
    try:
        ac = oa.prop.aircraft(actype, use_synonym=True)
        return True, actype
    except Exception:
        return False, None


def _detect_bada3_synonym(actype: str, bada_path: str) -> Tuple[bool, str | None]:
    """
    Check whether the BADA3 SYNONYM.NEW file maps *actype* to a
    different (equivalent) aircraft type's OPF data.

    Returns (used_synonym: bool, mapped_to: str | None).
    """
    try:
        from openap.addon.bada3 import read_synonym
        info = read_synonym(actype, bada_path)
        synonym_code = info["synonym"].strip().upper()
        if synonym_code != actype.strip().upper():
            return True, synonym_code
        return False, None
    except (ValueError, FileNotFoundError, ImportError, Exception):
        return False, None


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def load_flight_list(csv_path: str, m0: float = 0.85, sep: str = ";") -> Tuple[pd.DataFrame, list]:
    log(f"Loading flight list from {csv_path}")
    df = pd.read_csv(csv_path, sep=sep)
    required = {"flightid", "callsign", "actype", "origin", "destination", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["dep_time"] = pd.to_datetime(df["time"])

    flights_info = []
    for _, r in df.iterrows():
        flights_info.append({
            "flightid": str(r["flightid"]).strip(),
            "callsign": str(r["callsign"]).strip(),
            "actype": str(r["actype"]).strip(),
            "origin": str(r["origin"]).strip(),
            "destination": str(r["destination"]).strip(),
            "m0": m0,
            "dep_time": r["dep_time"].strftime("%Y-%m-%d %H:%M:%S"),
        })
    log(f"Loaded {len(flights_info)} flights")
    return df, flights_info


# ═══════════════════════════════════════════════════════════════════════════════
#  RESUME: check which flights already have trajectories
# ═══════════════════════════════════════════════════════════════════════════════
def _find_completed_flights(traj_dir: str) -> set:
    """
    Scan *traj_dir* for existing trajectory pickles and return the set of
    flightids that already have at least one optimised result.

    Naming convention: ``<flightid>_<trackname>_traj.pkl``
    """
    completed = set()
    if not os.path.isdir(traj_dir):
        return completed
    for p in glob.glob(os.path.join(traj_dir, "*_traj.pkl")):
        fname = os.path.basename(p)
        # flightid is everything before the last two underscore-separated tokens
        # e.g.  "42_gc_traj.pkl"  -> flightid = "42"
        #        "42_n500_traj.pkl" -> flightid = "42"
        parts = fname.replace("_traj.pkl", "").rsplit("_", 1)
        if len(parts) >= 2:
            completed.add(parts[0])
    return completed


def _load_existing_results(traj_dir: str, flightid: str) -> dict | None:
    """
    Reconstruct a result dict from persisted trajectory pickles and metadata
    JSONs for *flightid* so that scenario building can include them.
    """
    pattern = os.path.join(traj_dir, f"{flightid}_*_traj.pkl")
    pkl_files = sorted(glob.glob(pattern))
    if not pkl_files:
        return None

    results = []
    used_synonym = False
    for pkl in pkl_files:
        try:
            with open(pkl, "rb") as f:
                traj_df = pickle.load(f)
            # Load companion metadata
            json_path = pkl.replace("_traj.pkl", "_metadata.json")
            meta = {}
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    meta = json.load(f)
            ident = meta.get("identifier", os.path.basename(pkl).replace("_traj.pkl", ""))
            name = ident.replace(f"{flightid}_", "", 1)
            fuel = traj_df["mass"].iloc[0] - traj_df["mass"].iloc[-1] if len(traj_df) > 0 else 0
            if meta.get("used_synonym", False):
                used_synonym = True
            results.append({
                "success": True,
                "identifier": ident,
                "name": name,
                "trajectory": traj_df,
                "total_fuel": fuel,
                "nat_fuel": meta.get("nat_fuel", 0),
                "nat_distance": meta.get("nat_distance", 0),
                "deviation_nm": meta.get("deviation_nm", 0),
                "setup_s": 0,
                "opt_s": 0,
                "total_s": 0,
            })
        except Exception as exc:
            log(f"  [{flightid}] Could not reload {pkl}: {exc}", "WARN")

    if not results:
        return None

    results.sort(key=lambda x: x["total_fuel"])
    return {
        "success": True,
        "flightid": flightid,
        "callsign": "",
        "results": results,
        "best": results[0],
        "m0": 0,
        "mass_est": {},
        "nodes": 0,
        "error": None,
        "elapsed_s": 0,
        "used_synonym": used_synonym,
        "resumed": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  WEATHER DATA (LOCAL ERA5 NetCDF -- native 0.25deg resolution)
# ═══════════════════════════════════════════════════════════════════════════════
def _weather_time_range(flights_info, max_hours=MAX_FLIGHT_HOURS, buf=BUFFER_HOURS):
    dep_times = [pd.to_datetime(f["dep_time"]) for f in flights_info]
    start = min(dep_times).replace(minute=0, second=0, microsecond=0)
    end = (max(dep_times) + timedelta(hours=max_hours + buf)).replace(minute=0, second=0, microsecond=0)
    return start, end, pd.date_range(start, end, freq="1h")


def load_wind_from_local_netcdf(
    flights_info: list,
    netcdf_dir: str,
    flight_phase: str = "cruise",
) -> Tuple[pd.DataFrame | None, pd.Timestamp | None]:
    """Load ERA5 wind at native 0.25deg resolution."""
    log("Loading ERA5 wind data from local NetCDF (native 0.25deg) ...")

    airports = {f["origin"] for f in flights_info} | {f["destination"] for f in flights_info}
    lats, lons = [], []
    for icao in airports:
        info = oa.nav.airport(icao)
        if info is None:
            log(f"Airport '{icao}' not found!", "ERROR")
            return None, None
        lats.append(float(info["lat"]))
        lons.append(float(info["lon"]))

    latmin, latmax = min(lats) - 2, max(lats) + 2
    lonmin, lonmax = min(lons) - 2, max(lons) + 2
    log(f"  Geographic bounds: lat({latmin:.1f}, {latmax:.1f}), lon({lonmin:.1f}, {lonmax:.1f})")

    weather_start, weather_end, timestamps = _weather_time_range(flights_info)
    log(f"  Time range: {weather_start} -> {weather_end} ({len(timestamps)} hours)")

    dates_needed = sorted({ts.strftime("%Y%m%d") for ts in timestamps})
    log(f"  NetCDF dates needed: {dates_needed}")

    datasets = []
    for d in dates_needed:
        nc = os.path.join(netcdf_dir, f"p_levels_{d}.nc")
        if not os.path.exists(nc):
            log(f"File not found: {nc}", "ERROR")
            return None, None
        log(f"    Loading {nc}")
        datasets.append(xr.open_dataset(nc))

    time_coord = "valid_time" if "valid_time" in datasets[0].coords else "time"
    ds = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim=time_coord)

    lat_c = "latitude" if "latitude" in ds.coords else "lat"
    lon_c = "longitude" if "longitude" in ds.coords else "lon"
    lev_c = "pressure_level" if "pressure_level" in ds.coords else "level"
    u_var = "u" if "u" in ds.data_vars else "u_component_of_wind"
    v_var = "v" if "v" in ds.data_vars else "v_component_of_wind"

    native_lats = ds[lat_c].values
    native_res = abs(native_lats[1] - native_lats[0]) if len(native_lats) > 1 else 0.25
    log(f"  Native resolution: {native_res:.4f}deg -- using as-is")

    lat_slice = slice(latmax, latmin) if native_lats[0] > native_lats[-1] else slice(latmin, latmax)
    lon_vals = ds[lon_c].values
    if lon_vals.max() > 180:
        ds = ds.assign_coords(**{lon_c: (ds[lon_c].values + 180) % 360 - 180}).sortby(lon_c)

    ds_sub = ds.sel(**{
        lat_c: lat_slice,
        lon_c: slice(lonmin, lonmax),
        time_coord: slice(np.datetime64(weather_start), np.datetime64(weather_end)),
    })

    for d in datasets:
        d.close()
    del ds, datasets
    gc.collect()

    log(f"  Subset dims: {dict(ds_sub.dims)}")

    avail = ds_sub[lev_c].values
    if flight_phase == "cruise":
        levels = [l for l in avail if 100 <= l <= 400]
        log(f"  Pressure levels: cruise (100-400 hPa) -> {len(levels)} levels")
    elif flight_phase == "full":
        levels = list(avail)
        log(f"  Pressure levels: full -> {len(levels)} levels")
    else:
        levels = list(avail)

    if levels:
        ds_sub = ds_sub.sel(**{lev_c: levels})

    log("  Extracting wind vectors ...")
    times_nc = pd.to_datetime(ds_sub[time_coord].values)
    levels_nc = ds_sub[lev_c].values
    lats_nc = ds_sub[lat_c].values
    lons_nc = ds_sub[lon_c].values
    n_t, n_l, n_la, n_lo = len(times_nc), len(levels_nc), len(lats_nc), len(lons_nc)
    total = n_t * n_l * n_la * n_lo
    log(f"  Grid: {n_t}t x {n_l}lev x {n_la}lat x {n_lo}lon = {total:,} points")

    u_data = ds_sub[u_var].values
    v_data = ds_sub[v_var].values
    ds_sub.close()
    del ds_sub
    gc.collect()

    level_heights = np.array([pressure_to_height_m(l) for l in levels_nc])
    time_offsets = np.array([(t - times_nc[0]).total_seconds() for t in times_nc])

    ts_m, h_m, lat_m, lon_m = np.meshgrid(time_offsets, level_heights, lats_nc, lons_nc, indexing="ij")

    df_wind = pd.DataFrame({
        "ts": ts_m.ravel(),
        "h": h_m.ravel(),
        "latitude": lat_m.ravel(),
        "longitude": lon_m.ravel(),
        "u": u_data.ravel(),
        "v": v_data.ravel(),
    }).sort_values(["ts", "h", "latitude", "longitude"]).reset_index(drop=True)

    del u_data, v_data, ts_m, h_m, lat_m, lon_m
    gc.collect()

    mem_mb = df_wind.memory_usage(deep=True).sum() / 1024 ** 2
    log(f"  Wind data ready: {len(df_wind):,} records ({mem_mb:.1f} MB)")
    return df_wind, weather_start


# ═══════════════════════════════════════════════════════════════════════════════
#  MASS ESTIMATION  (BADA3 analytical climb integration)
# ═══════════════════════════════════════════════════════════════════════════════
def _isa_density(h_m: float) -> float:
    import math
    g0 = 9.80665
    if h_m <= 11_000:
        T = T0 - 0.0065 * h_m
        return RHO0 * (T / T0) ** (g0 / (0.0065 * R) - 1)
    else:
        rho_11 = RHO0 * (T_STRAT / T0) ** (g0 / (0.0065 * R) - 1)
        return rho_11 * math.exp(-g0 * (h_m - 11_000) / (R * T_STRAT))


def _bada3_climb_fuel(optimizer, takeoff_mass: float, target_alt_ft: float = None,
                      n_steps: int = 100) -> dict:
    """
    Estimate climb fuel from 1 500 ft to *target_alt_ft* using BADA3 thrust,
    drag, and fuel-flow coefficients via numerical integration of the TEM.
    """
    import math
    g0 = 9.80665
    data = optimizer.aircraft
    limits = data.get("limits", {})
    ceiling_m = limits.get("ceiling", data.get("ceiling", 13_000))
    ceiling_ft = ceiling_m / ft

    if target_alt_ft is None:
        # Fixed reasonable cruise entry altitude (FL200 = 20 000 ft).
        # This is the altitude we assume the aircraft climbs to before
        # starting cruise.  The climb fuel is estimated from ground to
        # this altitude, giving us the cruise entry mass.  The same
        # value is used as h_min in the cruise optimizer so the NLP
        # cannot descend below the altitude the aircraft was "delivered"
        # to by the climb phase.
        target_alt_ft = 20_000

    Ct = Cf = S = cd0 = k_drag = None
    engine_type = "turbofan"

    if hasattr(optimizer, "thrust") and hasattr(optimizer.thrust, "_Ct0"):
        thr = optimizer.thrust
        Ct = [thr._Ct0, thr._Ct1, thr._Ct2]
        engine_type = getattr(thr, "engine_type", "turbofan")

    if hasattr(optimizer, "fuelflow") and hasattr(optimizer.fuelflow, "_Cf1"):
        ff_a = optimizer.fuelflow
        Cf = [ff_a._Cf1, ff_a._Cf2]

    if hasattr(optimizer, "drag") and hasattr(optimizer.drag, "_cd0_cr"):
        drg = optimizer.drag
        cd0 = drg._cd0_cr
        k_drag = drg._k_cr
        S = drg._S

    if Ct is None or Cf is None or cd0 is None:
        mtow = limits.get("MTOW", data.get("mtow", 300_000))
        if mtow > 250_000:
            fb = 10_000
        elif mtow > 150_000:
            fb = 7_000
        else:
            fb = 4_000
        return {"climb_fuel_kg": fb, "method": "category_fallback",
                "target_alt_ft": target_alt_ft}

    Ct0, Ct1, Ct2 = Ct
    Cf1, Cf2 = Cf

    h_start_ft = 1_500.0
    dh_ft = (target_alt_ft - h_start_ft) / n_steps

    mass = takeoff_mass
    total_fuel = 0.0
    total_time_s = 0.0

    for i in range(n_steps):
        h_ft = h_start_ft + (i + 0.5) * dh_ft
        h_m = h_ft * ft
        T_isa = (T0 - 0.0065 * h_m) if h_m <= 11_000 else T_STRAT
        a = math.sqrt(1.4 * R * T_isa)
        rho = _isa_density(h_m)

        if h_ft < 10_000:
            tas_kts = 250.0
        elif h_ft < 28_000:
            tas_kts = 300.0
        else:
            tas_kts = 0.82 * a / kts
        tas_ms = tas_kts * kts

        if engine_type == "turbofan":
            T_climb = Ct0 * (1.0 - h_ft / Ct1 + Ct2 * h_ft ** 2)
        elif engine_type == "turboprop":
            T_climb = Ct0 / max(tas_kts, 1.0) * (1.0 - h_ft / Ct1) + Ct2
        else:
            T_climb = Ct0 * (1.0 - h_ft / Ct1 + Ct2 * h_ft ** 2)

        W = mass * g0
        q = 0.5 * rho * tas_ms ** 2
        CL = W / max(q * S, 1e-6)
        CD = cd0 + k_drag * CL ** 2
        D = q * S * CD
        net = max(T_climb - D, 0.05 * T_climb)
        f_esf = 1.25 if h_ft < 28_000 else 1.0
        roc_ms = net * tas_ms / (mass * g0 * f_esf)
        roc_ftmin = max(roc_ms / fpm, 200.0)
        eta = Cf1 * (1.0 + tas_kts / max(Cf2, 1.0)) * 1e-3
        ff_kgmin = eta * T_climb
        dt_min = abs(dh_ft) / roc_ftmin
        step_fuel = ff_kgmin * dt_min
        total_fuel += step_fuel
        total_time_s += dt_min * 60
        mass -= step_fuel

    return {
        "climb_fuel_kg": total_fuel,
        "climb_time_s": total_time_s,
        "target_alt_ft": target_alt_ft,
        "method": "bada3_integration",
        "engine_type": engine_type,
    }


def estimate_cruise_mass(origin, destination, optimizer, load_factor=0.85):
    limits = optimizer.aircraft.get("limits", {})
    mtow = limits.get("MTOW", optimizer.aircraft.get("mtow", 300_000))
    oew = optimizer.oew
    dist_nm = gc_distance_nm(origin, destination)
    takeoff_mass = mtow * load_factor

    climb_info = _bada3_climb_fuel(optimizer, takeoff_mass)
    climb_fuel = climb_info["climb_fuel_kg"]

    cruise_entry = takeoff_mass - climb_fuel
    return {
        "distance_nm": dist_nm,
        "mtow": mtow,
        "oew": oew,
        "load_factor": load_factor,
        "takeoff_mass": takeoff_mass,
        "climb_fuel_kg": climb_fuel,
        "climb_estimation_method": climb_info.get("method", "unknown"),
        "climb_target_alt_ft": climb_info.get("target_alt_ft"),
        "climb_time_s": climb_info.get("climb_time_s"),
        "cruise_entry_mass": cruise_entry,
        "m0_estimate": cruise_entry / mtow,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NAT FUEL CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_nat_fuel(traj_df: pd.DataFrame, nat_polygon) -> Tuple[float, float]:
    in_nat = traj_df.apply(
        lambda r: nat_polygon.contains(Point(r["longitude"], r["latitude"])), axis=1
    )
    if not in_nat.any():
        return 0.0, 0.0
    idx = traj_df.index[in_nat].tolist()
    nat_fuel = traj_df.loc[idx[0], "mass"] - traj_df.loc[idx[-1], "mass"]
    pts = traj_df.loc[in_nat]
    nat_dist = sum(
        gc_distance_nm_coords(
            pts.iloc[i - 1]["latitude"], pts.iloc[i - 1]["longitude"],
            pts.iloc[i]["latitude"], pts.iloc[i]["longitude"],
        )
        for i in range(1, len(pts))
    )
    return nat_fuel, nat_dist


# ═══════════════════════════════════════════════════════════════════════════════
#  INITIAL-GUESS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def _gc_intermediate_points_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg, n):
    """Great-circle Slerp returning *n* waypoints in **degrees** (endpoints incl.)."""
    lat1, lon1 = np.deg2rad(lat1_deg), np.deg2rad(lon1_deg)
    lat2, lon2 = np.deg2rad(lat2_deg), np.deg2rad(lon2_deg)
    d = 2.0 * np.arcsin(np.sqrt(
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    ))
    if d < 1e-12:
        return np.full(n, lat1_deg), np.full(n, lon1_deg)
    f = np.linspace(0.0, 1.0, n)
    A = np.sin((1 - f) * d) / np.sin(d)
    B = np.sin(f * d) / np.sin(d)
    x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
    y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
    z = A * np.sin(lat1) + B * np.sin(lat2)
    return np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2))), np.rad2deg(np.arctan2(y, x))


def _cross_track_offset(lat_deg, lon_deg, bearing_deg, offset_m):
    """Move each waypoint *offset_m* metres perpendicular to *bearing_deg*.

    Positive offset → left of bearing → poleward for westbound NAT tracks.
    Uses the spherical direct formula (Vincenty-style on a sphere).
    """
    R = 6.371e6
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    # Perpendicular bearing: rotate left (+90° towards north for westbound)
    perp_brg = np.deg2rad(np.asarray(bearing_deg, dtype=float)) - np.pi / 2
    d_R = np.asarray(offset_m, dtype=float) / R

    new_lat = np.arcsin(
        np.sin(lat) * np.cos(d_R) + np.cos(lat) * np.sin(d_R) * np.cos(perp_brg)
    )
    new_lon = lon + np.arctan2(
        np.sin(perp_brg) * np.sin(d_R) * np.cos(lat),
        np.cos(d_R) - np.sin(lat) * np.sin(new_lat),
    )
    return np.rad2deg(new_lat), np.rad2deg(new_lon)


def generate_track_variations(
    origin: str,
    destination: str,
    n_tracks: int = 7,
    max_deviation_nm: float = 500,
    n_points: int = 61,
) -> List[dict]:
    """Generate multistart initial-guess tracks using great-circle offsets.

    Each track is a sinusoidal cross-track deviation from the GC baseline.
    Positive deviation → poleward (northward for westbound NAT tracks).
    Output lat/lon in **degrees** (the optimiser's ``initial_guess()``
    converts to radians internally).
    """
    lat1, lon1 = get_airport_coords(origin)
    lat2, lon2 = get_airport_coords(destination)

    # GC baseline waypoints (degrees)
    gc_lat, gc_lon = _gc_intermediate_points_deg(lat1, lon1, lat2, lon2, n_points)

    # Forward bearing at each GC waypoint (for perpendicular offset direction)
    gc_brg = np.array([
        oa.aero.bearing(gc_lat[i], gc_lon[i],
                        gc_lat[min(i + 1, n_points - 1)],
                        gc_lon[min(i + 1, n_points - 1)])
        for i in range(n_points)
    ])
    gc_brg[-1] = gc_brg[-2]  # copy forward bearing for last point

    # Build deviation list – always include GC (0 nm) as the first track
    if n_tracks <= 1:
        deviations_nm = np.array([0.0])
    elif n_tracks == 2:
        deviations_nm = np.array([0.0, max_deviation_nm])
    else:
        # Symmetric spread with 0 guaranteed
        deviations_nm = np.linspace(-max_deviation_nm, max_deviation_nm, n_tracks)
        if not np.any(np.abs(deviations_nm) < 1):
            deviations_nm = np.sort(np.append(deviations_nm, 0.0))

    t = np.linspace(0, 1, n_points)
    prof = np.sin(t * np.pi)  # sinusoidal envelope (0 at endpoints, 1 at midpoint)

    tracks = []
    for dev_nm in deviations_nm:
        dev_m = dev_nm * 1852  # nm → m
        offset = dev_m * prof  # per-waypoint offset (m)

        if abs(dev_nm) < 1:
            la, lo = gc_lat.copy(), gc_lon.copy()
        else:
            la, lo = _cross_track_offset(gc_lat, gc_lon, gc_brg, offset)

        init_df = pd.DataFrame({
            "latitude": la,
            "longitude": lo,
            "altitude": np.full(n_points, 35_000.0),
            "mass": np.full(n_points, 300_000.0),
            "ts": np.linspace(0, 8 * 3600, n_points),
        })
        if abs(dev_nm) < 1:
            name = "gc"
        else:
            name = f"{'n' if dev_nm > 0 else 's'}{int(abs(dev_nm))}"

        tracks.append({
            "name": name,
            "init_guess_df": init_df,
            "deviation_nm": dev_nm,
        })
    return tracks


def update_init_guess(init_df: pd.DataFrame, optimizer,
                      *, flst_based: bool = False) -> pd.DataFrame:
    """Update the lateral initial-guess DataFrame with feasible altitude,
    realistic mass profile, and correct flight-time estimate.

    When *flst_based* is True the altitude, mass, and ts columns already
    contain BADA3-consistent values derived from the FLST log.  In this
    case only light validation is applied (ceiling clamp, mass floor).
    When False, the columns are overwritten with a generic ramp (as in v8).
    """
    df = init_df.copy()
    n = len(df)
    ceiling = optimizer.aircraft.get("cruise", {}).get(
        "ceiling", optimizer.aircraft["cruise"]["height"]
    )

    if flst_based:
        # --- FLST path: altitude/mass/ts are already realistic ------------
        # Clamp altitude to aircraft ceiling × 0.95 (as base.py does)
        h_max_ft = ceiling * 0.95 / ft
        h_min_ft = 20_000.0  # FL200
        df["altitude"] = np.clip(df["altitude"].values, h_min_ft, h_max_ft)

        # Mass floor = OEW
        oew = getattr(optimizer, "oew", optimizer.mass_init * 0.45)
        df["mass"] = np.maximum(df["mass"].values, oew)

        # Ensure ts is monotonically increasing (safety)
        ts = df["ts"].values.copy()
        for i in range(1, n):
            if ts[i] <= ts[i - 1]:
                ts[i] = ts[i - 1] + 1.0
        df["ts"] = ts
        return df

    # --- GC path: overwrite with generic ramp (v8 behaviour) --------------
    h_cr = optimizer.aircraft["cruise"]["height"]
    h_cr = min(h_cr, ceiling * 0.85, 12_500)

    mach_guess = optimizer.mach_max - 0.03
    h_rough = h_cr
    tas_guess = float(oa.aero.mach2tas(mach_guess, h_rough))  # m/s
    t_flight = float(optimizer.range) / max(tas_guess, 100)   # seconds

    ff_guess = 2.5  # kg/s (conservative average for wide-body cruise)
    fuel_total_guess = min(ff_guess * t_flight, optimizer.mass_init * 0.6)

    h_start = h_cr * 0.85
    h_end = h_cr
    h_profile = np.linspace(h_start, h_end, n)

    df["altitude"] = h_profile / ft  # feet
    df["mass"] = np.linspace(optimizer.mass_init,
                             optimizer.mass_init - fuel_total_guess, n)
    df["ts"] = np.linspace(0, t_flight, n)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  FLST-BASED INITIAL GUESS
#  Read a BlueSky FLST log file, extract cruise segments, and build
#  initial guesses from simulated reference flights.
# ═══════════════════════════════════════════════════════════════════════════════

# Column definitions for the two FLST formats
_FLST_HEADER_COMPACT = [
    "simt", "flightid", "ac_type", "flighttime", "distanceflown",
    "totalfuel", "positivefuelflow", "latitude", "longitude",
    "altitude", "tas", "vs", "spawntime",
]

_FLST_HEADER_FULL = [
    "simt", "flightid", "ac_type", "flighttime", "currentmass",
    "distanceflown", "totalfuel", "latitude", "longitude", "spawntime",
    "actualdistance2D", "actualdistance3D", "workdone",
    "positivefuelflow", "rawfuelflow", "thrust", "altitude", "tas",
    "gs", "cas", "mach", "vs", "heading", "asasactive", "pilotalt",
    "pilottas", "pilothdg", "pilotvs", "n_active_conflicts",
    "n_active_intrusions", "gsnorth", "gseast", "windnorth", "windeast",
    "headwind", "crosswind", "pilotcas", "pilotmach", "selalt", "selvs",
    "swlnav", "swvnav", "swvnavspd", "swats", "throttle", "temp",
    "rho", "flight_phase", "drag",
]


def _detect_flst_format(path: str) -> Tuple[list, int]:
    """Detect FLST format (compact vs full) by counting columns in data.

    Returns (header_list, n_header_comment_lines).
    """
    n_comment = 0
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                n_comment += 1
                continue
            n_cols = len(stripped.split(","))
            if n_cols <= 15:
                return _FLST_HEADER_COMPACT, n_comment
            else:
                return _FLST_HEADER_FULL, n_comment
    return _FLST_HEADER_COMPACT, n_comment


def _read_flst_file(
    path: str,
    target_flightids: set[str] | None = None,
    subsample_s: int = 10,
) -> dict[str, pd.DataFrame]:
    """Read an FLST log file and return per-flight DataFrames.

    Parameters
    ----------
    path : str
        Path to the FLST log file.
    target_flightids : set of str, optional
        If given, only return data for these flight IDs.
        If None, return all flights.
    subsample_s : int
        Keep one row every *subsample_s* seconds (based on ``flighttime``).
        Reduces memory for 1 Hz logs.  Default 10.

    Returns
    -------
    dict : ``{flightid_str: DataFrame}``
    """
    header, n_skip = _detect_flst_format(path)
    log(f"  FLST format: {'compact' if len(header) <= 15 else 'full'} "
        f"({len(header)} cols, {n_skip} comment lines)")

    # Determine which columns to actually load (for speed)
    usecols_names = ["simt", "flightid", "flighttime", "distanceflown",
                     "totalfuel", "latitude", "longitude", "altitude",
                     "tas", "vs"]
    if "currentmass" in header:
        usecols_names.append("currentmass")
    usecols_idx = [header.index(c) for c in usecols_names if c in header]

    chunk_size = 500_000
    flights: dict[str, list[pd.DataFrame]] = {}

    for chunk in pd.read_csv(
        path,
        comment="#",
        header=None,
        names=header,
        usecols=usecols_idx,
        dtype={c: "float64" for c in usecols_names if c != "flightid"},
        chunksize=chunk_size,
    ):
        chunk["flightid"] = chunk["flightid"].astype(str).str.strip()

        if target_flightids is not None:
            chunk = chunk[chunk["flightid"].isin(target_flightids)]

        if chunk.empty:
            continue

        # Subsample: keep every N-th second per flight
        if subsample_s > 1:
            chunk = chunk[
                (chunk["flighttime"] % subsample_s < 0.5) |
                chunk.groupby("flightid")["flighttime"].transform("min").eq(chunk["flighttime"]) |
                chunk.groupby("flightid")["flighttime"].transform("max").eq(chunk["flighttime"])
            ]

        for fid, grp in chunk.groupby("flightid"):
            flights.setdefault(str(fid), []).append(grp)

    # Concatenate chunks per flight
    result = {}
    for fid, chunks in flights.items():
        df = pd.concat(chunks, ignore_index=True).sort_values("flighttime").reset_index(drop=True)
        result[fid] = df

    log(f"  FLST loaded: {len(result)} flights from {path}")
    return result


def _trim_flst_cruise(
    df_flight: pd.DataFrame,
    trim_nm: float = TRIM_DISTANCE_NM,
) -> dict | None:
    """Trim *trim_nm* from start and end of a flight using ``distanceflown``.

    Returns a dict with:
      - ``cruise_df``: trimmed DataFrame (cruise segment)
      - ``origin_opt``: (lat, lon) of cruise start
      - ``dest_opt``: (lat, lon) of cruise end
      - ``trim_start_alt_ft``: altitude at cruise start
      - ``trim_end_alt_ft``: altitude at cruise end
      - ``cruise_dist_nm``: distance of the cruise segment

    Returns None if the flight is shorter than 2 × trim_nm.
    """
    if df_flight.empty:
        return None

    total_dist = df_flight["distanceflown"].iloc[-1]
    if total_dist < 2.2 * trim_nm:
        return None  # flight too short for meaningful cruise

    # Find trim indices using distanceflown
    start_mask = df_flight["distanceflown"] >= trim_nm
    end_mask = df_flight["distanceflown"] <= (total_dist - trim_nm)

    if not start_mask.any() or not end_mask.any():
        return None

    start_idx = start_mask.idxmax()  # first True
    end_idx = end_mask[::-1].idxmax()  # last True

    if start_idx >= end_idx:
        return None

    cruise_df = df_flight.loc[start_idx:end_idx].copy().reset_index(drop=True)

    if len(cruise_df) < 5:
        return None

    # Validate altitude: cruise should be above FL200
    median_alt = cruise_df["altitude"].median()
    if median_alt < 20_000:
        log(f"    FLST cruise median alt {median_alt:.0f} ft < FL200, "
            f"segment may include climb/descent", "WARN")

    origin_opt = (float(cruise_df["latitude"].iloc[0]),
                  float(cruise_df["longitude"].iloc[0]))
    dest_opt = (float(cruise_df["latitude"].iloc[-1]),
                float(cruise_df["longitude"].iloc[-1]))

    return {
        "cruise_df": cruise_df,
        "origin_opt": origin_opt,
        "dest_opt": dest_opt,
        "trim_start_alt_ft": float(cruise_df["altitude"].iloc[0]),
        "trim_end_alt_ft": float(cruise_df["altitude"].iloc[-1]),
        "trim_start_dist_nm": float(cruise_df["distanceflown"].iloc[0]),
        "trim_end_dist_nm": float(cruise_df["distanceflown"].iloc[-1]),
        "cruise_dist_nm": float(
            cruise_df["distanceflown"].iloc[-1] - cruise_df["distanceflown"].iloc[0]
        ),
    }


def _resample_to_n_points(
    cruise_df: pd.DataFrame,
    n_points: int,
) -> pd.DataFrame:
    """Resample a cruise-segment DataFrame to exactly *n_points* using
    distance-based **cubic spline** interpolation.

    CubicSpline gives C² continuity so that IPOPT sees smooth first
    derivatives (ẋ) of the lat/lon/alt/ts initial guess — critical for
    collocation-based NLP convergence.

    The time profile is re-derived from the TAS profile after resampling
    to guarantee that ``Δdist / Δts ≈ TAS`` at every node.  This avoids
    inconsistency between the spatial and temporal state components.

    Returns a DataFrame with columns:
      latitude, longitude, altitude (ft), ts (s), mass (NaN),
      _fuel_from_start, _tas_kt.
    ``mass`` is set to NaN (to be filled by the caller).
    """
    # Cumulative distance within the cruise segment (relative to segment start)
    cum_dist_raw = cruise_df["distanceflown"].values - cruise_df["distanceflown"].values[0]

    # Remove duplicate distance values (can happen in 1 Hz data at rest)
    mask_unique = np.diff(cum_dist_raw, prepend=-1) > 0
    mask_unique[0] = True
    cum_dist = cum_dist_raw[mask_unique]
    if len(cum_dist) < 4:
        return None  # need at least 4 points for cubic spline

    total_cruise_dist = cum_dist[-1]
    if total_cruise_dist < 1.0:
        return None  # degenerate segment

    # Target distances for evenly spaced points
    target_dists = np.linspace(0, total_cruise_dist, n_points)

    # Extract columns aligned with unique mask
    lat_raw = cruise_df["latitude"].values[mask_unique]
    lon_raw = cruise_df["longitude"].values[mask_unique]
    alt_raw = cruise_df["altitude"].values[mask_unique]
    ts_raw  = cruise_df["flighttime"].values[mask_unique] - cruise_df["flighttime"].values[0]

    # Build C²-continuous cubic spline interpolators
    # bc_type="clamped" pins first derivatives at endpoints to the data slope
    lat_fn = CubicSpline(cum_dist, lat_raw, bc_type="not-a-knot")
    lon_fn = CubicSpline(cum_dist, lon_raw, bc_type="not-a-knot")
    alt_fn = CubicSpline(cum_dist, alt_raw, bc_type="not-a-knot")
    ts_fn  = CubicSpline(cum_dist, ts_raw,  bc_type="not-a-knot")

    lat_resampled = lat_fn(target_dists)
    lon_resampled = lon_fn(target_dists)
    alt_resampled = alt_fn(target_dists)

    # --- Re-derive time from TAS to ensure dt/ds consistency -------------
    # TAS interpolation (if available in FLST)
    if "tas" in cruise_df.columns:
        tas_raw = cruise_df["tas"].values[mask_unique]  # knots
        # Guard: replace zeros/NaN with sensible cruise speed
        tas_raw = np.where((tas_raw > 50) & np.isfinite(tas_raw), tas_raw, 450)
        tas_fn = CubicSpline(cum_dist, tas_raw, bc_type="not-a-knot")
        tas_resampled = np.maximum(tas_fn(target_dists), 200)  # floor at 200 kt
    else:
        # Fallback: derive TAS from ts spline
        ts_resampled_raw = ts_fn(target_dists)
        ds = np.diff(target_dists)  # NM
        dt_raw = np.diff(ts_resampled_raw)  # seconds
        dt_raw = np.maximum(dt_raw, 1.0)
        tas_derived = ds / dt_raw * 3600  # NM/h → kt
        tas_resampled = np.concatenate([[tas_derived[0]], tas_derived])
        tas_resampled = np.clip(tas_resampled, 200, 600)

    # Integrate time from TAS: dt_i = ds_i / (TAS_i × kts→NM/s)
    # TAS in kt, distance in NM → dt = ds_NM / (TAS_kt / 3600) = ds × 3600 / TAS
    ds = np.diff(target_dists)  # NM
    tas_mid = 0.5 * (tas_resampled[:-1] + tas_resampled[1:])  # mid-segment TAS
    dt_segments = ds * 3600.0 / np.maximum(tas_mid, 200)  # seconds
    ts_resampled = np.concatenate([[0.0], np.cumsum(dt_segments)])

    # Fuel profile (relative to segment start)
    if "totalfuel" in cruise_df.columns:
        fuel_raw = cruise_df["totalfuel"].values[mask_unique] - cruise_df["totalfuel"].values[0]
        fuel_fn = CubicSpline(cum_dist, fuel_raw, bc_type="not-a-knot")
        fuel_profile = np.maximum(fuel_fn(target_dists), 0.0)
    else:
        fuel_profile = np.zeros(n_points)

    # Mass profile from FLST currentmass column (full format only).
    # Resampled via cubic spline so IPOPT gets an observed mass profile
    # rather than one computed from a fuel-flow model.
    if "currentmass" in cruise_df.columns:
        mass_raw_flst = cruise_df["currentmass"].values[mask_unique]
        valid_flst = (mass_raw_flst > 0) & np.isfinite(mass_raw_flst)
        if valid_flst.sum() >= 4:
            mass_flst_fn = CubicSpline(cum_dist, mass_raw_flst, bc_type="not-a-knot")
            mass_profile_flst = mass_flst_fn(target_dists)
        else:
            mass_profile_flst = np.full(n_points, np.nan)
    else:
        mass_profile_flst = np.full(n_points, np.nan)

    resampled = pd.DataFrame({
        "latitude": lat_resampled,
        "longitude": lon_resampled,
        "altitude": alt_resampled,  # feet
        "ts": ts_resampled,
        "mass": np.full(n_points, np.nan),  # filled later
        "_fuel_from_start": fuel_profile,  # relative fuel for mass shaping
        "_mass_from_flst": mass_profile_flst,  # direct mass from FLST currentmass
        "_tas_kt": tas_resampled if "tas" in cruise_df.columns else np.full(n_points, 450),
    })

    return resampled


def _fill_mass_profile(
    resampled_df: pd.DataFrame,
    mass_init: float,
    fuel_total_estimate: float,
    optimizer=None,
) -> pd.DataFrame:
    """Fill the mass column so that dm/dt is consistent with BADA3 fuel flow.

    Strategy (in priority order):
    1. FLST ``currentmass`` column (full format only): directly observed
       aircraft mass, resampled via cubic spline.  Most accurate starting
       point for IPOPT as it reflects the actual simulated trajectory.
    2. BADA3 fuel-flow forward integration: if ``currentmass`` is absent
       or invalid, integrate ``ff.enroute()`` forward from *mass_init*.
       Gives IPOPT a guess whose mass derivative satisfies the ODE
       ``ṁ = −ff(m, TAS, h)``.
    3. FLST fuel-burn shape (``_fuel_from_start``): scale cumulative fuel
       burned to *fuel_total_estimate* and subtract from *mass_init*.
    4. Last resort: linear mass decay from *mass_init*.
    """
    df = resampled_df.copy()
    n = len(df)

    # --- Strategy 1: FLST currentmass (directly observed, highest priority) ---
    if "_mass_from_flst" in df.columns:
        mass_flst = df["_mass_from_flst"].values
        if np.all(np.isfinite(mass_flst)) and np.all(mass_flst > 0):
            oew = getattr(optimizer, "oew", mass_init * 0.45) if optimizer else mass_init * 0.45
            df["mass"] = np.maximum(mass_flst, oew)
            log(f"    FLST currentmass used for mass profile "
                f"(m0={mass_flst[0]:.0f} kg -> mf={mass_flst[-1]:.0f} kg, "
                f"burn={mass_flst[0]-mass_flst[-1]:.0f} kg)")
            for col in ("_fuel_from_start", "_mass_from_flst", "_tas_kt"):
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            return df
        else:
            log("    FLST currentmass invalid (NaN/zero) -- falling back to BADA3", "WARN")

    # --- Strategy 2: BADA3 fuel-flow forward integration ------------------
    bada3_ok = False
    if optimizer is not None:
        try:
            ff_model = optimizer.fuelflow
            ts_arr = df["ts"].values
            alt_ft = df["altitude"].values
            # TAS from resampled data (kt) — stored by _resample_to_n_points
            if "_tas_kt" in df.columns:
                tas_kt = df["_tas_kt"].values
            else:
                # Estimate from Mach at altitude
                mach_guess = optimizer.mach_max - 0.03
                h_m = alt_ft * 0.3048
                tas_kt = np.array([
                    float(oa.aero.mach2tas(mach_guess, hi)) / 0.5144
                    for hi in h_m
                ])

            mass_profile = np.zeros(n)
            mass_profile[0] = mass_init
            for i in range(1, n):
                dt = ts_arr[i] - ts_arr[i - 1]
                if dt < 0.1:
                    mass_profile[i] = mass_profile[i - 1]
                    continue
                # Numeric BADA3 fuel flow: kg/s at previous node
                try:
                    ff_kgs = float(ff_model.enroute(
                        mass_profile[i - 1], tas_kt[i - 1],
                        alt_ft[i - 1], vs=0,
                    ))
                except Exception:
                    ff_kgs = 2.5  # fallback
                ff_kgs = max(ff_kgs, 0.3)   # floor to avoid zero fuel burn
                ff_kgs = min(ff_kgs, 10.0)  # cap for sanity
                mass_profile[i] = mass_profile[i - 1] - ff_kgs * dt

            # Sanity: mass should not go below OEW
            oew = getattr(optimizer, "oew", mass_init * 0.45)
            mass_profile = np.maximum(mass_profile, oew)
            df["mass"] = mass_profile
            bada3_ok = True
        except Exception as exc:
            log(f"    BADA3 mass integration failed ({exc}), using FLST fuel shape", "WARN")

    # --- Strategy 3: FLST fuel-burn shape ---------------------------------
    if not bada3_ok:
        if "_fuel_from_start" in df.columns:
            flst_total_fuel = df["_fuel_from_start"].iloc[-1]
            if flst_total_fuel > 0:
                scale = fuel_total_estimate / flst_total_fuel
                df["mass"] = mass_init - df["_fuel_from_start"] * scale
            else:
                # --- Strategy 4: linear decay (last resort) ---------------
                df["mass"] = np.linspace(mass_init, mass_init - fuel_total_estimate, n)
        else:
            # --- Strategy 4: linear decay (last resort) -------------------
            df["mass"] = np.linspace(mass_init, mass_init - fuel_total_estimate, n)

    # --- Clean up helper columns ------------------------------------------
    for col in ("_fuel_from_start", "_mass_from_flst", "_tas_kt"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def generate_flst_track_variations(
    flst_base_df: pd.DataFrame,
    n_tracks: int = 7,
    max_deviation_nm: float = 500,
) -> List[dict]:
    """Generate multistart initial-guess tracks from an FLST-based cruise path.

    Uses the same sinusoidal cross-track offset approach as
    ``generate_track_variations()`` but applies it to the FLST path
    instead of the great-circle baseline.

    Parameters
    ----------
    flst_base_df : pd.DataFrame
        Resampled FLST cruise path (from ``_resample_to_n_points``).
        Must have columns: latitude, longitude, altitude, ts, mass.
    n_tracks : int
        Number of initial-guess variants (including the original path).
    max_deviation_nm : float
        Maximum cross-track deviation in NM.

    Returns
    -------
    list of dict
        Each dict has keys: name, init_guess_df, deviation_nm.
    """
    n_points = len(flst_base_df)
    base_lat = flst_base_df["latitude"].values
    base_lon = flst_base_df["longitude"].values

    # Compute forward bearing at each point (for perpendicular offset)
    bearings = np.zeros(n_points)
    for i in range(n_points - 1):
        bearings[i] = _compute_bearing(
            base_lat[i], base_lon[i],
            base_lat[i + 1], base_lon[i + 1],
        )
    bearings[-1] = bearings[-2] if n_points > 1 else 0.0

    # Build deviation list (same logic as generate_track_variations)
    if n_tracks <= 1:
        deviations_nm = np.array([0.0])
    elif n_tracks == 2:
        deviations_nm = np.array([0.0, max_deviation_nm])
    else:
        deviations_nm = np.linspace(-max_deviation_nm, max_deviation_nm, n_tracks)
        if not np.any(np.abs(deviations_nm) < 1):
            deviations_nm = np.sort(np.append(deviations_nm, 0.0))

    # Sinusoidal envelope (0 at endpoints, 1 at midpoint)
    t = np.linspace(0, 1, n_points)
    prof = np.sin(t * np.pi)

    tracks = []
    for dev_nm in deviations_nm:
        if abs(dev_nm) < 1:
            # Original FLST path (no offset)
            init_df = flst_base_df[["latitude", "longitude", "altitude", "mass", "ts"]].copy()
            name = "flst"
        else:
            dev_m = dev_nm * 1852  # NM → m
            offset = dev_m * prof  # per-waypoint offset (m)
            la, lo = _cross_track_offset(base_lat, base_lon, bearings, offset)

            # --- Task C fix: re-estimate ts for the offset path ----------
            # The offset path is longer/shorter than the original.  If we
            # keep the original ts, IPOPT sees Δdist/Δts that violate the
            # kinematic ODE.  Re-derive ts from the new path length.
            offset_dist_nm = np.zeros(n_points)
            for j in range(1, n_points):
                offset_dist_nm[j] = offset_dist_nm[j - 1] + gc_distance_nm_coords(
                    la[j - 1], lo[j - 1], la[j], lo[j],
                )
            # Estimate TAS from original ts/distance or use stored _tas_kt
            if "_tas_kt" in flst_base_df.columns:
                tas_kt = flst_base_df["_tas_kt"].values.copy()
            else:
                # Derive from original spacing
                orig_total_time = flst_base_df["ts"].iloc[-1]
                orig_total_dist = 0.0
                for j in range(1, n_points):
                    orig_total_dist += gc_distance_nm_coords(
                        base_lat[j-1], base_lon[j-1], base_lat[j], base_lon[j],
                    )
                avg_tas_kt = max(orig_total_dist / max(orig_total_time, 1) * 3600, 400)
                tas_kt = np.full(n_points, avg_tas_kt)

            # Integrate time along the offset path
            ds_offset = np.diff(offset_dist_nm)
            tas_mid = 0.5 * (tas_kt[:-1] + tas_kt[1:])
            dt_offset = ds_offset * 3600.0 / np.maximum(tas_mid, 200)
            ts_offset = np.concatenate([[0.0], np.cumsum(dt_offset)])

            # --- Task C fix: blend altitude toward cruise ceiling ---------
            # For large deviations the FLST altitude profile may be
            # physically impossible at a very different latitude/wind.
            # Blend toward a "safe" flat-cruise altitude so the optimizer
            # has a feasible starting point.
            blend = min(abs(dev_nm) / max_deviation_nm, 1.0) if max_deviation_nm > 0 else 0.0
            # Scale factor: 0 at baseline, 1 at max deviation
            # Target: median altitude of the original profile (flat cruise)
            flat_alt = np.median(flst_base_df["altitude"].values)
            blended_alt = (1 - blend * 0.5) * flst_base_df["altitude"].values + blend * 0.5 * flat_alt

            # Mass: keep shape but stretch to match new flight time
            mass_stretch = flst_base_df["mass"].values.copy()

            init_df = pd.DataFrame({
                "latitude": la,
                "longitude": lo,
                "altitude": blended_alt,
                "mass": mass_stretch,
                "ts": ts_offset,
            })
            name = f"{'n' if dev_nm > 0 else 's'}{int(abs(dev_nm))}"

        tracks.append({
            "name": name,
            "init_guess_df": init_df,
            "deviation_nm": dev_nm,
        })

    return tracks


def load_flst_initial_guesses(
    flst_path: str,
    flights_info: list,
    trim_nm: float = TRIM_DISTANCE_NM,
) -> dict[str, dict]:
    """Load FLST file and extract cruise segments for all matching flights.

    Returns
    -------
    dict : ``{flightid: flst_data_dict}``
        Each value is a dict with keys:
          - ``cruise_df``: trimmed (but not yet resampled) DataFrame
          - ``origin_opt``: (lat, lon) tuple for optimizer origin
          - ``dest_opt``: (lat, lon) tuple for optimizer destination
          - plus metadata from ``_trim_flst_cruise()``

        Flights not found in the FLST or with too-short trajectories
        are silently omitted (the caller will fall back to GC guess).
    """
    target_ids = {str(f["flightid"]) for f in flights_info}
    log(f"Loading FLST initial guesses from {flst_path}")
    log(f"  Looking for {len(target_ids)} flight IDs")

    raw = _read_flst_file(flst_path, target_flightids=target_ids)

    flst_guesses: dict[str, dict] = {}
    for fid, df_flight in raw.items():
        trimmed = _trim_flst_cruise(df_flight, trim_nm=trim_nm)
        if trimmed is None:
            log(f"  [{fid}] FLST trim failed (flight too short or no cruise segment)", "WARN")
            continue
        flst_guesses[fid] = trimmed
        log(f"  [{fid}] FLST cruise: {trimmed['cruise_dist_nm']:.0f} NM, "
            f"{len(trimmed['cruise_df'])} rows, "
            f"origin_opt=({trimmed['origin_opt'][0]:.2f}, {trimmed['origin_opt'][1]:.2f}), "
            f"dest_opt=({trimmed['dest_opt'][0]:.2f}, {trimmed['dest_opt'][1]:.2f})")

    n_found = len(flst_guesses)
    n_missing = len(target_ids) - n_found
    log(f"  FLST guesses ready: {n_found} flights "
        f"({n_missing} will use GC fallback)")

    return flst_guesses


# ═══════════════════════════════════════════════════════════════════════════════
#  WIND INTERPOLANT CACHING  (v11 — pre-build per departure-hour bucket)
#
#  Instead of each worker building its own BSplineWind from the raw wind
#  DataFrame (5–30 s per flight), we:
#   1. Compute all unique departure-hour offsets across the flight list.
#   2. For each offset, shift the wind DataFrame and build one BSplineWind
#      covering the entire flight region (global bounding box + margin).
#   3. Save the BSplineWind to disk (two .casadi files + meta.json).
#   4. Workers load the cached BSplineWind in < 0.5 s.
#
#  Cache key: integer offset in seconds  (dep_hour – weather_start).
#  Cache dir: <output>/wind_cache/offset_<offset_s>/
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_dep_hour_offset(dep_time, weather_start) -> int:
    """Round departure time to the nearest hour and return the offset
    in seconds from *weather_start*.  Same logic as v9's wind-shift."""
    dep = pd.to_datetime(dep_time)
    dep_truncated = dep.replace(minute=0, second=0, microsecond=0)
    dep_hour = dep_truncated + timedelta(hours=1) if dep.minute >= 30 else dep_truncated
    return int((dep_hour - weather_start).total_seconds())


def _global_flight_bbox(flights_info: list, margin: float = 7.0) -> tuple:
    """Compute a bounding box covering all flight endpoints + *margin* degrees."""
    import openap.nav as nav
    lats, lons = [], []
    for f in flights_info:
        try:
            o = nav.airport(f["origin"])
            d = nav.airport(f["destination"])
            lats += [o["lat"], d["lat"]]
            lons += [o["lon"], d["lon"]]
        except Exception:
            pass
    if not lats:
        # Fallback: NAT region
        return 20.0, -80.0, 75.0, 30.0
    return (
        min(lats) - margin,
        min(lons) - margin,
        max(lats) + margin,
        max(lons) + margin,
    )


def precompute_wind_cache(
    wind_data: pd.DataFrame,
    weather_start,
    flights_info: list,
    cache_dir: str,
    wind_method: str = "linear",
    bspline_degree: int = 3,
    bspline_subsample: int = 1,
    time_subsample: int = 1,
) -> dict:
    """Pre-build and cache BSplineWind objects for each departure-hour bucket.

    Parameters
    ----------
    wind_data : pd.DataFrame
        Full wind DataFrame with columns ``ts, h, latitude, longitude, u, v``.
    weather_start : pd.Timestamp
        The earliest timestamp in the wind data.
    flights_info : list[dict]
        Flight list; each dict must have ``dep_time``.
    cache_dir : str
        Directory to write cached files into.
    wind_method, bspline_degree, bspline_subsample, time_subsample
        Same as the BSplineWind constructor parameters.

    Returns
    -------
    dict : ``{offset_s: cache_subdir_path}``
        Mapping from integer offset (seconds) to the directory containing
        the cached BSplineWind files.
    """
    from top.tools import BSplineWind

    os.makedirs(cache_dir, exist_ok=True)

    # -- Compute unique departure-hour offsets ----------------------------
    offsets = sorted({
        _compute_dep_hour_offset(f["dep_time"], weather_start)
        for f in flights_info
    })
    log(f"  Wind cache: {len(offsets)} departure-hour buckets to build")

    # -- Global bounding box covering all flights -------------------------
    lat_lo, lon_lo, lat_hi, lon_hi = _global_flight_bbox(flights_info)
    log(f"  Wind cache: global bbox lat=[{lat_lo:.1f},{lat_hi:.1f}] "
        f"lon=[{lon_lo:.1f},{lon_hi:.1f}]")

    cache_map: dict[int, str] = {}

    for i, offset_s in enumerate(offsets):
        subdir = os.path.join(cache_dir, f"offset_{offset_s}")

        # Check if already cached (supports resume across runs)
        meta_path = os.path.join(subdir, "meta.json")
        if os.path.isfile(meta_path):
            log(f"  Wind cache [{i+1}/{len(offsets)}]: offset={offset_s}s "
                f"-> CACHED (reusing)")
            cache_map[offset_s] = subdir
            continue

        log(f"  Wind cache [{i+1}/{len(offsets)}]: offset={offset_s}s "
            f"-> building ...")
        t0 = time.time()

        # -- shift time axis -----------------------------------------------
        ts_arr = wind_data["ts"].values
        ts_shifted = ts_arr - offset_s
        mask = ts_shifted >= 0
        df_shifted = wind_data.loc[mask].copy()
        df_shifted["ts"] = ts_shifted[mask]
        del ts_arr, ts_shifted, mask

        if df_shifted.empty:
            log(f"    WARNING: no wind after shift by {offset_s}s — skipping", "WARN")
            continue

        # -- build BSplineWind with global bbox ----------------------------
        # We pass two fake "airport" corners to the constructor that span
        # the entire flight region; the margin is already included in the
        # bbox, so we set margin=0.
        try:
            bsw = BSplineWind(
                df_shifted,
                lat1=lat_lo, lon1=lon_lo,
                lat2=lat_hi, lon2=lon_hi,
                margin=0,
                method=wind_method,
                degree=bspline_degree,
                subsample=bspline_subsample,
                time_subsample=time_subsample,
            )
        except (ValueError, RuntimeError) as exc:
            if wind_method == "bspline" and "Not implemented" in str(exc):
                log(f"    BSpline failed — retrying with method='linear'", "WARN")
                bsw = BSplineWind(
                    df_shifted,
                    lat1=lat_lo, lon1=lon_lo,
                    lat2=lat_hi, lon2=lon_hi,
                    margin=0,
                    method="linear",
                    degree=bspline_degree,
                    subsample=1,
                    time_subsample=1,
                )
            else:
                raise

        # -- save to disk --------------------------------------------------
        bsw.save(subdir)
        cache_map[offset_s] = subdir
        build_s = time.time() - t0
        log(f"    saved -> {subdir}  ({build_s:.1f}s, {bsw})")

        del df_shifted, bsw
        gc.collect()

    log(f"  Wind cache: {len(cache_map)} buckets ready in {cache_dir}")
    return cache_map


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS POOL WORKER CONTEXT
#  Each worker process stores shared (read-only) data here via the
#  pool initializer so that wind data is pickled only once per worker,
#  not once per submitted task.
# ═══════════════════════════════════════════════════════════════════════════════
_WORKER_CTX: dict = {}


def _init_flight_worker(
    wind_path, weather_start, perf_model, bada3_path,
    polydeg, ipopt_kwargs, load_factor, n_tracks,
    max_deviation_nm, nodes,
    wind_method, bspline_degree, bspline_subsample, time_subsample,
    no_wind,
    flst_guesses_path=None,
    wind_cache_dir=None,
):
    """Called once when each worker process starts.

    **v11 change**: when *wind_cache_dir* is provided (for ``linear`` /
    ``bspline`` wind methods), the raw wind DataFrame is **not** loaded.
    Workers will load the pre-built BSplineWind from the cache on a
    per-flight basis (keyed by departure-hour offset).

    For ``poly`` wind method the old behaviour is kept: the raw DataFrame
    is loaded from *wind_path* and each flight builds its own PolyWind.

    When *no_wind* is True, ``wind_path`` is None and no wind data is
    loaded.

    When *flst_guesses_path* is provided, FLST cruise-segment data is
    loaded from a pickle file (serialised dict of per-flight cruise info).
    """
    # Only load the raw DataFrame when we actually need it (poly method).
    # For linear/bspline, workers load cached BSplineWind objects on demand.
    if wind_cache_dir is not None:
        # v11 cached path — no raw wind DataFrame needed
        wind_df = None
    elif no_wind or wind_path is None:
        wind_df = None
    else:
        wind_df = pd.read_pickle(wind_path)

    # Load FLST guesses (if provided)
    if flst_guesses_path is not None and os.path.isfile(flst_guesses_path):
        with open(flst_guesses_path, "rb") as fh:
            flst_guesses = pickle.load(fh)
    else:
        flst_guesses = {}

    _WORKER_CTX.update({
        "wind_data": wind_df,
        "weather_start": weather_start,
        "perf_model": perf_model,
        "bada3_path": bada3_path,
        "polydeg": polydeg,
        "ipopt_kwargs": ipopt_kwargs,
        "load_factor": load_factor,
        "n_tracks": n_tracks,
        "max_deviation_nm": max_deviation_nm,
        "nodes": nodes,
        "wind_method": wind_method,
        "bspline_degree": bspline_degree,
        "bspline_subsample": bspline_subsample,
        "time_subsample": time_subsample,
        "no_wind": no_wind,
        "flst_guesses": flst_guesses,
        "wind_cache_dir": wind_cache_dir,  # v11
    })


def _flight_worker(flight: dict) -> dict:
    """Top-level function executed by ProcessPoolExecutor.

    Must be a plain module-level function (picklable).  Reads shared
    data from ``_WORKER_CTX`` (populated by ``_init_flight_worker``)
    and delegates to ``optimise_flight``.
    """
    return optimise_flight(
        flight,
        _WORKER_CTX["wind_data"],
        _WORKER_CTX["weather_start"],
        perf_model=_WORKER_CTX["perf_model"],
        bada3_path=_WORKER_CTX["bada3_path"],
        load_factor=_WORKER_CTX["load_factor"],
        n_tracks=_WORKER_CTX["n_tracks"],
        max_deviation_nm=_WORKER_CTX["max_deviation_nm"],
        nodes=_WORKER_CTX["nodes"],
        polydeg=_WORKER_CTX["polydeg"],
        ipopt_kwargs=_WORKER_CTX["ipopt_kwargs"],
        wind_method=_WORKER_CTX["wind_method"],
        bspline_degree=_WORKER_CTX["bspline_degree"],
        bspline_subsample=_WORKER_CTX["bspline_subsample"],
        time_subsample=_WORKER_CTX["time_subsample"],
        no_wind=_WORKER_CTX["no_wind"],
        flst_guesses=_WORKER_CTX.get("flst_guesses", {}),
        wind_cache_dir=_WORKER_CTX.get("wind_cache_dir"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE-FLIGHT OPTIMISATION
#  Creates the optimizer ONCE per flight and loops tracks sequentially.
#  Flight-level parallelism is handled by ProcessPoolExecutor in run_batch.
# ═══════════════════════════════════════════════════════════════════════════════
def optimise_flight(
    flight: dict,
    wind_data: pd.DataFrame | None,
    weather_start: pd.Timestamp | None,
    *,
    perf_model: str,
    bada3_path: str,
    load_factor: float,
    n_tracks: int,
    max_deviation_nm: float,
    nodes: int | None,
    polydeg: int,
    ipopt_kwargs: dict,
    wind_method: str = "poly",
    bspline_degree: int = 3,
    bspline_subsample: int = 1,
    time_subsample: int = 1,
    no_wind: bool = False,
    flst_guesses: dict | None = None,
    wind_cache_dir: str | None = None,
) -> dict:
    t_start = time.time()
    fid = flight["flightid"]
    csn = flight["callsign"]
    dep = pd.to_datetime(flight["dep_time"])

    log(f"  [{fid}] Starting optimisation  {flight['origin']}->{flight['destination']}  "
        f"({flight['actype']})  dep={dep}  wind_method={'NONE (no-wind)' if no_wind else wind_method}")

    # -- wind preparation --------------------------------------------------
    # v11: if a wind cache is available (linear/bspline), load the pre-built
    # BSplineWind instead of shifting + rebuilding from raw data.
    # For poly wind method, the old shift-from-DataFrame path is kept.
    wind_shifted = None
    cached_bsw = None      # will hold the loaded BSplineWind when cached
    dep_hour_offset = None  # departure-hour offset in seconds

    if not no_wind:
        dep_truncated = dep.replace(minute=0, second=0, microsecond=0)
        dep_hour = dep_truncated + timedelta(hours=1) if dep.minute >= 30 else dep_truncated
        dep_hour_offset = int((dep_hour - weather_start).total_seconds())

        if wind_cache_dir is not None and wind_method in ("linear", "bspline"):
            # ── v11 CACHED PATH ──────────────────────────────────────────
            from top.tools import BSplineWind
            cache_subdir = os.path.join(wind_cache_dir, f"offset_{dep_hour_offset}")
            t_load = time.time()
            cached_bsw = BSplineWind.load(cache_subdir)
            load_s = time.time() - t_load
            log(f"  [{fid}] Wind loaded from cache in {load_s:.2f}s "
                f"(offset={dep_hour_offset}s, {cached_bsw})")
        else:
            # ── LEGACY PATH (poly, or no cache available) ────────────────
            offset = float(dep_hour_offset)
            ts_arr = wind_data["ts"].values
            ts_shifted = ts_arr - offset
            mask = ts_shifted >= 0
            wind_shifted = wind_data.loc[mask].copy()
            wind_shifted["ts"] = ts_shifted[mask]
            del ts_arr, ts_shifted, mask
            log(f"  [{fid}] Wind shifted by {offset:.0f}s -> {len(wind_shifted):,} rows remaining")

            if wind_shifted.empty:
                return {"success": False, "flightid": fid, "callsign": csn,
                        "results": [], "best": None, "error": "no wind after shift",
                        "used_synonym": False}
    else:
        log(f"  [{fid}] No-wind mode: skipping wind shift")

    try:
        # -- detect synonym ------------------------------------------------
        if perf_model == "bada3":
            used_synonym, mapped_to = _detect_bada3_synonym(flight["actype"], bada3_path)
            if used_synonym:
                log(f"  [{fid}] Aircraft {flight['actype']} uses BADA3 synonym -> {mapped_to}", "WARN")
        else:
            used_synonym, _ = _detect_openap_synonym(flight["actype"])
            if used_synonym:
                log(f"  [{fid}] Aircraft {flight['actype']} uses OpenAP synonym data", "WARN")

        # -- determine initial-guess source (FLST vs GC) ------------------
        use_flst = bool(flst_guesses and fid in flst_guesses)
        flst_info = flst_guesses.get(fid) if use_flst else None

        # -- mass estimation with airport ICAO codes (always needed) -------
        # A temporary optimizer with airport endpoints gives us the
        # mass estimation via BADA3 climb integration.
        opt_for_mass = top.Cruise(
            flight["actype"], flight["origin"], flight["destination"],
            m0=0.85, perf_model=perf_model, bada3_path=bada3_path, debug=False,
        )
        mass_est = estimate_cruise_mass(
            flight["origin"], flight["destination"], opt_for_mass, load_factor,
        )
        m0 = mass_est["m0_estimate"]
        aircraft_data = opt_for_mass.aircraft
        mtow = aircraft_data.get("mtow", aircraft_data.get("limits", {}).get("MTOW", 300_000))

        # -- create the actual optimisation optimizer ----------------------
        if use_flst:
            origin_opt = flst_info["origin_opt"]
            dest_opt = flst_info["dest_opt"]
            opt = top.Cruise(
                flight["actype"], origin_opt, dest_opt,
                m0=0.85, perf_model=perf_model, bada3_path=bada3_path,
                debug=False,
            )
            del opt_for_mass
            log(f"  [{fid}] FLST optimizer: "
                f"origin=({origin_opt[0]:.2f}, {origin_opt[1]:.2f}), "
                f"dest=({dest_opt[0]:.2f}, {dest_opt[1]:.2f})")
        else:
            opt = opt_for_mass  # reuse — airport-to-airport optimisation

        # Update the optimizer's initial mass to the estimated value
        # (init_conditions in trajectory() reads self.mass_init)
        opt.mass_init = m0 * opt.aircraft["mtow"]

        # -- altitude floor: cruise optimizer must not go below climb target
        h_min_m = mass_est["climb_target_alt_ft"] * ft
        log(f"  [{fid}] Cruise entry: m0={m0:.4f} ({opt.mass_init:.0f} kg), "
            f"climb target FL{mass_est['climb_target_alt_ft']/100:.0f}, "
            f"h_min={h_min_m:.0f} m")

        # -- dynamic nodes -------------------------------------------------
        if nodes is None:
            dyn_nodes = max(20, min(120, int(opt.range / 50_000)))
            log(f"  [{fid}] Dynamic nodes: {dyn_nodes}  (range {opt.range/1000:.0f} km)")
        else:
            dyn_nodes = nodes

        # -- enable wind & setup ONCE (shared across all tracks) -----------
        t_setup = time.time()

        if no_wind:
            # No-wind mode: skip wind entirely
            log(f"  [{fid}] Wind method: NONE (no-wind mode -- enable_wind NOT called)")
        elif cached_bsw is not None:
            # ── v11 CACHED PATH: assign the pre-loaded BSplineWind directly ──
            opt.wind = cached_bsw
            log(f"  [{fid}] Wind method: BSplineWind FROM CACHE "
                f"(method={cached_bsw.method}, offset={dep_hour_offset}s)")
        else:
            # ── LEGACY PATH (poly, or no cache available) ────────────────
            # Determine wind parameters based on --wind-method
            if wind_method == "poly":
                # Legacy PolyWind (2nd-order polynomial regression)
                log(f"  [{fid}] Wind method: PolyWind (polynomial regression)")
                opt.enable_wind(wind_shifted, use_bspline=False)
            elif wind_method in ("linear", "bspline"):
                # BSplineWind with CasADi grid interpolation
                # Auto-compute max flight time from route distance:
                # distance / 800 km/h (typical cruise speed) + 4h buffer
                max_flight_time_s = None
                if wind_method == "bspline":
                    cruise_speed_kmh = 800
                    buffer_h = 4
                    est_flight_h = opt.range / 1000 / cruise_speed_kmh + buffer_h
                    max_flight_time_s = est_flight_h * 3600
                    log(f"  [{fid}] Wind method: BSplineWind (method={wind_method}, "
                        f"degree={bspline_degree}, subsample={bspline_subsample}, "
                        f"time_sub={time_subsample}, "
                        f"max_time={est_flight_h:.0f}h)")
                else:
                    log(f"  [{fid}] Wind method: BSplineWind (method={wind_method}, "
                        f"degree={bspline_degree}, subsample={bspline_subsample})")
                try:
                    opt.enable_wind(
                        wind_shifted,
                        use_bspline=True,
                        wind_method=wind_method,
                        bspline_degree=bspline_degree,
                        bspline_subsample=bspline_subsample,
                        max_flight_time_s=max_flight_time_s,
                        time_subsample=time_subsample,
                    )
                except RuntimeError as e:
                    if "Not implemented" in str(e):
                        log(f"  [{fid}] BSpline init failed (degenerate grid axis) "
                            f"-- retrying with method='linear'", "WARN")
                        opt.enable_wind(
                            wind_shifted,
                            use_bspline=True,
                            wind_method="linear",
                            bspline_degree=bspline_degree,
                            bspline_subsample=1,
                            max_flight_time_s=max_flight_time_s,
                            time_subsample=1,
                        )
                    else:
                        raise
            else:
                raise ValueError(f"Unknown wind method: {wind_method!r}")

        opt.setup(nodes=dyn_nodes, polydeg=polydeg, ipopt_kwargs=ipopt_kwargs)
        setup_s = time.time() - t_setup
        log(f"  [{fid}] Optimizer setup done in {setup_s:.1f}s "
            f"({'no wind' if no_wind else 'wind'} + NLP)")

        # -- generate track variations ------------------------------------
        if use_flst:
            n_points = dyn_nodes + 1
            resampled = _resample_to_n_points(flst_info["cruise_df"], n_points)
            if resampled is None:
                raise ValueError(
                    f"FLST resample to {n_points} points failed "
                    f"(cruise segment too short?)"
                )
            # Estimate cruise fuel from the FLST reference simulation
            cruise_df_raw = flst_info["cruise_df"]
            cruise_fuel_est = float(
                cruise_df_raw["totalfuel"].iloc[-1]
                - cruise_df_raw["totalfuel"].iloc[0]
            )
            if cruise_fuel_est < 100:
                # Fallback: rough fuel-flow × cruise time
                cruise_fuel_est = 2.5 * float(resampled["ts"].iloc[-1])
            resampled = _fill_mass_profile(
                resampled, opt.mass_init, cruise_fuel_est,
                optimizer=opt,
            )
            tracks = generate_flst_track_variations(
                resampled,
                n_tracks=n_tracks,
                max_deviation_nm=max_deviation_nm,
            )
            log(f"  [{fid}] {len(tracks)} FLST-based initial guesses "
                f"(+/-{max_deviation_nm} NM)")
        else:
            tracks = generate_track_variations(
                flight["origin"], flight["destination"],
                n_tracks=n_tracks,
                max_deviation_nm=max_deviation_nm,
                n_points=dyn_nodes + 1,
            )
            log(f"  [{fid}] {len(tracks)} GC-based initial guesses "
                f"(+/-{max_deviation_nm} NM)")

        # -- sequential track optimisation (optimizer reused) --------------
        results: list[dict] = []
        for i_trk, track in enumerate(tracks):
            name = track["name"]
            ident = f"{fid}_{name}"
            t0 = time.time()
            try:
                init_df = update_init_guess(
                    track["init_guess_df"], opt, flst_based=use_flst,
                )
                t_opt = time.time()
                df = opt.trajectory(objective="fuel", initial_guess=init_df,
                                    h_min=h_min_m, fixed_mach=0.82)
                opt_s = time.time() - t_opt

                # -- extract IPOPT convergence stats ----------------------
                conv_status = "unknown"
                ipopt_iter = -1
                try:
                    if hasattr(opt, "opti"):
                        _stats = opt.opti.debug.stats()
                        conv_status = str(_stats.get("return_status", "unknown"))
                        ipopt_iter = int(_stats.get("iter_count", -1))
                except Exception:
                    pass

                if df is not None and len(df) > 0:
                    if "tas" in df.columns and "h" in df.columns:
                        df["cas"] = tas_kn_to_cas_kn(df["tas"], df["h"])
                    fuel = df["mass"].iloc[0] - df["mass"].iloc[-1]
                    nat_fuel, nat_dist = calculate_nat_fuel(df, NAT_POLYGON)
                    results.append({
                        "success": True,
                        "identifier": ident,
                        "name": name,
                        "trajectory": df,
                        "total_fuel": fuel,
                        "nat_fuel": nat_fuel,
                        "nat_distance": nat_dist,
                        "deviation_nm": track.get("deviation_nm", 0),
                        "setup_s": setup_s,
                        "opt_s": opt_s,
                        "total_s": time.time() - t0,
                        "convergence_status": conv_status,
                        "ipopt_iterations": ipopt_iter,
                    })
                    log(f"  [{fid}] [{i_trk+1}/{len(tracks)}] OK {ident:>20s}  "
                        f"fuel={fuel:,.0f} kg  (opt {opt_s:.1f}s  "
                        f"conv={conv_status}  iter={ipopt_iter})")
                else:
                    log(f"  [{fid}] [{i_trk+1}/{len(tracks)}] FAIL {ident:>20s}  "
                        f"empty trajectory  conv={conv_status}  iter={ipopt_iter}", "WARN")
            except Exception as exc:
                log(f"  [{fid}] [{i_trk+1}/{len(tracks)}] FAIL {ident:>20s}  "
                    f"{exc}", "WARN")

        if not no_wind:
            if wind_shifted is not None:
                del wind_shifted
            if cached_bsw is not None:
                del cached_bsw
        gc.collect()

        results.sort(key=lambda x: x["total_fuel"])
        best = results[0] if results else None

        elapsed = time.time() - t_start
        if best:
            log(f"  [{fid}] Best: {best['identifier']}  fuel={best['total_fuel']:,.0f} kg  "
                f"(total {elapsed:.1f}s)")

        return {
            "success": bool(results),
            "flightid": fid,
            "callsign": csn,
            "results": results,
            "best": best,
            "m0": m0,
            "mass_est": mass_est,
            "nodes": dyn_nodes,
            "error": None if results else "all tracks failed",
            "elapsed_s": elapsed,
            "used_synonym": used_synonym,
            "mtow": mtow,
            "aircraft_data": aircraft_data,
            "flst_based": use_flst,
        }

    except Exception as exc:
        if not no_wind:
            for _var in ("wind_shifted", "cached_bsw"):
                try:
                    del locals()[_var]
                except (NameError, KeyError):
                    pass
        gc.collect()
        log(f"  [{fid}] EXCEPTION: {exc}", "ERROR")
        traceback.print_exc()
        return {"success": False, "flightid": fid, "callsign": csn,
                "results": [], "best": None, "error": str(exc),
                "elapsed_s": time.time() - t_start,
                "used_synonym": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIMB / DESCENT PROFILE GENERATION
#  Uses BADA3-sourced aircraft data instead of oa.prop.aircraft()
# ═══════════════════════════════════════════════════════════════════════════════
def _climb_descent_params_from_aircraft(aircraft_data: dict) -> dict:
    """
    Derive climb/descent speed targets and distances from the optimiser's
    ``aircraft`` dict (which is built from BADA3 data in base.py).

    This replaces the old ``_climb_descent_params(actype)`` which called
    ``oa.prop.aircraft(actype)`` and would fail for synonym types.
    """
    mtow = aircraft_data.get("mtow", aircraft_data.get("limits", {}).get("MTOW", 80_000))

    if mtow > 250_000:
        return {"climb_cas": 310, "descent_cas": 290, "approach_cas": 250,
                "final_cas": 145, "toc_nm": 180, "tod_nm": 150}
    if mtow > 150_000:
        return {"climb_cas": 300, "descent_cas": 280, "approach_cas": 250,
                "final_cas": 140, "toc_nm": 150, "tod_nm": 120}
    if mtow > 80_000:
        return {"climb_cas": 290, "descent_cas": 280, "approach_cas": 250,
                "final_cas": 135, "toc_nm": 100, "tod_nm": 80}
    return {"climb_cas": 250, "descent_cas": 250, "approach_cas": 200,
            "final_cas": 120, "toc_nm": 60, "tod_nm": 50}


def _create_climb_profile(origin, first_cruise, aircraft_data):
    p = _climb_descent_params_from_aircraft(aircraft_data)
    o = oa.nav.airport(origin)
    olat, olon = o["lat"], o["lon"]
    clat, clon = first_cruise["latitude"], first_cruise["longitude"]
    calt = first_cruise["altitude"]
    ccas = first_cruise.get("cas", p["climb_cas"])

    targets = [
        (0.15, 10_000, 250),
        (0.35, 24_000, p["climb_cas"]),
        (0.60, calt * 0.85, p["climb_cas"]),
        (0.85, calt * 0.95, ccas),
    ]
    wps = []
    for frac, alt, cas in targets:
        la, lo = gc_intermediate_point(olat, olon, clat, clon, frac)
        wps.append({"latitude": la, "longitude": lo, "altitude": min(alt, calt), "cas": cas})
    return wps


def _create_descent_profile(last_cruise, destination, aircraft_data):
    p = _climb_descent_params_from_aircraft(aircraft_data)
    d = oa.nav.airport(destination)
    dlat, dlon = d["lat"], d["lon"]
    dalt = d.get("alt", 0)
    clat, clon = last_cruise["latitude"], last_cruise["longitude"]
    calt = last_cruise["altitude"]
    ccas = last_cruise.get("cas", p["descent_cas"])

    targets = [
        (0.15, calt * 0.90, ccas),
        (0.35, 24_000, p["descent_cas"]),
        (0.55, 18_000, p["descent_cas"]),
        (0.70, 11_000, 250),
        (0.85, 6000, p["approach_cas"]),
        (0.95, 3000, p["approach_cas"]),
    ]
    wps = []
    for frac, alt, cas in targets:
        la, lo = gc_intermediate_point(clat, clon, dlat, dlon, frac)
        wps.append({"latitude": la, "longitude": lo, "altitude": max(alt, dalt + 500), "cas": cas})
    wps.append({"latitude": dlat, "longitude": dlon, "altitude": dalt, "cas": p["final_cas"]})
    return wps


# ═══════════════════════════════════════════════════════════════════════════════
#  BLUESKY SCENARIO EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
def _format_continuous_time(dep_dt: datetime, reference_midnight: datetime) -> str:
    """Format departure time as continuous timestamp relative to reference midnight.

    Hours can exceed 24 for flights departing on subsequent days.
    Example: reference = 2025-07-17 00:00, dep = 2025-07-18 01:00 -> '25:00:00.00>'
    """
    total_seconds = (dep_dt - reference_midnight).total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    frac_seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{frac_seconds:05.2f}>"


def _format_continuous_time_nav(dep_dt: datetime, reference_midnight: datetime,
                                offset_s: float = 60) -> str:
    """Like _format_continuous_time but with an extra offset (for LNAV/VNAV commands)."""
    total_seconds = (dep_dt - reference_midnight).total_seconds() + offset_s
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}>"


def _build_block_with_climb_descent(identifier, flight_info, traj_df, aircraft_data,
                                    reference_midnight):
    """
    Build BlueSky .scn block with synthetic climb + cruise + descent.
    Aircraft created at airport ground level.
    """
    origin = flight_info["origin"]
    destination = flight_info["destination"]
    dep = flight_info["dep_time"]
    if isinstance(dep, str):
        t_dep = datetime.strptime(dep, "%Y-%m-%d %H:%M:%S")
    else:
        t_dep = dep
    dep_str = _format_continuous_time(t_dep, reference_midnight)

    o = oa.nav.airport(origin)
    d = oa.nav.airport(destination)
    olat, olon, oalt = o["lat"], o["lon"], o.get("alt", 0)
    dlat, dlon, dalt = d["lat"], d["lon"], d.get("alt", 0)

    # Trim cruise ends overlapping with climb/descent
    p = _climb_descent_params_from_aircraft(aircraft_data)
    dists_orig, dists_dest = [], []
    for _, row in traj_df.iterrows():
        dists_orig.append(gc_distance_nm_coords(olat, olon, row["latitude"], row["longitude"]))
        dists_dest.append(gc_distance_nm_coords(row["latitude"], row["longitude"], dlat, dlon))
    traj_w = traj_df.copy()
    traj_w["d_orig"] = dists_orig
    traj_w["d_dest"] = dists_dest
    trimmed = traj_w[
        (traj_w["d_orig"] >= p["toc_nm"] * 0.8) &
        (traj_w["d_dest"] >= p["tod_nm"] * 0.8)
    ]
    if len(trimmed) < 5:
        trimmed = traj_df.iloc[2:-2]

    first_cr = {
        "latitude": trimmed["latitude"].iloc[0],
        "longitude": trimmed["longitude"].iloc[0],
        "altitude": trimmed["altitude"].iloc[0],
        "cas": trimmed["cas"].iloc[0] if "cas" in trimmed.columns else 280,
    }
    last_cr = {
        "latitude": trimmed["latitude"].iloc[-1],
        "longitude": trimmed["longitude"].iloc[-1],
        "altitude": trimmed["altitude"].iloc[-1],
        "cas": trimmed["cas"].iloc[-1] if "cas" in trimmed.columns else 280,
    }
    climb = _create_climb_profile(origin, first_cr, aircraft_data)
    descent = _create_descent_profile(last_cr, destination, aircraft_data)

    block = [
        f"# Flight {identifier} from {origin} to {destination}\n",
        f"# Optimised cruise with synthetic climb/descent\n",
        f"{dep_str}CRE {identifier}, {flight_info['actype']}, {olat}, {olon}, 90, {oalt}, 100\n",
    ]

    block += [
        f"{dep_str}ORIG {identifier} {origin}\n",
        f"{dep_str}DEST {identifier} {destination}\n",
    ]

    block.append(f"# === CLIMB ({len(climb)} wpts) ===\n")
    for wp in climb:
        block.append(
            f"{dep_str}ADDWPT {identifier}, {wp['latitude']:.6f}, {wp['longitude']:.6f}, "
            f"{wp['altitude']:.0f}, {wp['cas']:.1f}\n"
        )
    block.append(f"# === CRUISE ({len(trimmed)} wpts) ===\n")
    for _, row in trimmed.iterrows():
        spd = row["cas"] if "cas" in row else 280
        block.append(
            f"{dep_str}ADDWPT {identifier}, {row['latitude']:.6f}, {row['longitude']:.6f}, "
            f"{row['altitude']:.0f}, {spd:.1f}\n"
        )
    block.append(f"# === DESCENT ({len(descent)} wpts) ===\n")
    for wp in descent:
        block.append(
            f"{dep_str}ADDWPT {identifier}, {wp['latitude']:.6f}, {wp['longitude']:.6f}, "
            f"{wp['altitude']:.0f}, {wp['cas']:.1f}\n"
        )

    nav_t = _format_continuous_time_nav(t_dep, reference_midnight)
    block += [
        f"{nav_t}SWTOC {identifier}, OFF\n",
        f"{nav_t}LNAV {identifier}, ON\n",
        f"{nav_t}VNAV {identifier}, ON\n",
    ]
    return block


def _build_block_cruisestart(identifier, flight_info, traj_df, cruise_mass_kg, aircraft_data,
                              reference_midnight, flst_based=False):
    """
    Build BlueSky .scn block where the aircraft is CREated at the 120 NM
    trim waypoint (first point where cumulative GC distance from WP0
    exceeds TRIM_DISTANCE_NM).  Uses the trajectory mass at the trim
    point for SETMASS.

    When *flst_based* is True the trajectory already starts at the
    trim point (Option A optimisation from FLST cruise segment), so
    no additional trimming is applied — WP0 is used directly.

    If the trajectory is shorter than TRIM_DISTANCE_NM, falls back to
    the first waypoint (v6 behaviour).

    Returns ``(block_lines, trim_info)`` where *trim_info* is a dict
    with trim-point metadata (or ``None`` if no trim was applied).
    """
    origin = flight_info["origin"]
    destination = flight_info["destination"]
    dep = flight_info["dep_time"]
    if isinstance(dep, str):
        t_dep = datetime.strptime(dep, "%Y-%m-%d %H:%M:%S")
    else:
        t_dep = dep
    dep_str = _format_continuous_time(t_dep, reference_midnight)

    # --- Find trim point -------------------------------------------------
    if flst_based:
        # FLST-based: trajectory already starts at the 120 NM cruise trim
        # point.  Use WP0 directly — no additional trimming needed.
        trim_info = {
            "trim_index": 0,
            "trim_lat": float(traj_df["latitude"].iloc[0]),
            "trim_lon": float(traj_df["longitude"].iloc[0]),
            "trim_alt_ft": float(traj_df["altitude"].iloc[0]),
            "trim_cas": float(traj_df["cas"].iloc[0]) if "cas" in traj_df.columns else 280.0,
            "trim_heading": _compute_bearing(
                float(traj_df["latitude"].iloc[0]),
                float(traj_df["longitude"].iloc[0]),
                float(traj_df["latitude"].iloc[min(1, len(traj_df) - 1)]),
                float(traj_df["longitude"].iloc[min(1, len(traj_df) - 1)]),
            ),
            "trim_mass_kg": float(traj_df["mass"].iloc[0]),
            "cumulative_dist_nm": 0.0,
            "flst_based": True,
        }
        trimmed_traj = traj_df
    else:
        trim_info = _find_trim_point(traj_df, trim_nm=TRIM_DISTANCE_NM)

    if flst_based:
        # FLST: trim_info + trimmed_traj already set above
        trim_lat = trim_info["trim_lat"]
        trim_lon = trim_info["trim_lon"]
        trim_alt = trim_info["trim_alt_ft"]
        trim_spd = trim_info["trim_cas"]
        trim_hdg = trim_info["trim_heading"]
        trim_mass = trim_info["trim_mass_kg"]
    elif trim_info is not None:
        trim_idx = trim_info["trim_index"]
        trimmed_traj = traj_df.iloc[trim_idx:].reset_index(drop=True)
        trim_lat = trim_info["trim_lat"]
        trim_lon = trim_info["trim_lon"]
        trim_alt = trim_info["trim_alt_ft"]
        trim_spd = trim_info["trim_cas"]
        trim_hdg = trim_info["trim_heading"]
        trim_mass = trim_info["trim_mass_kg"]
    else:
        # Fallback: use first waypoint (v6 behaviour)
        trimmed_traj = traj_df
        trim_lat = float(traj_df["latitude"].iloc[0])
        trim_lon = float(traj_df["longitude"].iloc[0])
        trim_alt = float(traj_df["altitude"].iloc[0])
        trim_spd = float(traj_df["cas"].iloc[0]) if "cas" in traj_df.columns else 280.0
        trim_hdg = 90.0
        trim_mass = cruise_mass_kg

    block = [
        f"# Flight {identifier} from {origin} to {destination}\n",
        f"# Cruise-start: aircraft created at {TRIM_DISTANCE_NM} NM trim waypoint with optimiser mass\n",
        f"{dep_str}CRE {identifier}, {flight_info['actype']}, "
        f"{trim_lat:.6f}, {trim_lon:.6f}, {trim_hdg:.0f}, {trim_alt:.0f}, {trim_spd:.1f}\n",
    ]

    block += [
        f"{dep_str}ORIG {identifier} {origin}\n",
        f"{dep_str}DEST {identifier} {destination}\n",
        f"# SETMASS {identifier} {trim_mass:.1f}\n",
        f"{dep_str}SETMASS {identifier}, {trim_mass:.1f}\n",
    ]

    block.append(f"# === CRUISE ({len(trimmed_traj)} wpts from {TRIM_DISTANCE_NM} NM onward) ===\n")
    for _, row in trimmed_traj.iterrows():
        spd = row["cas"] if "cas" in row else 280
        block.append(
            f"{dep_str}ADDWPT {identifier}, {row['latitude']:.6f}, {row['longitude']:.6f}, "
            f"{row['altitude']:.0f}, {spd:.1f}\n"
        )

    # Synthetic descent profile from last cruise waypoint to destination
    last_cr = {
        "latitude": float(trimmed_traj["latitude"].iloc[-1]),
        "longitude": float(trimmed_traj["longitude"].iloc[-1]),
        "altitude": float(trimmed_traj["altitude"].iloc[-1]),
        "cas": float(trimmed_traj["cas"].iloc[-1]) if "cas" in trimmed_traj.columns else 280,
    }
    descent = _create_descent_profile(last_cr, destination, aircraft_data)

    block.append(f"# === DESCENT ({len(descent)} wpts) ===\n")
    for wp in descent:
        block.append(
            f"{dep_str}ADDWPT {identifier}, {wp['latitude']:.6f}, {wp['longitude']:.6f}, "
            f"{wp['altitude']:.0f}, {wp['cas']:.1f}\n"
        )

    nav_t = _format_continuous_time_nav(t_dep, reference_midnight)
    block += [
        f"{nav_t}SWTOC {identifier}, OFF\n",
        f"{nav_t}LNAV {identifier}, ON\n",
        f"{nav_t}VNAV {identifier}, ON\n",
    ]
    return block, trim_info


# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINE SCENARIO ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_baseline_scenario(scn_path: str) -> Tuple[dict, list]:
    """Parse a baseline BlueSky ``.scn`` file into per-flight line blocks.

    Returns
    -------
    blocks : dict
        ``{flightid: [line, ...]}`` preserving original line content.
    order : list
        Ordered list of flight IDs as they appear in the file.
    """
    with open(scn_path, "r", encoding="utf-8") as fh:
        all_lines = fh.readlines()

    blocks: dict[str, list[str]] = {}
    order: list[str] = []
    current_fid: str | None = None

    for line in all_lines:
        # Detect CRE line -> start of a new flight block
        if ">CRE " in line:
            # Parse flight ID: everything between ">CRE " and the first comma
            after_cre = line.split(">CRE ", 1)[1]
            fid = after_cre.split(",")[0].strip()
            current_fid = fid
            if fid not in blocks:
                blocks[fid] = []
                order.append(fid)

        if current_fid is not None:
            blocks[current_fid].append(line)

    return blocks, order


def _align_baseline_flight(lines: list[str],
                           trim_alt_ft: float,
                           trim_mass_kg: float) -> list[str]:
    """Modify a single flight's baseline scenario block for aligned comparison.

    1. Find the first ``ADDWPT`` where altitude >= *trim_alt_ft*.
    2. Rewrite the ``CRE`` line to spawn at that waypoint.
    3. Insert a ``SETMASS`` line right after ``CRE`` with *trim_mass_kg*.
    4. Remove all ``ADDWPT`` lines before the trim point.
    5. Update ``ALT`` and ``SPD`` lines to match the trim point.
    6. Keep everything from the trim point onward unchanged.

    Returns the modified list of lines.  If no matching altitude is found,
    returns the original lines unchanged.
    """
    # --- Collect ADDWPT info (line index, altitude, raw parts) -----------
    addwpt_info: list[Tuple[int, float, list[str]]] = []
    for i, line in enumerate(lines):
        if ">ADDWPT " in line:
            after = line.split(">ADDWPT ", 1)[1]
            parts = [p.strip() for p in after.split(",")]
            # parts: [fid, lat, lon, alt, spd]
            if len(parts) >= 5:
                try:
                    alt = float(parts[3])
                    addwpt_info.append((i, alt, parts))
                except ValueError:
                    continue

    if not addwpt_info:
        return lines

    # --- Find the first ADDWPT where altitude >= trim_alt_ft -------------
    trim_wp_idx = None
    for k, (line_idx, alt, parts) in enumerate(addwpt_info):
        if alt >= trim_alt_ft:
            trim_wp_idx = k
            break

    if trim_wp_idx is None:
        # trim_alt_ft higher than any baseline altitude — skip alignment
        return lines

    _, trim_alt, trim_parts = addwpt_info[trim_wp_idx]
    trim_fid = trim_parts[0]
    trim_lat = float(trim_parts[1])
    trim_lon = float(trim_parts[2])
    # Speed could be CAS (integer-ish) or Mach (decimal < 1)
    trim_spd_raw = trim_parts[4].rstrip("\n").strip()

    # --- Compute heading from trim WP to next WP ------------------------
    heading = 90.0
    if trim_wp_idx + 1 < len(addwpt_info):
        next_parts = addwpt_info[trim_wp_idx + 1][2]
        next_lat = float(next_parts[1])
        next_lon = float(next_parts[2])
        heading = _compute_bearing(trim_lat, trim_lon, next_lat, next_lon)

    # --- Set of ADDWPT line indices to remove (before trim point) --------
    remove_indices = {addwpt_info[k][0] for k in range(trim_wp_idx)}

    # --- Build modified block --------------------------------------------
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        if i in remove_indices:
            continue  # skip pre-trim ADDWPT lines

        if ">CRE " in line:
            # Rewrite CRE with trim-point position/altitude/speed
            ts_prefix = line.split(">")[0] + ">"
            cre_content = line.split(">CRE ", 1)[1]
            cre_parts = [p.strip() for p in cre_content.split(",")]
            fid = cre_parts[0]
            actype = cre_parts[1]
            new_lines.append(
                f"{ts_prefix}CRE {fid}, {actype}, {trim_lat:.6f}, {trim_lon:.6f}, "
                f"{heading:.0f}, {trim_alt:.0f}, {trim_spd_raw}\n"
            )
            # Insert SETMASS right after CRE
            new_lines.append(
                f"{ts_prefix}SETMASS {fid}, {trim_mass_kg:.1f}\n"
            )

        elif ">ALT " in line:
            # Update ALT command to trim altitude
            ts_prefix = line.split(">")[0] + ">"
            after_alt = line.split(">ALT ", 1)[1]
            fid = after_alt.split()[0]
            new_lines.append(f"{ts_prefix}ALT {fid} {trim_alt:.0f}\n")

        elif ">SPD " in line:
            # Update SPD command to trim speed
            ts_prefix = line.split(">")[0] + ">"
            after_spd = line.split(">SPD ", 1)[1]
            fid = after_spd.split()[0]
            new_lines.append(f"{ts_prefix}SPD {fid} {trim_spd_raw}\n")

        else:
            new_lines.append(line)

    return new_lines


def _write_aligned_baseline(blocks: dict[str, list[str]],
                            order: list[str],
                            output_path: str) -> str:
    """Write the (possibly modified) baseline blocks back to a .scn file."""
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("# Aligned baseline scenario — trimmed to match optimised "
                 "cruise-start altitude\n")
        fh.write(f"# Generated by batch_optimise_v11.py on "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for fid in order:
            if fid in blocks:
                fh.writelines(blocks[fid])
    log(f"Aligned baseline saved -> {output_path}  ({len(order)} flights)")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
def build_scenarios(flights_info, all_results, top_k=3):
    """
    Build THREE scenario dicts, sorted by departure time (continuous time):
      1. scenario_all        — all successful flights
      2. scenario_wosyn      — exclude flights that used synonym data
      3. scenario_cruisestart — aircraft created at 120 NM trim waypoint

    Also collects trim info from the GC variant of each flight for use
    in baseline alignment.

    Returns
    -------
    scenario_all, scenario_wosyn, scenario_cruisestart : dict
    reference_midnight : datetime
    gc_trim_info : dict
        ``{flightid: trim_info_dict}`` — trim-point metadata from the
        GC variant (deviation_nm ≈ 0) of each flight.  Used for
        baseline alignment.
    """
    # Compute reference midnight (midnight of the earliest departure date)
    dep_datetimes = [
        datetime.strptime(f["dep_time"], "%Y-%m-%d %H:%M:%S") for f in flights_info
    ]
    reference_midnight = min(dep_datetimes).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    log(f"  Scenario reference midnight: {reference_midnight}")

    # Collect entries with departure time for sorting
    entries = []
    gc_trim_info: dict[str, dict | None] = {}

    for res in all_results:
        if not res["success"]:
            continue
        fid = res["flightid"]
        flight = next((f for f in flights_info if f["flightid"] == fid), None)
        if flight is None:
            continue

        used_synonym = res.get("used_synonym", False)
        aircraft_data = res.get("aircraft_data", {})
        is_flst_based = res.get("flst_based", False)
        dep_dt = datetime.strptime(flight["dep_time"], "%Y-%m-%d %H:%M:%S")

        for track_res in res["results"][:top_k]:
            ident = track_res["identifier"]
            traj_df = track_res["trajectory"]

            block = None
            block_cs = None
            trim_info_cs = None

            try:
                block = _build_block_with_climb_descent(
                    ident, flight, traj_df, aircraft_data, reference_midnight,
                )
            except Exception as exc:
                log(f"  Scenario build failed for {ident}: {exc}", "WARN")

            # Cruise-start scenario: mass from first row of trajectory
            try:
                cruise_mass = float(traj_df["mass"].iloc[0])
                block_cs, trim_info_cs = _build_block_cruisestart(
                    ident, flight, traj_df, cruise_mass, aircraft_data,
                    reference_midnight, flst_based=is_flst_based,
                )
            except Exception as exc:
                log(f"  Cruise-start scenario build failed for {ident}: {exc}", "WARN")

            # Store GC/FLST variant trim info for baseline alignment
            is_baseline_variant = (
                abs(track_res.get("deviation_nm", 999)) < 1.0
                or track_res.get("name", "") in ("gc", "flst")
            )
            if is_baseline_variant and fid not in gc_trim_info:
                gc_trim_info[fid] = trim_info_cs

            entries.append({
                "dep_dt": dep_dt,
                "ident": ident,
                "block": block,
                "block_cs": block_cs,
                "used_synonym": used_synonym,
            })

    # Sort all entries by departure time, then by identifier for stable ordering
    entries.sort(key=lambda e: (e["dep_dt"], e["ident"]))

    # Build ordered scenario dicts
    scenario_all = {}
    scenario_wosyn = {}
    scenario_cruisestart = {}

    for e in entries:
        if e["block"] is not None:
            scenario_all[e["ident"]] = e["block"]
            if not e["used_synonym"]:
                scenario_wosyn[e["ident"]] = e["block"]
        if e["block_cs"] is not None:
            scenario_cruisestart[e["ident"]] = e["block_cs"]

    return scenario_all, scenario_wosyn, scenario_cruisestart, reference_midnight, gc_trim_info


def _sort_scenario_lines(scenario: dict) -> list[str]:
    """Flatten scenario blocks and sort all lines by BlueSky timestamp.

    Comment lines are grouped with the next timestamped line so that
    flight headers and section markers stay attached to their commands.
    This prevents out-of-order timestamps when LNAV/VNAV lines
    (departure + 60 s) interleave with CRE lines of other flights.
    """
    all_lines: list[str] = []
    for block in scenario.values():
        all_lines.extend(block)

    groups: list[tuple[float, list[str]]] = []
    pending: list[str] = []

    for line in all_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            pending.append(line)
            continue
        # Parse timestamp: HH:MM:SS.ss> or HH:MM:SS>
        m = re.match(r'(\d+):(\d+):([\d.]+)>', stripped)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            sort_key = h * 3600 + mi * 60 + s
        else:
            sort_key = 0.0
        groups.append((sort_key, pending + [line]))
        pending = []

    if pending:
        groups.append((float('inf'), pending))

    # Stable sort by timestamp
    groups.sort(key=lambda g: g[0])

    result: list[str] = []
    for _, lines in groups:
        result.extend(lines)
    return result


def save_scenario(scenario: dict, path: str,
                  reference_midnight: datetime = None) -> str:
    sorted_lines = _sort_scenario_lines(scenario)
    with open(path, "w", encoding="utf-8") as fh:
        if reference_midnight is not None:
            fh.write(f"# Reference date (midnight): "
                     f"{reference_midnight.strftime('%Y-%m-%d')}\n")
            fh.write(f"# Timestamps are continuous time relative to "
                     f"{reference_midnight.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"# Hours may exceed 24:00 for flights departing "
                     f"on subsequent days\n\n")
        fh.writelines(sorted_lines)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  ARTIFACT PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════
def save_artifacts(traj_dir, identifier, flight, traj_df, deviation_nm,
                   extra_meta, cfg, used_synonym=False):
    os.makedirs(traj_dir, exist_ok=True)
    pkl = os.path.join(traj_dir, f"{identifier}_traj.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(traj_df, f)

    first_wp = {
        "altitude_ft": float(traj_df["altitude"].iloc[0]),
        "latitude": float(traj_df["latitude"].iloc[0]),
        "longitude": float(traj_df["longitude"].iloc[0]),
        "mass_kg": float(traj_df["mass"].iloc[0]),
    }
    last_wp = {
        "altitude_ft": float(traj_df["altitude"].iloc[-1]),
        "latitude": float(traj_df["latitude"].iloc[-1]),
        "longitude": float(traj_df["longitude"].iloc[-1]),
        "mass_kg": float(traj_df["mass"].iloc[-1]),
    }

    meta = {
        "identifier": identifier,
        "flightid": flight["flightid"],
        "callsign": flight["callsign"],
        "actype": flight["actype"],
        "origin": flight["origin"],
        "destination": flight["destination"],
        "dep_time": str(flight["dep_time"]),
        "m0": float(flight.get("m0", cfg["m0"])),
        "method": "Cruise_MultiStart",
        "perf_model": cfg["perf_model"],
        "no_wind": cfg.get("no_wind", False),
        "wind_method": cfg.get("wind_method", "poly"),
        "bspline_degree": cfg.get("bspline_degree"),
        "bspline_subsample": cfg.get("bspline_subsample"),
        "time_subsample": cfg.get("time_subsample", 1),
        "deviation_nm": deviation_nm,
        "n_tracks": cfg["n_tracks"],
        "max_deviation_nm": cfg["max_dev"],
        "nodes": cfg.get("nodes_used"),
        "polydeg": cfg["polydeg"],
        "ipopt": cfg["ipopt"],
        "first_waypoint": first_wp,
        "last_waypoint": last_wp,
        "used_synonym": used_synonym,
        "timestamp": datetime.now().isoformat(),
    }
    meta.update(extra_meta or {})
    json_path = os.path.join(traj_dir, f"{identifier}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return pkl, json_path


def save_ranking(all_results, output_dir):
    rows = []
    for fr in all_results:
        if not fr["success"]:
            continue
        for rank, r in enumerate(fr["results"], 1):
            rows.append({
                "flightid": fr["flightid"],
                "callsign": fr["callsign"],
                "rank": rank,
                "track": r["name"],
                "identifier": r["identifier"],
                "total_fuel_kg": r["total_fuel"],
                "nat_fuel_kg": r.get("nat_fuel", 0),
                "nat_distance_nm": r.get("nat_distance", 0),
                "deviation_nm": r.get("deviation_nm", 0),
                "used_synonym": fr.get("used_synonym", False),
            })
    df = pd.DataFrame(rows)
    p = os.path.join(output_dir, "optimization_rankings.csv")
    df.to_csv(p, index=False, sep=";")
    log(f"Rankings saved -> {p}")
    return df


def save_failed_flights(failed_flights: list, output_dir: str):
    """Persist the list of flights that failed optimisation."""
    if not failed_flights:
        log("No failed flights to record.")
        return
    df = pd.DataFrame(failed_flights)
    p = os.path.join(output_dir, "failed_flights.csv")
    df.to_csv(p, index=False, sep=";")
    log(f"Failed flights saved -> {p}  ({len(failed_flights)} flights)")


def save_run_metadata(output_dir: str, cfg: dict, flights_info: list,
                      all_results: list, failed_flights: list, skipped: int):
    n_synonym = sum(1 for r in all_results if r.get("used_synonym", False) and r["success"])
    n_flst = sum(1 for r in all_results if r.get("flst_based", False) and r["success"])
    meta = {
        "run_timestamp": datetime.now().isoformat(),
        "config": cfg,
        "no_wind": cfg.get("no_wind", False),
        "flst_log": cfg.get("flst_log"),
        "n_flights": len(flights_info),
        "flights": [
            {"flightid": f["flightid"], "callsign": f["callsign"],
             "actype": f["actype"], "origin": f["origin"],
             "destination": f["destination"], "dep_time": f["dep_time"]}
            for f in flights_info
        ],
        "summary": {
            "successful": sum(1 for r in all_results if r["success"]),
            "failed": len(failed_flights),
            "skipped_already_done": skipped,
            "used_synonym": n_synonym,
            "flst_based": n_flst,
            "total_elapsed_s": sum(r.get("elapsed_s", 0) for r in all_results),
        },
    }
    p = os.path.join(output_dir, "run_metadata.json")
    with open(p, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log(f"Run metadata saved -> {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_batch(cfg: dict) -> list:
    wall_t0 = time.time()
    clear_memory()

    no_wind = cfg.get("no_wind", False)

    _, flights_info = load_flight_list(cfg["csv"], m0=cfg["m0"])

    # ── Load baseline scenario (if provided) ──────────────────────────
    baseline_blocks = None
    baseline_order = None
    baseline_scn_path = cfg.get("baseline_scenario")
    if baseline_scn_path:
        if os.path.isfile(baseline_scn_path):
            baseline_blocks, baseline_order = _parse_baseline_scenario(baseline_scn_path)
            log(f"Loaded baseline scenario: {len(baseline_blocks)} flights "
                f"from {baseline_scn_path}")
        else:
            log(f"Baseline scenario file not found: {baseline_scn_path}", "WARN")

    # ── Load FLST initial guesses (if --flst-log provided) ────────────
    flst_guesses: dict[str, dict] = {}
    flst_log_path = cfg.get("flst_log")
    if flst_log_path:
        if os.path.isfile(flst_log_path):
            flst_guesses = load_flst_initial_guesses(
                flst_log_path, flights_info,
                trim_nm=TRIM_DISTANCE_NM,
            )
        else:
            log(f"FLST log file not found: {flst_log_path} "
                f"— all flights will use GC initial guess", "WARN")

    # ── Resume: find already-completed flights ─────────────────────────
    traj_dir = os.path.join(cfg["output"], "trajectories")
    completed_fids = _find_completed_flights(traj_dir)
    if completed_fids:
        log(f"Resume: found {len(completed_fids)} already-completed flights in {traj_dir}")

    # ── Load wind (skip entirely when --no-wind) ──────────────────────
    if no_wind:
        log("No-wind mode: skipping wind data loading entirely.")
        wind_data = None
        weather_start = None
    else:
        wind_data, weather_start = load_wind_from_local_netcdf(
            flights_info,
            netcdf_dir=cfg["wind_dir"],
            flight_phase=cfg.get("flight_phase", "cruise"),
        )
        if wind_data is None:
            log("No wind data -- aborting.", "ERROR")
            return []

    ipopt_kw = {
        "max_iter": cfg["max_iter"],
        "tol": cfg["tol"],
        "acceptable_tol": cfg["acceptable_tol"],
        "print_level": cfg.get("ipopt_print", 0),
        "hessian_approximation": "exact",
        "mu_strategy": "adaptive",
    }

    # ── Wind method parameters ─────────────────────────────────────────
    wind_method = cfg.get("wind_method", "poly")
    bspline_degree = cfg.get("bspline_degree", 3)
    bspline_subsample = cfg.get("bspline_subsample", 1)
    time_subsample = cfg.get("time_subsample", 1)

    # CasADi 3.7 limitation: 'bspline' method with degree>=3 crashes
    # (SystemError) for 4-D grids above ~8,000 points.  Warn the user.
    if not no_wind and wind_method == "bspline" and bspline_degree >= 3:
        grid_size = len(wind_data)
        log(f"WARNING: --wind-method=bspline with --bspline-degree={bspline_degree} "
            f"is known to crash in CasADi 3.7 for large 4-D grids (>{8_000:,} pts). "
            f"Current wind grid has {grid_size:,} rows.  "
            f"Consider --bspline-degree=1 or --wind-method=linear instead.", "WARN")

    if no_wind:
        wind_info = "NONE (no-wind mode)"
    else:
        wind_info = wind_method
        if wind_method in ("linear", "bspline"):
            wind_info += f" (degree={bspline_degree}, sub={bspline_subsample})"

    # ── v11: Pre-build wind interpolant cache per departure-hour bucket ──
    wind_cache_dir = None
    wind_cache_map = None
    if not no_wind and wind_method in ("linear", "bspline"):
        wind_cache_dir = os.path.join(cfg["output"], "wind_cache")
        log(f"\n  Pre-building wind interpolant cache -> {wind_cache_dir}")
        wind_cache_map = precompute_wind_cache(
            wind_data, weather_start, flights_info,
            cache_dir=wind_cache_dir,
            wind_method=wind_method,
            bspline_degree=bspline_degree,
            bspline_subsample=bspline_subsample,
            time_subsample=time_subsample,
        )
        log(f"  Wind cache: {len(wind_cache_map)} buckets ready")

    log(f"\n{'=' * 72}")
    flst_info_str = f"  | FLST: {len(flst_guesses)} flights" if flst_guesses else ""
    log(f"BATCH OPTIMISATION  {len(flights_info)} flights  "
        f"| {cfg['n_tracks']} tracks  +/-{cfg['max_dev']} NM  "
        f"| {cfg['workers']} worker processes  "
        f"| wind: {wind_info}{flst_info_str}")
    log(f"{'=' * 72}")

    all_results: list[dict] = []
    failed_flights: list[dict] = []
    skipped_count = 0

    # ── Handle resumed flights (main process, no pool needed) ─────────
    flights_to_process: list[dict] = []
    for flight in flights_info:
        fid = flight["flightid"]
        if fid in completed_fids:
            log(f"  [{fid}] SKIP: already optimised (resuming)")
            existing = _load_existing_results(traj_dir, fid)
            if existing:
                if cfg["perf_model"] == "bada3":
                    syn, _ = _detect_bada3_synonym(flight["actype"], cfg["bada_path"])
                else:
                    syn, _ = _detect_openap_synonym(flight["actype"])
                existing["used_synonym"] = syn
                existing["callsign"] = flight["callsign"]
                try:
                    opt_tmp = top.Cruise(
                        flight["actype"], flight["origin"], flight["destination"],
                        m0=0.85, perf_model=cfg["perf_model"],
                        bada3_path=cfg["bada_path"], debug=False,
                    )
                    existing["aircraft_data"] = opt_tmp.aircraft
                    del opt_tmp
                except Exception:
                    existing["aircraft_data"] = {}
                all_results.append(existing)

                # ── Baseline alignment for resumed flights ────────────
                if baseline_blocks is not None and existing["success"]:
                    _try_align_baseline(existing, fid, baseline_blocks)

            skipped_count += 1
        else:
            flights_to_process.append(flight)

    log(f"\n  {len(flights_to_process)} flights to optimise, "
        f"{skipped_count} resumed, {cfg['workers']} worker processes")

    # ── Flight-level parallel processing ──────────────────────────────
    wind_tmp = None
    flst_tmp = None
    if flights_to_process:
        # -- Save wind data to a temp file so workers load from disk ----
        #    v11: Only needed for poly wind method (no cache).  For
        #    linear/bspline, workers load from the wind_cache_dir.
        if not no_wind and wind_cache_dir is None:
            # poly method or fallback — workers need the raw DataFrame
            wind_tmp = os.path.join(
                tempfile.gettempdir(), f"batch_wind_{os.getpid()}.pkl",
            )
            wind_mb = wind_data.memory_usage(deep=True).sum() / 1024**2
            wind_data.to_pickle(wind_tmp)
            log(f"  Wind data saved to temp file ({wind_mb:.0f} MB)")
        elif not no_wind:
            # v11 cached path — no temp file needed, workers load from cache
            wind_mb = wind_data.memory_usage(deep=True).sum() / 1024**2
            log(f"  Wind cache active — no temp wind file needed "
                f"(original data was {wind_mb:.0f} MB)")
        else:
            wind_mb = 0
            log("  No-wind mode: no temp wind file needed.")

        # -- Save FLST guesses to temp file (if any) -------------------
        if flst_guesses:
            flst_tmp = os.path.join(
                tempfile.gettempdir(), f"batch_flst_{os.getpid()}.pkl",
            )
            with open(flst_tmp, "wb") as fh:
                pickle.dump(flst_guesses, fh)
            log(f"  FLST guesses saved to temp file "
                f"({len(flst_guesses)} flights)")

        # -- Auto-cap workers based on available RAM --------------------
        mem = psutil.virtual_memory()
        avail_mb = mem.available / 1024**2
        if no_wind:
            per_worker_mb = 500
        elif wind_cache_dir is not None:
            # v11: cached path — each worker only loads lightweight CasADi
            # Function objects (~50–200 MB) instead of the full DataFrame
            per_worker_mb = 300
        else:
            # Legacy: each worker loads wind DataFrame + optimizer overhead
            per_worker_mb = wind_mb * 2.5
        reserve_mb = 4096          # keep 4 GB for OS + main process
        max_safe = max(1, int((avail_mb - reserve_mb) / max(per_worker_mb, 1)))
        n_workers = min(cfg["workers"], max_safe)
        if n_workers < cfg["workers"]:
            log(f"  Memory guard: {avail_mb:.0f} MB available, "
                f"~{per_worker_mb:.0f} MB/worker -> capping at "
                f"{n_workers} workers (requested {cfg['workers']})", "WARN")
        else:
            n_workers = cfg["workers"]

        # Free wind DataFrame in main process (workers load from file/cache)
        if not no_wind:
            del wind_data
        gc.collect()

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_flight_worker,
            initargs=(
                wind_tmp, weather_start,
                cfg["perf_model"], cfg["bada_path"],
                cfg["polydeg"], ipopt_kw,
                cfg["m0"], cfg["n_tracks"],
                cfg["max_dev"], cfg.get("nodes"),
                wind_method, bspline_degree, bspline_subsample,
                time_subsample,
                no_wind,
                flst_tmp,
                wind_cache_dir,  # v11
            ),
        ) as pool:
            future_to_flight = {
                pool.submit(_flight_worker, flight): flight
                for flight in flights_to_process
            }

            done_count = 0
            for fut in as_completed(future_to_flight):
                done_count += 1
                flight = future_to_flight[fut]
                fid = flight["flightid"]

                try:
                    result = fut.result()
                except Exception as exc:
                    log(f"  [{fid}] PROCESS EXCEPTION: {exc}", "ERROR")
                    traceback.print_exc()
                    result = {
                        "success": False, "flightid": fid,
                        "callsign": flight["callsign"],
                        "results": [], "best": None, "error": str(exc),
                        "elapsed_s": 0, "used_synonym": False,
                    }

                all_results.append(result)
                log(f"\n-- Completed {done_count}/{len(flights_to_process)} "
                    f"[{fid}] -----------------------------------")

                if result["success"]:
                    nodes_used = result.get("nodes", cfg.get("nodes"))
                    mass_est = result.get("mass_est", {})
                    used_syn = result.get("used_synonym", False)
                    for tr in result["results"]:
                        save_artifacts(
                            traj_dir, tr["identifier"], flight, tr["trajectory"],
                            deviation_nm=tr.get("deviation_nm", 0),
                            extra_meta={"total_fuel": tr["total_fuel"],
                                        "nat_fuel": tr["nat_fuel"],
                                        "nat_distance": tr["nat_distance"],
                                        "mass_estimation": mass_est,
                                        "convergence_status": tr.get("convergence_status", "unknown"),
                                        "ipopt_iterations": tr.get("ipopt_iterations", -1),
                                        "flst_based": result.get("flst_based", False)},
                            cfg={**cfg, "ipopt": ipopt_kw, "nodes_used": nodes_used},
                            used_synonym=used_syn,
                        )

                    # ── Baseline alignment (incremental) ──────────────
                    if baseline_blocks is not None:
                        _try_align_baseline(result, fid, baseline_blocks)

                else:
                    failed_flights.append({
                        "flightid": fid,
                        "callsign": flight["callsign"],
                        "actype": flight["actype"],
                        "origin": flight["origin"],
                        "destination": flight["destination"],
                        "error": result.get("error", "unknown"),
                    })

                gc.collect()

        # -- Clean up temp files ----------------------------------------
        if wind_tmp and os.path.exists(wind_tmp):
            os.remove(wind_tmp)
            log("  Temp wind file removed.")
        if flst_tmp and os.path.exists(flst_tmp):
            os.remove(flst_tmp)
            log("  Temp FLST file removed.")

    # ── Build scenarios ────────────────────────────────────────────────
    log(f"\nBuilding BlueSky scenarios (top {cfg['top_k']} per flight) ...")
    scn_all, scn_wosyn, scn_cruisestart, ref_midnight, gc_trim = build_scenarios(
        flights_info, all_results, top_k=cfg["top_k"],
    )

    scn_dir = cfg["output"]
    p1 = save_scenario(scn_all, os.path.join(scn_dir, "scenario_all.scn"),
                       reference_midnight=ref_midnight)
    p2 = save_scenario(scn_wosyn, os.path.join(scn_dir, "scenario_wosyn.scn"),
                       reference_midnight=ref_midnight)
    p3 = save_scenario(scn_cruisestart, os.path.join(scn_dir, "scenario_cruisestart.scn"),
                       reference_midnight=ref_midnight)
    log(f"Scenario ALL saved -> {p1}  ({len(scn_all)} flights)")
    log(f"Scenario w/o synonym saved -> {p2}  ({len(scn_wosyn)} flights)")
    log(f"Scenario cruise-start saved -> {p3}  ({len(scn_cruisestart)} flights)")

    # ── Write aligned baseline (if requested) ─────────────────────────
    n_aligned = 0
    if baseline_blocks is not None:
        aligned_path = os.path.join(scn_dir, "scenario_baseline_aligned.scn")
        _write_aligned_baseline(baseline_blocks, baseline_order, aligned_path)
        n_aligned = len(baseline_order)

    # ── Persist rankings, failures, metadata ──────────────────────────
    save_ranking(all_results, cfg["output"])
    save_failed_flights(failed_flights, cfg["output"])
    save_run_metadata(cfg["output"], cfg, flights_info, all_results,
                      failed_flights, skipped_count)

    # ── Summary ───────────────────────────────────────────────────────
    ok = sum(1 for r in all_results if r["success"])
    n_syn = sum(1 for r in all_results if r.get("used_synonym") and r["success"])
    n_flst = sum(1 for r in all_results if r.get("flst_based") and r["success"])
    wall = time.time() - wall_t0
    log(f"\n{'=' * 72}")
    log(f"DONE  {ok}/{len(flights_info)} flights optimised  "
        f"| {skipped_count} resumed  "
        f"| {len(failed_flights)} failed  "
        f"| {n_syn} used synonym  "
        f"| {n_flst} FLST-based")
    log(f"      {len(scn_all)} in scenario_all  "
        f"| {len(scn_wosyn)} in scenario_wosyn  "
        f"| {len(scn_cruisestart)} in scenario_cruisestart  "
        f"| wind: {wind_info}  "
        f"| {wall:.0f}s wall time")
    if baseline_blocks is not None:
        log(f"      Aligned baseline: {n_aligned} flights "
            f"-> scenario_baseline_aligned.scn")
    log(f"{'=' * 72}")

    return all_results


def _try_align_baseline(result: dict, fid: str,
                        baseline_blocks: dict[str, list[str]]) -> None:
    """Extract GC/FLST-variant trim info from *result* and align the baseline.

    For FLST-based flights the trajectory already starts at the trim
    point, so WP0 is used directly.  For GC-based flights the 120 NM
    trim point is located via ``_find_trim_point()``.

    Modifies *baseline_blocks* in-place for the given *fid*.
    """
    if not result.get("success"):
        return

    is_flst_based = result.get("flst_based", False)

    # Find the baseline variant (GC or FLST, deviation_nm ≈ 0)
    base_result = None
    for tr in result["results"]:
        if abs(tr.get("deviation_nm", 999)) < 1.0 or tr.get("name") in ("gc", "flst"):
            base_result = tr
            break
    # Fallback to best if no explicit baseline variant
    if base_result is None and result["results"]:
        base_result = result["results"][0]

    if base_result is None:
        return

    traj = base_result["trajectory"]

    if is_flst_based:
        # FLST-based: trajectory starts at the trim point already
        trim_info = {
            "trim_alt_ft": float(traj["altitude"].iloc[0]),
            "trim_mass_kg": float(traj["mass"].iloc[0]),
            "cumulative_dist_nm": 0.0,
        }
    else:
        trim_info = _find_trim_point(traj, trim_nm=TRIM_DISTANCE_NM)
        if trim_info is None:
            log(f"  [{fid}] Baseline alignment skipped: trajectory shorter "
                f"than {TRIM_DISTANCE_NM} NM", "WARN")
            return

    if fid not in baseline_blocks:
        log(f"  [{fid}] Baseline alignment skipped: flight not found "
            f"in baseline scenario", "WARN")
        return

    baseline_blocks[fid] = _align_baseline_flight(
        baseline_blocks[fid],
        trim_info["trim_alt_ft"],
        trim_info["trim_mass_kg"],
    )
    log(f"  [{fid}] Baseline aligned: trim_alt={trim_info['trim_alt_ft']:.0f} ft, "
        f"trim_mass={trim_info['trim_mass_kg']:.0f} kg"
        f"{', flst_based' if is_flst_based else ''}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description="Batch-optimise flights and export BlueSky scenarios (v11). "
                    "Supports --flst-log for FLST-based initial guesses, "
                    "--no-wind mode, and --baseline-scenario alignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument("--csv", required=True,
                    help="Path to the flight list CSV (semicolon-separated).")
    ap.add_argument("--output", required=True,
                    help="Output directory for scenarios, trajectories, metadata.")
    ap.add_argument("--wind-dir", required=False, default=None,
                    help="Directory with local ERA5 NetCDF files (p_levels_YYYYMMDD.nc). "
                         "Required unless --no-wind is set.")
    ap.add_argument("--bada-path", required=True,
                    help="Path to BADA3 performance data directory.")

    ap.add_argument("--flst-log", default=None,
                    help="Path to a BlueSky FLST log file (compact or full format). "
                         "When provided, each flight's reference trajectory from "
                         "the FLST is used as the initial guess for the optimizer "
                         "instead of a great-circle guess.  The cruise segment is "
                         "extracted by trimming 120 NM from both ends of the "
                         "trajectory.  The optimizer's origin/destination are set "
                         "to the trim-point coordinates (lat,lon tuples).  "
                         "Flights not found in the FLST file automatically fall "
                         "back to the GC-based initial guess.  "
                         "Track variations still respect --n-tracks and --max-dev.")

    ap.add_argument("--baseline-scenario", default=None,
                    help="Path to the baseline BlueSky .scn file.  When provided, "
                         "the script aligns the baseline scenario flight-by-flight "
                         "so that each flight spawns at the same altitude as the "
                         "optimised trajectory's 120 NM trim point, with the same "
                         "mass (SETMASS).  The aligned baseline is written to "
                         "scenario_baseline_aligned.scn in the output directory.  "
                         "The original baseline file is NOT modified.")

    ap.add_argument("--no-wind", action="store_true", default=False,
                    help="Run optimisation WITHOUT wind.  When set, no ERA5 data "
                         "is loaded and opt.enable_wind() is never called.  "
                         "--wind-dir is not required in this mode.  "
                         "To also disable wind in BlueSky simulation, simply "
                         "do not load the windecmwfup plugin.")

    ap.add_argument("--perf-model", default="bada3", choices=["bada3", "openap"],
                    help="Performance model (default: bada3).")
    ap.add_argument("--wind-method", default="poly",
                    choices=["poly", "linear", "bspline"],
                    help="Wind interpolation method (default: poly). "
                         "Ignored when --no-wind is set. "
                         "'poly'    = PolyWind (2nd-order polynomial regression). "
                         "'linear'  = BSplineWind multilinear C0 CasADi interpolation "
                         "            (RECOMMENDED -- handles full-res grids in <1s). "
                         "'bspline' = BSplineWind B-spline CasADi interpolation "
                         "            (limited by CasADi 3.7 bug for degree>=3 + large grids).")
    ap.add_argument("--bspline-degree", type=int, default=3,
                    help="B-spline degree per axis (1=linear, 3=cubic). "
                         "Only used when --wind-method=bspline. "
                         "NOTE: CasADi 3.7 crashes with degree>=3 on 4-D grids "
                         "above ~8k points -- use degree=1 or --wind-method=linear "
                         "for large grids (default: 3).")
    ap.add_argument("--bspline-subsample", type=int, default=1,
                    help="Take every N-th lat/lon point to reduce grid size. "
                         "Only relevant for --wind-method=bspline. Helps stay "
                         "below the ~8k point CasADi 3.7 limit (default: 1).")
    ap.add_argument("--time-subsample", type=int, default=1,
                    help="Take every N-th time step to reduce grid size. "
                         "Only relevant for --wind-method=bspline. With hourly "
                         "ERA5 data, --time-subsample=3 gives 3h resolution "
                         "(sufficient for transatlantic flights). (default: 1).")
    ap.add_argument("--m0", type=float, default=0.85,
                    help="Initial mass ratio x MTOW (default: 0.85).")
    ap.add_argument("--n-tracks", type=int, default=7,
                    help="Number of initial-guess variants (default: 7).")
    ap.add_argument("--max-dev", type=float, default=500,
                    help="Max lateral deviation in NM (default: 500).")
    ap.add_argument("--nodes", type=int, default=None,
                    help="Number of collocation nodes.  Omit for dynamic "
                         "(~1 per 50 km, clamped 20-120).")
    ap.add_argument("--polydeg", type=int, default=3,
                    help="Collocation polynomial degree (default: 3).")
    ap.add_argument("--max-iter", type=int, default=10_000,
                    help="IPOPT max iterations (default: 10000).")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="IPOPT convergence tolerance (default: 1e-6).")
    ap.add_argument("--acceptable-tol", type=float, default=1e-4,
                    help="IPOPT acceptable tolerance (default: 1e-4).")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Keep best K results per flight in scenario (default: 3).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Worker processes for flight-level parallelism (default: 4).")
    ap.add_argument("--flight-phase", default="cruise",
                    choices=["cruise", "full"],
                    help="ERA5 pressure level filter (default: cruise).")
    ap.add_argument("--wind-resolution", type=float, default=None,
                    help="Wind grid resolution in degrees. None = native resolution "
                         "(e.g. 0.25). Set to 0.5 or 1.0 to coarsen for speed.")
    ap.add_argument("--ipopt-print", type=int, default=0,
                    help="IPOPT print level 0-12 (default: 0 = quiet).")

    args = ap.parse_args(argv)

    # Validate: --wind-dir is required unless --no-wind is set
    if not args.no_wind and args.wind_dir is None:
        ap.error("--wind-dir is required unless --no-wind is set.")

    os.makedirs(args.output, exist_ok=True)

    return vars(args)


if __name__ == "__main__":
    cfg = parse_args()
    log("Configuration:")
    for k, v in sorted(cfg.items()):
        log(f"  {k:20s} = {v}")
    run_batch(cfg)
