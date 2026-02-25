from typing import Optional
import warnings

import casadi as ca
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
from openap import aero


def read_grids(paths: str | list[str], engine=None) -> pd.DataFrame:
    """
    Parameters:
    paths (str or list of str): The paths can be a single path or a list of paths.
        You must ensure the file with the lowest `time` value corresponds to the
        take-off time of your flight.
    engine (str, optional): The engine to use for reading the grib files.
        Defaults to None. Options are 'cfgrib' and 'netcdf4', etc.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the data from the grib files.

    The DataFrame includes the following transformations:
    - Adjusts the 'longitude' column to be within the range [-180, 180].
    - Adds a column 'h' calculated using the 'isobaricInhPa' column.
    - Adds a column 'ts' representing the total seconds from the minimum time.
    """
    df = (
        xr.open_mfdataset(paths, engine=engine)
        .to_dataframe()
        .reset_index()
        .drop(columns=["step", "valid_time"])
        .assign(longitude=lambda d: (d.longitude + 180) % 360 - 180)
        .assign(h=lambda d: aero.h_isa(d.isobaricInhPa * 100))
        .assign(ts=lambda d: (d.time - d.time.min()).dt.total_seconds())
    )
    return df


def pressure_to_height_m(p_hpa):
    """Convert pressure level (hPa) to altitude (m) using ISA troposphere formula.

    Valid for the troposphere (up to ~11 km / above ~226 hPa).

    Example: 250 hPa → ~10 363 m (FL340)
    """
    p_pa = p_hpa * 100.0
    return 44330.76923 * (1.0 - (p_pa / 101325.0) ** 0.190264)


def height_to_pressure_pa(h_m):
    """Convert altitude (m) to pressure (Pa) using inverted ISA troposphere formula.

    Compatible with CasADi MX symbolic types for use inside the NLP.

    Example: 10 363 m → ~25 000 Pa (250 hPa)
    """
    return 101325.0 * (1.0 - h_m / 44330.76923) ** 5.2559


class PolyWind:
    """
    A class to model wind fields using second order polynomial regression.
    """

    def __init__(self, windfield: pd.DataFrame, proj, lat1, lon1, lat2, lon2, margin=5):
        self.wind = windfield

        # select region based on airports
        df = (
            self.wind.query(f"longitude <= {max(lon1, lon2) + margin}")
            .query(f"longitude >= {(min(lon1, lon2)) - margin}")
            .query(f"latitude <= {max(lat1, lat2) + margin}")
            .query(f"latitude >= {min(lat1, lat2) - margin}")
            .query("h <= 13000")
        )

        x, y = proj(df.longitude, df.latitude)

        df = df.assign(x=x, y=y)

        model = make_pipeline(PolynomialFeatures(2), Ridge())
        model.fit(df[["x", "y", "h", "ts"]], df[["u", "v"]])

        features = model["polynomialfeatures"].get_feature_names_out()
        features = [string.replace("^", "**") for string in features]
        features = [string.replace(" ", "*") for string in features]

        self.features = features
        self.coef_u, self.coef_v = model["ridge"].coef_

    def calc_u(self, x, y, h, ts):
        u = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h, "ts": ts}) * c
                for (f, c) in zip(self.features, self.coef_u)
            ]
        )
        return u

    def calc_v(self, x, y, h, ts):
        v = sum(
            [
                eval(f, {}, {"x": x, "y": y, "h": h, "ts": ts}) * c
                for (f, c) in zip(self.features, self.coef_v)
            ]
        )
        return v

class BSplineWind:
    """Wind field using CasADi B-spline interpolation on a regular
    (lat, lon, h, ts) grid.

    Preserves the full spatial structure of the wind field (jet-stream
    cores, shear zones) that a polynomial regression would smooth out.
    Builds two 4-D CasADi ``interpolant`` objects — one for *u* (eastward)
    and one for *v* (northward) wind.

    Interface is drop-in compatible with :class:`PolyWind`:
    ``calc_u(x, y, h, ts)`` and ``calc_v(x, y, h, ts)`` accept the
    optimizer's projected coordinates and internally convert back to
    geographic coordinates before querying the interpolant.

    Parameters
    ----------
    windfield : pd.DataFrame
        Must contain columns ``ts, h, latitude, longitude, u, v``.
        Must be a **complete regular grid** (every combination of the four
        coordinate axes is present exactly once).
    proj : callable
        Projection function from :class:`Base` (supports
        ``inverse=True, symbolic=True``).
    lat1, lon1 : float
        Origin airport coordinates (degrees).
    lat2, lon2 : float
        Destination airport coordinates (degrees).
    margin : int
        Bounding-box margin in degrees around the route (default 5).
    degree : int
        B-spline degree per axis: 1 = linear, 3 = cubic (default 3).
        Automatically reduced for axes with fewer than ``degree + 1``
        unique values.
    """

    def __init__(
        self,
        windfield: pd.DataFrame,
        proj,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        margin: int = 5,
        degree: int = 3,
    ):
        self.proj = proj
        self.degree = degree

        # ---- bounding box ------------------------------------------------
        lat_lo = min(lat1, lat2) - margin
        lat_hi = max(lat1, lat2) + margin
        lon_lo = min(lon1, lon2) - margin
        lon_hi = max(lon1, lon2) + margin

        # ---- filter wind data to domain ----------------------------------
        df = windfield.query(
            f"latitude  >= {lat_lo} and latitude  <= {lat_hi} and "
            f"longitude >= {lon_lo} and longitude <= {lon_hi} and "
            f"h <= 13000"
        ).copy()

        if df.empty:
            raise ValueError(
                f"BSplineWind: no wind data in bounding box "
                f"lat[{lat_lo:.1f},{lat_hi:.1f}] "
                f"lon[{lon_lo:.1f},{lon_hi:.1f}]"
            )

        # ---- NaN check ---------------------------------------------------
        n_nan = df[["u", "v"]].isna().sum().sum()
        if n_nan > 0:
            raise ValueError(
                f"BSplineWind: {n_nan} NaN value(s) in wind data — "
                f"clean the input DataFrame before constructing the "
                f"interpolant."
            )

        # ---- build CasADi interpolants -----------------------------------
        self._interp_u, self._interp_v, self._bounds, self._grid_info = \
            self._build_interpolants(df, degree)

    # ------------------------------------------------------------------
    #  Interpolant construction
    # ------------------------------------------------------------------
    def _build_interpolants(self, df, degree):
        """Build 4-D CasADi B-spline interpolants for *u* and *v* wind.

        Grid axes: ``[lat, lon, h, ts]``  (lat varies fastest in the
        flattened value vector — CasADi column-major convention).
        """
        # Unique, sorted grid axes
        lats = np.sort(df.latitude.unique()).astype(np.float64)
        lons = np.sort(df.longitude.unique()).astype(np.float64)
        hs   = np.sort(df.h.unique()).astype(np.float64)
        tss  = np.sort(df.ts.unique()).astype(np.float64)

        # Validate completeness
        expected = len(lats) * len(lons) * len(hs) * len(tss)
        actual   = len(df)
        if actual != expected:
            raise ValueError(
                f"BSplineWind: incomplete grid — expected {expected} points "
                f"({len(lats)} lat × {len(lons)} lon × {len(hs)} h "
                f"× {len(tss)} ts), got {actual} rows. "
                f"The wind DataFrame must be a complete regular grid."
            )

        # Adapt degree per axis (need at least degree+1 knots)
        axes = [("lat", lats), ("lon", lons), ("h", hs), ("ts", tss)]
        deg_per_axis = []
        for name, vec in axes:
            max_deg = max(len(vec) - 1, 0)
            d = min(degree, max_deg)
            if d < degree:
                warnings.warn(
                    f"BSplineWind: '{name}' axis has only {len(vec)} "
                    f"point(s) — reducing B-spline degree from "
                    f"{degree} to {d}"
                )
            deg_per_axis.append(d)

        # Sort DataFrame so that lat varies fastest (CasADi convention).
        # Grid = [lat, lon, h, ts] → sort by [ts, h, lon, lat] ascending
        # so that the first axis (lat) changes with every row.
        df_sorted = df.sort_values(
            ["ts", "h", "longitude", "latitude"], ascending=True
        ).reset_index(drop=True)

        u_vals = df_sorted["u"].values.astype(np.float64)
        v_vals = df_sorted["v"].values.astype(np.float64)

        opts = {"degree": deg_per_axis}

        interp_u = ca.interpolant(
            "u_wind", "bspline", [lats, lons, hs, tss], u_vals, opts
        )
        interp_v = ca.interpolant(
            "v_wind", "bspline", [lats, lons, hs, tss], v_vals, opts
        )

        bounds = {
            "lat": (float(lats[0]),  float(lats[-1])),
            "lon": (float(lons[0]),  float(lons[-1])),
            "h":   (float(hs[0]),    float(hs[-1])),
            "ts":  (float(tss[0]),   float(tss[-1])),
        }

        grid_info = {
            "n_lat": len(lats), "n_lon": len(lons),
            "n_h":   len(hs),   "n_ts":  len(tss),
            "degree": deg_per_axis,
            "total_points": actual,
        }

        return interp_u, interp_v, bounds, grid_info

    # ------------------------------------------------------------------
    #  Clamping (prevents extrapolation)
    # ------------------------------------------------------------------
    def _clamp(self, lat, lon, h, ts):
        """Clamp query coordinates to grid bounds.

        Uses ``ca.fmin`` / ``ca.fmax`` which are CasADi-compatible and
        (sub-)differentiable, so IPOPT handles the kinks gracefully.
        """
        b = self._bounds
        lat_c = ca.fmin(ca.fmax(lat, b["lat"][0]), b["lat"][1])
        lon_c = ca.fmin(ca.fmax(lon, b["lon"][0]), b["lon"][1])
        h_c   = ca.fmin(ca.fmax(h,   b["h"][0]),   b["h"][1])
        ts_c  = ca.fmin(ca.fmax(ts,  b["ts"][0]),   b["ts"][1])
        return lat_c, lon_c, h_c, ts_c

    # ------------------------------------------------------------------
    #  Main query interface  (drop-in for PolyWind)
    # ------------------------------------------------------------------
    def calc_u(self, x, y, h, ts):
        """Evaluate eastward wind (m/s) at projected coordinates.

        Parameters match :class:`PolyWind`: ``(x, y)`` in projected
        metres, ``h`` in metres, ``ts`` in seconds.
        """
        lon, lat = self.proj(x, y, inverse=True, symbolic=True)
        lat_c, lon_c, h_c, ts_c = self._clamp(lat, lon, h, ts)
        return self._interp_u(ca.vertcat(lat_c, lon_c, h_c, ts_c))

    def calc_v(self, x, y, h, ts):
        """Evaluate northward wind (m/s) at projected coordinates."""
        lon, lat = self.proj(x, y, inverse=True, symbolic=True)
        lat_c, lon_c, h_c, ts_c = self._clamp(lat, lon, h, ts)
        return self._interp_v(ca.vertcat(lat_c, lon_c, h_c, ts_c))

    # ------------------------------------------------------------------
    #  Direct geographic query  (for analysis / testing)
    # ------------------------------------------------------------------
    def eval_uv(self, lat, lon, h, ts):
        """Evaluate *(u, v)* directly at geographic coordinates.

        Accepts numpy scalars or 1-D arrays.  Returns two numpy arrays.
        No projection is involved — useful for comparing against ERA5.
        """
        pts = np.atleast_2d(
            np.column_stack(
                [np.atleast_1d(lat), np.atleast_1d(lon),
                 np.atleast_1d(h),   np.atleast_1d(ts)]
            )
        ).T  # shape (4, N)
        u = np.array(self._interp_u(pts)).flatten()
        v = np.array(self._interp_v(pts)).flatten()
        return u, v

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    @property
    def bounds(self):
        """Grid bounds dict with keys lat, lon, h, ts."""
        return self._bounds

    @property
    def grid_info(self):
        """Grid shape / degree metadata dict."""
        return self._grid_info

    def __repr__(self):
        g = self._grid_info
        b = self._bounds
        return (
            f"BSplineWind("
            f"{g['n_lat']}×{g['n_lon']}×{g['n_h']}×{g['n_ts']} grid, "
            f"degree={g['degree']}, "
            f"lat=[{b['lat'][0]:.1f},{b['lat'][1]:.1f}], "
            f"lon=[{b['lon'][0]:.1f},{b['lon'][1]:.1f}], "
            f"h=[{b['h'][0]:.0f},{b['h'][1]:.0f}]m, "
            f"ts=[{b['ts'][0]:.0f},{b['ts'][1]:.0f}]s)"
        )

def construct_interpolant(
    longitude: np.array,
    latitude: np.array,
    height: np.array,
    grid_value: np.array,
    timestamp: Optional[np.array] = None,
    shape: str = "linear",
):
    """
    This function is used to create the 3d or 4d grid based cost function.

    It interpolates grid values based on the given longitude, latitude, height,
        timestamp, and grid_value arrays.

    Parameters:
        longitude (np.array): Array of longitudes.
        latitude (np.array): Array of latitudes.
        height (np.array): Array of heights (in meters).
        grid_value (np.array): Array of grid values.
        timestamp (Optional[np.array], optional): Array of timestamps. Defaults to None.
        shape (str, optional): Interpolation shape. Defaults to "linear".

    Returns:
        ca.interpolant: Casadi interpolant object representing the grid values.
    """

    assert shape in ["linear", "bspline"]

    if max(height) > 20_000:
        raise Warning(
            """Grid contains heights above 20,000 meters. You 'height' might be feet
            Make sure the 'height' values are in meters."""
        )

    if timestamp is None:
        return ca.interpolant(
            "grid_cost", shape, [longitude, latitude, height], grid_value
        )
    else:
        return ca.interpolant(
            "grid_cost", shape, [longitude, latitude, height, timestamp], grid_value
        )


def interp_grid(
    longitude, latitude, height, grid_value, timestamp=None, shape="linear"
):
    raise DeprecationWarning(
        "Function interp_grid() is deprecated, "
        "use interpolant_from_dataframe() instead."
    )


def interpolant_from_dataframe(
    df: pd.DataFrame, shape: str = "linear"
) -> ca.interpolant:
    """
    This function is used to create the 3d or 4d grid based cost function.

    It interpolates grid values based on the given DataFrame. The DataFrame must
    contain columns 'longitude', 'latitude', 'height' (meters), and 'cost'.

    If the DataFrame contains a 'ts' column, it will be used as the timestamp,
    and the grid will be treated as 4d.

    Parameters:
        df (pd.DataFrame): DataFrame containing the grid values.
        shape (str, optional): Interpolation shape. Defaults to "linear".

    Returns:
        ca.interpolant: Casadi interpolant object representing the grid values.
    """

    assert shape in ["linear", "bspline"], "Shape must be 'linear' or 'bspline'"
    assert "longitude" in df.columns, "Missing 'longitude' column in DataFrame"
    assert "latitude" in df.columns, "Missing 'latitude' column in DataFrame"
    assert "height" in df.columns, "Missing 'height' column in DataFrame"

    if "ts" in df.columns:
        df = df.sort_values(["ts", "height", "latitude", "longitude"], ascending=True)
        return construct_interpolant(
            df.longitude.unique(),
            df.latitude.unique(),
            df.height.unique(),
            df.cost.values,
            df.ts.unique(),
            shape=shape,
        )
    else:
        df = df.sort_values(["height", "latitude", "longitude"], ascending=True)
        return construct_interpolant(
            df.longitude.unique(),
            df.latitude.unique(),
            df.height.unique(),
            df.cost.values,
            shape=shape,
        )
