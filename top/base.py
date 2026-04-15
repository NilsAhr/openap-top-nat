import warnings
from typing import Callable, Union

import casadi as ca
import openap.casadi as oc
from openap.extra.aero import fpm, ft, kts

import numpy as np
import openap
import pandas as pd

# bada3_adapter import
from .perf import make_bada3_backend

# Mean Earth radius (m) for spherical geodesic ODE
R_EARTH = 6.371e6

# ── NLP state-variable scaling ──────────────────────────────────────────
# Physical → Scaled:  x_s  = x_phys * S_X
# Scaled → Physical:  x_ph = x_s    * S_X_INV
#
# Target: every scaled state ≈ O(1) so that the Hessian is well-conditioned.
#   lat (rad) ≈ 0.7–1.1  →  × 10     →  7–11
#   lon (rad) ≈ −1.2–0.1 →  × 10     →  −12–1
#   h   (m)   ≈ 6 000–13 000  → × 1e-4 →  0.6–1.3
#   m   (kg)  ≈ 50 000–350 000 → × 1/70 000 → 0.7–5
#   ts  (s)   – not scaled (user may add later)
S_X     = np.array([10.0,  10.0,  1e-4,       1.0 / 70_000, 1.0])
S_X_INV = 1.0 / S_X   # [0.1, 0.1, 10000, 70000, 1]

try:
    from . import tools
except Exception:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")


class Base:
    def __init__(
        self,
        actype: str,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        m0: float = 0.85,
        dT: float = 0.0,
        use_synonym=False,
        perf_model: str = "openap",
        bada3_path: str | None = None,
        debug: bool = False,
    ):
        """OpenAP trajectory optimizer.

        Args:
            actype (str): ICAO aircraft type code
            origin (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            destination (Union[str, tuple]): ICAO or IATA code of airport, or tuple (lat, lon)
            m0 (float, optional): Takeoff mass factor. Defaults to 0.85 (of MTOW).
            dT (float, optional): Temperature shift from standard ISA. Default = 0
            use_synonym (bool, optional): Use aircraft synonym database to find similar aircraft if actype is not found. Defaults to False.
            perf_model (str, optional): Performance model to use ('openap' or 'bada3'). Defaults to 'openap'.
            bada3_path (str, optional): Path to BADA3 performance data files. Required if perf_model is 'bada3'. Defaults to None.
        """

        self.debug = debug

        # Validate performance model
        if perf_model not in ["openap", "bada3"]:
            raise ValueError("perf_model must be either 'openap' or 'bada3'")

        if perf_model == "bada3" and bada3_path is None:
            raise ValueError("bada3_path must be provided when using BADA3 performance model")

        self.perf_model = perf_model
        self.bada3_path = bada3_path

        #print(f"Debug - origin: {origin}, type: {type(origin)}, isinstance(str): {isinstance(origin, str)}")
        # ORIGIN airport data
        if isinstance(origin, str):
            ap1 = openap.nav.airport(origin)
            self.lat1, self.lon1 = ap1["lat"], ap1["lon"]
            #print(f"Debug - airport data: lat={self.lat1}, lon={self.lon1}")
        else:
            #print(f"Debug - treating as coordinates: {origin}")
            self.lat1, self.lon1 = origin

        #print(f"Debug - destination: {destination}, type: {type(destination)}")
        # DESTINATION airport data
        if isinstance(destination, str):
            ap2 = openap.nav.airport(destination)
            self.lat2, self.lon2 = ap2["lat"], ap2["lon"]
        else:
            self.lat2, self.lon2 = destination

        # aircraft data
        self.actype = actype

        if self.perf_model.lower() == "bada3":
            # Import BADA3 here to avoid circular imports
            from .perf import load_model
            
            try:
                # Use the encoding-safe loader from bada3_adapter
                bada3_model = load_model(self.actype, self.bada3_path)
                bada3_aircraft_data = bada3_model.data
                self.aircraft = self._create_aircraft_from_bada3(bada3_aircraft_data)
                
                # Fill missing fields with OpenAP data if available
                try:
                    openap_aircraft = oc.prop.aircraft(self.actype, use_synonym=use_synonym)
                    self._fill_missing_aircraft_data(openap_aircraft)
                except Exception:
                    pass
            except Exception as e:
                raise RuntimeError(f"Failed to load BADA3 data for {actype}: {e}")
        else:
            # Standard OpenAP mode
            self.aircraft = oc.prop.aircraft(self.actype, use_synonym=use_synonym)

        # Engine setup - simplified for BADA3
        if self.perf_model.lower() == "bada3":
            # BADA3 uses aircraft-level performance
            self.engtype = self.aircraft["engine"]["type"] # "turbofan", "turboprop", etc.
            self.engine = {
                "name": self.engtype,
                "type": self.aircraft["engine"]["type"],
                "number": self.aircraft["engine"]["number"]
            }
        else:
            # OpenAP engine validation and loading
            self.engtype = self.aircraft["engine"]["default"] # "CFM56-7B24", etc.
            self.engine = oc.prop.engine(self.aircraft["engine"]["default"])

        # Initialize aircraft parameters from the aircraft dict
        self.mass_init = m0 * self.aircraft["mtow"]
        self.oew = self.aircraft["oew"]
        self.mlw = self.aircraft.get("mlw", self.aircraft["mtow"])  # fallback
        self.fuel_max = self.aircraft.get("mfc", self.aircraft["mtow"] - self.aircraft["oew"])  # fallback
        self.mach_max = self.aircraft.get("mmo", 0.85)  # fallback
        self.dT = dT
        self.use_synonym = use_synonym

        # Performance model initialization
        if self.perf_model.lower() == "bada3":

            self.thrust, self.drag, self.fuelflow, _ = make_bada3_backend(
                self.actype, self.bada3_path
            )
            # For BADA3, we don't have WRAP or Emission models
            self.wrap = None
            self.emission = None
        else:
            # Existing OpenAP behavior
            self.thrust = oc.Thrust(actype, use_synonym=self.use_synonym)
            self.wrap = openap.WRAP(actype, use_synonym=self.use_synonym)
            self.drag = oc.Drag(actype, wave_drag=True, use_synonym=self.use_synonym)
            self.fuelflow = oc.FuelFlow(actype, wave_drag=True, use_synonym=self.use_synonym)
            self.emission = oc.Emission(actype, use_synonym=self.use_synonym)

        ########
        # Test coordinate projection
        ########
        # self.proj = Proj(
        #     proj="lcc",
        #     ellps="WGS84",
        #     lat_1=min(self.lat1, self.lat2),
        #     lat_2=max(self.lat1, self.lat2),
        #     lat_0=(self.lat1 + self.lat2) / 2,
        #     lon_0=(self.lon1 + self.lon2) / 2,
        # )

        self.wind = None

        # ── Objective scaling ────────────────────────────────────────────
        # IPOPT converges best when J ≈ O(1–100).  Raw fuel objective is
        # O(30 000–50 000) kg for a wide-body transatlantic flight.
        # Default 1e-4 maps 38 000 kg → 3.8.
        self.obj_scale = 1e-4

        # Check cruise range
        self.range = oc.aero.distance(self.lat1, self.lon1, self.lat2, self.lon2)
        if self.wrap is not None:  # Add this check
            max_range = self.wrap.cruise_range()["maximum"] * 1.2
            if self.range > max_range * 1000:
                warnings.warn("The destination is likely out of maximum cruise range.")
        # For BADA3, we don't have WRAP, so skip range check or implement BADA3 range check
        # #else:  
            #if self.debug:
            #    print("Cruise range check skipped for BADA3 mode (WRAP not available)")
        self.setup()

    def _create_aircraft_from_bada3(self, bada3_data: dict) -> dict:
        """Create aircraft dictionary from BADA3 data with OpenAP structure"""
        # Calculate derived parameters
        mtow = bada3_data["mtow"]
        oew = bada3_data["oew"]
        mpl = bada3_data["mpl"]
        
        # Calculate MFC using MTOW-based proxy
        alpha = 0.28 if mtow / 1000 < 100 else (0.38 if mtow / 1000 < 200 else 0.48)
        mfc_proxy = alpha * mtow
        mfc_structural_limit = mtow - oew  # Structural limit (no payload)
        mfc = min(mfc_proxy, mfc_structural_limit)
        
        # Calculate other mass parameters
        mlw = oew + mpl  # Maximum Landing Weight
        mzfw = oew + mpl  # Maximum Zero Fuel Weight

        # altitude
        ceiling_m = bada3_data["ceiling"] # in meters
        # Typical cruise altitude: 85% of ceiling, capped at FL410
        typical_cruise_m = min(ceiling_m * 0.85, 12500)  # ~FL410 max

        aircraft = {
            # Mass parameters
            "mtow": mtow,  # already in kg
            "oew": oew,    # already in kg
            "mpl": mpl,    # already in kg
            "mzfw": mzfw,  # OEW + max payload

            # Flight envelope
            "mmo": bada3_data["mmo"],
            "vmo": bada3_data["vmo"],
            "ceiling": bada3_data["ceiling"], # in meters
            
            # Geometry
            "wing": {
                "area": bada3_data["wing"]["area"],
                "span": bada3_data["wing"]["span"],
            },
            "fuselage": {
                "length": bada3_data["fuselage"]["length"], 
            },
            
            # Engine
            "engine": {
                "type": bada3_data["engine"]["type"],
                "number": bada3_data["engine"]["number"],
                "default": bada3_data['engine']['type'],
            },
            
            # Cruise parameters
            "cruise": {
                "height": typical_cruise_m,  # meters
                "ceiling": ceiling_m,  # already converted to meters
            },
            "limits": {
                "MTOW": mtow,           # kg
                "OEW": oew,             # kg
                "MLW": mlw,             # OEW + max payload
                "MZFW": mzfw,           # OEW + max payload
                "MFC_lower": mtow - (oew + mpl),  # lower bound MTOW - MZFW
                "MFC_upper": mtow - oew,  # upper bound MTOW - OEW
                # max fuel capacity (proxy based on MTOW fraction)
                "MFC": mfc,               # kg  
                "VMO": bada3_data["vmo"],             # knots
                "MMO": bada3_data["mmo"],             # Mach
                "ceiling": bada3_data["ceiling"],     # meters, h(altitude - state variable treat in meters)
                "h_cruise": typical_cruise_m,   # meters
            }

        }
        
        return aircraft

    def _fill_missing_aircraft_data(self, openap_aircraft: dict):
        """Fill missing aircraft data with OpenAP values where BADA3 data is incomplete"""
        
        # List of fields that might be missing in BADA3 but available in OpenAP
        openap_fields = [
            "limits", "flaps", "gear", "approach", "landing", 
            "takeoff", "climb", "descent", "service"
        ]
        
        for field in openap_fields:
            if field in openap_aircraft and field not in self.aircraft:
                self.aircraft[field] = openap_aircraft[field]
        
        # Fill missing engine data
        if "engine" in openap_aircraft:
            openap_engine = openap_aircraft["engine"]
            if "default" not in self.aircraft["engine"] and "default" in openap_engine:
                self.aircraft["engine"]["default"] = openap_engine["default"]
            
            # Add other engine parameters that might be missing
            for eng_param in ["options", "max_thrust", "bypass_ratio"]:
                if eng_param in openap_engine and eng_param not in self.aircraft["engine"]:
                    self.aircraft["engine"][eng_param] = openap_engine[eng_param]
        
        # Fill missing performance limits if available
        if "limits" in openap_aircraft:
            self.aircraft.setdefault("limits", openap_aircraft["limits"])

    def _gc_intermediate_points(self, n):
        """Return *n* great-circle intermediate points (including endpoints).

        Returns
        -------
        lat_rad, lon_rad : np.ndarray, shape (n,)
            Latitude and longitude in **radians**.
        """
        lat1 = np.deg2rad(self.lat1)
        lon1 = np.deg2rad(self.lon1)
        lat2 = np.deg2rad(self.lat2)
        lon2 = np.deg2rad(self.lon2)

        # Angular distance (Haversine)
        d = 2.0 * np.arcsin(np.sqrt(
            np.sin((lat2 - lat1) / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
        ))

        fracs = np.linspace(0.0, 1.0, n)

        if d < 1e-12:
            return np.full(n, np.deg2rad(self.lat1)), np.full(n, np.deg2rad(self.lon1))

        A = np.sin((1.0 - fracs) * d) / np.sin(d)
        B = np.sin(fracs * d) / np.sin(d)

        x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
        y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
        z = A * np.sin(lat1) + B * np.sin(lat2)

        lats = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))   # radians
        lons = np.arctan2(y, x)                             # radians
        return lats, lons

    def proj(self, lon, lat, inverse=False, symbolic=False):
        """Legacy azimuthal-equidistant projection (DEPRECATED).

        Kept only for backward compatibility with Climb / Descent / Full.
        The Cruise optimiser no longer uses this — it works in geographic
        coordinates (lat/lon degrees) directly.
        """
        lat0 = (self.lat1 + self.lat2) / 2
        lon0 = (self.lon1 + self.lon2) / 2

        if not inverse:
            if symbolic:
                bearings = oc.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
                distances = oc.aero.distance(lat0, lon0, lat, lon)
                x = distances * ca.sin(bearings)
                y = distances * ca.cos(bearings)
            else:
                bearings = openap.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
                distances = openap.aero.distance(lat0, lon0, lat, lon)
                x = distances * np.sin(bearings)
                y = distances * np.cos(bearings)

            return x, y
        else:
            x, y = lon, lat
            if symbolic:
                distances = ca.sqrt(x**2 + y**2)
                bearing = ca.arctan2(x, y) * 180 / 3.14159
                lat, lon = oc.aero.latlon(lat0, lon0, distances, bearing)
            else:
                distances = np.sqrt(x**2 + y**2)
                bearing = np.arctan2(x, y) * 180 / 3.14159
                lat, lon = openap.aero.latlon(lat0, lon0, distances, bearing)

            return lon, lat

    def _max_feasible_altitude(self, mass, mach=None):
        """Find the highest altitude where thrust and lift constraints
        are satisfied for the given *mass* and *mach*.

        Scans from FL420 downward in 100 ft steps.  Returns height in
        metres, or ``h_min`` (FL200) if nothing is feasible.
        """
        from openap.extra.aero import ft as _ft, kts as _kts
        if mach is None:
            mach = self.mach_max - 0.03  # same as control initial guess

        S = self.aircraft["wing"]["area"]
        cd0 = self.drag.polar["clean"]["cd0"]
        ck = self.drag.polar["clean"]["k"]
        g0 = 9.80665
        W = mass * g0

        for alt_fl in range(420, 199, -1):            # FL420 → FL200
            h = alt_fl * 100 * _ft
            v = float(openap.aero.mach2tas(mach, h))
            tas_kn = v / _kts
            rho = float(openap.aero.density(h))
            T = float(np.asarray(self.thrust.cruise(tas_kn, alt_fl * 100)).flatten()[0])
            D = float(np.asarray(self.drag.clean(mass, tas_kn, alt_fl * 100)).flatten()[0])

            # Same margins as the NLP constraints
            if T * 0.95 - D <= 0:
                continue
            D_max = T * 0.9
            cd_max = D_max / (0.5 * rho * v ** 2 * S + 1e-10)
            cl_max_sq = (cd_max - cd0) / ck
            if cl_max_sq <= 0:
                continue
            L_max = np.sqrt(cl_max_sq) * 0.5 * rho * v ** 2 * S
            if L_max * 0.8 > W:
                return h

        return 20_000 * _ft          # fallback to FL200

    def initial_guess(self, flight: pd.DataFrame = None):
        # --- Estimate flight time from cruise TAS -------------------------
        mach_guess = self.mach_max - 0.03          # same Mach as control guess
        # Use a rough initial-guess altitude (will be refined below)
        h_rough = min(self.aircraft["cruise"]["height"], 11000)
        tas_guess = float(openap.aero.mach2tas(mach_guess, h_rough))  # m/s
        t_flight = float(self.range) / max(tas_guess, 100)            # seconds
        ts_guess = np.linspace(0, t_flight, self.nodes + 1)

        # --- Estimate mass profile (linear burn) --------------------------
        # Rough fuel-flow estimate: ~2.5 kg/s for a wide-body at cruise
        ff_guess = 2.5  # kg/s  (conservative average for B772/B77W class)
        fuel_total_guess = ff_guess * t_flight
        # Don't let the guess burn more than 60% of initial mass
        fuel_total_guess = min(fuel_total_guess, self.mass_init * 0.6)
        m_guess = np.linspace(self.mass_init,
                              self.mass_init - fuel_total_guess,
                              self.nodes + 1)

        if flight is None:
            h_cr = self.aircraft["cruise"]["height"]

            # Sanity check for BADA3: don't use ceiling as cruise
            if self.perf_model.lower() == "bada3":
                ceiling = self.aircraft.get("cruise", {}).get("ceiling", 
                        self.aircraft.get("ceiling", h_cr))
                h_cr = min(h_cr, ceiling * 0.85, 12500)  # Cap at ~FL410

            # Ensure the initial-guess altitude is actually feasible for
            # the initial mass.  The database cruise height is often only
            # reachable at mid-flight mass, not at heavy MTOW fractions.
            h_feas = self._max_feasible_altitude(self.mass_init)
            if h_feas < h_cr:
                if hasattr(self, 'debug') and self.debug:
                    print(f"initial_guess: h_cr={h_cr:.0f} m (FL{h_cr/0.3048/100:.0f}) "
                          f"infeasible at {self.mass_init:.0f} kg – "
                          f"capping to {h_feas:.0f} m (FL{h_feas/0.3048/100:.0f})")
                h_cr = h_feas

            # Great-circle intermediate points in radians
            lat_guess, lon_guess = self._gc_intermediate_points(self.nodes + 1)

            # Use a gently rising altitude profile: start at the
            # mass-feasible altitude, end at the database cruise height
            # (the aircraft gets lighter as it burns fuel).
            h_start = self._max_feasible_altitude(self.mass_init)
            h_end = min(h_cr, self._max_feasible_altitude(
                self.mass_init - fuel_total_guess))
            h_guess = np.linspace(h_start, h_end, self.nodes + 1)
            if hasattr(self, 'debug') and self.debug:
                print(f"initial_guess: altitude ramp FL{h_start/0.3048/100:.0f} "
                      f"→ FL{h_end/0.3048/100:.0f}")
        else:
            lat_guess = np.deg2rad(np.asarray(flight.latitude, dtype=float))
            lon_guess = np.deg2rad(np.asarray(flight.longitude, dtype=float))
            #Altitude sanity check
            h_guess = flight.altitude * ft

            # Clamp to reasonable bounds
            if self.perf_model.lower() == "bada3":
                ceiling = self.aircraft.get("ceiling", 13000)
                h_guess = np.clip(h_guess, 3000, ceiling * 0.95)

            if "mass" in flight:
                m_guess = flight.mass

            if "ts" in flight:
                ts_guess = flight.ts
            elif "timestamp" in flight:
                ts_guess = (
                    flight.timestamp - flight.timestamp.min()
                ).dt.total_seconds()

        # ── Resample to self.nodes + 1 points if needed ─────────────────
        # The NLP expects exactly (self.nodes + 1) state samples.  An
        # external DataFrame may have a different number of rows (e.g. 61
        # points while the NLP uses ~110 nodes).  Linearly interpolate all
        # five state channels so the guess and the NLP match.
        arrays = [np.asarray(a, dtype=float) for a in
                  [lat_guess, lon_guess, h_guess, m_guess, ts_guess]]
        n_have   = len(arrays[0])
        n_target = self.nodes + 1
        if n_have != n_target:
            if hasattr(self, 'debug') and self.debug:
                print(f"initial_guess: resampling external guess from "
                      f"{n_have} → {n_target} points")
            t_old = np.linspace(0, 1, n_have)
            t_new = np.linspace(0, 1, n_target)
            arrays = [np.interp(t_new, t_old, a) for a in arrays]
        lat_guess, lon_guess, h_guess, m_guess, ts_guess = arrays

        # Scale to NLP units: x_s = x_phys * S_X
        x_phys = np.vstack([lat_guess, lon_guess, h_guess, m_guess, ts_guess]).T
        return x_phys * S_X[np.newaxis, :]   # broadcast (N, 5) * (1, 5)

    def enable_wind(self, windfield: pd.DataFrame, use_bspline=False,
                    wind_method="linear", bspline_degree=3, bspline_subsample=1,
                    max_flight_time_s=None, time_subsample=1):
        """Enable wind effects in the trajectory optimisation.

        Parameters
        ----------
        windfield : pd.DataFrame
            DataFrame with columns ``ts, h, latitude, longitude, u, v``.
        use_bspline : bool
            If *True*, use :class:`tools.BSplineWind` (CasADi grid
            interpolation on the native lat/lon/h/ts grid).
            If *False* (default), use the legacy :class:`tools.PolyWind`
            (2nd-order polynomial regression).
        wind_method : str
            CasADi interpolation method (default ``'linear'``).

            * ``'linear'`` -- 4-D multilinear (C0, fast, full resolution)
            * ``'bspline'`` -- B-spline (C2, slow, limited to small grids)
        bspline_degree : int
            B-spline degree per axis (1 = linear, 3 = cubic). Only used
            when ``wind_method='bspline'``.
        bspline_subsample : int
            Take every *n*-th lat/lon point to reduce grid size and
            build time. Only relevant for ``wind_method='bspline'``.
        max_flight_time_s : float or None
            Clip the time axis to ``ts <= max_flight_time_s`` before
            building the interpolant.  Critical for batch runs.
        time_subsample : int
            Take every *n*-th time step (default 1).
        """
        if use_bspline:
            self.wind = tools.BSplineWind(
                windfield,
                self.lat1,
                self.lon1,
                self.lat2,
                self.lon2,
                method=wind_method,
                degree=bspline_degree,
                subsample=bspline_subsample,
                max_flight_time_s=max_flight_time_s,
                time_subsample=time_subsample,
            )
        else:
            self.wind = tools.PolyWind(
                windfield, self.lat1, self.lon1, self.lat2, self.lon2
            )

    def change_engine(self, engtype):
        self.engtype = engtype
        # bada3 case
        if self.perf_model.lower() == "bada3":
            # For BADA3, engine changes might not be supported
            warnings.warn("Engine change with BADA3 performance model may not be fully supported")
            # Try to update engine info in aircraft dict
            try:
                self.engine = oc.prop.engine(engtype)
            except Exception:
                self.engine = {"name": engtype, "type": "turbofan"}
        else:
            # Original OpenAP behavior
            self.engine = oc.prop.engine(engtype)
            self.thrust = oc.Thrust(
                self.actype,
                engtype,
                use_synonym=self.use_synonym,
                force_engine=True,
            )
            self.fuelflow = oc.FuelFlow(
                self.actype,
                engtype,
                wave_drag=True,
                use_synonym=self.use_synonym,
                force_engine=True,
            )
            self.emission = oc.Emission(self.actype, engtype, use_synonym=self.use_synonym)

    def collocation_coeff(self):
        # Get collocation points using Legendre polynomials
        tau_root = np.append(0, ca.collocation_points(self.polydeg, "legendre"))

        # C[i,j] = time derivative of Lagrange polynomial i evaluated at collocation point j
        C = np.zeros((self.polydeg + 1, self.polydeg + 1))

        # D[j] = Lagrange polynomial j evaluated at final time (t=1)
        D = np.zeros(self.polydeg + 1)

        # B[j] = integral of Lagrange polynomial j from 0 to 1
        B = np.zeros(self.polydeg + 1)

        # For each collocation point, construct Lagrange polynomial and calculate coefficients
        for j in range(self.polydeg + 1):
            # Construct Lagrange polynomial that is 1 at tau_root[j] and 0 at tau_root[r] where r != j
            p = np.poly1d([1])
            for r in range(self.polydeg + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate polynomial at t=1 for continuity constraints
            D[j] = p(1.0)

            # Get time derivative coefficients for collocation constraints
            pder = np.polyder(p)
            for r in range(self.polydeg + 1):
                C[j, r] = pder(tau_root[r])

            # Get integral coefficients for cost function quadrature
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return C, D, B

    def xdot(self, x, u) -> ca.MX:
        """Spherical-earth ODE for cruising flight.

        States are geographic: lat (rad), lon (rad), h (m), m (kg), ts (s).
        Controls: mach, vs (m/s), psi – true heading from north (rad, CW).

        Returns
        -------
        ca.MX  :  [dlat, dlon, dh, dm, dt]  (rad/s, rad/s, m/s, kg/s, s/s)
        """
        lat, lon, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        v = oc.aero.mach2tas(mach, h, dT=self.dT)
        gamma = ca.arctan2(vs, v)

        # Ground-speed components (m/s) before wind
        v_north = v * ca.cos(psi) * ca.cos(gamma)   # northward
        v_east  = v * ca.sin(psi) * ca.cos(gamma)   # eastward

        if self.wind is not None:
            # Wind interpolants expect degrees – convert from radian states
            lat_deg = lat * (180.0 / np.pi)
            lon_deg = lon * (180.0 / np.pi)
            v_north += self.wind.calc_v(lat_deg, lon_deg, h, ts)  # northward wind
            v_east  += self.wind.calc_u(lat_deg, lon_deg, h, ts)  # eastward  wind

        # Spherical geodesic rates (rad/s)
        dlat = v_north / (R_EARTH + h)
        dlon = v_east  / ((R_EARTH + h) * ca.cos(lat))

        dh = vs

        dm = -self.fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)

        dt = 1

        return ca.vertcat(dlat, dlon, dh, dm, dt)

    def setup(
        self,
        nodes: int | None = None,
        polydeg: int = 3,
        debug=False,
        ipopt_kwargs={},
        **kwargs,
    ):
        if nodes is not None:
            self.nodes = nodes
        else:
            self.nodes = int(self.range / 50_000)  # node every 50km

        max_nodes = kwargs.get("max_nodes", 120)

        self.nodes = max(20, self.nodes)
        self.nodes = min(max_nodes, self.nodes)

        self.polydeg = polydeg

        max_iteration = kwargs.get("max_iteration", kwargs.get("max_iterations", 3000))
        tol = kwargs.get("tol", 1e-6)
        acceptable_tol = kwargs.get("acceptable_tol", 1e-3)
        alpha_for_y = kwargs.get("alpha_for_y", "primal-and-full")
        #hessian_approximation = kwargs.get("hessian_approximation", "limited-memory")
        hessian_approximation = kwargs.get("hessian_approximation", "exact")

        self.debug = debug

        if debug:
            print("Calculating optimal trajectory...")
            ipopt_print = 5
            print_time = 1
        else:
            ipopt_print = 0
            print_time = 0

        self.solver_options = {
            "print_time": print_time,
            "calc_lam_p": False,
            "ipopt.print_level": ipopt_print,
            "ipopt.sb": "yes",
            "ipopt.max_iter": max_iteration,
            "ipopt.fixed_variable_treatment": "relax_bounds",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": acceptable_tol,
            "ipopt.acceptable_iter": 15,            # accept after 15 near-converged iters
            "ipopt.acceptable_constr_viol_tol": 1e-4,  # ~100 m acceptable constraint viol
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.alpha_for_y": alpha_for_y,
            "ipopt.hessian_approximation": hessian_approximation,
        }

        for key, value in ipopt_kwargs.items():
            self.solver_options[f"ipopt.{key}"] = value

        # Allow override of objective scaling per setup() call
        self.obj_scale = kwargs.get("obj_scale", self.obj_scale)

    def init_model(self, objective, **kwargs):
        autoscale_cost = kwargs.get("auto_scale_cost", False)

        # ── Scaling vectors (CasADi) ─────────────────────────────────────
        s_x     = ca.vertcat(*S_X.tolist())       # physical → scaled
        s_x_inv = ca.vertcat(*S_X_INV.tolist())    # scaled → physical

        # ── Scaled state symbol (NLP decision variable) ──────────────────
        # The NLP sees x_s = x_phys * s_x.
        self.x = ca.MX.sym("x_s", 5)

        # Controls (not scaled)
        mach = ca.MX.sym("mach")
        vs   = ca.MX.sym("vs")
        psi  = ca.MX.sym("psi")
        self.u = ca.vertcat(mach, vs, psi)

        self.ts_final = ca.MX.sym("ts_final")

        # Control discretization
        self.dt = self.ts_final / self.nodes

        # ── Unscale to physical for dynamics & objective ──────────────────
        x_phys = self.x * s_x_inv  # element-wise: x_phys_i = x_s_i / S_i

        # Handle objective function
        if isinstance(objective, Callable):
            self.objective = objective
        elif objective.lower().startswith("ci:"):
            ci = int(objective[3:])
            kwargs["ci"] = ci
            self.objective = self.obj_ci
        else:
            self.objective = getattr(self, f"obj_{objective}")

        # Objective evaluated on *physical* states, then scaled so that
        # J ≈ O(1–100) for good IPOPT conditioning.  The default
        # obj_scale = 1e-4 maps a typical fuel burn of ~38 000 kg to ~3.8.
        L = self.objective(x_phys, self.u, self.dt, **kwargs) * self.obj_scale

        if autoscale_cost:
            # Normalise objective by initial-guess cost
            x0_s = self.x_guess.T                     # scaled
            x0_phys = x0_s * S_X_INV[:, np.newaxis]   # unscale for numeric eval
            u0  = self.u_guess
            dt0 = self.range / 200 / self.nodes
            cost = np.sum(self.objective(x0_phys, u0, dt0, symbolic=False, **kwargs))
            L = L / cost * 1e3

        # ── Dynamics in physical space, then scale the rates ─────────────
        f_phys  = self.xdot(x_phys, self.u)   # d(x_phys)/dt
        f_scaled = f_phys * s_x                # d(x_s)/dt = S * d(x_phys)/dt

        # ── func_dynamics: maps (x_scaled, u) → (xdot_scaled, L) ─────────
        self.func_dynamics = ca.Function(
            "f",
            [self.x, self.u],
            [f_scaled, L],
            ["x", "u"],
            ["xdot", "L"],
            {"allow_free": True},
        )

    def _calc_emission(self, x, u, symbolic=True):
        if self.perf_model.lower() == "bada3":
            raise NotImplementedError("Emission calculations not available with BADA3 performance model")
        xp, yp, h, m = x[0], x[1], x[2], x[3]  # lat_deg, lon_deg, h, mass
        mach, vs, psi = u[0], u[1], u[2]

        if symbolic:
            fuelflow = self.fuelflow
            emission = self.emission
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
        else:
            fuelflow = openap.FuelFlow(
                self.actype, self.engtype, polydeg=2, use_synonym=self.use_synonym
            )
            emission = openap.Emission(
                self.actype, self.engtype, use_synonym=self.use_synonym
            )
            v = openap.aero.mach2tas(mach, h, dT=self.dT)

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        co2 = emission.co2(ff)
        h2o = emission.h2o(ff)
        sox = emission.sox(ff)
        soot = emission.soot(ff)
        nox = emission.nox(ff, v / kts, h / ft, dT=self.dT)

        return co2, h2o, sox, soot, nox

    def obj_fuel(self, x, u, dt, symbolic=True, **kwargs):
        """
        Fuel objective (kg) over one collocation interval.
        x = [lat, lon, h, m, ts], u = [mach, vs, psi]
        """
        # unpack states and controls
        lat, lon, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        # Choose aero conversion and fuelflow backend based on mode
        if symbolic:
            # CasADi-safe conversions
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
            tas_kt = v / kts
            alt_ft = h / ft

            if self.perf_model.lower() == "bada3":
                fuelflow = self.fuelflow  # BADA3 adapter (symbolic-safe)
            else:
                # Use self.fuelflow (wave_drag=True) so the objective
                # is consistent with the mass dynamics in xdot().
                fuelflow = self.fuelflow

        else:
            # Numeric conversions
            v = openap.aero.mach2tas(mach, h, dT=self.dT)
            tas_kt = v / kts
            alt_ft = h / ft

            if self.perf_model.lower() == "bada3":
                # Use the same BADA3 adapter for numeric pre-scaling too
                fuelflow = self.fuelflow
            else:
                fuelflow = openap.FuelFlow(
                    self.actype,
                    self.engtype,
                    use_synonym=self.use_synonym,
                    force_engine=True
                    )

        # Fuel flow (kg/s), note BADA3 adapter supports dT and numeric/symbolic inputs
        ff = fuelflow.enroute(m, tas_kt, alt_ft, vs / fpm, dT=self.dT)

        # Quadrature: fuel burned over interval = ff * dt
        return ff * dt

        # old
        #if symbolic:
        #    fuelflow = self.fuelflow
        #    v = oc.aero.mach2tas(mach, h, dT=self.dT)
        #else:
        #    fuelflow = openap.FuelFlow(
        #        self.actype,
        #        self.engtype,
        #        use_synonym=self.use_synonym,
        #        force_engine=True,
        #    )
        #    v = openap.aero.mach2tas(mach, h, dT=self.dT)

        #ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        #return ff * dt

    def obj_time(self, x, u, dt, **kwargs):
        return dt

    def obj_ci(self, x, u, dt, ci, time_price=25, fuel_price=0.8, **kwargs):
        """
        Calculate the objective cost index (CI) based on time and fuel costs.

        Parameters:
        x (ca.MX): state vector.
        u (ca.MX): control vector.
        dt (ca.MX): time step.
        ci (float): Cost index, a percentage value between 0 and 100.
        time_price (float): optional, cost of time per minute (default is 25 EUR/min).
        fuel_price (float): optional, cost of fuel per liter (default is 0.8 EUR/L).

        Returns:
        ca.MX: cost index objective.
        """

        fuel = self.obj_fuel(x, u, dt, **kwargs)

        # time cost 25 eur/min
        time_cost = (dt / 60) * time_price

        # fuel cost 0.8 eur/L, Jet A density 0.82
        fuel_cost = fuel * (fuel_price / 0.82)

        obj = ci / 100 * time_cost + (1 - ci / 100) * fuel_cost
        return obj

    def obj_gwp20(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.22 * h2o + 619 * nox - 832 * sox + 4288 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gwp50(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.1 * h2o + 205 * nox - 392 * sox + 2018 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gwp100(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.06 * h2o + 114 * nox - 226 * sox + 1166 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp20(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.07 * h2o - 222 * nox - 241 * sox + 1245 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp50(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.01 * h2o - 69 * nox - 38 * sox + 195 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_gtp100(self, x, u, dt, **kwargs):
        co2, h2o, sox, soot, nox = self._calc_emission(x, u, **kwargs)
        cost = co2 + 0.008 * h2o + 13 * nox - 31 * sox + 161 * soot
        # cost = cost * 1e-3
        return cost * dt

    def obj_grid_cost(self, x, u, dt, **kwargs):
        """
        Calculate the cost of the grid object.

        Parameters:
        x (ca.MX): State vector [lat, lon, h, m, ts] (lat/lon in radians).
        u (ca.MX): Control vector [mach, vs, psi].
        dt (ca.MX): Time step.

        **kwargs (dict): Additional keyword arguments.
            - interpolant (function): Interpolant function.
            - symbolic (bool): Flag indicating whether to use symbolic computation.
            - n_dim (int): Dimension of the input data (3 or 4), default to 3.
            - time_dependent (bool): Flag indicating whether the cost is time dependent.
            The cost will be multiplied by dt if true.

        Returns:
        cost (ca.MX): cost objective.

        Raises:
        AssertionError: If n_dim is not 3 or 4.
        """

        lat_rad, lon_rad, h, m, ts = x[0], x[1], x[2], x[3], x[4]

        interpolant = kwargs.get("interpolant", None)
        symbolic = kwargs.get("symbolic", True)
        n_dim = kwargs.get("n_dim", 3)
        time_dependent = kwargs.get("time_dependent", True)
        assert n_dim in [3, 4]

        #self.solver_options["ipopt.hessian_approximation"] = "limited-memory"
        self.solver_options["ipopt.hessian_approximation"] = "exact"

        # Convert radian states to degrees for the interpolant grid
        lon = lon_rad * (180.0 / np.pi)
        lat = lat_rad * (180.0 / np.pi)

        if n_dim == 3:
            input_data = [lon, lat, h]
        else:
            input_data = [lon, lat, h, ts]

        if symbolic:
            input_data = ca.vertcat(*input_data)
        else:
            input_data = np.array(input_data)

        cost = interpolant(input_data)

        if not symbolic:
            cost = cost.full()[0]

        if time_dependent:
            cost *= dt

        return cost

    def obj_combo(self, x, u, dt, obj1, obj2, ratio=0.5, **kwargs):
        if isinstance(obj1, str):
            obj1 = getattr(self, f"obj_{obj1}")

        if isinstance(obj2, str):
            obj2 = getattr(self, f"obj_{obj2}")

        # x_guess is in scaled NLP units; objectives expect physical states
        x0_phys = self.x_guess.T * S_X_INV[:, np.newaxis]
        u0 = self.u_guess
        dt0 = self.range / 200 / self.nodes

        kwargs_ = kwargs.copy()
        kwargs_["symbolic"] = False

        n1 = obj1(x0_phys, u0, dt0, **kwargs_).sum()
        n2 = obj2(x0_phys, u0, dt0, **kwargs_).sum()

        c1 = obj1(x, u, dt, **kwargs)
        c2 = obj2(x, u, dt, **kwargs)

        return ratio * c1 / n1 + (1 - ratio) * c2 / n2

    def to_trajectory(self, ts_final, x_opt, u_opt, **kwargs):
        """Convert optimization results to a trajectory DataFrame.

        Args:
            ts_final: Final timestamp
            x_opt: Optimized states
            u_opt: Optimized controls
            **kwargs: Additional arguments including:
                - interpolant: Grid cost interpolant function
                - time_dependent: Whether grid cost is time dependent (default True)
                - n_dim: Dimension of grid cost, 3 or 4 (default 4)

        Returns:
            pd.DataFrame: Trajectory with columns including fuel_cost and grid_cost
        """
        interpolant = kwargs.get("interpolant", None)
        time_dependent = kwargs.get("time_dependent", True)
        n_dim = kwargs.get("n_dim", 4)

        # Extract optimised states (scaled) and unscale to physical
        X = x_opt.full()                           # [lat_s, lon_s, h_s, m_s, ts]
        X = X * S_X_INV[:, np.newaxis]              # → [lat_rad, lon_rad, h_m, m_kg, ts_s]
        U = u_opt.full()                            # [mach, vs, psi] (not scaled)

        # Extrapolate the final control point, Uf
        U2 = U[:, -2:-1]
        U1 = U[:, -1:]
        Uf = U1 + (U1 - U2)

        U = np.append(U, Uf, axis=1)
        n = self.nodes + 1

        self.X = X
        self.U = U
        self.dt = ts_final / (n - 1)

        xp, yp, h, mass, ts = X  # xp=lat_rad, yp=lon_rad
        mach, vs, psi = U
        # Convert radian states to degrees for output
        lat = np.rad2deg(np.asarray(xp).squeeze())
        lon = np.rad2deg(np.asarray(yp).squeeze())
        ts_ = np.linspace(0, ts_final, n).round(4)
        tas = (openap.aero.mach2tas(mach, h, dT=self.dT) / kts).round(4) # Convert Mach to TAS
        alt = (h / ft).round() # Convert to feet
        vertrate = (vs / fpm).round()

        def _as_1d(arr, n=None):
            if hasattr(arr, "full"):
                arr = arr.full()
            arr = np.asarray(arr).squeeze()
            if arr.ndim == 0:
                arr = np.full(n, float(arr)) if n is not None else np.array([float(arr)])
            arr = arr.astype(float)
            if n is not None and arr.size != n:
                if arr.size == n - 1:
                    arr = np.append(arr, np.nan)
                else:
                    raise ValueError(f"Unexpected array size {arr.size}, expected {n} or {n-1}")
            return arr

        # Calculate fuel_cost per segment
        fuel_cost = self.obj_fuel(X, U, self.dt, symbolic=False)
        fuel_cost = _as_1d(fuel_cost, n)

        # Calculate grid_cost per segment (NaN if no interpolant)
        if interpolant is not None:
            grid_cost = self.obj_grid_cost(
                X,
                U,
                self.dt,
                interpolant=interpolant,
                time_dependent=time_dependent,
                n_dim=n_dim,
                symbolic=False,
            )
            grid_cost = _as_1d(grid_cost, n)
        else:
            grid_cost = np.full(n, np.nan)

        df = pd.DataFrame(
            dict(
                mass=mass,
                ts=ts_,
                x=xp,
                y=yp,
                h=h,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                mach=mach.round(6),
                tas=tas,
                vertical_rate=vertrate,
                heading=(np.rad2deg(psi) % 360).round(4),
                fuel_cost=fuel_cost,
                grid_cost=grid_cost,
            )
        )

        # Handle fuel flow calculation based on performance model
        if self.perf_model.lower() == "bada3":
            # Use BADA3 fuel flow for trajectory output
            ff_values = []
            for i in range(len(df)):
                try:
                    ff = self.fuelflow.enroute(
                        mass=df.iloc[i].mass, 
                        tas_kt=df.iloc[i].tas, 
                        alt_ft=df.iloc[i].altitude, 
                        vs=df.iloc[i].vertical_rate
                    )
                    # Convert CasADi array to float if needed
                    if hasattr(ff, 'full'):
                        ff_values.append(float(ff.full().flatten()[0]))
                    else:
                        ff_values.append(float(ff))
                except Exception as e:
                    print(f"Warning: BADA3 fuel flow calculation failed at step {i}: {e}")
                    ff_values.append(0.0)
            
            df = df.assign(fuelflow=ff_values)
        else:
            # Original OpenAP fuel flow calculation
            fuelflow = openap.FuelFlow(
                self.actype,
                self.engtype,
                use_synonym=self.use_synonym,
                force_engine=True,
            )
            df = df.assign(
                fuelflow=(
                    fuelflow.enroute(
                        mass=df.mass, tas=tas, alt=alt, vs=vertrate, dT=self.dT
                    )
                )
            )   

        if self.wind:
            wu = self.wind.calc_u(lat, lon, h, ts)
            wv = self.wind.calc_v(lat, lon, h, ts)
            df = df.assign(wu=wu, wv=wv)

        return df
