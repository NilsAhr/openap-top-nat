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
        m0: float = 0.95,
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
            m0 (float, optional): Takeoff mass factor. Defaults to 0.95 (of MTOW).
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
                "height": bada3_data["ceiling"],  # already converted to meters
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

    def proj(self, lon, lat, inverse=False, symbolic=False):
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
                #bearing = ca.atan2(x, y) * 180 / 3.14159
                bearing = ca.arctan2(x, y) * 180 / 3.14159
                lat, lon = oc.aero.latlon(lat0, lon0, distances, bearing)
            else:
                distances = np.sqrt(x**2 + y**2)
                bearing = np.arctan2(x, y) * 180 / 3.14159
                lat, lon = openap.aero.latlon(lat0, lon0, distances, bearing)

            return lon, lat

    def initial_guess(self, flight: pd.DataFrame = None):
        m_guess = self.mass_init * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 12 * 3600, self.nodes + 1)

        if flight is None:
            h_cr = self.aircraft["cruise"]["height"]
            xp_0, yp_0 = self.proj(self.lon1, self.lat1)
            xp_f, yp_f = self.proj(self.lon2, self.lat2)
            xp_guess = np.linspace(xp_0, xp_f, self.nodes + 1)
            yp_guess = np.linspace(yp_0, yp_f, self.nodes + 1)
            h_guess = h_cr * np.ones(self.nodes + 1)
        else:
            xp_guess, yp_guess = self.proj(flight.longitude, flight.latitude)
            h_guess = flight.altitude * ft
            if "mass" in flight:
                m_guess = flight.mass

            if "ts" in flight:
                ts_guess = flight.ts
            elif "timestamp" in flight:
                ts_guess = (
                    flight.timestamp - flight.timestamp.min()
                ).dt.total_seconds()

        return np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

    def enable_wind(self, windfield: pd.DataFrame):
        self.wind = tools.PolyWind(
            windfield, self.proj, self.lat1, self.lon1, self.lat2, self.lon2
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
        """Ordinary differential equation for cruising

        Args:
            x (ca.MX): States [x position (m), y position (m), height (m), mass (kg)]
            u (ca.MX): Controls [mach number, vertical speed (m/s), heading (rad)]

        Returns:
            ca.MX: State direvatives
        """
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]

        v = oc.aero.mach2tas(mach, h, dT=self.dT)
        #gamma = ca.atan2(vs, v)
        gamma = ca.arctan2(vs, v)

        dx = v * ca.sin(psi) * ca.cos(gamma)
        if self.wind is not None:
            dx += self.wind.calc_u(xp, yp, h, ts)

        dy = v * ca.cos(psi) * ca.cos(gamma)
        if self.wind is not None:
            dy += self.wind.calc_v(xp, yp, h, ts)

        dh = vs

        dm = -self.fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)

        dt = 1

        return ca.vertcat(dx, dy, dh, dm, dt)

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
        acceptable_tol = kwargs.get("acceptable_tol", 1e-4)
        alpha_for_y = kwargs.get("alpha_for_y", "primal-and-full")
        hessian_approximation = kwargs.get("hessian_approximation", "limited-memory")

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
            "ipopt.mu_strategy": "adaptive",
            "ipopt.alpha_for_y": alpha_for_y,
            "ipopt.hessian_approximation": hessian_approximation,
        }

        for key, value in ipopt_kwargs.items():
            self.solver_options[f"ipopt.{key}"] = value

    def init_model(self, objective, **kwargs):
        autoscale_cost = kwargs.get("auto_scale_cost", False)

        # Model variables
        xp = ca.MX.sym("xp")
        yp = ca.MX.sym("yp")
        h = ca.MX.sym("h")
        m = ca.MX.sym("m")
        ts = ca.MX.sym("ts")

        mach = ca.MX.sym("mach")
        vs = ca.MX.sym("vs")
        psi = ca.MX.sym("psi")

        self.x = ca.vertcat(xp, yp, h, m, ts)
        self.u = ca.vertcat(mach, vs, psi)

        self.ts_final = ca.MX.sym("ts_final")

        # Control discretization
        self.dt = self.ts_final / self.nodes

        # Handel objective function
        if isinstance(objective, Callable):
            self.objective = objective
        elif objective.lower().startswith("ci:"):
            ci = int(objective[3:])
            kwargs["ci"] = ci
            self.objective = self.obj_ci
        else:
            self.objective = getattr(self, f"obj_{objective}")

        L = self.objective(self.x, self.u, self.dt, **kwargs)

        if autoscale_cost:
            # scale objective based on initial guess
            x0 = self.x_guess.T
            u0 = self.u_guess
            dt0 = self.range / 200 / self.nodes
            cost = np.sum(self.objective(x0, u0, dt0, symbolic=False, **kwargs))
            L = L / cost * 1e3

        # Continuous time dynamics
        self.func_dynamics = ca.Function(
            "f",
            [self.x, self.u],
            [self.xdot(self.x, self.u), L],
            ["x", "u"],
            ["xdot", "L"],
            {"allow_free": True},
        )

    def _calc_emission(self, x, u, symbolic=True):
        if self.perf_model.lower() == "bada3":
            raise NotImplementedError("Emission calculations not available with BADA3 performance model")
        xp, yp, h, m = x[0], x[1], x[2], x[3]
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
        x = [xp, yp, h, m, ts], u = [mach, vs, psi]
        """
        # unpack states and controls
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
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
                fuelflow = oc.FuelFlow(
                    self.actype,
                    self.engtype,
                    use_synonym=self.use_synonym,
                    force_engine=True
                    )

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
        x (ca.MX): State vector [xp, yp, h, m, ts].
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

        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]

        interpolant = kwargs.get("interpolant", None)
        symbolic = kwargs.get("symbolic", True)
        n_dim = kwargs.get("n_dim", 3)
        time_dependent = kwargs.get("time_dependent", True)
        assert n_dim in [3, 4]

        self.solver_options["ipopt.hessian_approximation"] = "limited-memory"

        lon, lat = self.proj(xp, yp, inverse=True, symbolic=symbolic)

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

        x0 = self.x_guess.T
        u0 = self.u_guess
        dt0 = self.range / 200 / self.nodes

        kwargs_ = kwargs.copy()
        kwargs_["symbolic"] = False

        n1 = obj1(x0, u0, dt0, **kwargs_).sum()
        n2 = obj2(x0, u0, dt0, **kwargs_).sum()

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

        # Extract optimised states and controls
        X = x_opt.full() # [xp, yp, h, mass, ts]
        U = u_opt.full() # [mach, vs, psi]

        # Extrapolate the final control point, Uf
        U2 = U[:, -2:-1]
        U1 = U[:, -1:]
        Uf = U1 + (U1 - U2)

        U = np.append(U, Uf, axis=1)
        n = self.nodes + 1

        self.X = X
        self.U = U
        self.dt = ts_final / (n - 1)

        xp, yp, h, mass, ts = X
        mach, vs, psi = U
        # Convert to readable format
        lon, lat = self.proj(xp, yp, inverse=True) # Convert back to lat/lon
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
            wu = self.wind.calc_u(xp, yp, h, ts)
            wv = self.wind.calc_v(xp, yp, h, ts)
            df = df.assign(wu=wu, wv=wv)

        return df
