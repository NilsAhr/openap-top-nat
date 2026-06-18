import warnings
from math import pi

import casadi as ca
import numpy as np
import openap
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base, R_EARTH, S_X, S_X_INV
from .climb import Climb
from .cruise import Cruise
from .descent import Descent

try:
    from . import tools
except Exception:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")


class CompleteFlight(Base):
    """Single-NLP takeoff-to-landing trajectory optimiser.

    This optimiser solves climb, cruise and descent in one monolithic
    direct-collocation NLP.  It uses the same spherical, scaled framework
    as :class:`Cruise` / :class:`Base`:

    * **Coordinate system** -- geographic ``[lat (rad), lon (rad), h (m),
      m (kg), ts (s)]`` integrated with the spherical-earth ODE in
      :meth:`Base.xdot` (no Cartesian projection).
    * **NLP scaling** -- all state decision variables and bounds are in
      scaled units (``x_s = x_phys * S_X``); collocation residuals are
      scaled by ``_cscale``; the objective uses ``obj_scale``.
    * **Performance model** -- honours ``perf_model='bada3'`` via the
      inherited BADA3 adapters and the BlueSky-equivalent altitude ceiling
      (:meth:`Base._use_bluesky_ceiling`).
    * **Wind** -- inherits :meth:`Base.enable_wind`, so both ``PolyWind``
      and ``BSplineWind`` (4-D CasADi interpolant) work transparently
      through :meth:`Base.xdot`.

    A dedicated full-flight initial guess
    (:meth:`_full_flight_initial_guess`) provides a
    ground -> climb -> cruise -> descent -> ground vertical profile so the
    solver starts from a feasible takeoff-to-landing shape.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    def _phase_indices(self):
        """Return ``(idx_toc, idx_tod)`` node indices delimiting the
        climb / cruise / descent phases, derived from along-track distance.

        Long flights (> 1500 km) use fixed climb/descent distances; shorter
        flights fall back to a percentage split.  The same logic is reused
        for the phase constraints in :meth:`trajectory` so the guess and the
        constraints agree.
        """
        max_climb_range = 500_000    # m, heavy aircraft climb distance
        max_descent_range = 400_000  # m, oceanic arrival descent distance
        dd = self.range / (self.nodes + 1)

        if self.range > 1500_000:
            idx_toc = int(max_climb_range / dd)
            idx_tod = int((self.range - max_descent_range) / dd)
        else:
            idx_toc = int(self.nodes * 0.15)
            idx_tod = int(self.nodes * 0.85)

        idx_toc = max(3, min(idx_toc, self.nodes // 3))
        idx_tod = max(idx_toc + 5, min(idx_tod, self.nodes - 3))
        return idx_toc, idx_tod

    def _full_flight_initial_guess(self, flight=None):
        """Dedicated takeoff-to-landing state guess in **scaled** units.

        Builds a ground -> climb -> cruise -> descent -> ground vertical
        profile along the great circle, with a linear mass burn and a
        range-based timeline.  Returns an ``(nodes + 1, 5)`` array already
        multiplied by ``S_X``.

        When *flight* is provided the base interpolating guess is used
        instead (it already handles external DataFrames and scaling).
        """
        if flight is not None:
            return self.initial_guess(flight)

        n = self.nodes + 1

        # Great-circle horizontal path (radians)
        lat_guess, lon_guess = self._gc_intermediate_points(n)

        # Vertical profile: ground -> cruise -> ground
        h_min = 100 * ft
        h_cruise = self._max_feasible_altitude(self.mass_init)
        idx_toc, idx_tod = self._phase_indices()

        h_guess = np.full(n, float(h_cruise))
        h_guess[: idx_toc + 1] = np.linspace(h_min, h_cruise, idx_toc + 1)
        h_guess[idx_tod:] = np.linspace(h_cruise, h_min, n - idx_tod)

        # Mass: linear burn estimate (same rough model as base.initial_guess)
        mach_guess = self.mach_max - 0.03
        tas_guess = float(openap.aero.mach2tas(mach_guess, min(float(h_cruise), 11000)))
        t_flight = float(self.range) / max(tas_guess, 100)
        ff_guess = 2.5  # kg/s, conservative wide-body cruise average
        fuel_total_guess = min(ff_guess * t_flight, self.mass_init * 0.6)
        m_guess = np.linspace(
            self.mass_init, self.mass_init - fuel_total_guess, n
        )

        # Timeline
        ts_guess = np.linspace(0, t_flight, n)

        if self.debug:
            print(
                f"full-flight guess: climb 0->{idx_toc}, cruise->{idx_tod}, "
                f"descent->{self.nodes}; h_cruise=FL{h_cruise/0.3048/100:.0f}, "
                f"t_flight={t_flight/3600:.2f} h"
            )

        # Scale to NLP units: x_s = x_phys * S_X
        x_phys = np.vstack([lat_guess, lon_guess, h_guess, m_guess, ts_guess]).T
        return x_phys * S_X[np.newaxis, :]

    # ------------------------------------------------------------------
    # Bounds & guesses
    # ------------------------------------------------------------------
    def init_conditions(self, **kwargs):
        """Initialise direct-collocation bounds and guesses.

        States are **scaled**:
          ``[lat_rad*S_lat, lon_rad*S_lon, h_m*S_h, m_kg*S_m, ts_s*S_ts]``.
        Controls are unscaled ``[mach, vs (m/s), psi (rad)]``.
        """
        Sl, So, Sh, Sm, St = S_X  # 10, 10, 1e-4, 1/70000, 1
        d2r = pi / 180.0

        # Origin / destination in radians
        lat_0 = self.lat1 * d2r
        lon_0 = self.lon1 * d2r
        lat_f = self.lat2 * d2r
        lon_f = self.lon2 * d2r

        # Bounding box (radians) with generous margins
        lat_margin = 10.0 * d2r
        lon_margin = 15.0 * d2r
        lat_min = min(lat_0, lat_f) - lat_margin
        lat_max = max(lat_0, lat_f) + lat_margin
        lon_min = min(lon_0, lon_f) - lon_margin
        lon_max = max(lon_0, lon_f) + lon_margin

        ts_min = 0
        ts_max = max(5, self.range / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = 100 * ft

        # Great-circle initial (departure) and final (arrival) bearings
        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180
        hdg_final = (
            oc.aero.bearing(self.lat2, self.lon2, self.lat1, self.lon1) + 180
        ) % 360
        psi_f = hdg_final * pi / 180
        psi_mid = (psi + psi_f) / 2

        # Cruise Mach band, with optional LRC-style cap (APF for BADA3).
        mach_hi = self.mach_max
        mach_cruise_lo = 0.78
        mach_guess = self.mach_max - 0.03
        import os

        _cap = os.environ.get("OPT_MACH_CAP", None)
        if _cap is None:
            cap_val = (
                self._apf_cruise_mach()
                if getattr(self, "perf_model", None) == "bada3"
                else None
            )
        elif _cap in ("", "0"):
            cap_val = None
        elif _cap.lower() == "apf":
            cap_val = self._apf_cruise_mach()
        else:
            cap_val = float(_cap)
        if cap_val:
            mach_hi = min(mach_hi, cap_val)
            mach_cruise_lo = min(mach_cruise_lo, mach_hi)
            mach_guess = min(mach_guess, mach_hi)

        # ── state bounds (SCALED) ────────────────────────────────────
        # Initial: on the ground at the origin, full mass, t = 0
        self.x_0_lb = [lat_0 * Sl, lon_0 * So, h_min * Sh, self.mass_init * Sm, ts_min * St]
        self.x_0_ub = [lat_0 * Sl, lon_0 * So, h_min * Sh, self.mass_init * Sm, ts_min * St]

        # Final: on the ground at the destination
        self.x_f_lb = [lat_f * Sl, lon_f * So, h_min * Sh, self.oew * 0.5 * Sm, ts_min * St]
        self.x_f_ub = [lat_f * Sl, lon_f * So, h_min * Sh, self.mass_init * Sm, ts_max * St]

        # Path bounds
        self.x_lb = [lat_min * Sl, lon_min * So, h_min * Sh, self.oew * 0.5 * Sm, ts_min * St]
        self.x_ub = [lat_max * Sl, lon_max * So, h_max * Sh, self.mass_init * Sm, ts_max * St]

        # ── control bounds (unscaled: mach, vs [m/s], psi [rad]) ─────
        # Initial control (climb): low Mach, positive ROC, depart on GC bearing
        self.u_0_lb = [0.1, 500 * fpm, psi - pi / 4]
        self.u_0_ub = [0.4, 2500 * fpm, psi + pi / 4]

        # Final control (descent): low Mach, negative ROC, arrive on GC bearing
        self.u_f_lb = [0.1, -2500 * fpm, psi_f - pi / 4]
        self.u_f_ub = [0.4, -300 * fpm, psi_f + pi / 4]

        # Path control: span climb -> cruise -> descent
        self.u_lb = [0.1, -2500 * fpm, psi_mid - pi / 2]
        self.u_ub = [mach_hi, 2500 * fpm, psi_mid + pi / 2]

        # ── guesses ──────────────────────────────────────────────────
        self.x_guess = self._full_flight_initial_guess(
            kwargs.get("initial_guess", None)
        )

        # Linearly varying heading guess from departure to arrival bearing
        psi_guess = np.linspace(float(psi), float(psi_f), self.nodes)
        self.u_guess_array = [
            [mach_guess, 0, psi_guess[i]] for i in range(self.nodes)
        ]
        self.u_guess = [mach_guess, 1000 * fpm, float(psi)]  # scalar fallback

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """Compute the optimal complete-flight trajectory.

        Parameters
        ----------
        objective : str | Callable
            Optimisation objective (default ``"fuel"``).
        **kwargs
            ``max_fuel`` (float), ``initial_guess`` (pd.DataFrame),
            ``return_failed`` (bool), ``warm_start`` (dict),
            ``h_max`` (float) ...
            ``reg_weights`` (dict): Soft-penalty (regularisation) weights on
                consecutive control changes.  Adds
                  w * Σ(ΔU_k)²  to the objective for each channel.
                Keys / typical values:
                  "mach"    : 10.0   (Mach number change)
                  "vs"      : 1e-3   (vertical speed change, m/s)
                  "heading" : 0.0    (heading change, rad)
                Default is {} (no regularisation).

        Returns
        -------
        pd.DataFrame | None
            Optimised trajectory, or ``None`` if it failed sanity checks
            (unless ``return_failed=True``).
        """
        self.init_conditions(**kwargs)
        self.init_model(objective, **kwargs)

        if self.debug:
            print(f"Using performance model: {self.perf_model.upper()}")

        customized_max_fuel = kwargs.get("max_fuel", None)
        return_failed = kwargs.get("return_failed", False)

        # Scaling shortcuts
        Sl, So, Sh, Sm, St = S_X
        _Sh_inv = float(S_X_INV[2])   # 10 000  (scaled -> m)
        _Sm_inv = float(S_X_INV[3])   # 70 000  (scaled -> kg)
        _Sm = float(S_X[3])           # 1 / 70 000

        C, D, B = self.collocation_coeff()

        # Legendre roots for consistent interior-point guesses
        _tau_root = np.append(0, ca.collocation_points(self.polydeg, "legendre"))

        # Diagonal scaling for collocation residuals (see cruise.py)
        _cscale = ca.vertcat(1.0, 1.0, 10.0, 50.0, 1e-3)

        # Empty NLP
        w, w0, lbw, ubw = [], [], [], []
        J = 0
        g, lbg, ubg = [], [], []
        X, U = [], []

        # Initial state
        nstates = self.x.shape[0]
        Xk = ca.MX.sym("X0", nstates, self.x.shape[1])
        w.append(Xk)
        lbw.append(self.x_0_lb)
        ubw.append(self.x_0_ub)
        w0.append(self.x_guess[0])
        X.append(Xk)

        # Formulate the NLP
        for k in range(self.nodes):
            # Control variable
            Uk = ca.MX.sym("U_" + str(k), self.u.shape[0])
            U.append(Uk)
            w.append(Uk)

            if k == 0:
                lbw.append(self.u_0_lb)
                ubw.append(self.u_0_ub)
            elif k == self.nodes - 1:
                lbw.append(self.u_f_lb)
                ubw.append(self.u_f_ub)
            else:
                lbw.append(self.u_lb)
                ubw.append(self.u_ub)

            if self.u_guess_array is not None:
                w0.append(self.u_guess_array[k])
            else:
                w0.append(self.u_guess)

            # State at collocation points (linear interp at Legendre roots)
            x_k = self.x_guess[k]
            x_kp1 = self.x_guess[min(k + 1, len(self.x_guess) - 1)]
            Xc = []
            for j in range(self.polydeg):
                Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), nstates)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
                tau_j = _tau_root[j + 1]
                w0.append((1.0 - tau_j) * x_k + tau_j * x_kp1)

            # Collocation equations
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                xpc = C[0, j] * Xk
                for r in range(self.polydeg):
                    xpc = xpc + C[r + 1, j] * Xc[r]

                fj, qj = self.func_dynamics(Xc[j - 1], Uk)
                g.append(_cscale * (self.dt * fj - xpc))
                lbg.append([0] * nstates)
                ubg.append([0] * nstates)

                Xk_end = Xk_end + D[j] * Xc[j - 1]
                J = J + B[j] * qj

            # State at end of interval
            Xk = ca.MX.sym("X_" + str(k + 1), nstates)
            w.append(Xk)
            X.append(Xk)

            if k < self.nodes - 1:
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
            else:
                lbw.append(self.x_f_lb)
                ubw.append(self.x_f_ub)

            w0.append(self.x_guess[min(k + 1, len(self.x_guess) - 1)])

            # Continuity constraint (scaled)
            g.append(_cscale * (Xk_end - Xk))
            lbg.append([0] * nstates)
            ubg.append([0] * nstates)

        w.append(self.ts_final)
        lbw.append([0])
        ubw.append([ca.inf])
        # self.range in metres; ~200 m/s typical cruise ground speed
        w0.append([self.range / 200])

        # ============================================================
        # PHASE CONSTRAINTS: climb / cruise / descent boundaries
        # ============================================================
        idx_toc, idx_tod = self._phase_indices()
        h_cruise_min = 30000 * ft  # FL300 minimum cruise altitude

        if self.debug:
            print(
                f"Phase boundaries: TOC node {idx_toc}, TOD node {idx_tod} "
                f"(of {self.nodes})"
            )

        # ----- CLIMB (0 .. idx_toc): positive ROC -----
        for k in range(0, idx_toc):
            g.append(U[k][1])
            lbg.append([0])
            ubg.append([ca.inf])

        # ----- CRUISE (idx_toc .. idx_tod): min FL300 + |VS| <= 500 fpm -----
        for k in range(idx_toc, idx_tod):
            # Minimum cruise altitude (state is SCALED -> compare in scaled units)
            g.append(X[k][2] - h_cruise_min * Sh)
            lbg.append([0])
            ubg.append([ca.inf])

            g.append(U[k][1])
            lbg.append([-500 * fpm])
            ubg.append([500 * fpm])

        # ----- DESCENT (idx_tod .. end): negative ROC -----
        for k in range(idx_tod, self.nodes):
            g.append(U[k][1])
            lbg.append([-ca.inf])
            ubg.append([0])

        # ============================================================
        # FORCE & ENERGY CONSTRAINTS (physics in unscaled units)
        # ============================================================
        _use_bs_ceiling = self._use_bluesky_ceiling()
        if _use_bs_ceiling:
            _bs_hMO, _bs_hmax, _bs_gw, _bs_mmax = self._bluesky_ceiling_coeffs()

        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            h_phys = X[k][2] * _Sh_inv        # scaled -> m
            mass_phys = X[k][3] * _Sm_inv     # scaled -> kg
            v = oc.aero.mach2tas(U[k][0], h_phys, dT=self.dT)
            tas = v / kts
            alt = h_phys / ft
            rho = oc.aero.density(h_phys, dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self.drag.clean(mass_phys, tas, alt, dT=self.dT)

            if _use_bs_ceiling:
                # BlueSky-equivalent BADA3 OPF ceiling (matches the simulator)
                hmaxact = ca.fmin(
                    _bs_hMO, _bs_hmax + _bs_gw * (_bs_mmax - mass_phys)
                )
                g.append(hmaxact - h_phys)
                lbg.append([0])
                ubg.append([ca.inf])

                g.append(thrust_max - drag)
                lbg.append([0])
                ubg.append([ca.inf])
            else:
                # max thrust * 95% > drag (5% margin)
                g.append(thrust_max * 0.95 - drag)
                lbg.append([0])
                ubg.append([ca.inf])

                # max lift * 80% > weight (20% margin)
                cd0 = self.drag.polar["clean"]["cd0"]
                ck = self.drag.polar["clean"]["k"]
                drag_max = thrust_max * 0.9
                cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
                cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
                L_max = cl_max * 0.5 * rho * v**2 * S
                g.append(L_max * 0.8 - mass_phys * oc.aero.g0)
                lbg.append([0])
                ubg.append([ca.inf])

            # excess energy >= change in potential energy (climb feasibility)
            excess_energy = (thrust_max - drag) * v - mass_phys * oc.aero.g0 * U[k][1]
            g.append(excess_energy)
            lbg.append([0])
            ubg.append([ca.inf])

        # ts and dt consistency
        for k in range(self.nodes - 1):
            g.append(X[k + 1][4] - X[k][4] - self.dt)
            lbg.append([-1])
            ubg.append([1])

        # smooth Mach change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][0] - U[k][0])
            lbg.append([-0.2])
            ubg.append([0.2])

        # smooth vertical-rate change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][1] - U[k][1])
            lbg.append([-500 * fpm])
            ubg.append([500 * fpm])

        # smooth heading change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][2] - U[k][2])
            lbg.append([-15 * pi / 180])
            ubg.append([15 * pi / 180])

        # fuel constraint (mass states are SCALED)
        g.append(X[0][3] - X[-1][3])
        lbg.append([0])
        ubg.append([self.fuel_max * _Sm])

        if customized_max_fuel is not None:
            g.append(X[0][3] - X[-1][3] - customized_max_fuel * _Sm)
            lbg.append([-ca.inf])
            ubg.append([0])

        # ── Soft regularisation on consecutive control changes ────────
        reg_weights = kwargs.get("reg_weights", {})
        _w_mach = float(reg_weights.get("mach", 0.0))
        _w_vs = float(reg_weights.get("vs", 0.0))
        _w_psi = float(reg_weights.get("heading", 0.0))
        if _w_mach > 0 or _w_vs > 0 or _w_psi > 0:
            dU_mach = ca.vertcat(*[U[k + 1][0] - U[k][0] for k in range(self.nodes - 1)])
            dU_vs = ca.vertcat(*[U[k + 1][1] - U[k][1] for k in range(self.nodes - 1)])
            dU_psi = ca.vertcat(*[U[k + 1][2] - U[k][2] for k in range(self.nodes - 1)])
            J += _w_mach * ca.sumsqr(dU_mach)
            J += _w_vs * ca.sumsqr(dU_vs)
            J += _w_psi * ca.sumsqr(dU_psi)
            if self.debug:
                print(
                    f"  Regularisation: w_mach={_w_mach}, "
                    f"w_vs={_w_vs}, w_heading={_w_psi}"
                )

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        X = ca.horzcat(*X)
        U = ca.horzcat(*U)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Create NLP solver
        nlp = {"f": J, "x": w, "g": g}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options)

        # warm-start support
        warm = kwargs.get("warm_start", None)
        if warm is not None:
            self.solution = self.solver(
                x0=warm["x0"], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg,
                lam_x0=warm["lam_x0"], lam_g0=warm["lam_g0"],
            )
        else:
            self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt, **kwargs)
        df_copy = df.copy()

        # Sanity checks
        if not self.solver.stats()["success"]:
            warnings.warn("flight might be infeasible.")

        if df.altitude.max() < 5000:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            df = None

        if df is not None:
            final_mass = df.mass.iloc[-1]
            if final_mass < self.oew:
                warnings.warn("final mass condition violated (smaller than OEW).")
                df = None
            if final_mass > self.mlw:
                warnings.warn("final mass condition violated (larger than MLW).")
                df = None

        if return_failed:
            return df_copy

        return df


class MultiPhase(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cruise = Cruise(*args, **kwargs)
        self.climb = Climb(*args, **kwargs)
        self.descent = Descent(*args, **kwargs)

    def enable_wind(self, windfield: pd.DataFrame):
        w = tools.PolyWind(
            windfield, self.proj, self.lat1, self.lon1, self.lat2, self.lon2
        )
        self.cruise.wind = w
        self.climb.wind = w
        self.descent.wind = w

    def change_engine(self, engtype):
        self.cruise.engtype = engtype
        self.cruise.engine = oc.prop.engine(engtype)
        self.cruise.thrust = oc.Thrust(self.actype, engtype)
        self.cruise.fuelflow = oc.FuelFlow(self.actype, engtype, polydeg=2)
        self.cruise.emission = oc.Emission(self.actype, engtype)

        self.climb.engtype = engtype
        self.climb.engine = oc.prop.engine(engtype)
        self.climb.thrust = oc.Thrust(self.actype, engtype)
        self.climb.fuelflow = oc.FuelFlow(self.actype, engtype, polydeg=2)
        self.climb.emission = oc.Emission(self.actype, engtype)

        self.descent.engtype = engtype
        self.descent.engine = oc.prop.engine(engtype)
        self.descent.thrust = oc.Thrust(self.actype, engtype)
        self.descent.fuelflow = oc.FuelFlow(self.actype, engtype, polydeg=2)
        self.descent.emission = oc.Emission(self.actype, engtype)

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """
        Calculate the optimal trajectory including climb, cruise, and descent phases.

        Parameters:
        objective (str or tuple): The optimization objective for the trajectory.
            It can be a string or a tuple of strings specifying the objective for
            climb, cruise, and descent respectively. Default is "fuel".
        **kwargs: Additional keyword arguments.

        Returns:
        pd.DataFrame: A DataFrame containing the combined trajectory data.

        The DataFrame includes columns for mass, latitude, longitude, true airspeed,
            and timestamp (ts) among others.

        The method performs the following steps:
        1. Calculate the preliminary optimal cruise trajectory parameters.
        2. Calculate the optimal climb trajectory.
        3. Update the cruise parameters based on the climb results and recalculate the cruise trajectory.
        4. Calculate the optimal descent trajectory.
        5. Determine the top of descent (TOD) point.
        6. Adjust the timestamps for the cruise and descent phases.
        7. Concatenate the climb, cruise, and descent data into a single DataFrame.

        If the `debug` attribute is set to True, debug information will be printed.
        """
        if isinstance(objective, str):
            obj_cl = obj_cr = obj_de = objective
        else:
            obj_cl, obj_cr, obj_de = objective

        if self.debug:
            print("Finding the preliminary optimal cruise trajectory parameters...")
            # Print performance model info
            print(f"Using performance model: {self.perf_model.upper()}")

        dfcr = self.cruise.trajectory(obj_cr, **kwargs)

        # climb
        if self.debug:
            print("Finding optimal climb trajectory...")

        dfcl = self.climb.trajectory(obj_cl, dfcr, **kwargs)

        # cruise
        if self.debug:
            print("Finding optimal cruise trajectory...")

        self.cruise.mass_init = dfcl.mass.iloc[-1]
        self.cruise.lat1 = dfcl.latitude.iloc[-1]
        self.cruise.lon1 = dfcl.longitude.iloc[-1]
        dfcr = self.cruise.trajectory(obj_cr, **kwargs)

        # descent
        if self.debug:
            print("Finding optimal descent trajectory...")

        self.descent.mass_init = dfcr.mass.iloc[-1]
        dfde = self.descent.trajectory(obj_de, dfcr, **kwargs)

        # find top of descent
        dbrg = np.array(
            openap.aero.bearing(
                dfde.latitude.iloc[0],
                dfde.longitude.iloc[0],
                dfcr.latitude,
                dfcr.longitude,
            )
        )
        ddbrg = np.abs((dbrg[1:] - dbrg[:-1]).round())
        idx = np.where(ddbrg > 90)[0]
        idx_tod = idx[0] if len(idx) > 0 else -1

        dfcr = dfcr.iloc[: idx_tod + 1]

        # time at top of climb
        dfcr.ts = dfcl.ts.iloc[-1] + dfcr.ts

        # time at top of descent, considering the distant between last point in cruise and tod

        x1, y1 = self.proj(dfcr.longitude.iloc[-1], dfcr.latitude.iloc[-1])
        x2, y2 = self.proj(dfde.longitude.iloc[0], dfde.latitude.iloc[0])

        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        v = dfcr.tas.iloc[-1] * kts
        dt = np.round(d / v)
        dfde.ts = dfcr.ts.iloc[-1] + dt + dfde.ts

        df_full = pd.concat([dfcl, dfcr, dfde], ignore_index=True)

        return df_full

    def get_solver_stats(self):
        """Get solver statistics for all phases.

        Returns:
            dict: Solver statistics for climb, cruise, and descent phases.
        """
        return {
            "climb": self.climb.solver.stats(),
            "cruise": self.cruise.solver.stats(),
            "descent": self.descent.solver.stats(),
        }
