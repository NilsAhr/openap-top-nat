import warnings
from math import pi

import casadi as ca
import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from .base import Base, R_EARTH, S_X, S_X_INV


class Cruise(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fix_mach = False
        self.fix_alt = False
        self.fix_track = False
        self.allow_descent = True

    def fix_mach_number(self):
        self.fix_mach = True

    def fix_cruise_altitude(self):
        self.fix_alt = True

    def fix_track_angle(self):
        self.fix_track = True

    def allow_cruise_descent(self):
        self.allow_descent = True

    def init_conditions(self, **kwargs):
        """Initialize direct collocation bounds and guesses.

        States are now **scaled**:
          [lat_rad*S_lat, lon_rad*S_lon, h_m*S_h, m_kg*S_m, ts_s*S_ts]
        See ``S_X`` / ``S_X_INV`` in base.py for the numeric values.
        """

        # ── helpers ─────────────────────────────────────────────────
        Sl, So, Sh, Sm, St = S_X          # 10, 10, 1e-4, 1/70000, 1

        # Origin / destination in radians (state coordinates)
        d2r = pi / 180.0
        lat_0 = self.lat1 * d2r
        lon_0 = self.lon1 * d2r
        lat_f = self.lat2 * d2r
        lon_f = self.lon2 * d2r

        # Bounding box in radians with generous margins
        lat_margin = 10.0 * d2r   # ~10 deg (~1100 km)
        lon_margin = 15.0 * d2r   # ~15 deg (wider – longitude shrinks at high lat)
        lat_min = min(lat_0, lat_f) - lat_margin
        lat_max = max(lat_0, lat_f) + lat_margin
        lon_min = min(lon_0, lon_f) - lon_margin
        lon_max = max(lon_0, lon_f) + lon_margin

        ts_min = 0
        ts_max = max(5, self.range / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        #h_max = kwargs.get("h_cruise", self.aircraft["limits"]["h_cruise"]) # 0.85 from ceiling
        #h_min = kwargs.get("h_min", 15_000 * ft)
        h_min = kwargs.get("h_min", 20_000 * ft)

        # Optional: pin Mach to a specific value (overrides normal bounds)
        fixed_mach = kwargs.get("fixed_mach", None)

        # Initial bearing (departure) and final bearing (arrival) on GC
        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180
        hdg_final = (oc.aero.bearing(self.lat2, self.lon2, self.lat1, self.lon1) + 180) % 360
        psi_f = hdg_final * pi / 180

        # ── ALL state bounds are in SCALED units ────────────────────
        # Initial conditions - Lower upper bounds
        self.x_0_lb = [lat_0*Sl, lon_0*So, h_min*Sh, self.mass_init*Sm, ts_min*St]
        self.x_0_ub = [lat_0*Sl, lon_0*So, h_max*Sh, self.mass_init*Sm, ts_min*St]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [lat_f*Sl, lon_f*So, h_min*Sh, self.oew*Sm, ts_min*St]
        self.x_f_ub = [lat_f*Sl, lon_f*So, h_max*Sh, self.mass_init*Sm, ts_max*St]

        # States - Lower and upper bounds
        self.x_lb = [lat_min*Sl, lon_min*So, h_min*Sh, self.oew*Sm, ts_min*St]
        self.x_ub = [lat_max*Sl, lon_max*So, h_max*Sh, self.mass_init*Sm, ts_max*St]

        if fixed_mach is not None:
            # Pin Mach: lb == ub == fixed_mach on every node
            self.fix_mach = True
            mach_lo = fixed_mach
            mach_hi = fixed_mach
            mach_guess = fixed_mach
        else:
            mach_lo_init = self.mach_max - 0.06
            mach_hi = self.mach_max
            mach_lo_mid = 0.78
            mach_guess = self.mach_max - 0.03

        # Control init - lower and upper bounds (use initial bearing)
        self.u_0_lb = [fixed_mach or mach_lo_init, -500 * fpm, psi - pi / 4]
        self.u_0_ub = [fixed_mach or mach_hi, 500 * fpm, psi + pi / 4]

        # Control final - lower and upper bounds (use FINAL bearing)
        self.u_f_lb = [fixed_mach or mach_lo_mid, -500 * fpm, psi_f - pi / 4]
        self.u_f_ub = [fixed_mach or mach_hi, 500 * fpm, psi_f + pi / 4]

        # Control - Lower and upper bound (wide enough for full GC arc)
        psi_mid = (psi + psi_f) / 2
        self.u_lb = [fixed_mach or mach_lo_mid, -500 * fpm, psi_mid - pi / 2]
        self.u_ub = [fixed_mach or mach_hi, 500 * fpm, psi_mid + pi / 2]

        # Initial guess - states  (already in SCALED units from base.py)
        self.x_guess = self.initial_guess()

        # Initial guess - controls
        # Use linearly varying heading from initial to final GC bearing
        # so that the guess is consistent with the GC state guess.
        psi_guess = np.linspace(float(psi), float(psi_f), self.nodes)
        self.u_guess_array = [
            [mach_guess, 0, psi_guess[i]]
            for i in range(self.nodes)
        ]
        self.u_guess = [mach_guess, 0, float(psi)]  # scalar fallback

    def trajectory(self, objective="fuel", **kwargs) -> pd.DataFrame:
        """
        Computes the optimal trajectory for the aircraft based on the given objective.

        Parameters:
        - objective (str): The objective of the optimization, default is "fuel".
        - **kwargs: Additional keyword arguments.
            - max_fuel (float): Customized maximum fuel constraint.
            - initial_guess (pd.DataFrame): Initial guess for the trajectory. This is
                usually a exsiting flight trajectory.
            - return_failed (bool): If True, returns the DataFrame even if the
                optimization fails. Default is False.

        Returns:
        - pd.DataFrame: A DataFrame containing the optimized trajectory.

        Note:
        - The function uses CasADi for symbolic computation and optimization.
        - The constraints and bounds are defined based on the aircraft's performance
            and operational limits.
        """

        # arguments passed init_condition to overwright h_min and h_max
        self.init_conditions(**kwargs)

        self.init_model(objective, **kwargs)

        # Print performance model info
        if self.debug:
            print(f"Using performance model: {self.perf_model.upper()}")

        customized_max_fuel = kwargs.get("max_fuel", None)

        initial_guess = kwargs.get("initial_guess", None)

        # Only reset u_guess_array when an external initial_guess is
        # provided (the block below will recompute it from the DataFrame).
        # Otherwise keep the linearly-varying heading array from
        # init_conditions().
        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

            # Compute per-node heading guesses from the deviated positions.
            # x_guess is in SCALED units; unscale lat/lon for trig.
            n_pts = len(self.x_guess)
            if n_pts >= 2:
                headings = np.zeros(self.nodes)
                for i in range(self.nodes):
                    i_next = min(i + 1, n_pts - 1)
                    # Unscale lat/lon to radians for heading computation
                    lat_i  = self.x_guess[i, 0] * S_X_INV[0]   # / S_lat
                    lat_nx = self.x_guess[i_next, 0] * S_X_INV[0]
                    lon_i  = self.x_guess[i, 1] * S_X_INV[1]
                    lon_nx = self.x_guess[i_next, 1] * S_X_INV[1]
                    dlat = lat_nx - lat_i
                    dlon = lon_nx - lon_i
                    # Approximate east/north displacements from radian diffs
                    dx = dlon * np.cos(lat_i)  # east
                    dy = dlat                    # north
                    headings[i] = np.arctan2(dx, dy)  # radians, from north
                self.u_guess_array = [
                    [self.u_guess[0], self.u_guess[1], headings[i]]
                    for i in range(self.nodes)
                ]

        return_failed = kwargs.get("return_failed", False)

        C, D, B = self.collocation_coeff()

        # Start with an empty NLP
        w = []  # Containing all the states & controls generated
        w0 = []  # Containing the initial guess for w
        lbw = []  # Lower bound constraints on the w variable
        ubw = []  # Upper bound constraints on the w variable
        J = 0  # Objective function
        g = []  # Constraint function
        lbg = []  # Constraint lb value
        ubg = []  # Constraint ub value

        # Diagonal scaling for collocation constraint residuals.
        # With NLP variable scaling (S_X), the dynamics rates are:
        #   dlat_s ~ 4e-4,  dlon_s ~ 8e-4,  dh_s ~ 1e-4,
        #   dm_s ~ 4e-5,    dts ~ 1
        # Multiplied by dt (~600 s) the residuals are O(0.02–0.5)
        # for lat/lon/h/m but O(600) for ts.  This _cscale brings
        # all rows into the same order of magnitude.
        _cscale = ca.vertcat(1.0, 1.0, 10.0, 50.0, 1e-3)

        # For plotting x and u given w
        X = []
        U = []

        # Apply initial conditions
        # Create Xk such that it is the same length as x
        nstates = self.x.shape[0]
        Xk = ca.MX.sym("X0", nstates, self.x.shape[1])
        w.append(Xk)
        lbw.append(self.x_0_lb)
        ubw.append(self.x_0_ub)
        w0.append(self.x_guess[0])
        X.append(Xk)

        # Formulate the NLP
        for k in range(self.nodes):
            # New NLP variable for the control
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

            # State at collocation points
            Xc = []
            for j in range(self.polydeg):
                Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), nstates)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
                w0.append(self.x_guess[k])

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                # Expression for the state derivative at the collocation point
                xpc = C[0, j] * Xk
                for r in range(self.polydeg):
                    xpc = xpc + C[r + 1, j] * Xc[r]

                # Append collocation equations (scaled)
                fj, qj = self.func_dynamics(Xc[j - 1], Uk)
                g.append(_cscale * (self.dt * fj - xpc))
                lbg.append([0] * nstates)
                ubg.append([0] * nstates)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Add contribution to quadrature function
                # J = J + B[j] * qj * dt
                J = J + B[j] * qj

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym("X_" + str(k + 1), nstates)
            w.append(Xk)
            X.append(Xk)

            # lbw.append(x_lb)
            # ubw.append(x_ub)

            if k < self.nodes - 1:
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
            else:
                # Final conditions
                lbw.append(self.x_f_lb)
                ubw.append(self.x_f_ub)

            w0.append(self.x_guess[k + 1])

            # Add equality constraint (scaled)
            g.append(_cscale * (Xk_end - Xk))
            lbg.append([0] * nstates)
            ubg.append([0] * nstates)

        w.append(self.ts_final)
        lbw.append([0])
        ubw.append([ca.inf])
        w0.append([self.range * 1000 / 200])

        # aircraft performance constraints
        # (states are in scaled NLP units – unscale for physics)
        _Sh_inv = float(S_X_INV[2])   # 10 000
        _Sm_inv = float(S_X_INV[3])   # 70 000
        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            h_phys    = X[k][2] * _Sh_inv            # scaled → m
            mass_phys = X[k][3] * _Sm_inv            # scaled → kg
            v = oc.aero.mach2tas(U[k][0], h_phys, dT=self.dT)
            tas = v / kts
            alt = h_phys / ft
            rho = oc.aero.density(h_phys, dT=self.dT)
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)

            # max_thrust * 95% > drag (5% margin)
            g.append(thrust_max * 0.95 - self.drag.clean(mass_phys, tas, alt, dT=self.dT))
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift * 80% > weight (20% margin)
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v**2 * S
            g.append(L_max * 0.8 - mass_phys * oc.aero.g0)
            lbg.append([0])
            ubg.append([ca.inf])

        # ts and dt should be consistent
        for k in range(self.nodes - 1):
            g.append(X[k + 1][4] - X[k][4] - self.dt)
            lbg.append([-1])
            ubg.append([1])

        # # smooth Mach number change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][0] - U[k][0])
        #     lbg.append([-0.2])
        #     ubg.append([0.2])  # to be tunned

        # # smooth vertical rate change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][1] - U[k][1])
        #     lbg.append([-500 * fpm])
        #     ubg.append([500 * fpm])  # to be tunned

        # smooth heading change
        for k in range(self.nodes - 1):
            g.append(U[k + 1][2] - U[k][2])
            #lbg.append([-15 * pi / 180])
            #ubg.append([15 * pi / 180])
            lbg.append([-5 * pi / 180])
            ubg.append([5 * pi / 180])


        # optional constraints
        if self.fix_mach:
            for k in range(self.nodes - 1):
                g.append(U[k + 1][0] - U[k][0])
                lbg.append([0])
                ubg.append([0])

        if self.fix_alt:
            for k in range(self.nodes):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([0])

        if self.fix_track:
            for k in range(self.nodes - 1):
                g.append(U[k + 1][2] - U[k][2])
                lbg.append([0])
                ubg.append([0])

        if not self.allow_descent:
            for k in range(self.nodes):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([ca.inf])

        # add fuel constraint  (X[·][3] is in scaled mass units)
        _Sm = float(S_X[3])  # 1 / 70 000
        g.append(X[0][3] - X[-1][3])
        lbg.append([0])
        ubg.append([self.fuel_max * _Sm])

        if customized_max_fuel is not None:
            g.append(X[0][3] - X[-1][3] - customized_max_fuel * _Sm)
            lbg.append([-ca.inf])
            ubg.append([0])

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

        # Create an NLP solver
        nlp = {"f": J, "x": w, "g": g}

        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options)
        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt, **kwargs)

        df_copy = df.copy()

        # check if the optimizer has failed
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

        if return_failed:
            return df_copy

        return df
