"""
BADA3 performance backend adapters for the top-nat optimizer.

This module adapts BADA3 Drag/Thrust/FuelFlow to the minimal interface that
openap-top-nat expects, without changing the optimizer formulation.

Requirements:
- A BADA3 dataset folder containing SYNONYM.NEW and matching *.OPF files.
- The OpenAP BADA3 addon available for import: openap.extra.bada3

If openap.extra.bada3 is not available in your environment, vendor its bada3.py
into your project and update the import below accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import tempfile
import shutil
from pathlib import Path

from openap.base import FuelFlowBase
from openap.extra.aero import ft, fpm, kts
import casadi as ca

# Try importing BADA3 addon from OpenAP
try:
    from openap.addon import bada3 as _bada3
except Exception as e:  # pragma: no cover
    _bada3 = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# bada3 model wrapper around the parsed BADA3 data dictionary
@dataclass
class BADA3Model:
    """Light wrapper for the parsed BADA3 aircraft model dictionary."""
    data: dict


def _require_bada3():
    if _bada3 is None:
        raise ImportError(
            "openap.extra.bada3 is not available. Install a version of OpenAP "
            "that includes the BADA3 addon, or vendor bada3.py and update the import. "
            f"Original import error: {_IMPORT_ERROR}"
        )


def load_model(actype: str, bada3_path: str) -> BADA3Model:
    """Load and parse the BADA3 aircraft model for actype from bada3_path."""
    _require_bada3()
    
    try:
        # Try loading with default UTF-8 encoding first
        model = _bada3.load_bada3(actype, bada3_path)
        return BADA3Model(model)
    except UnicodeDecodeError as e:
        #print(f"UTF-8 encoding failed for {actype}: {e}")
        #print("Attempting to fix SYNONYM.NEW encoding issue...")
        
        # Handle the SYNONYM.NEW encoding issue specifically
        return _load_model_with_synonym_fix(actype, bada3_path)


def _load_model_with_synonym_fix(actype: str, bada3_path: str) -> BADA3Model:
    """Load BADA3 model with SYNONYM.NEW encoding conversion."""
    bada3_dir = Path(bada3_path)
    synonym_file = bada3_dir / "SYNONYM.NEW"
    
    if not synonym_file.exists():
        raise FileNotFoundError(f"SYNONYM.NEW not found in {bada3_path}")
    
    # Create a temporary directory with a UTF-8 converted SYNONYM.NEW
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Convert SYNONYM.NEW from latin-1 to UTF-8
        try:
            #("Converting SYNONYM.NEW from latin-1 to UTF-8...")
            with open(synonym_file, 'r', encoding='latin-1') as f:
                content = f.read()
            with open(temp_path / "SYNONYM.NEW", 'w', encoding='utf-8') as f:
                f.write(content)
            #print("✓ SYNONYM.NEW converted successfully")
        except Exception as e:
            raise ValueError(f"Failed to convert SYNONYM.NEW: {e}")
        
        # Copy the aircraft-specific OPF file (these are usually UTF-8 compatible)
        opf_file = bada3_dir / f"{actype.upper()}__.OPF"
        if opf_file.exists():
            shutil.copy2(opf_file, temp_path)
        else:
            raise FileNotFoundError(f"OPF file not found for aircraft {actype}")
        
        # Copy other aircraft files if they exist
        for suffix in ['APF', 'PTD', 'PTF']:
            src_file = bada3_dir / f"{actype.upper()}__.{suffix}"
            if src_file.exists():
                shutil.copy2(src_file, temp_path)
        
        # Try loading with the converted files
        try:
            model = _bada3.load_bada3(actype, str(temp_path))
            print(f"✓ Successfully loaded {actype} with fixed encoding")
            return BADA3Model(model)
        except Exception as load_error:
            raise ValueError(f"Failed to load {actype} even after encoding fix: {load_error}")


class BADA3DragAdapter:
    """
    Adapter to match the optimizer's Drag interface.

    Must provide:
    - clean(mass, tas_kt, alt_ft, [pa|vs], [dT]) -> D [N]
    - attribute 'polar' with at least: polar['clean']['cd0'] and ['k']
    """

    def __init__(self, actype: str, bada3_path: str, model: Optional[BADA3Model] = None):
        _require_bada3()
        self.actype = actype.upper()
        self._model = model or load_model(actype, bada3_path)
        self._drag = _bada3.Drag(actype, bada3_path, model=self._model.data)

        # Inject CasADi overrides into the addon instance for numeric path consistency
        try:
            import openap.casadi as oc
            if hasattr(oc, "numpy_override"):
                self.sci = oc.numpy_override
                self.aero = oc.aero_override
                self._drag.sci = oc.numpy_override
                self._drag.aero = oc.aero_override
        except Exception:
            # Fall back to numpy/aero; we'll only use this in numeric branch
            self.sci = getattr(self._drag, "sci", None)
            self.aero = getattr(self._drag, "aero", None)

        # Cache polar for symbolic path and constraints
        data = self._model.data
        self._S = data["wing"]["area"]
        self._cd0_cr = data["CD0"]["CR"]
        self._k_cr = data["CD2"]["CR"]

        self.polar = {
            "clean": {"cd0": self._cd0_cr, "k": self._k_cr},
            "ap": {"cd0": data["CD0"]["AP"], "k": data["CD2"]["AP"]},
            "ld": {"cd0": data["CD0"]["LD"], "k": data["CD2"]["LD"]},
        }

    def _is_symbolic(self, *vals):
        for v in vals:
            if isinstance(v, (ca.MX, ca.SX)):
                return True
        return False

    def _clean_symbolic(self, mass, tas_kt, alt_ft, vs_ftmin=0):
        # Compute clean drag symbolically (branch-free, CasADi-safe)
        tas = tas_kt * kts
        alt = alt_ft * ft
        rho = self.aero.density(alt)  # CasADi-safe density
        gamma = self.sci.arctan2(vs_ftmin * fpm, tas)

        g0 = getattr(self.aero, "g0", 9.80665)  # in case
        weight = mass * g0
        L = weight * self.sci.cos(gamma)

        q = 0.5 * rho * tas**2
        # Guard against zero dynamic pressure
        qS = self.sci.maximum(q * self._S, 1e-9)
        CL = L / qS
        CD = self._cd0_cr + self._k_cr * CL**2
        D = q * self._S * CD
        return D

    def clean(self, mass: Any, tas_kt: Any, alt_ft: Any, *args, **kwargs) -> Any:
        """
        Return drag force [N] in clean config.

        Accepts extra args/kwargs for compatibility (e.g., dT passed by optimizer),
        but only vs is used here, default 0 for cruise constraints.
        """
        # Try to extract vertical speed if provided (optimizer usually passes dT only)
        vs = 0
        if len(args) > 0 and args[0] is not None:
            # If caller passed 'vs' as positional arg 4 (rare), accept it
            vs = args[0]
        vs = kwargs.get("vs", vs)

        if self._is_symbolic(mass, tas_kt, alt_ft, vs):
            return self._clean_symbolic(mass, tas_kt, alt_ft, vs)
        else:
            # Numeric path: delegate to addon implementation
            return self._drag.clean(mass, tas_kt, alt_ft)


class BADA3ThrustAdapter:
    """
    Adapter to match the optimizer's Thrust interface.

    Provides CasADi-safe climb/cruise/idle: returns MX when given MX inputs,
    and delegates to the addon for numeric (numpy) inputs.
    """

    def __init__(self, actype: str, bada3_path: str, model: Optional[BADA3Model] = None):
        _require_bada3()
        self.actype = actype.upper()
        self._model = model or load_model(actype, bada3_path)
        self._thr = _bada3.Thrust(actype, bada3_path, model=self._model.data)

        # Inject CasADi overrides into both the adapter and underlying model
        try:
            import openap.casadi as oc
            if hasattr(oc, "numpy_override"):
                self.sci = oc.numpy_override
                self.aero = oc.aero_override
                self._thr.sci = oc.numpy_override
                self._thr.aero = oc.aero_override
        except Exception:
            # Fallback (won't be used in symbolic branch)
            self.sci = getattr(self._thr, "sci", None)
            self.aero = getattr(self._thr, "aero", None)

        # Cache coefficients from model for symbolic path
        data = self._model.data
        self.engine_type = data["engine"]["type"]
        self._Ct0, self._Ct1, self._Ct2, self._Ct3, self._Ct4 = data["Ct"]
        self._ctdeshigh = data["CTdeshigh"]
        self._ctdeslow  = data["CTdeslow"]
        self._ctdesapp  = data["CTdesapp"]
        self._ctdesld   = data["CTdesld"]
        # hpdes used only to choose descent coeff; keep from addon to match
        self._hpdes = self._thr.hpdes

    def _is_symbolic(self, *vals):
        for v in vals:
            if isinstance(v, (ca.MX, ca.SX)):
                return True
        return False

    def _thrust_climb_symbolic(self, tas_kt, alt_ft, dT=0):
        # BADA3 eq. (3.7-1..7) by engine type; inputs in kt and ft (as in addon)
        if self.engine_type == "turbofan":
            thr_isa = self._Ct0 * (1 - alt_ft / self._Ct1 + self._Ct2 * alt_ft**2)
        elif self.engine_type == "turboprop":
            thr_isa = self._Ct0 / self.sci.maximum(tas_kt, 1e-6) * (1 - alt_ft / self._Ct1) + self._Ct2
        elif self.engine_type in ("piston", "electric"):
            thr_isa = self._Ct0 * (1 - alt_ft / self._Ct1) + self._Ct2 / self.sci.maximum(tas_kt, 1e-6)
        else:
            thr_isa = self._Ct0 * (1 - alt_ft / self._Ct1 + self._Ct2 * alt_ft**2)

        dT_eff = dT - self._Ct3
        c_tc5  = self.sci.maximum(self._Ct4, 0.0)
        dT_lim = self.sci.maximum(0.0, self.sci.minimum(c_tc5 * dT_eff, 0.4))
        return thr_isa * (1 - dT_lim)

    def climb(self, tas_kt: Any, alt_ft: Any, dT: Any = 0) -> Any:
        if self._is_symbolic(tas_kt, alt_ft, dT):
            return self._thrust_climb_symbolic(tas_kt, alt_ft, dT)
        return self._thr.climb(tas_kt, alt_ft, dT=dT)

    def cruise(self, tas_kt: Any, alt_ft: Any, dT: Any = 0) -> Any:
        # BADA3 eq. (3.7-8): 0.95 * climb
        if self._is_symbolic(tas_kt, alt_ft, dT):
            return 0.95 * self._thrust_climb_symbolic(tas_kt, alt_ft, dT)
        return self._thr.cruise(tas_kt, alt_ft, dT=dT)

    def descent_idle(self, tas_kt: Any, alt_ft: Any, dT: Any = 0, config: str = "CR") -> Any:
        if config not in ("CR", "AP", "LD"):
            raise ValueError(f"Config '{config}' unknown (expected CR/AP/LD).")
        thr_cl_max = self.climb(tas_kt, alt_ft, dT=dT)
        ctdes_cfg = self._ctdeslow if config == "CR" else (self._ctdesapp if config == "AP" else self._ctdesld)
        thr_des_high = self._ctdeshigh * thr_cl_max
        thr_des_low  = ctdes_cfg * thr_cl_max
        if self._is_symbolic(alt_ft):
            return ca.if_else(alt_ft > self._hpdes, thr_des_high, thr_des_low)
        # Numeric fallback
        return self._thr.idle(tas_kt, alt_ft, dT=dT, config=config)


class BADA3FuelFlowAdapter(FuelFlowBase):
    def __init__(self, actype: str, bada3_path: str, model: Optional[BADA3Model] = None):
        super().__init__(actype)
        # Ensure CasADi context (numpy/aero overrides) if available
        try:
            import openap.casadi as oc  # noqa
            if hasattr(oc, "numpy_override"):
                self.sci = oc.numpy_override
                self.aero = oc.aero_override
        except Exception:
            pass

        _require_bada3()
        self.actype = actype.upper()
        self._model = model or load_model(actype, bada3_path)
        self._ff = _bada3.FuelFlow(actype, bada3_path, model=self._model.data)
        self._ff.sci = self.sci
        self._ff.aero = self.aero

        # Cache coefficients — mirrors addon fields
        data = self._model.data
        # Drag polar for required-thrust
        self._cd0_cr = data["CD0"]["CR"]
        self._k_cr   = data["CD2"]["CR"]
        self._S      = data["wing"]["area"]

        # Engine and fuel coefficients
        self.engine_type = data["engine"]["type"]
        self._Cf0, self._Cf1 = data["Cf"]                # cf1, cf2 in addon naming
        self._CfDes0, self._CfDes1 = data["CfDes"]       # cf3, cf4
        self._CfCrz  = data.get("CfCrz", 1.0)            # cfcr (cruise correction)
        self._Ct0, self._Ct1, self._Ct2, self._Ct3, self._Ct4 = data["Ct"]  # thrust climb coeffs

        self._g = 9.81
        self._symbolic_func_enroute = None
        self._symbolic_func_nominal = None

    def _is_symbolic(self, *vals):
        for v in vals:
            if isinstance(v, (ca.MX, ca.SX)):
                return True
        return False

    # --- symbolic building blocks ---

    def _drag_clean_symbolic(self, mass, tas_kt, alt_ft, vs_ftmin=0):
        tas = tas_kt * kts
        alt = alt_ft * ft
        rho = self.aero.density(alt)
        gamma = self.sci.arctan2(vs_ftmin * fpm, tas)

        weight = mass * self._g
        L = weight * self.sci.cos(gamma)

        q = 0.5 * rho * tas**2
        CL = L / self.sci.maximum(q * self._S, 1e-9)
        CD = self._cd0_cr + self._k_cr * CL**2
        D  = q * self._S * CD
        return D

    def _thrust_climb_symbolic(self, tas_kt, alt_ft, dT=0):
        # BADA3 climb thrust (eq. 3.7) with engine-type branches
        if self.engine_type == "turbofan":
            thr_isa = self._Ct0 * (1 - alt_ft / self._Ct1 + self._Ct2 * alt_ft**2)
        elif self.engine_type == "turboprop":
            thr_isa = self._Ct0 / self.sci.maximum(tas_kt, 1e-6) * (1 - alt_ft / self._Ct1) + self._Ct2
        elif self.engine_type in ("piston", "electric"):
            thr_isa = self._Ct0 * (1 - alt_ft / self._Ct1) + self._Ct2 / self.sci.maximum(tas_kt, 1e-6)
        else:
            # Fallback to turbofan shape if unknown
            thr_isa = self._Ct0 * (1 - alt_ft / self._Ct1 + self._Ct2 * alt_ft**2)

        # Temperature deviation correction and limits (eq. 3.7.4–3.7.7)
        dT_eff = dT - self._Ct3
        c_tc5  = self.sci.maximum(self._Ct4, 0.0)
        dT_lim = self.sci.maximum(0.0, self.sci.minimum(c_tc5 * dT_eff, 0.4))
        return thr_isa * (1 - dT_lim)

    def _idle_symbolic(self, alt_ft):
        # BADA3 idle fuel (kg/min) then convert to kg/s
        if self.engine_type in ("turbofan", "turboprop"):
            f_min = self._CfDes0 * (1 - alt_ft / self._CfDes1)
        elif self.engine_type in ("piston", "electric"):
            f_min = self._CfDes0
        else:
            f_min = self._CfDes0
        return f_min / 60.0

    def _eta_turbofan(self, tas_kt):
        # eta in kg/(N*min); addon divides by 60 later
        return (self._Cf0 * (1 + tas_kt / self._Cf1)) * 1e-3

    def _eta_turboprop(self, tas_kt):
        # addon: eta = cf1 * (1 - tas/cf2) * (tas/1e3) * 1e-3
        return (self._Cf0 * (1 - tas_kt / self._Cf1) * (tas_kt / 1e3)) * 1e-3

    def _nominal_symbolic(self, mass, tas_kt, alt_ft, vs_ftmin=0, acc=0, dT=0):
        # Required thrust with climb-thrust cap and idle floor
        D     = self._drag_clean_symbolic(mass, tas_kt, alt_ft, vs_ftmin)
        gamma = self.sci.arctan2(vs_ftmin * fpm, tas_kt * kts)
        T_dem = D + mass * self._g * self.sci.sin(gamma) + mass * acc
        T_cap = self._thrust_climb_symbolic(tas_kt, alt_ft, dT)
        T     = self.sci.minimum(T_dem, T_cap)

        if self.engine_type == "turbofan":
            eta = self._eta_turbofan(tas_kt)
            f_nom_minute = eta * T                   # kg/min
            f_nom = f_nom_minute / 60.0              # kg/s
        elif self.engine_type == "turboprop":
            eta = self._eta_turboprop(tas_kt)
            f_nom_minute = eta * T
            f_nom = f_nom_minute / 60.0
        elif self.engine_type in ("piston", "electric"):
            # addon uses cf1 as constant (kg/min), convert to kg/s
            f_nom = self._Cf0 / 60.0
        else:
            # Safe fallback
            eta = self._eta_turbofan(tas_kt)
            f_nom = (eta * T) / 60.0

        # Idle floor
        f_idle = self._idle_symbolic(alt_ft)
        return self.sci.maximum(f_nom, f_idle)

    def _enroute_symbolic(self, mass, tas_kt, alt_ft, vs_ftmin=0, acc=0, dT=0):
        # Cruise enroute fuel = cfcr * nominal
        return self._CfCrz * self._nominal_symbolic(mass, tas_kt, alt_ft, vs_ftmin, acc, dT)

    # --- public API ---

    def enroute(self, mass: Any, tas_kt: Any, alt_ft: Any, vs: Any = 0, acc: Any = 0, dT: Any = 0, **kwargs):
        if self._is_symbolic(mass, tas_kt, alt_ft, vs, acc, dT):
            if self._symbolic_func_enroute is None:
                m  = ca.MX.sym("mass")
                v  = ca.MX.sym("tas")
                h  = ca.MX.sym("alt")
                vz = ca.MX.sym("vs")
                ac = ca.MX.sym("acc")
                tD = ca.MX.sym("dT")
                ff_expr = self._enroute_symbolic(m, v, h, vz, ac, tD)
                self._symbolic_func_enroute = ca.Function(
                    "bada3_ff_enroute",
                    [m, v, h, vz, ac, tD],
                    [ff_expr],
                    ["mass","tas","alt","vs","acc","dT"],
                    ["fuelflow"],
                )
            return self._symbolic_func_enroute(mass, tas_kt, alt_ft, vs, acc, dT)
        else:
            return self._ff.enroute(mass, tas_kt, alt_ft, vs)

    def nominal(self, mass: Any, tas_kt: Any, alt_ft: Any, vs: Any = 0, acc: Any = 0, dT: Any = 0, **kwargs) -> Any:
        if self._is_symbolic(mass, tas_kt, alt_ft, vs, acc, dT):
            if self._symbolic_func_nominal is None:
                m  = ca.MX.sym("mass")
                v  = ca.MX.sym("tas")
                h  = ca.MX.sym("alt")
                vz = ca.MX.sym("vs")
                ac = ca.MX.sym("acc")
                tD = ca.MX.sym("dT")
                ff_expr = self._nominal_symbolic(m, v, h, vz, ac, tD)
                self._symbolic_func_nominal = ca.Function(
                    "bada3_ff_nominal",
                    [m, v, h, vz, ac, tD],
                    [ff_expr],
                    ["mass","tas","alt","vs","acc","dT"],
                    ["fuelflow"],
                )
            return self._symbolic_func_nominal(mass, tas_kt, alt_ft, vs, acc, dT)
        return self._ff.nominal(mass, tas_kt, alt_ft, vs)

    def idle(self, mass: Any, tas_kt: Any, alt_ft: Any, **kwargs) -> Any:
        # Numeric idle is fine; add symbolic when needed by the optimizer
        return self._ff.idle(mass, tas_kt, alt_ft)


def make_bada3_backend(actype: str, bada3_path: str):
    """Factory returning (thrust, drag, fuelflow) adapters and the parsed model dict."""
    model = load_model(actype, bada3_path)  # load once
    thrust = BADA3ThrustAdapter(actype, bada3_path, model=model)
    drag = BADA3DragAdapter(actype, bada3_path, model=model)
    fuelflow = BADA3FuelFlowAdapter(actype, bada3_path, model=model)
    return thrust, drag, fuelflow, model