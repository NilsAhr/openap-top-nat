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
    model = _bada3.load_bada3(actype, bada3_path)
    return BADA3Model(model)


# Each adapter class wraps the corresponding BADA3 performance class and exposes the exact interface top-nat optimizer expects:
class BADA3DragAdapter:
    """
    Adapter to match the optimizer's Drag interface.

    Must provide:
    - clean(mass, tas_kt, alt_ft, [pa|vs], [dT]) -> D [N]
    - attribute 'polar' with at least: polar['clean']['cd0'] and ['k'] for constraints
    """

    def __init__(self, actype: str, bada3_path: str, model: Optional[BADA3Model] = None):
        _require_bada3()
        self.actype = actype.upper()
        self._model = model or load_model(actype, bada3_path)
        self._drag = _bada3.Drag(actype, bada3_path, model=self._model.data)

        # Provide the polar dict expected by constraints in full.py
        cd0_cr = self._model.data["CD0"]["CR"]
        k_cr = self._model.data["CD2"]["CR"]
        # Optional: expose AP/LD if later needed
        cd0_ap = self._model.data["CD0"]["AP"]
        k_ap = self._model.data["CD2"]["AP"]
        cd0_ld = self._model.data["CD0"]["LD"]
        k_ld = self._model.data["CD2"]["LD"]

        self.polar = {
            "clean": {"cd0": cd0_cr, "k": k_cr}, # Used by optimizer constraints
            "ap": {"cd0": cd0_ap, "k": k_ap},
            "ld": {"cd0": cd0_ld, "k": k_ld},
        }

    def clean(self, mass: Any, tas_kt: Any, alt_ft: Any, *_, **__) -> Any:
        """
        Return drag force [N] in clean config.

        Accepts extra positional/keyword args to be compatible with calls like:
        drag.clean(mass, tas_kt, alt_ft, pa, dT=...)
        """
        return self._drag.clean(mass, tas_kt, alt_ft)


class BADA3ThrustAdapter:
    """
    Adapter to match the optimizer's Thrust interface.

    Must provide:
    - climb(tas_kt, alt_ft, dT=0) -> N
    - descent_idle(tas_kt, alt_ft, dT=0) -> N
    """

    def __init__(self, actype: str, bada3_path: str, model: Optional[BADA3Model] = None):
        _require_bada3()
        self.actype = actype.upper()
        self._model = model or load_model(actype, bada3_path)
        self._thr = _bada3.Thrust(actype, bada3_path, model=self._model.data)

    def climb(self, tas_kt: Any, alt_ft: Any, dT: Any = 0) -> Any:
        return self._thr.climb(tas_kt, alt_ft, dT=dT)

    def descent_idle(self, tas_kt: Any, alt_ft: Any, dT: Any = 0) -> Any:
        # Use cruise ("CR") configuration for descent-idle thrust by default.
        # This matches typical BADA3 usage, where idle thrust during descent is modeled using the cruise config,
        # and aligns with optimizer expectations for descent-idle calculations.
        return self._thr.idle(tas_kt, alt_ft, dT=dT, config="CR")


class BADA3FuelFlowAdapter:
    """
    Adapter to match the optimizer's FuelFlow interface.

    Must provide:
    - nominal(mass, tas_kt, alt_ft, vs=0, dT=? ignored) -> kg/s
    - enroute(mass, tas_kt, alt_ft, vs=0, dT=? ignored) -> kg/s
    - idle(mass, tas_kt, alt_ft, vs=? ignored, dT=? ignored) -> kg/s
    """

    def __init__(self, actype: str, bada3_path: str, model: Optional[BADA3Model] = None):
        _require_bada3()
        self.actype = actype.upper()
        self._model = model or load_model(actype, bada3_path)
        self._ff = _bada3.FuelFlow(actype, bada3_path, model=self._model.data)

    # Accept **kwargs to remain compatible with calls that pass dT or other args
    def nominal(self, mass: Any, tas_kt: Any, alt_ft: Any, vs: Any = 0, **kwargs) -> Any:
        return self._ff.nominal(mass, tas_kt, alt_ft, vs)

    def enroute(self, mass: Any, tas_kt: Any, alt_ft: Any, vs: Any = 0, **kwargs) -> Any:
        return self._ff.enroute(mass, tas_kt, alt_ft, vs)

    def idle(self, mass: Any, tas_kt: Any, alt_ft: Any, **kwargs) -> Any:
        return self._ff.idle(mass, tas_kt, alt_ft)


def make_bada3_backend(actype: str, bada3_path: str):
    """Factory returning (thrust, drag, fuelflow) adapters and the parsed model dict."""
    thrust = BADA3ThrustAdapter(actype, bada3_path)
    drag = BADA3DragAdapter(actype, bada3_path)
    fuelflow = BADA3FuelFlowAdapter(actype, bada3_path)
    return thrust, drag, fuelflow, None