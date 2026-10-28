"""Fuel-saving estimates for Flettner rotor sails, checked against operational data.

A learning / portfolio project, not a design tool. See the README for the
theory, assumptions and limitations.
"""

from .aero import RHO_AIR, RotorCoefficients, RotorConfig, RotorForces, apparent_wind, rotor_forces
from .savings import KNOT, ShipConfig, VoyageResult, estimate_voyage, predict_savings

__all__ = [
    "KNOT",
    "RHO_AIR",
    "RotorCoefficients",
    "RotorConfig",
    "RotorForces",
    "ShipConfig",
    "VoyageResult",
    "apparent_wind",
    "estimate_voyage",
    "predict_savings",
    "rotor_forces",
]

__version__ = "0.1.0"
