"""Aerodynamics of Flettner rotor sails.

Conventions used throughout the package
---------------------------------------
* SI units: speeds in m/s, forces in N, power in W (kW where the name says so).
* Compass angles (true wind direction, ship heading) are in degrees, measured
  clockwise from north. True wind direction is the meteorological "coming
  from" direction: a wind of 270 deg blows from the west towards the east.
* Apparent wind angle (AWA) is the direction the apparent wind comes *from*,
  relative to the bow, in degrees in (-180, 180]. 0 = head wind,
  +90 = wind from starboard, -90 = wind from port, 180 = wind from astern.
* Ship-fixed force components: thrust is positive forwards along the
  heading, side force is positive towards starboard.

Assumptions
-----------
* Ship speed is speed through the air-relevant frame, i.e. ocean current
  is ignored (speed over ground == speed through water).
* All rotors see the same apparent wind; rotor-rotor and rotor-superstructure
  interaction, the atmospheric boundary layer (wind shear over rotor height)
  and ship motions are ignored.
* Each rotor spins in the sense that makes its Magnus lift point forwards
  (the controller reverses rotation when the wind changes side).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

RHO_AIR = 1.225
"""Air density in kg/m^3 (ISA sea level, 15 degC)."""

DEFAULT_COEFFICIENTS_FILE = "rotor_coefficients_illustrative.csv"


def wrap_angle_deg(angle: ArrayLike) -> NDArray[np.float64]:
    """Wrap angles in degrees to the interval (-180, 180]."""
    wrapped = (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(wrapped == -180.0, 180.0, wrapped)


def apparent_wind(
    true_wind_speed: ArrayLike,
    true_wind_direction_deg: ArrayLike,
    ship_speed: ArrayLike,
    ship_heading_deg: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve the wind triangle.

    The apparent wind is the air velocity felt on board, i.e. the true wind
    velocity minus the ship velocity (both as vectors)::

        v_true = -V_tw * (sin(theta_tw), cos(theta_tw))    # (east, north), blowing *towards*
        v_ship =  V_s  * (sin(psi),      cos(psi))
        v_app  = v_true - v_ship
        V_aw   = |v_app|
        AWA    = wrap(direction_from(v_app) - psi)

    Args:
        true_wind_speed: True wind speed V_tw [m/s].
        true_wind_direction_deg: True wind "from" direction theta_tw [deg from north].
        ship_speed: Ship speed V_s [m/s].
        ship_heading_deg: Ship heading psi [deg from north].

    Returns:
        ``(apparent_wind_speed [m/s], apparent_wind_angle [deg])``. Below
        1e-9 m/s the apparent wind is treated as calm: speed 0 and angle 0
        (the angle is undefined there).
    """
    tws = np.asarray(true_wind_speed, dtype=float)
    twd = np.radians(np.asarray(true_wind_direction_deg, dtype=float))
    vs = np.asarray(ship_speed, dtype=float)
    heading_deg = np.asarray(ship_heading_deg, dtype=float)
    heading = np.radians(heading_deg)

    app_east = -tws * np.sin(twd) - vs * np.sin(heading)
    app_north = -tws * np.cos(twd) - vs * np.cos(heading)

    aws = np.hypot(app_east, app_north)
    # Direction the apparent wind comes from (the opposite of where it blows to).
    app_from_deg = np.degrees(np.arctan2(-app_east, -app_north))
    awa = wrap_angle_deg(app_from_deg - heading_deg)
    # Round away floating-point noise so that e.g. a pure following wind is 180, not -179.9999.
    awa = wrap_angle_deg(np.round(awa, 9))
    calm = aws < 1e-9
    return np.where(calm, 0.0, aws), np.where(calm, 0.0, awa)


@dataclass(frozen=True, eq=False)
class RotorCoefficients:
    """Non-dimensional rotor characteristics as a function of spin ratio.

    Values between table points are linearly interpolated. Outside the table
    range the end values are held constant (no extrapolation), which is why
    :func:`rotor_forces` reports the spin ratio so callers can check it.
    """

    spin_ratio: NDArray[np.float64]
    cl: NDArray[np.float64]
    cd: NDArray[np.float64]
    cp: NDArray[np.float64]
    source: str = "unknown"

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> RotorCoefficients:
        """Load a coefficient table with columns ``spin_ratio, cl, cd, cp``.

        Lines starting with ``#`` are comments. With ``path=None`` the bundled
        **illustrative placeholder** table is loaded.
        """
        if path is None:
            ref = resources.files("rotor_sail_performance") / "data" / DEFAULT_COEFFICIENTS_FILE
            with resources.as_file(ref) as p:
                table = pd.read_csv(p, comment="#")
            source = f"bundled placeholder ({DEFAULT_COEFFICIENTS_FILE}) - illustrative, not measured"
        else:
            table = pd.read_csv(path, comment="#")
            source = str(path)

        missing = {"spin_ratio", "cl", "cd", "cp"} - set(table.columns)
        if missing:
            raise ValueError(f"coefficient table is missing columns: {sorted(missing)}")
        table = table.sort_values("spin_ratio")
        if table["spin_ratio"].duplicated().any():
            raise ValueError("coefficient table has duplicate spin ratios")
        return cls(
            spin_ratio=table["spin_ratio"].to_numpy(float),
            cl=table["cl"].to_numpy(float),
            cd=table["cd"].to_numpy(float),
            cp=table["cp"].to_numpy(float),
            source=source,
        )

    @property
    def max_spin_ratio(self) -> float:
        """Largest spin ratio covered by the table."""
        return float(self.spin_ratio[-1])

    def lift_coefficient(self, spin_ratio: ArrayLike) -> NDArray[np.float64]:
        """CL(SR) by linear interpolation."""
        return np.interp(spin_ratio, self.spin_ratio, self.cl)

    def drag_coefficient(self, spin_ratio: ArrayLike) -> NDArray[np.float64]:
        """CD(SR) by linear interpolation."""
        return np.interp(spin_ratio, self.spin_ratio, self.cd)

    def power_coefficient(self, spin_ratio: ArrayLike) -> NDArray[np.float64]:
        """Electrical power coefficient Cp(SR) by linear interpolation."""
        return np.interp(spin_ratio, self.spin_ratio, self.cp)

    def scaled(
        self, cl_factor: float = 1.0, cd_factor: float = 1.0, cp_factor: float = 1.0
    ) -> RotorCoefficients:
        """Return a copy with every coefficient multiplied by a constant factor.

        Used for sensitivity studies and Monte Carlo uncertainty propagation.
        """
        return RotorCoefficients(
            spin_ratio=self.spin_ratio.copy(),
            cl=self.cl * cl_factor,
            cd=self.cd * cd_factor,
            cp=self.cp * cp_factor,
            source=f"{self.source} (scaled CL x{cl_factor:.3f}, CD x{cd_factor:.3f}, Cp x{cp_factor:.3f})",
        )


@dataclass(frozen=True)
class RotorConfig:
    """Geometry and control settings of the rotor installation.

    The defaults describe a generic 30 m x 5 m rotor, a size class seen on
    merchant ships; they are not a specific product.
    """

    height_m: float = 30.0
    diameter_m: float = 5.0
    n_rotors: int = 2
    target_spin_ratio: float = 3.0
    max_rpm: float = 180.0

    @property
    def area_m2(self) -> float:
        """Reference (projected) area A = height * diameter [m^2]."""
        return self.height_m * self.diameter_m

    @property
    def radius_m(self) -> float:
        """Rotor radius R [m]."""
        return 0.5 * self.diameter_m

    @property
    def max_omega_rad_s(self) -> float:
        """Maximum rotational speed [rad/s]."""
        return self.max_rpm * 2.0 * np.pi / 60.0


@dataclass(frozen=True)
class RotorForces:
    """Forces and power for the whole installation (all N rotors summed).

    ``spin_ratio`` and ``rpm`` are per rotor (all rotors run identically).
    """

    spin_ratio: NDArray[np.float64]
    rpm: NDArray[np.float64]
    lift_n: NDArray[np.float64]
    drag_n: NDArray[np.float64]
    thrust_n: NDArray[np.float64]
    side_force_n: NDArray[np.float64]
    power_w: NDArray[np.float64]


def controlled_spin_ratio(apparent_wind_speed: ArrayLike, rotor: RotorConfig) -> NDArray[np.float64]:
    """Spin ratio reached by a simple controller.

    The controller aims for ``rotor.target_spin_ratio`` but cannot exceed the
    maximum rotational speed::

        omega = min(SR_target * V_aw / R, omega_max)
        SR    = omega * R / V_aw

    At zero apparent wind the rotor is idle and SR is reported as 0.
    """
    aws = np.asarray(apparent_wind_speed, dtype=float)
    omega = np.minimum(rotor.target_spin_ratio * aws / rotor.radius_m, rotor.max_omega_rad_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(aws > 0.0, omega * rotor.radius_m / aws, 0.0)
    return sr


def rotor_forces(
    apparent_wind_speed: ArrayLike,
    apparent_wind_angle_deg: ArrayLike,
    rotor: RotorConfig,
    coefficients: RotorCoefficients,
    rho_air: float = RHO_AIR,
    spinning: ArrayLike = True,
) -> RotorForces:
    """Magnus forces on the rotors and their projection onto the ship axes.

    Per rotor, with q = 0.5 * rho_air * V_aw^2 and A = height * diameter::

        SR = omega * R / V_aw
        L  = q * A * CL(SR)          (perpendicular to the apparent wind)
        D  = q * A * CD(SR)          (along the apparent wind)
        P  = q * A * V_aw * Cp(SR)   (electrical power to spin the rotor)

    With beta = |AWA| and lift oriented to have a forward component::

        thrust     =  L * sin(beta) - D * cos(beta)
        side force = -sign(AWA) * (L * cos(beta) + D * sin(beta))

    so a beam wind (beta = 90 deg) gives thrust = L, a head wind gives
    thrust = -D and a following wind gives thrust = +D. Forward of the beam
    the side force pushes the ship to leeward; well abaft the beam the
    forward-tilted lift can make it point to windward. The installation
    total is N times the single-rotor value: rotor-rotor interaction
    (wake shadowing, mutual induction) is **ignored**.

    Args:
        apparent_wind_speed: V_aw [m/s].
        apparent_wind_angle_deg: AWA [deg], positive from starboard.
        rotor: Rotor geometry and control settings.
        coefficients: CL/CD/Cp table.
        rho_air: Air density [kg/m^3].
        spinning: ``False`` (or a boolean array) for rotors that are stopped.
            A stopped rotor has SR = 0: it draws no power but still has drag.

    Returns:
        :class:`RotorForces` summed over ``rotor.n_rotors``.
    """
    aws = np.asarray(apparent_wind_speed, dtype=float)
    awa = np.asarray(apparent_wind_angle_deg, dtype=float)
    aws, awa, spin_on = np.broadcast_arrays(aws, awa, np.asarray(spinning, dtype=bool))

    sr = np.where(spin_on, controlled_spin_ratio(aws, rotor), 0.0)
    omega = np.where(aws > 0.0, sr * aws / rotor.radius_m, 0.0)
    rpm = omega * 60.0 / (2.0 * np.pi)

    q_area = 0.5 * rho_air * aws**2 * rotor.area_m2
    lift = q_area * coefficients.lift_coefficient(sr)
    drag = q_area * coefficients.drag_coefficient(sr)
    power = q_area * aws * coefficients.power_coefficient(sr)

    beta = np.radians(np.abs(awa))
    thrust = lift * np.sin(beta) - drag * np.cos(beta)
    side = -np.sign(awa) * (lift * np.cos(beta) + drag * np.sin(beta))

    n = rotor.n_rotors
    return RotorForces(
        spin_ratio=sr,
        rpm=rpm,
        lift_n=n * lift,
        drag_n=n * drag,
        thrust_n=n * thrust,
        side_force_n=n * side,
        power_w=n * power,
    )
