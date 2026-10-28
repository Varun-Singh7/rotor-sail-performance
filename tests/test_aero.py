import numpy as np
import pytest

from rotor_sail_performance.aero import (
    RotorCoefficients,
    apparent_wind,
    controlled_spin_ratio,
    rotor_forces,
    wrap_angle_deg,
)


class TestWindTriangle:
    def test_head_wind_adds_speeds(self):
        # Ship heading north into a northerly wind.
        aws, awa = apparent_wind(10.0, 0.0, 6.0, 0.0)
        assert aws == pytest.approx(16.0)
        assert awa == pytest.approx(0.0)

    def test_following_wind_stronger_than_ship(self):
        # Ship heading north, wind from the south.
        aws, awa = apparent_wind(10.0, 180.0, 6.0, 0.0)
        assert aws == pytest.approx(4.0)
        assert abs(awa) == pytest.approx(180.0)

    def test_following_wind_weaker_than_ship_feels_like_head_wind(self):
        aws, awa = apparent_wind(4.0, 180.0, 6.0, 0.0)
        assert aws == pytest.approx(2.0)
        assert awa == pytest.approx(0.0)

    def test_following_wind_equal_to_ship_speed_is_calm(self):
        aws, awa = apparent_wind(6.0, 180.0, 6.0, 0.0)
        assert aws == pytest.approx(0.0, abs=1e-12)
        assert awa == 0.0

    @pytest.mark.parametrize("heading", [0.0, 37.0, 250.0])
    def test_beam_wind_from_starboard(self, heading):
        aws, awa = apparent_wind(8.0, heading + 90.0, 6.0, heading)
        assert aws == pytest.approx(10.0)
        # Apparent wind moves forward of the beam: atan(8/6) from the bow.
        assert awa == pytest.approx(np.degrees(np.arctan2(8.0, 6.0)))

    def test_beam_wind_from_port_is_negative(self):
        _, awa = apparent_wind(8.0, 270.0, 6.0, 0.0)
        assert awa == pytest.approx(-np.degrees(np.arctan2(8.0, 6.0)))

    def test_stationary_ship_sees_true_wind(self):
        aws, awa = apparent_wind(7.0, 135.0, 0.0, 90.0)
        assert aws == pytest.approx(7.0)
        assert awa == pytest.approx(45.0)

    def test_zero_true_wind_gives_head_wind_equal_to_ship_speed(self):
        aws, awa = apparent_wind(0.0, 0.0, 6.0, 123.0)
        assert aws == pytest.approx(6.0)
        assert awa == pytest.approx(0.0)

    def test_vectorised(self):
        aws, awa = apparent_wind([10.0, 10.0], [0.0, 180.0], [6.0, 6.0], [0.0, 0.0])
        np.testing.assert_allclose(aws, [16.0, 4.0])
        np.testing.assert_allclose(np.abs(awa), [0.0, 180.0])


def test_wrap_angle_range():
    np.testing.assert_allclose(wrap_angle_deg([-180.0, 180.0, 190.0, -190.0, 540.0, 0.0]),
                               [180.0, 180.0, -170.0, 170.0, 180.0, 0.0])


class TestRotorForces:
    def test_zero_apparent_wind_gives_zero_forces(self, rotor, coeffs):
        f = rotor_forces(0.0, 90.0, rotor, coeffs)
        assert f.thrust_n == 0.0
        assert f.side_force_n == 0.0
        assert f.power_w == 0.0

    def test_zero_wind_stationary_ship_gives_zero_thrust(self, rotor, coeffs):
        aws, awa = apparent_wind(0.0, 0.0, 0.0, 45.0)
        assert rotor_forces(aws, awa, rotor, coeffs).thrust_n == 0.0

    def test_beam_wind_thrust_equals_lift(self, rotor, coeffs):
        f = rotor_forces(10.0, 90.0, rotor, coeffs)
        assert f.thrust_n == pytest.approx(f.lift_n)
        expected_lift = rotor.n_rotors * 0.5 * 1.225 * rotor.area_m2 * coeffs.lift_coefficient(3.0) * 10.0**2
        assert f.lift_n == pytest.approx(expected_lift)

    def test_head_wind_thrust_is_minus_drag(self, rotor, coeffs):
        f = rotor_forces(10.0, 0.0, rotor, coeffs)
        assert f.thrust_n == pytest.approx(-f.drag_n)
        assert f.thrust_n < 0

    def test_following_wind_thrust_is_drag(self, rotor, coeffs):
        f = rotor_forces(10.0, 180.0, rotor, coeffs)
        assert f.thrust_n == pytest.approx(f.drag_n)

    @pytest.mark.parametrize("awa", [30.0, 60.0, 90.0, 120.0, 150.0])
    def test_thrust_forward_on_both_sides_and_side_force_mirrored(self, rotor, coeffs, awa):
        stbd = rotor_forces(12.0, awa, rotor, coeffs)
        port = rotor_forces(12.0, -awa, rotor, coeffs)
        assert stbd.thrust_n > 0
        assert port.thrust_n == pytest.approx(stbd.thrust_n)
        assert port.side_force_n == pytest.approx(-stbd.side_force_n)
        if awa <= 90.0:
            # Wind from forward of the beam on starboard pushes the ship to port.
            assert stbd.side_force_n < 0
        else:
            # Abaft the beam the forward-tilted lift has a windward component.
            beta = np.radians(awa)
            assert stbd.side_force_n == pytest.approx(-(stbd.lift_n * np.cos(beta) + stbd.drag_n * np.sin(beta)))

    def test_stopped_rotor_has_drag_but_no_lift_or_power(self, rotor, coeffs):
        f = rotor_forces(10.0, 90.0, rotor, coeffs, spinning=False)
        assert f.lift_n == 0.0
        assert f.power_w == 0.0
        assert f.side_force_n < 0

    def test_forces_scale_with_number_of_rotors(self, coeffs):
        from rotor_sail_performance import RotorConfig

        one = rotor_forces(10.0, 70.0, RotorConfig(n_rotors=1), coeffs)
        four = rotor_forces(10.0, 70.0, RotorConfig(n_rotors=4), coeffs)
        assert four.thrust_n == pytest.approx(4 * one.thrust_n)
        assert four.power_w == pytest.approx(4 * one.power_w)
        assert four.spin_ratio == pytest.approx(one.spin_ratio)


def test_spin_ratio_capped_by_max_rpm(rotor):
    low = controlled_spin_ratio(5.0, rotor)
    high = controlled_spin_ratio(40.0, rotor)
    assert low == pytest.approx(rotor.target_spin_ratio)
    assert high == pytest.approx(rotor.max_omega_rad_s * rotor.radius_m / 40.0)
    assert high < rotor.target_spin_ratio


def test_coefficient_table_loads_and_interpolates(coeffs, tmp_path):
    assert "illustrative" in coeffs.source
    assert coeffs.lift_coefficient(0.0) == 0.0
    mid = coeffs.lift_coefficient(2.25)
    assert coeffs.lift_coefficient(2.0) < mid < coeffs.lift_coefficient(2.5)
    # Clamped, not extrapolated, beyond the table.
    assert coeffs.lift_coefficient(99.0) == coeffs.cl[-1]

    path = tmp_path / "custom.csv"
    path.write_text("# test\nspin_ratio,cl,cd,cp\n0,0,0.5,0\n4,8,1,1\n")
    custom = RotorCoefficients.from_csv(path)
    assert custom.lift_coefficient(2.0) == pytest.approx(4.0)


def test_coefficient_table_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("spin_ratio,cl\n0,0\n")
    with pytest.raises(ValueError, match="missing"):
        RotorCoefficients.from_csv(path)
