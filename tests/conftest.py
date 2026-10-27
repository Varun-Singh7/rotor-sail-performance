import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from rotor_sail_performance import RotorCoefficients, RotorConfig, ShipConfig  # noqa: E402


@pytest.fixture(scope="session")
def coeffs() -> RotorCoefficients:
    return RotorCoefficients.from_csv()


@pytest.fixture
def rotor() -> RotorConfig:
    return RotorConfig()


@pytest.fixture
def ship() -> ShipConfig:
    return ShipConfig()
