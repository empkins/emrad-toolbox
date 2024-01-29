import pytest
import numpy as np
from emrad_toolbox.radar_preprocessing.radar import RadarPreprocessor


@pytest.fixture
def radar_data_small():
    return {
        'i': np.array([1, 2, 3, 4, 5]),
        'q': np.array([1, 2, 3, 4, 5])
    }

@pytest.fixture
def radar_data_small_power():
    return np.array([2, 8, 18, 32, 50], dtype='float64')

@pytest.fixture
def radar_data_large_numbers():
    return {
        'i': np.array([1e9, 2e9, 3e7, 4e6, 5e6]).astype('int32'),
        'q': np.array([1e8, 2e9, 3e7, 4e6, 5e6]).astype('int32')
    }


def test_calculate_power_overflow(radar_data_small, radar_data_large_numbers):
    large_numbers_power = RadarPreprocessor.calculate_power(radar_data_large_numbers['i'], radar_data_large_numbers['q'])
    assert all(x >= 0 for x in large_numbers_power)


def test_calculate_power(radar_data_small, radar_data_small_power):
    small_numbers_power = RadarPreprocessor.calculate_power(radar_data_small['i'], radar_data_small['q'])
    assert all(x >= 0 for x in small_numbers_power)
    small_numbers_power_squared = np.square(small_numbers_power)
    assert np.allclose(small_numbers_power_squared, radar_data_small_power)

