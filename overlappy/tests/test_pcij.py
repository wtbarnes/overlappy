"""
Tests for PCij matrix for different combinations of roll and grating
"""
import pytest
import numpy as np
import astropy.units as u

from overlappy.wcs import pcij_matrix


@pytest.mark.parametrize('roll_angle,grating_angle,order,expected_matrix', [
    (0*u.deg, 0*u.deg, 0, np.eye(3)),
    (0*u.deg, 0*u.deg, 1, [[1, 0, -1], [0, 1, 0], [0, 0, 1]]),
    (0*u.deg, 0*u.deg, 2, [[1, 0, -2], [0, 1, 0], [0, 0, 1]]),
    (0*u.deg, 0*u.deg, -1, [[1, 0, 1], [0, 1, 0], [0, 0, 1]]),
    (90*u.deg, 0*u.deg, 0, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    (90*u.deg, 0*u.deg, 1, [[0, -1, 0], [1, 0, -1], [0, 0, 1]]),
    (0*u.deg, 45*u.deg, 0, np.eye(3)),
    (0*u.deg, 90*u.deg, 1, [[1, 0, 0], [0, 1, 1], [0, 0, 1]]),
    (30*u.deg, 30*u.deg, 1, [[np.sqrt(3)/2, -1/2, -1], [1/2, np.sqrt(3)/2, 0], [0, 0, 1]]),
    (60*u.deg, 30*u.deg, 3, [[1/2, -np.sqrt(3)/2, -3*np.sqrt(3)/2], [np.sqrt(3)/2, 1/2, -3/2], [0, 0, 1]])
])
def test_pcij_simple(roll_angle, grating_angle, order, expected_matrix):
    pcij = pcij_matrix(roll_angle=roll_angle,
                       dispersion_angle=grating_angle,
                       order=order)
    assert np.allclose(pcij, np.array(expected_matrix))
