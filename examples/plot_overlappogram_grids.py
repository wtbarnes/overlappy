"""
WCS Grid Examples
=================

A bunch of examples of different kinds of overlappogram WCS that can be created.
This is mainly a convenient way to visualize how these different WCS behave.
"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import ndcube
from sunpy.coordinates import get_earth
from sunpy.visualization.wcsaxes_compat import wcsaxes_heliographic_overlay
from overlappy.wcs import overlappogram_fits_wcs, pcij_matrix
from overlappy.util import strided_array, color_lat_lon_axes

#################################################################################
# First, set up some of the base parameters for our WCS and the data array
#
shape = (50, 50)
spectral_platescale = 10 * u.Angstrom / u.pix
spatial_platescale = [150, 150] * u.arcsec / u.pix
wavelength = np.arange(0, 150, spectral_platescale.value) * spectral_platescale.unit * u.pix
data = np.zeros(shape)
data = strided_array(data, wavelength.shape[0])
observer = get_earth('2020-01-01')


def make_grid_plot(alpha, gamma, mu):
    pcij = pcij_matrix(alpha, gamma, mu)
    wcs = overlappogram_fits_wcs(shape,
                                 wavelength,
                                 (spatial_platescale[0], spatial_platescale[1], spectral_platescale),
                                 pc_matrix=pcij,
                                 observer=observer)
    cube = ndcube.NDCube(data, wcs=wcs)
    fig = plt.figure(figsize=(5, 5*wavelength.shape[0]))
    for i in range(wavelength.shape[0]):
        ax = fig.add_subplot(wavelength.shape[0], 1, i+1, projection=cube[i].wcs)
        cube[i].plot(axes=ax)
        ax.set_title(wavelength[i])
        color_lat_lon_axes(ax)
        wcsaxes_heliographic_overlay(ax, grid_spacing=20*u.deg, annotate=False)
    plt.show()


# %%
# Example 1: :math:`\alpha=0,\gamma=0,\mu=0`
# --------------------------------------------
#
make_grid_plot(0*u.deg, 0*u.deg, 0)

# %%
# Example 2: :math:`\alpha=0,\gamma=0,\mu=1`
# --------------------------------------------
#
make_grid_plot(0*u.deg, 0*u.deg, 1)

# %%
# Example 3: :math:`\alpha=90,\gamma=0,\mu=1`
# --------------------------------------------
#
make_grid_plot(90*u.deg, 0*u.deg, 1)

# %%
# Example 4: :math:`\alpha=90,\gamma=0,\mu=3`
# --------------------------------------------
#
make_grid_plot(90*u.deg, 0*u.deg, 3)

# %%
# Example 5: :math:`\alpha=90,\gamma=45,\mu=1`
# --------------------------------------------
#
make_grid_plot(90*u.deg, 45*u.deg, 1)

# %%
# Example 6: :math:`\alpha=90,\gamma=45,\mu=-1`
# ---------------------------------------------
#
make_grid_plot(90*u.deg, 45*u.deg, -1)

# %%
# Example 6: :math:`\alpha=30,\gamma=30,\mu=2`
# ------------------------------------------
#
make_grid_plot(30*u.deg, 30*u.deg, 2)

# %%
# Example 7: :math:`\alpha=0,\gamma=-90,\mu=1`
# ------------------------------------------
#
make_grid_plot(0*u.deg, -90*u.deg, 1)
