"""
WCS Grid Examples
=================

A bunch of examples of different kinds of overlappogram WCS that can be created.
This is mainly a convenient way to visualize how these different WCS behave.
"""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from mpl_animators import ArrayAnimatorWCS
from sunpy.coordinates import get_earth
from sunpy.visualization import drawing
from sunpy.visualization.wcsaxes_compat import wcsaxes_heliographic_overlay
from overlappy.wcs import overlappogram_fits_wcs, pcij_matrix
from overlappy.util import strided_array, color_lat_lon_axes

#################################################################################
# First, set up some of the base parameters for our WCS and the data array
#
shape = (10, 10)
spectral_platescale = 15 * u.Angstrom / u.pix
spatial_platescale = [500, 500] * u.arcsec / u.pix
wavelength = np.arange(0, 150, spectral_platescale.value) * spectral_platescale.unit * u.pix
data = np.zeros(shape)
data = strided_array(data, wavelength.shape[0])
observer = get_earth('2020-01-01')


class CustomArrayAnimatorWCS(ArrayAnimatorWCS):
    def update_plot_2d(self, val, im, slider):
        """
        Update the image plot.
        """
        if len(self.axes.patches):
            self.axes.patches[0].remove()
        super().update_plot_2d(val, im, slider)
        color_lat_lon_axes(self.axes, alpha=0)
        drawing.limb(self.axes, observer, color='w')

    def _compute_slider_labels_from_wcs(self, slices):
        labels = super()._compute_slider_labels_from_wcs(slices)
        labels[-1] = 'Wavelength'
        return labels


def make_grid_plot(alpha, gamma, mu):
    pcij = pcij_matrix(alpha, gamma, mu)
    print('PC_ij = ', pcij)
    wcs = overlappogram_fits_wcs(wavelength.shape+shape,
                                 (spatial_platescale[0], spatial_platescale[1], spectral_platescale),
                                 pc_matrix=pcij,
                                 observer=observer)
    coord_params = {
        "hpln": {"axislabel": "Helioprojective Longitude"},
        "hplt": {"axislabel": "Helioprojective Latitude"},
    }
    animator = CustomArrayAnimatorWCS(data, wcs, ["x", "y", 0], coord_params=coord_params)
    animation = animator.get_animation()
    return animation


# %%
# Example 1: :math:`\alpha=0,\gamma=0,\mu=0`
# --------------------------------------------
#
ani = make_grid_plot(0*u.deg, 0*u.deg, 0)

# %%
# Example 2: :math:`\alpha=0,\gamma=0,\mu=1`
# --------------------------------------------
#
ani = make_grid_plot(0*u.deg, 0*u.deg, 1)

# %%
# Example 3: :math:`\alpha=90,\gamma=0,\mu=1`
# --------------------------------------------
#
ani = make_grid_plot(90*u.deg, 0*u.deg, 1)

# %%
# Example 4: :math:`\alpha=90,\gamma=0,\mu=3`
# --------------------------------------------
#
ani = make_grid_plot(90*u.deg, 0*u.deg, 3)

# %%
# Example 5: :math:`\alpha=90,\gamma=45,\mu=1`
# --------------------------------------------
#
ani = make_grid_plot(90*u.deg, 45*u.deg, 1)

# %%
# Example 6: :math:`\alpha=90,\gamma=45,\mu=-1`
# ---------------------------------------------
#
ani = make_grid_plot(90*u.deg, 45*u.deg, -1)

# %%
# Example 6: :math:`\alpha=30,\gamma=30,\mu=2`
# --------------------------------------------
#
ani = make_grid_plot(30*u.deg, 30*u.deg, 2)

# %%
# Example 7: :math:`\alpha=0,\gamma=-90,\mu=1`
# --------------------------------------------
#
ani = make_grid_plot(0*u.deg, -90*u.deg, 1)

plt.show()