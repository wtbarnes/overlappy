"""
Utility functions
"""
import numpy as np
import astropy.units as u
import astropy.constants

__all__ = [
    "draw_hgs_grid",
    "color_lat_lon_axes",
    "hgs_observer_to_keys",
    "pcij_to_keys",
    "strided_array",
]


def draw_hgs_grid(ax, observer):
    from sunpy.visualization.wcsaxes_compat import wcsaxes_heliographic_overlay
    hgs_grid = wcsaxes_heliographic_overlay(
        ax,
        obstime=observer.obstime,
        rsun=observer.rsun,
    )
    hgs_grid[0].grid(grid_type='contours')
    hgs_grid[1].grid(grid_type='contours')
    return hgs_grid


def color_lat_lon_axes(ax,
                       lon_color='C0',
                       lat_color='C3',
                       wvl_color='C4',
                       lat_tick_ops=None,
                       lon_tick_ops=None,
                       wvl_tick_ops=None,
                       alpha=1):
    lat_tick_ops = {} if lat_tick_ops is None else lat_tick_ops
    lon_tick_ops = {} if lon_tick_ops is None else lon_tick_ops
    lat_tick_ops['color'] = lat_color
    lon_tick_ops['color'] = lon_color
    lon = ax.coords[0]
    lat = ax.coords[1]
    # Ticks-lon
    lon.set_ticklabel_position('lb')
    lon.set_axislabel(ax.get_xlabel(), color=lon_color)
    lon.set_ticklabel(color=lon_color)
    lon.set_ticks(**lon_tick_ops)
    # Ticks-lat
    lat.set_ticklabel_position('lb')
    lat.set_axislabel(ax.get_ylabel(), color=lat_color)
    lat.set_ticklabel(color=lat_color)
    lat.set_ticks(**lat_tick_ops)
    # Grid
    lon.grid(color=lon_color, grid_type='contours', alpha=alpha)
    lat.grid(color=lat_color, grid_type='contours', alpha=alpha)
    # If wavelength axis is set, do some styling there too
    if len(ax.coords.get_coord_range()) > 2:
        wvl = ax.coords[2]
        wvl_tick_ops = {} if wvl_tick_ops is None else wvl_tick_ops
        wvl_tick_ops['color'] = wvl_color
        wvl.set_ticklabel_position('rt')
        wvl.set_format_unit(u.angstrom)
        wvl.set_major_formatter('x.x')
        wvl.set_ticklabel(color=wvl_color)
        wvl.set_ticks(color=wvl_color)
        wvl.set_axislabel('Wavelength [Angstrom]', color=wvl_color)
        wvl.grid(color=wvl_color, grid_type='contours', alpha=alpha)
        return lon, lat, wvl

    return lon, lat


def hgs_observer_to_keys(observer):
    return {
        'DATE-OBS': observer.obstime.isot,
        'HGLN_OBS': observer.lon.to('deg').value,
        'HGLT_OBS': observer.lat.to('deg').value,
        'DSUN_OBS': observer.radius.to('m').value,
        'RSUN_REF': astropy.constants.R_sun.to('m').value,
    }


def pcij_to_keys(pcij_matrix):
    pcij_keys = {}
    for i in range(pcij_matrix.shape[0]):
        for j in range(pcij_matrix.shape[1]):
            pcij_keys[f'PC{i+1}_{j+1}'] = pcij_matrix[i,j]
    return pcij_keys


def strided_array(array, N, **kwargs):
    """
    Return a "strided" version of the array.

    For an array of shape (N_2, N_1), this
    function creates an array of dimension (N, N_2, N_1)
    where each layer is a view of the original array.
    In other words, the values at (k,i,j) and (k+1,i,j)
    point to the same place in memory.
    """
    return np.lib.stride_tricks.as_strided(
        array,
        shape=(N,)+array.shape,
        strides=(0,)+array.strides,
        **kwargs,
    )
