"""
Tools for reprojecting spectral cubes to overlappograms
"""
import numpy as np
import astropy.units as u
import ndcube
import reproject

from .wcs import pcij_matrix, overlappogram_fits_wcs
from .util import strided_array


def reproject_to_overlappogram(cube,
                               detector_shape,
                               reference_pixel=None,
                               scale=None,
                               roll_angle=0*u.deg,
                               dispersion_angle=0*u.deg,
                               order=1,
                               observer=None,
                               sum_over_lambda=True,
                               reproject_kwargs=None):
    """
    Reproject a spectral cube to an overlappogram.

    Parameters
    ----------
    cube : `ndcube.NDCube`
        Spectral cube with dimensions (wave, lat, lon)
    detector_shape : `tuple`
        Shape of the 2D detector to project onto
    reference_pixel : `~astropy.units.Quantity`
        The pixel that corresponds to (0,0,wave[0]). This will default
        to the middle of the detector.
    scale : `tuple`
        The scale of the the spatial and spectral axes
    roll_angle : `~astropy.units.Quantity`
        Angle between y-like pixel axis and y-like world axis.
    dispersion_angle : `~astropy.units.Quantity`
        Angle between dispersion direction and the y-like pixel axis.
    order : `int`
        Spectral order to model
    observer : `~astropy.coordinates.SkyCoord`
        Location of observer in HGS coordinates
    sum_over_lambda : `bool`
        If True, sum over all layers in wavelength to create overlapped
        images. If False, each layer will be the reprojection of the FOV
        at that wavelength and everything else will be NaN.

    Returns
    --------
    `ndcube.NDCube`
    """
    wavelength = cube.axis_world_coords(0)[0].to('angstrom')
    pc_matrix = pcij_matrix(roll_angle, dispersion_angle, order=order)
    if scale is None:
        scale = [u.Quantity(cd, f'{cu} / pix') for cd,cu in
                 zip(cube.wcs.wcs.cdelt, cube.wcs.wcs.cunit)]
    overlap_wcs = overlappogram_fits_wcs(
        detector_shape,
        wavelength,
        scale,
        reference_pixel=reference_pixel,
        pc_matrix=pc_matrix,
        observer=observer,
    )

    reproject_kwargs = {} if reproject_kwargs is None else reproject_kwargs
    overlap_data = reproject.reproject_interp(
        cube,
        overlap_wcs,
        shape_out=wavelength.shape + detector_shape,
        return_footprint=False,
        **reproject_kwargs,
    )

    if sum_over_lambda:
        isnan = np.where(np.isnan(overlap_data))
        overlap_data[isnan] = 0.0
        overlap_data = overlap_data.sum(axis=0)
        overlap_data = strided_array(overlap_data, wavelength.shape[0])
    
    return ndcube.NDCube(overlap_data, wcs=overlap_wcs, unit=cube.unit)