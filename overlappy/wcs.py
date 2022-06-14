"""
Functions for building overlappogram WCSs
"""
import numpy as np
import astropy.units as u
import astropy.wcs

from .util import pcij_to_keys, hgs_observer_to_keys


@u.quantity_input
def rotation_matrix(angle: u.deg):
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


@u.quantity_input
def dispersion_matrix(order, dispersion_axis=0):
    dispersion_array = np.eye(3)
    dispersion_array[dispersion_axis, 2] = -order
    return dispersion_array


@u.quantity_input
def pcij_matrix(roll_angle: u.deg,
                dispersion_angle: u.deg,
                order=1,
                dispersion_axis=0,
                align_p2_wave=False):
    """
    Parameters
    ----------
    roll_angle: 
        Angle between the second pixel axis and the y-like
        world axis.
    dispersion_angle:
        Angle between the wavelength (dispersion) axis and the second pixel axis.
    order:
        Order of the dispersion. Default is 1.
    """
    R_2 = rotation_matrix(roll_angle - dispersion_angle)
    D = dispersion_matrix(order, dispersion_axis=dispersion_axis)
    if align_p2_wave:
        # This aligns the dispersion axis with the wavelength axis
        # and decorrelates wavelength with the the third "fake"
        # pixel axis. This means that wavelength *does not* vary
        # as you increment that third axis.
        # This is not strictly correct as the world grid at each p3
        # should only correspond to a particular wavelength, not
        # all of them.
        # This may be useful when plotting two slices of the overlappogram and displaying a wavelength axis.
        # However, this will not work when performing reprojections.
        # If you want this capability, you should apply this PCij to your WCS after
        # the fact.
        D[2, dispersion_axis] = 1
        D[2, 2] = 0
    R_1 = rotation_matrix(dispersion_angle)
    # The operation here is (from R to L):
    # -- align dispersion axis with p2 pixel axis
    # -- disperse in spectral order along p2 axis
    # -- apply roll angle rotation
    return R_2 @ D @ R_1


@u.quantity_input
def overlappogram_fits_wcs(detector_shape,
                           wavelength: u.m,
                           scale,
                           reference_pixel: u.pix = None,
                           reference_coord=None,
                           pc_matrix=None,
                           observer=None):
    # NOTE: We assume that the reference coord is (0,0,wave[0]) such that
    # for the default reference pixel, the zeroth order images falls in the
    # middle of the array and the dispersed images "start" from the middle
    # of the detector such that the +/- orders are symmetric about the middle
    # of the detector.
    if reference_pixel is None:
        reference_pixel = (
            (detector_shape[1] + 1) / 2,
            (detector_shape[0] + 1) / 2,
            1,
        ) * u.pix
    # FIXME: I don't think this is strictly correct to allow. Really, we should always
    # be setting crval[1,2] to (0,0) (the center of the coordinate frame) and then 
    # calculating the reference pixel appropriately.
    if reference_coord is None:
        reference_coord = (
            0 * u.arcsec,
            0 * u.arcsec,
            wavelength[0],
        )
    wcs_keys = {
        'WCSAXES': 3,
        'NAXIS1': detector_shape[1],
        'NAXIS2': detector_shape[0],
        'NAXIS3': wavelength.shape[0],
        'CDELT1': scale[0].to('arcsec / pix').value,
        'CDELT2': scale[1].to('arcsec / pix').value,
        'CDELT3': scale[2].to('Angstrom / pix').value,
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CUNIT3': 'Angstrom',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CTYPE3': 'WAVE',
        'CRPIX1': reference_pixel[0].value,
        'CRPIX2': reference_pixel[1].value,
        'CRPIX3': reference_pixel[2].value,
        'CRVAL1': reference_coord[0].to('arcsec').value,
        'CRVAL2': reference_coord[1].to('arcsec').value,
        'CRVAL3': reference_coord[2].to('angstrom').value,
    }
    wcs_keys = {**wcs_keys, **pcij_to_keys(pc_matrix)}
    if observer is not None:
        wcs_keys = {**wcs_keys, **hgs_observer_to_keys(observer)}
    wcs = astropy.wcs.WCS(wcs_keys)
    return wcs
