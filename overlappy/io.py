"""
Utility functions for writing overlappogram data I/O
"""
import astropy.wcs
import sunpy.io._fits as sunpy_fits
import ndcube

from .util import strided_array


def write_overlappogram(cube, filename):
    header = cube.wcs.to_header()
    # Encode wave dimension in extra keyword as NAXIS3 will not be included
    # in the header because our array is 2D
    # Note that we only write one slice of the overlappogram as each
    # wavelength contains the same data array.
    header['NWAVE'] = cube.data.shape[0]
    header['BUNIT'] = cube.unit.to_string()
    sunpy_fits.write(
        filename,
        cube.data[0],
        header,
    )


def read_overlappogram(filename, flat=True):
    pair = sunpy_fits.read(filename)
    data, header = pair[0]
    if flat:
        header['NAXIS'] = 3
        header['NAXIS3'] = header['NWAVE']
        data = strided_array(data, header['NWAVE'])
    header.pop('KEYCOMMENTS', None)
    wcs = astropy.wcs.WCS(header=header)
    unit = header.get('BUNIT', None)

    return ndcube.NDCube(data, wcs=wcs, meta=header, unit=unit)
