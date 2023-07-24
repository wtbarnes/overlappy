"""
Tools for reprojecting spectral cubes to overlappograms
"""
import numpy as np
import astropy.units as u
import ndcube
import reproject

from .wcs import pcij_matrix, overlappogram_fits_wcs
from .util import strided_array

__all__ = [
    "reproject_to_overlappogram",
]


def reproject_to_overlappogram(cube,
                               detector_shape,
                               reference_pixel=None,
                               reference_coord=None,
                               scale=None,
                               roll_angle=0*u.deg,
                               dispersion_angle=0*u.deg,
                               order=1,
                               observer=None,
                               sum_over_lambda=True,
                               algorithm='interpolation',
                               reproject_kwargs=None,
                               meta_keys=None,
                               use_dask=False):
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
    meta_keys : `list`
        Keys from spectral cube metadata to copy into overlappogram metadata.
    use_dask : `bool`
        If True, parallelize the reprojection with Dask.
        Requires first starting a Dask client.

    Returns
    --------
    `ndcube.NDCube`
    """
    wavelength = cube.axis_world_coords(0)[0].to('angstrom')
    pc_matrix = pcij_matrix(roll_angle, dispersion_angle,
                            order=order)
    if scale is None:
        scale = [u.Quantity(cd, f'{cu} / pix') for cd, cu in
                 zip(cube.wcs.wcs.cdelt, cube.wcs.wcs.cunit)]
    overlap_wcs = overlappogram_fits_wcs(
        detector_shape,
        wavelength,
        scale,
        reference_pixel=reference_pixel,
        reference_coord=reference_coord,
        pc_matrix=pc_matrix,
        observer=observer,
    )

    functions = {
        'interpolation': reproject.reproject_interp,
        'adaptive': reproject.reproject_adaptive,
        'exact': reproject.reproject_exact
    }
    reproject_kwargs = {} if reproject_kwargs is None else reproject_kwargs

    if use_dask:
        import distributed
        import dask
        import dask.array
        client = distributed.get_client()
        # Lay out per slice reproject function

        @dask.delayed
        def _reproject_slice(cube_slice, wcs_slice):
            return functions[algorithm](
                cube_slice,
                wcs_slice,
                shape_out=wcs_slice.array_shape,
                return_footprint=False,
                **reproject_kwargs,
            ).squeeze()

        # Build WCS and data slices
        indices = list(range(cube.data.shape[0]))
        cube_slices = client.scatter([cube[i:i+1] for i in indices])
        wcs_slices = [overlap_wcs[i:i+1] for i in indices]
        # Map reproject to slice
        delayed_slices = [_reproject_slice(cs, ws)
                          for cs, ws in zip(cube_slices, wcs_slices)]
        # Stack resulting arrays
        overlap_data = dask.array.stack([
            dask.array.from_delayed(f, detector_shape, dtype=cube.data.dtype)
            for f in delayed_slices
        ])
    else:
        shape_out = wavelength.shape + detector_shape
        overlap_data = functions[algorithm](
            cube,
            overlap_wcs,
            shape_out=shape_out,
            return_footprint=False,
            **reproject_kwargs,
        )

    if sum_over_lambda:
        overlap_data = np.where(np.isnan(overlap_data), 0.0, overlap_data).sum(axis=0)
        if use_dask:
            overlap_data = overlap_data.compute()
        overlap_data = strided_array(overlap_data, wavelength.shape[0])

    meta = {}
    if meta_keys is not None:
        for k in meta_keys:
            meta[k] = cube.meta.get(k)

    return ndcube.NDCube(overlap_data, wcs=overlap_wcs, unit=cube.unit, meta=meta)
