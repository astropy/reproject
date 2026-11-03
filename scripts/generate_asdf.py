from collections.abc import Iterable

import astropy.modeling.models as m
import astropy.units as u
import gwcs.coordinate_frames as cf
from astropy.modeling import CompoundModel, Model
from gwcs import WCS
from numpy.typing import ArrayLike


def generate_celestial_transform(
    crpix: Iterable[float] | u.Quantity,
    cdelt: Iterable[float] | u.Quantity,
    pc: ArrayLike | u.Quantity,
    crval: Iterable[float] | u.Quantity,
    lon_pole: float | u.Quantity = None,
    projection: Model = None,
) -> CompoundModel:
    """
    Create a simple celestial transform from FITS like parameters.

    Supports unitful or unitless parameters, but if any parameters have units
    all must have units, if parameters are unitless they are assumed to be in
    degrees.

    Parameters
    ----------
    crpix
        The reference pixel (a length two array).
    crval
        The world coordinate at the reference pixel (a length two array).
    cdelt
        The sample interval along the pixel axis
    pc
        The rotation matrix for the affine transform. If specifying parameters
        with units this should have celestial (``u.deg``) units.
    lon_pole
        The longitude of the celestial pole, defaults to 180 degrees.
    projection
        The map projection to use, defaults to ``TAN``.

    Notes
    -----

    This function has not been tested with more complex projections. Ensure
    that your lon_pole is correct for your projection.
    """
    projection = m.Pix2Sky_TAN() if projection is None else None
    spatial_unit = None
    if hasattr(crval[0], "unit"):
        spatial_unit = crval[0].unit
    # TODO: Note this assumption is only valid for certain projections.
    if lon_pole is None:
        lon_pole = 180
    if spatial_unit is not None:
        # Lon pole should always have the units of degrees
        lon_pole = u.Quantity(lon_pole, unit=u.deg)

    # Make translation unitful if all parameters have units
    translation = (0, 0)
    if spatial_unit is not None:
        translation *= pc.unit
        # If we have units then we need to convert all things to Quantity
        # as they might be Parameter classes
        crpix = u.Quantity(crpix)
        cdelt = u.Quantity(cdelt)
        crval = u.Quantity(crval)
        lon_pole = u.Quantity(lon_pole)
        pc = u.Quantity(pc)

    shift = m.Shift(-crpix[0]) & m.Shift(-crpix[1])
    scale = m.Multiply(cdelt[0]) & m.Multiply(cdelt[1])
    rot = m.AffineTransformation2D(pc, translation=translation)
    skyrot = m.RotateNative2Celestial(crval[0], crval[1], lon_pole)
    return shift | rot | scale | projection | skyrot


def generate_asdf(input_file="aia_171_level1.fits", output_file="aia_171_level1.asdf"):
    # Put imports for optional or not dependencies here
    import asdf
    import sunpy.map

    aia_map = sunpy.map.Map(input_file)

    transform = generate_celestial_transform(
        crpix=aia_map.reference_pixel,
        cdelt=aia_map.scale,
        pc=aia_map.rotation_matrix * u.pix,
        crval=aia_map.wcs.wcs.crval * u.deg,
    )

    input_frame = cf.Frame2D()
    output_frame = cf.CelestialFrame(
        reference_frame=aia_map.coordinate_frame,
        unit=(u.arcsec, u.arcsec),
        axes_names=("Helioprojective Longitude", "Helioprojective Latitude"),
    )

    aia_gwcs = WCS(
        forward_transform=transform,
        input_frame=input_frame,
        output_frame=output_frame,
    )

    tree = {
        "data": aia_map.data,
        "meta": dict(aia_map.meta),
        "wcs": aia_gwcs,
    }

    af = asdf.AsdfFile(tree)
    af.write_to(output_file)


if __name__ == "__main__":
    generate_asdf()
