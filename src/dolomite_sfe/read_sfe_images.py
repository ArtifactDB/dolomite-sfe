import json
import os

import rasterio
from aicsimageio import AICSImage
from dolomite_base.read_object import read_object_registry
from spatialfeatureexperiment.aligned_spatialimage import BioFormatsImage, ExtImage, SpatRasterImage

read_object_registry["geotiff"] = "dolomite_sfe.read_spat_raster_image"
read_object_registry["bioformats_image"] = "dolomite_sfe.read_bio_infomats_image"
read_object_registry["ext_image"] = "dolomite_sfe.read_ext_image"


def read_spat_raster_image(path: str, metadata: dict, **kwargs) -> SpatRasterImage:
    """Load a
    :py:class:`~spatialfeatureexperiment.aligned_spatialimage.SpatRasterImage`
    from its on-disk representation.

    This method should generally not be called directly but instead be invoked by
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path:
            Path to the directory containing the object.

        metadata:
            Metadata for the object.

        kwargs:
            Further arguments.

    Returns:
        A
        :py:class:`~spatialfeatureexperiment.aligned_spatialimage.SpatRasterImage`.
    """
    image_files = [f for f in os.listdir(path) if f.startswith("image.")]

    if metadata is None:
        with open(os.path.join(path, "OBJECT"), "r") as f:
            metadata = json.load(f)["geotiff"]

    file_path = os.path.join(path, image_files[0])

    img = rasterio.open(file_path, **kwargs)

    return SpatRasterImage(img, extent=metadata.get("extent", None))


def read_bio_infomats_image(path: str, metadata: dict, **kwargs) -> BioFormatsImage:
    """Load a
    :py:class:`~spatialfeatureexperiment.aligned_spatialimage.BioFormatsImage`
    from its on-disk representation.

    This method should generally not be called directly but instead be invoked by
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path:
            Path to the directory containing the object.

        metadata:
            Metadata for the object.

        kwargs:
            Further arguments.

    Returns:
        A
        :py:class:`~spatialfeatureexperiment.aligned_spatialimage.BioFormatsImage`.
    """
    if metadata is None:
        with open(os.path.join(path, "OBJECT"), "r") as f:
            metadata = json.load(f)["bioformats_image"]

    if os.path.isdir(os.path.join(path, "image")):
        image_dir = os.path.join(path, "image")
        image_files = os.listdir(image_dir)
        if image_files:
            img_file = os.path.join(image_dir, image_files[0])
        else:
            img_file = None
    else:
        image_files = [f for f in os.listdir(path) if f.startswith("image.")]
        img_file = os.path.join(path, image_files[0]) if image_files else None

    if "transformation" in metadata and "v" in metadata["transformation"]:
        metadata["transformation"]["v"] = list(metadata["transformation"]["v"])

    # Create BioFormatsImage object
    return BioFormatsImage(
        img_file,
        extent=metadata.get("extent"),
        is_full=metadata.get("is_full", False),
        origin=metadata.get("origin"),
        transformation=metadata.get("transformation"),
    )


def read_ext_image(path: str, metadata: dict, **kwargs) -> ExtImage:
    """Load a
    :py:class:`~spatialfeatureexperiment.aligned_spatialimage.ExtImage`
    from its on-disk representation.

    This method should generally not be called directly but instead be invoked by
    :py:meth:`~dolomite_base.read_object.read_object`.

    Args:
        path:
            Path to the directory containing the object.

        metadata:
            Metadata for the object.

        kwargs:
            Further arguments.

    Returns:
        A
        :py:class:`~spatialfeatureexperiment.aligned_spatialimage.ExtImage`.
    """
    if metadata is None:
        with open(os.path.join(path, "OBJECT"), "r") as f:
            metadata = json.load(f)["ext_image"]

    img = AICSImage(os.path.join(path, "image.tiff"), **kwargs)
    return ExtImage(img, extent=metadata.get("extent", None))
