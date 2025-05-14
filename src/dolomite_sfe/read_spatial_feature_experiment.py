import json
import os

import dolomite_base as dl
import dolomite_spatial as dlspatial
import geopandas as gpd
import libpysal
import numpy as np
from dolomite_base.read_object import read_object_registry
from scipy import sparse as sp
from spatialfeatureexperiment import SpatialFeatureExperiment

read_object_registry["spatial_feature_experiment"] = "dolomite_sfe.read_spatial_feature_experiment"
read_object_registry["simple_feature"] = "dolomite_sfe.read_simple_feature"


def read_simple_feature(path: str, metadata: dict, **kwargs) -> gpd.GeoDataFrame:
    """Load a :py:class:`~geopandas.GeoDataFrame` from its on-disk representation.

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
        A :py:class:`~geopandas.GeoDataFrame` object containing the coordinates.
    """
    parquet_path = os.path.join(path, "map.parquet")
    gpdf = gpd.read_parquet(parquet_path)

    if "rownames" in gpdf.columns:
        gpdf = gpdf.set_index("rownames")

    fd_path = os.path.join(path, "feature_data")
    if os.path.isdir(fd_path):
        fd = dl.alt_read_object(fd_path, **kwargs)
        gpdf.attrs["featureData"] = fd

    param_path = os.path.join(path, "params")
    if os.path.isdir(param_path):
        params = dl.alt_read_object(param_path, **kwargs)
        gpdf.attrs["params"] = params

    return gpdf


def read_geometries(path: str, geom_type: str = None):
    """Read geometries from disk.

    Args:
        path:
            Path to the directory containing geometries.

        geom_type:
            Type of geometry ('col', 'row', or 'annot').

    Returns:
        Dictionary of geometries or None if directory doesn't exist.
    """
    if geom_type is None:
        raise ValueError("'geom_type' must be specified.")

    tg = f"{geom_type}geometries"
    geom_path = os.path.join(path, tg)

    if not os.path.isdir(geom_path):
        return None

    with open(os.path.join(geom_path, "names.json"), "r") as handle:
        _sf_names = json.load(handle)

    geoms = {}
    for i, name in enumerate(_sf_names):
        fp2 = os.path.join(geom_path, str(i))
        geoms[name] = dl.alt_read_object(fp2)

    return geoms


def read_graphs(path: str):
    """Read spatial graphs from disk.

    Args:
        path:
            Path to the directory containing spatial graphs.

    Returns:
        Dictionary of spatial graphs or None if directory doesn't exist.
    """
    _gph_path = os.path.join(path, "spatial_graphs")
    if not os.path.isdir(_gph_path):
        return None

    ms = ["row", "col", "annot"]

    with open(os.path.join(_gph_path, "names.json"), "r") as handle:
        _samples = json.load(handle)

    out = {}
    for i, sample in enumerate(_samples):
        _samp_gph_path = os.path.join(_gph_path, str(i))
        mlist = {m: {} for m in ms}

        for m in ms:
            _final_path = os.path.join(_samp_gph_path, m)
            if os.path.isdir(_final_path):
                with open(os.path.join(_final_path, "names.json"), "r") as handle:
                    _graph_names = json.load(handle)

                gs = {}
                for j, graph_name in enumerate(_graph_names):
                    _margin_path = os.path.join(_final_path, str(j))
                    method = dl.alt_read_object(os.path.join(_margin_path, "method"))

                    # Convert matrix to listw object
                    matrix = dl.alt_read_object(_margin_path)
                    if not isinstance(matrix, sp.csr_matrix):
                        matrix = sp.csr_matrix(matrix)

                    print(method, matrix)

                    _graph = libpysal.graph.Graph.from_sparse(matrix)
                    print(_graph)
                    gs[graph_name] = _graph

                mlist[m] = gs

        out[sample] = mlist

    return out


def read_local_results(sfe: SpatialFeatureExperiment, path: str) -> SpatialFeatureExperiment:
    """Read local results from disk and add them to the SFE object.

    Args:
        sfe:
            SpatialFeatureExperiment object.

        path:
            Path to the directory containing local results.

    Returns:
        Updated `SpatialFeatureExperiment` object.
    """
    _local_path = os.path.join(path, "local_results")
    if not os.path.isdir(_local_path):
        return sfe

    lrs = dl.alt_read_object(_local_path)
    sfe.column_data.set_column(column="localResults", value=lrs, in_place=True)
    return sfe


def read_reduced_dim_feature_data(sfe: SpatialFeatureExperiment, path: str) -> SpatialFeatureExperiment:
    """Read reduced dimension feature data from disk and add them to the SFE object.

    Args:
        sfe:
            SpatialFeatureExperiment object.

        path:
            Path to the directory containing reduced dimensions.

    Returns:
        Updated `SpatialFeatureExperiment` object.
    """
    _rd_path = os.path.join(path, "reduced_dimensions")
    if not os.path.isdir(_rd_path):
        return sfe

    with open(os.path.join(_rd_path, "names.json"), "r") as handle:
        _rd_names = json.load(handle)

    rds = sfe.get_reduced_dims()

    new_rds = {}
    for i, rd_name in enumerate(_rd_names):
        rd = np.array(rds[rd_name])
        _rd_attr_path = os.path.join(_rd_path, str(i), "attrs")

        if os.path.isdir(_rd_attr_path):
            a = dl.alt_read_object(_rd_attr_path)
            for key, value in a.items():
                if key not in ["dim", "dimnames"]:
                    setattr(rd, key, value)

        new_rds[rd_name] = rd

    sfe.set_reduced_dims(new_rds, in_place=True)
    return sfe


def read_spatial_feature_experiment(path: str, metadata: dict, **kwargs) -> SpatialFeatureExperiment:
    """Load a
    :py:class:`~spatialfeatureexperiment.SpatialFeatureExperiment.SpatialFeatureExperiment`
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
        :py:class:`~spatialexperiment.SpatialExperiment.SpatialExperiment`
        with file-backed arrays in the assays.
    """
    print(path, metadata, kwargs)
    spexp = dlspatial.read_spatial_experiment(path, metadata=metadata, **kwargs)

    print(spexp)
    print(spexp.get_image_data())

    spe = SpatialFeatureExperiment.from_spatial_experiment(
        spexp,
        row_geometries=read_geometries(path, "row"),
        column_geometries=read_geometries(path, "col"),
        annotation_geometries=read_geometries(path, "annot"),
        spatial_graphs=read_graphs(path),
    )

    return spe
