import os

from dolomite_base import read_object, save_object
import dolomite_sfe
from spatialfeatureexperiment import SpatialFeatureExperiment, BioFormatsImage


def test_basic_reader():
    rpath = os.getcwd() + "/tests/data/sfe_save"

    spe = read_object(rpath)
    assert isinstance(spe, SpatialFeatureExperiment)

    assert spe.shape == (398, 6272)
    assert "sample_id" in spe.get_column_data().get_column_names()
    assert len(spe.get_image_data()) == 1
    grp = spe.get_spatial_graphs()
    assert isinstance(grp, dict)
    assert "sample01" in grp
    assert len(spe.get_col_geometries()) == 3
    assert len(spe.get_row_geometries()) == 1
    assert isinstance(spe.get_image_data().get_column("data")[0], BioFormatsImage)

# def test_basic_writer():
#     rpath = os.getcwd() + "/tests/data"
#     dir = os.path.join(mkdtemp(), "spatial_rtrip")

#     spe = read_object(rpath)
#     assert isinstance(spe, SpatialExperiment)

#     save_object(spe, dir)

#     rtrip =  read_object(dir)
#     assert isinstance(rtrip, SpatialExperiment)
#     assert spe.shape == rtrip.shape
#     assert "sample_id" in rtrip.get_column_data().get_column_names()
#     assert len(set(rtrip.get_column_data().get_column("sample_id")).difference(["section1", "section2"])) == 0