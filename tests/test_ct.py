import pytest
import zipfile
from ct import Ct, irc2xyz, xyz2irc, IrcTuple, XyzTuple
import os
import numpy as np

TEST_DIR = os.path.dirname(__file__)


class Setup(object):
    def __init__(self, tmpdir):
      testdata_zip = os.path.join(TEST_DIR, 'testdata', 'testdata.zip')
      tmp_dir = tmpdir.mkdir('testdata')
      with zipfile.ZipFile(testdata_zip, 'r') as zipf:
          zipf.extractall(tmp_dir)
      for dirs, _, files in os.walk(str(tmp_dir)):
          for f in files:
              os.chmod(os.path.join(dirs, f), 0o777)
      self.data_dir = str(tmpdir)


@pytest.fixture()
def setup(tmpdir):
    return Setup(tmpdir)


def test_Ct(setup):
    ctscan = Ct(series_uid='ct1', data_dir=setup.data_dir)
    assert ctscan.ct_arr.shape == (140, 512, 512)


def test_irc2xyz_xyz2irc(setup):
    coord_irc = IrcTuple(1, 2, 3)
    origin_xyz = (1, 0, -1)
    voxel_size = (1, 1, 1.5)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    coord_xyz  = irc2xyz(coord_irc, origin_xyz, voxel_size, rotation_matrix)
    irc  = xyz2irc(coord_xyz, origin_xyz, voxel_size, rotation_matrix)
    assert irc  == coord_irc


 
