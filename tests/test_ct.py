import pytest
import zipfile
from ct import Ct
import os

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
    print(ctscan.ct_arr.shape)


        
