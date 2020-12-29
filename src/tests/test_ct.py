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
          zipf.extract_all(tmp_dir)
      self.data_dir = tmp_dir


@pytest.fixture()
def setup(tmpdir):
    return Setup(tmpdir)


def test_Ct(setup):
    ctscan = Ct(series_uid='ct1', data_dir=setup.data_dir)

        
