import os
import zipfile
import pytest


TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')


class LunaDataSetup(object):
    def __init__(self, tmpdir):
      testdata_zip = os.path.join(TEST_DIR, 'testdata', 'testdata.zip')
      data_dir = str(tmpdir)
      with zipfile.ZipFile(testdata_zip, 'r') as zipf:
          zipf.extractall(data_dir)
      for dirs, _, files in os.walk(data_dir):
          for f in files:
              os.chmod(os.path.join(dirs, f), 0o777)
      self.data_dir = data_dir


@pytest.fixture()
def luna_setup(tmpdir):
    return LunaDataSetup(tmpdir)


class MockDataSetup():
    def __init__(self, tmpdir):
        annotations_path = tmpdir.join("annotations.csv")
        candidates_path = tmpdir.join("candidates.csv")
        subset0 = tmpdir.mkdir("subset0")
        uid1 = subset0.join("uid1.mhd")
        uid2 = subset0.join("uid2.mhd")
        with open(uid1, 'w') as f:
            f.write(" ")
        with open(uid2, 'w') as f:
            f.write(" ")

        with open(annotations_path, 'w') as f:
            f.write('\n'.join([
                "seriesuid,coordX,coordY,coordZ,diameter_mm",
                "uid1,0,0,0,2",
                "uid1,3,3,3,0.5",
                "uid2,-3,-3,-3,1",
                "uid3,-10,-10,-10,1",
                ]))
        with open(candidates_path, 'w') as f:
            f.write('\n'.join([
                "seriesuid,coordX,coordY,coordZ,class",
                "uid1,0.01,0.01,0.01,1",
                "uid1,3.5,3.01,3.01,0"
                "uid3,-10,-10,-10,1"
                ]))
        self.data_dir = os.path.dirname(annotations_path)  
        self.annotations_path = annotations_path
        self.candidates_path = candidates_path


@pytest.fixture()
def mock_setup(tmpdir):
    return MockDataSetup(tmpdir)        


