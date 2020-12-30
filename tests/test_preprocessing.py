"""
Run from src/ directory with
PYTHONPATH=. pytest -rP tests/
"""
import os
from preprocessing import mapSeriesUIDsToAnnotations, getCandidateNoduleList
import pytest


class Setup():
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
def setup(tmpdir):
    return Setup(tmpdir)        


def test_mapSeriesUIDsToAnnotations(setup):
    annotations = mapSeriesUIDsToAnnotations(setup.annotations_path)
    assert 'uid1' in annotations.keys()
    assert 'uid2' in annotations.keys()
    assert len(annotations['uid1']) == 2
    annot_center, annot_diameter = annotations['uid1'][0]
    assert annot_center[0] == 0
    assert annot_center[1] == 0
    assert annot_center[2] == 0
    assert annot_diameter == 2
    annot_center, annot_diameter = annotations['uid1'][1]
    assert annot_center[0] == 3
    assert annot_center[1] == 3
    assert annot_center[2] == 3
    assert annot_diameter == 0.5


def test_getCandidateNoduleList(setup):
    nodules = getCandidateNoduleList(setup.data_dir)
    assert len(nodules) == 1
    assert nodules[0].series_uid == "uid1"
    assert nodules[0].isNodule
    assert nodules[0].center_xyz == tuple([0.01, 0.01, 0.01])
    



