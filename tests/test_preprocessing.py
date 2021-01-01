"""
Run from src/ directory with
PYTHONPATH=. pytest -rP tests/
"""
import os
from preprocessing import mapSeriesUIDsToAnnotations, getCandidateNoduleList
import pytest
from conftest import mock_setup as setup


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
    



