import pytest
import zipfile
from ct import Ct, IrcTuple, XyzTuple
import os
import numpy as np
from conftest import luna_setup as setup 


def test_Ct(setup):
    ctscan = Ct(series_uid='ct1', data_dir=setup.data_dir)
    assert ctscan.ct_arr.shape == (119, 512, 512)


def test_irc2xyz_xyz2irc(setup):
    coord_irc = IrcTuple(1, 2, 3)
    origin_xyz = XyzTuple(1, 0, -1)
    voxel_size = XyzTuple(1, 1, 1.5)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    coord_xyz = Ct.irc2xyz(coord_irc, origin_xyz, voxel_size, rotation_matrix)
    irc = Ct.xyz2irc(coord_xyz, origin_xyz, voxel_size, rotation_matrix)
    assert irc == coord_irc

def test_cropCtAtXYZLocation(setup):
    ctscan = Ct(series_uid='ct1', data_dir=setup.data_dir)
    point_irc = IrcTuple(70, 250, 250)
    point_xyz = Ct.irc2xyz(point_irc, ctscan.origin_xyz, 
            ctscan.voxel_size, ctscan.rotation_matrix)
    crop_width = [4, 4, 6]
    ct_crop, crop_center = ctscan.cropCtAtXYZLocation(point_xyz, crop_width)
    assert crop_center == point_irc
    assert ct_crop.shape == tuple(crop_width)

def test_cropCtAtXYZLocation_atedge(setup):
    ctscan = Ct(series_uid='ct1', data_dir=setup.data_dir)
    point_irc = IrcTuple(0, 0, 0)
    point_xyz = Ct.irc2xyz(point_irc, ctscan.origin_xyz, 
            ctscan.voxel_size, ctscan.rotation_matrix)
    crop_width = [100000, 100000, 10000]
    ct_crop, crop_center = ctscan.cropCtAtXYZLocation(point_xyz, crop_width)
    assert np.all(ct_crop == ctscan.ct_arr)


    

 
