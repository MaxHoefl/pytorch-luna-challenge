import os
import glob
import numpy as np
import SimpleITK as sitk
from collections import namedtuple


class Ct(object):
    def __init__(self, series_uid, data_dir):
        self.series_uid = series_uid
        ct_path = glob.glob(os.path.join(
            data_dir, '**', f'{series_uid}*.mhd'))
        assert len(ct_path) == 1, \
            f'Could not find path for CT data with series uid {series_uid}'
        ct_path = ct_path[0]
        assert os.path.exists(ct_path), \
                f'No mhd file found for {series_uid}. Tried path {ct_path}'
        self.ct_path = ct_path
        ct_mhd = sitk.ReadImage(ct_path)
        ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_arr.clip(-1000, 1000, ct_arr) # min/max Hounsfield unit
        self.ct_arr = ct_arr


IrcTuple = namedtuple('IrcTuple', ['idx', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc: IrcTuple, 
        origin_xyz: tuple, 
        voxel_size: tuple, 
        rotation_matrix: np.array):
    """
    Converts an array of voxels in I(ndex) x R(ows) x C(olumns) format)
    to XYZ coordinates in millimeters using the following steps
    1. Flip from IRC to CRI (as X direction goes across horizontall <-> columns,
        Y direction goes vertically <-> rows, Z direction <-> index
    2. Scale voxel indices with size of voxel in millimeters
    3. Matrix multiply with rotation matrix 
    4. Add offset from origin
    """
    cri_arr = np.array(coord_irc)[::-1]
    origin_xyz = np.array(origin_xyz)
    voxel_size = np.array(voxel_size)
    coord_xyz = (rotation_matrix @ (cri_arr * voxel_size)) + origin_xyz
    return XyzTuple(*coord_xyz)


def xyz2irc(coord_xyz: XyzTuple,
        origin_xyz: tuple,
        voxel_size: tuple,
        rotation_matrix: np.array):
    """
    Performs inverse operation of `irc2xyz`
    """
    origin_xyz = np.array(origin_xyz)
    voxel_size = np.array(voxel_size)
    coord_xyz = np.array(coord_xyz)
    cri_arr = ((coord_xyz - origin_xyz) @ np.linalg.inv(rotation_matrix)) / voxel_size
    cri_arr = np.round(cri_arr)
    return IrcTuple(int(cri_arr[2]), int(cri_arr[1]), int(cri_arr[0]))


if __name__ == '__main__':
    c = Ct('1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260', 
            '/home/ubuntu/workspace/pytorch-luna-challenge/data-unversioned/')
    print(c.ct_arr.shape)

