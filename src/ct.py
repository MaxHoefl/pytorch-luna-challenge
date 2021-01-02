import os
import glob
import numpy as np
import SimpleITK as sitk
from collections import namedtuple
import functools
from diskcache import Cache

IrcTuple = namedtuple('IrcTuple', ['idx', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])
raw_cache = Cache()


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
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.voxel_size = XyzTuple(*ct_mhd.GetSpacing())
        self.rotation_matrix = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    @staticmethod
    def irc2xyz(coord_irc: IrcTuple, 
            origin_xyz: XyzTuple, 
            voxel_size: XyzTuple, 
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

    @staticmethod
    def xyz2irc(coord_xyz: XyzTuple,
            origin_xyz: XyzTuple,
            voxel_size: XyzTuple,
            rotation_matrix: np.array):
        """
        Performs inverse operation of `irc2xyz`
        """
        origin_xyz = np.array(origin_xyz)
        voxel_size = np.array(voxel_size)
        coord_xyz = np.array(coord_xyz)
        cri_arr = ((coord_xyz - origin_xyz) @ np.linalg.inv(rotation_matrix)) / voxel_size
        cri_arr = np.round(cri_arr)
        assert np.all(cri_arr >= 0), f'Negative IRC index in {cri_arr}'
        return IrcTuple(int(cri_arr[2]), int(cri_arr[1]), int(cri_arr[0]))

    def cropCtAtXYZLocation(self, point_xyz, crop_width):
        """
        Crop the CT voxel array around a point (given in XYZ coordinates as
        nodules were annotated in XYZ coordinates) and yield a an area around 
        that point of width `crop_width`. The result will be still in IRC
        coordinates (i.e. voxel)
        The `crop_width` is an `IrcTuple` so the resulting area from the CT
        does not have to be cubic.
        """
        point_irc = self.xyz2irc(
                point_xyz, 
                self.origin_xyz, 
                self.voxel_size, 
                self.rotation_matrix)
        slice_list = []
        for axis, coordinate in enumerate(point_irc):
            start_idx = np.maximum(0, 
                    int(np.round(coordinate - crop_width[axis] / 2)))
            end_idx = np.minimum(self.ct_arr.shape[axis], 
                    int(start_idx + crop_width[axis]))
            slice_list.append(slice(start_idx, end_idx))
        ct_crop = self.ct_arr[tuple(slice_list)]
        return ct_crop, point_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid, data_dir):
    return Ct(series_uid, data_dir)


@raw_cache.memoize(typed=True)
def getCtCrop(series_uid, data_dir, center_xyz, crop_width):
    ct = getCt(series_uid, data_dir)
    return ct.cropCtAtXYZLocation(center_xyz, crop_width)


if __name__ == '__main__':
    c = Ct('1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260', 
            '/home/ubuntu/workspace/pytorch-luna-challenge/data-unversioned/')
    print(c.ct_arr.shape)
    coord_irc = IrcTuple(1, 2, 3)
    origin_xyz = XyzTuple(1, 0, -1)
    voxel_size = XyzTuple(1, 1, 1.5)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    coord_xyz = Ct.irc2xyz(coord_irc, origin_xyz, voxel_size, rotation_matrix)

