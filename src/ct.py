import os
import glob
import numpy as np
import SimpleITK as sitk


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
        self.ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

if __name__ == '__main__':
    c = Ct('1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260', 
            '/home/ubuntu/workspace/pytorch-luna-challenge/data-unversioned/')
    print(c.ct_arr.shape)

