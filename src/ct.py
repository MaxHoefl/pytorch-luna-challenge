import os
import numpy as np
import SimpleITK as sitk


class Ct(object):
    def __init__(self, series_uid, data_dir):
        self.series_uid = series_uid
        ct_path = glob.glob(os.path.join(
            data_dir, 'subset*', f'{series_uid}*.mhd'))
        assert os.path.exists(ct_path), \
                f'No mhd file found for {series_uid}. Tried path {ct_path}'
        self.ct_path = ct_path
        ct_mhd = sitk.ReadImage(ct_path)
        self.ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

