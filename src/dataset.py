import torch
import torch.nn as nn
from torch.utils.data import Dataset
import copy
from preprocessing import getCandidateNoduleList
from ct import Ct


class LunaDataset(Dataset):
    INPUT_CHANNELS = 1
    #CROP_HEIGHT = 48
    #CROP_WIDTH = 48
    #CROP_DEPTH = 32
    CROP_HEIGHT = 24
    CROP_WIDTH = 24
    CROP_DEPTH = 16

    def __init__(self, data_dir, series_uid=None, val_stride=0, is_val=None):
        super().__init__()
        self.data_dir = data_dir
        self.series_uid = series_uid
        self.is_val = is_val
        self.candidate_nodules = copy.copy(getCandidateNoduleList(data_dir))
        if series_uid:
            self.candidate_nodules = [
                x for x in self.candidate_nodules if x.series_uid == series_uid
            ]
        if is_val:
            assert val_stride > 0, f"Provide a val_stride >0, got {val_stride}"
            self.candidate_nodules = self.candidate_nodules[::val_stride]
        elif val_stride:
            if len(self.candidate_nodules) > val_stride:
                del self.candidate_nodules[::val_stride]
            assert self.candidate_nodules

    def __len__(self):
        return len(self.candidate_nodules)

    def __getitem__(self, idx):
        candidate = self.candidate_nodules[idx]
        crop_width = (self.CROP_DEPTH, self.CROP_HEIGHT, self.CROP_WIDTH)
        ct_scan = Ct(candidate.series_uid, self.data_dir)
        ct_crop, crop_center = \
                ct_scan.cropCtAtXYZLocation(candidate.center_xyz, crop_width)
        ct_crop = torch.from_numpy(ct_crop)
        ct_crop = ct_crop.to(torch.float32)
        ct_crop = ct_crop.unsqueeze(0)
        
        target = torch.tensor([
            not candidate.isNodule,
            candidate.isNodule
        ], dtype=torch.long)

        return ct_crop, target, candidate.series_uid, crop_center

