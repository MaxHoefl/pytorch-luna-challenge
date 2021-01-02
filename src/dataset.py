import torch
from torch.utils.data import Dataset
import copy
from preprocessing import getCandidateNoduleList
from ct import Ct


class LunaDataset(Dataset):
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

    def __len__(self):
        return len(self.candidate_nodules)

    def __getitem__(self, idx):
        candidate = self.candidate_nodules[idx]
        crop_width = (32, 48, 48)
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

         
