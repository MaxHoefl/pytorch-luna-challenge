import torch
from torch.utils.data import Dataset
from preprocessing import getCandidateNoduleList
from ct import Ct


class LunaDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.candidate_nodules = getCandidateNoduleList(data_dir)

    def __len__(self):
        return len(self.candidate_nodules)

    def __getitem__(self, idx):
        candidate = self.candidate_nodules[idx]
        crop_width = (32, 48, 48)
        ct_scan = Ct(candidate.series_uid, self.data_dir)
        ct_crop, crop_center = \
                ct_scan.cropCtAtXYZLocation(candidate.center_xyz, crop_width)
        ct_crop = torch.from_numy(ct_crop)
        ct_crop = ct_crop.to(torch.float32)
        ct_crop = ct_crop.unsqueeze(0)
        
        target = torch.tensor([
            not candidate.isNodule,
            candidate.isNodule
        ], dtype=torch.long)

        return ct_crop, target, candidate.series_uid, crop_center

         
