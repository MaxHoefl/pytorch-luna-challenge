import torch
from torch.utils.data import Dataset


class LunaDataset(Dataset):
    def __len__(self):
        return len(self.candidate_nodules)

    def __getitem__(self, idx):

