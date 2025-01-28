'''
Dataset class is meant to provide a torch dataset.
Each sample from a torch dataset return an array made of:
[ray_origin, ray_direction, channels_truth]
'''

# Import modules
from torch.utils.data import Dataset
import numpy as np
import torch


# Dataset class definition
class NeRFData(Dataset):
    def __init__(self,
                 data_path: str):

        # Read the data folder
        self.df = np.load(data_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample = self.df[index]
        return torch.tensor(sample, dtype=torch.float32)


# Test for usage
if __name__ == "__main__":
    ds = NeRFData(
        data_path="data/preprocessed_data/ir.npy",
    )
    print(ds[0])
