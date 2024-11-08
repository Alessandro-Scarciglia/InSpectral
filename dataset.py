'''
Dataset class is meant to provide a torch dataset.
Each sample from a torch dataset return an array made of:
[ray_origin, ray_direction, configuration_code, channels_truth]
'''

# Import modules
from torch.utils.data import Dataset
import numpy as np


# Dataset class definition
class InspectralData(Dataset):
    def __init__(self,
                 data_path: str,
                 mode: str = 'training',
                 valid_set_ratio: float = 0.2):

        # Apply default correction to mode
        if mode not in ["training", "validation"]:
            self.mode = "training"
            print(f"Warning: mode argument set to '{mode}', but only 'training' and 'validation' are allowed. "
                  "Default mode forced to 'training'.")
        else:
            self.mode = mode

        # Read the data folder
        df = np.load(data_path)
        
        # Split training and validation
        idx = int(len(df) * valid_set_ratio)
        self.training_set = df[idx:]
        self.validation_set = df[:idx]


    def __len__(self):

        # Choose length according to mode
        if self.mode == 'training':
            length = len(self.training_set)
        elif self.mode == "validation":
            length = len(self.validation_set)

        return length
    
    def __getitem__(self, index):

        # Choose sampling set according to mode
        if self.mode == 'training':
            sample = self.training_set[index]
        elif self.mode == 'validation':
            sample = self.validation_set[index]
        
        return sample


# Test for usage
if __name__ == "__main__":
    ds = InspectralData(
        data_path="data/preprocessed_data/rgb_True_size_256.npy",
        validset_ratio=0.2,
        mode='training'
    )
