# Import modules
from torch.utils.data import Dataset
import numpy as np
import torch


class NeRFData(Dataset):
    """
    Dataset class is meant to provide a torch dataset. This dataset comes from preprocessed data, thus samples are
    not images but single pixels described by rays origin, direction, channels ground truth (RGBA), driving mask, and image index. 
    This class reads the output of the script 'generate_postprocessed_dataset.py'.

    Attributes:
    ----------
    data_path: str
        the full path to the .npy file containing the postprocessed pixel-ray dataset.
    """
    def __init__(
            self,
            data_path: str
    ):
        # Read the data folder
        self.df = np.load(data_path)

    def __len__(
            self
    ) -> int:
        return len(self.df)
    
    def __getitem__(
            self,
            index: int
    ) -> torch.Tensor:
        # Whole row: 6 ray values + image channels (3 RGB, 1 IR or 4 RGB+IR) + 1 mask + 1 image index
        sample = self.df[index]
        return torch.tensor(sample, dtype=torch.float32)