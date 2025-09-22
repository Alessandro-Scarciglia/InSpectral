# Import modules
from torch.utils.data import Dataset
import numpy as np
import torch


class NeRFData(Dataset):
    """
    Dataset class is meant to provide a torch dataset. This dataset comes from preprocessed data, thus samples are
    not images but single pixels described by rays origin, direction, channels ground truth, light direction. This class
    reads the output of the script 'generate_nerf_dataset.py'.

    Attributes:
    ----------
    data_path: str
        the full path to the folder where metadata are stored in 'transforms.json' file
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
        sample = self.df[index]
        return torch.tensor(sample, dtype=torch.float32)

