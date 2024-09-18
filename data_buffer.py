# Import standard modules
import torch
from torch.utils.data import Dataset
import numpy as np
import json

# Import custom modules
from rays_generator import RaysGenerator
from virtual_sensors import VirtualSensors


class DataBuffer(Dataset):
    def __init__(self,
                 H: int,
                 W: int,
                 CH: int,
                 K: np.array):

        # Rays Generator
        self.rays_generator = RaysGenerator(H=H,
                                            W=W,
                                            CH=CH,
                                            K=K)

        # Attributes
        self.buffer = list()

    # Methods to fill the buffer
    def get_data(self,
                 c2w: torch.Tensor,
                 frame: torch.Tensor):

        # Generate rays
        rays = self.rays_generator(c2w)

        # Couple rays and frame
        length = rays.shape[-1] + frame.shape[-1]
        rays_and_labels = np.concatenate([rays, frame], axis=-1).reshape(-1, length).tolist()

        # Add to buffer
        self.buffer += rays_and_labels

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return torch.Tensor(self.buffer[index])


# Test
if __name__ == "__main__":

    # Retrieve intrinsic calibration matrix K
    with open("calibration/calibration.json", "r") as fopen:
        calib = json.load(fopen)
        K = np.array(calib["mtx"]).reshape(3, 3)

    # Create data buffer
    ds = DataBuffer(H=1024, W=1024, CH=3, K=K)

    # Instantiate the virtual sensor
    vsens = VirtualSensors(datapath="data/transforms.json",
                           roll_cfg="roll_0")
    
    # Fill the buffer
    for measurement in vsens.get_measurement():
        print(len(ds))
        ds.get_data(*measurement)
        print(ds[0])