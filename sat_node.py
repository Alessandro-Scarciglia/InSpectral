# Modules import
import json
import torch
import numpy as np

# Custom modules
from virtual_sensors import VirtualSensors
from data_buffer import DataBuffer
from rays_generator import RaysGenerator
from parameters import *
from rendering import NeuralRenderer
from trainer import Trainer


class SatNode:
    def __init__(self,
                 roll_cfg: str,
                 datapath: str,
                 calibration_path: str,
                 resolution: int = 1024,
                 channels: int = 1,
                 lin_drift: float = 0,
                 ang_drift: float = 0,
                 device: str = 'cpu'
                 ):

        # Retrieve intrinsic calibration matrix K
        with open(calibration_path, "r") as fopen:
            calib = json.load(fopen)
            self.k = np.array(calib["mtx"]).reshape(3, 3)
            self.k[:2, :3] /= (1024/resolution)

        # Initialize measurements generator
        vsens = VirtualSensors(datapath=datapath,
                               resolution=resolution,
                               roll_cfg=roll_cfg,
                               lin_drift=lin_drift,
                               ang_drift=ang_drift)
        self.measurements = vsens.get_measurement()

        # Define the image size
        self.H = self.W = resolution
        self.CH = channels

        # Create data buffers
        self.train_set = DataBuffer(**rays_parameters, K=self.k)
        self.valid_set = list()

        # Rays generator
        self.rays_generator = RaysGenerator(**rays_parameters, K=self.k)

        # Create NeRF Renderer
        self.renderer = NeuralRenderer(**sampler_parameters,
                                       **sh_parameters,
                                       **hash_parameters,
                                       **nerf_parameters,
                                       device=device)

        # Create Training routine
        self.trainer = Trainer(self.renderer, **optimizer_parameters)


    # Obtain one measurement 
    def get_measurement(self):
        return self.measurements
    

    def render(self, c2w: torch.Tensor):

        # Get rays from pose
        rays = self.rays_generator(c2w)
        rays = rays.reshape(-1, 6)
        
        # Rendering of the frame
        frame = self.renderer(rays)
        
        return frame
