# Modules import
import json
import torch
import cv2
import numpy as np

# Custom modules
from virtual_sensors import VirtualSensors
from parameters import RenderingParameters
from rendering import NeuralRenderer
from trainer import Trainer


class SatNode:
    def __init__(self,
                 roll_cfg: str,
                 datapath: str,
                 calibration_path: str,
                 resolution: int = 1024,
                 lin_drift: float = 0,
                 ang_drift: float = 0,
                 device: str = "cuda:0"
                 ):

        # Retrieve intrinsic calibration matrix K
        with open(calibration_path, "r") as fopen:
            calib = json.load(fopen)
            self.k = calib["mtx"]

        # Initialize measurements generator
        vsens = VirtualSensors(datapath=datapath,
                               resolution=resolution,
                               roll_cfg=roll_cfg,
                               lin_drift=lin_drift,
                               ang_drift=ang_drift)
        self.measurements = vsens.get_measurement()

        # Define the image size
        self.H = self.W = resolution

        # Create NeRF Renderer
        renderer_args = RenderingParameters()
        self.renderer = NeuralRenderer(**renderer_args.get_all_params())
        self.renderer.to(device)

        # Create Training routine
        self.trainer = Trainer(self.renderer)


    def get_measurement(self):
        return self.measurements
    

    def render(self,
               c2w: torch.Tensor,
               frame: torch.Tensor = torch.tensor([])):
        return self.renderer(c2w, frame)


    
# Run for usage
if __name__ == "__main__":
    sat1 = SatNode(roll_cfg="roll_120",
                  datapath='data/transforms.json',
                  calibration_path='calibration/calibration.json')

    for i, (c2w, frame) in enumerate(sat1.get_measurement()):
        c2w = torch.tensor(c2w)
        frame = torch.tensor(frame)
        sat1.trainer.train_one_frame(c2w=c2w, frame=frame, niter=i)