# Modules import
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from virtual_sensors import VirtualSensors
from parameters import RenderingParameters, TrainingParameters
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
                 device: str = "cpu"
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

        # Create NeRF Renderer
        renderer_args = RenderingParameters()
        self.renderer = NeuralRenderer(**renderer_args.get_all_params(), K=self.k)

        # Create Training routine
        training_args = TrainingParameters()
        self.trainer = Trainer(self.renderer, **training_args.get_all_params())


    def get_measurement(self):
        return self.measurements
    

    def render(self, c2w: torch.Tensor):
        return self.renderer(c2w)


    
# Run for test
if __name__ == "__main__":
    sat1 = SatNode(roll_cfg="roll_120",
                   resolution=32,
                   datapath='data/transforms.json',
                   calibration_path='calibration/calibration.json')

    for i, (c2w, frame) in enumerate(sat1.get_measurement()): 
        
        # Show every 10 samples
        if i % 10 == 0:
            with torch.no_grad():
                test_chs_map, _, __, = sat1.render(c2w)
                cv2.imshow("", test_chs_map.detach().numpy() * 255.)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        loss = sat1.trainer.train_one_frame(c2w=c2w, frame=frame, niter=i)
        print(loss.item())
