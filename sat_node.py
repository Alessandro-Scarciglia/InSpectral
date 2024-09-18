# Modules import
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2

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
        self.trainer = Trainer(self.renderer, **training_parameters)


    def get_measurement(self):
        return self.measurements
    

    def render(self, c2w: torch.Tensor):

        # Get rays from pose
        rays = self.rays_generator(c2w)
        rays = rays.reshape(-1, 6)
        
        # Rendering of the frame
        frame = self.renderer(rays)
        
        return frame


# Run for test
if __name__ == "__main__":

    # Instantiate the satellite object
    sat1 = SatNode(roll_cfg="roll_0",
                   resolution=128,
                   datapath='data/transforms.json',
                   calibration_path='calibration/calibration.json',
                   device='cuda:0')

    # Start acquisition of images
    for i, (c2w, frame) in enumerate(sat1.get_measurement()): 
      
        # Save a test frame every 5 acquisition
        if i % 5 == 0:
            sat1.valid_set.append((c2w, frame))
            continue

        # Populate the data buffer
        sat1.train_set.get_data(c2w=c2w, frame=frame)
        
        # Display acquisition
        cv2.imshow("Acquisition Phase", frame.detach().numpy())
        cv2.waitKey(1)
    
    # Close display windows
    cv2.destroyAllWindows()

    # Generate DataLoader from Dataset
    dataloader = DataLoader(dataset=sat1.train_set, batch_size=32*32, shuffle=True, num_workers=4)

    # Training loop
    for i, rays_batch in enumerate(dataloader):

        # Train one batch
        lossval = sat1.trainer.train_one_batch(rays=rays_batch[:, :6],
                                               labels=rays_batch[:, 6:],
                                               niter=i)
        
        print(f"Batch n.{i} | Loss: {lossval:.5f}")

        # Show rendering every 10 batches 
        if i % 500 == 0:
            with torch.no_grad():
                # Load test case
                test_c2w, test_frame = sat1.valid_set[7]

                # Render frame
                frame, _, _ = sat1.render(test_c2w)

                # Display result
                # frame = frame.reshape(sat1.H, sat1.W, 3)
                # disp = np.hstack([frame, test_frame.cpu()])
                # cv2.imshow("Rendering Test", disp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()