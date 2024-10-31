# Modules import
import cv2
import json
import torch
import numpy as np
from parameters import *


class VirtualSensors:
    def __init__(self,
                 datapath: str,
                 resolution: int = 1024,
                 roll_cfgs: list = ["roll_0"],
                 lin_drift: float = 0.,
                 ang_drift: float = 0.,
                 noise_mean: float = 0.,
                 noise_std: float = 1.,
                 is_rgb: bool = False):
        
        # Attributes
        self.datapath = datapath
        self.res = resolution
        self.roll_cfgs = roll_cfgs
        self.lin_drift = lin_drift
        self.ang_drift = ang_drift
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.is_rgb = is_rgb
        

    def get_measurement(self):
        '''
        This function is meant to be a virtualization of both Camera and IMU.
        This function reads a sample from a dataset and returns a tuple (image, pose).
        - image: the virtualization of an RGB frame captured from a camera sensor.
        - pose: the pose c2w (4x4 homography) where the frame is taken.

        Arguments:
        - datapath: path to dataset location.
        - resolution: frame resolution (<= original frame resolution). 
        - lin_drift: cumulative drift error on position at every step.
        - ang_drift: cumulative drift error on orientation at every step.
        '''
        
        # Read the JSON file where images path and pose labels are stored
        with open(self.datapath, "r") as fopen:
            orbits = json.load(fopen)

        # For each selected orbit
        for roll_cfg in self.roll_cfgs:

            # Select specific orbit
            orbit = orbits[roll_cfg]

            # Loop through every item
            for imagepath, c2w in orbit.items():
                
                # Jump on similar positions
                if imagepath.split(".")[-2] in JUMP:
                    continue

                # Retrieve image
                frame = cv2.imread(imagepath)
                resized_frame = cv2.resize(frame, (self.res, self.res))

                # If rgb flag is False, convert in grayscale
                if not self.is_rgb:
                    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                    out_frame = np.expand_dims(gray_frame, axis=-1)
                else:
                    out_frame = resized_frame

                # Change variable format in torch.tensor
                out_frame = torch.tensor(out_frame) / 255.
                c2w = torch.tensor(c2w)

                # Add noise in a [-noise_mult, noise_mult] range
                noise_src = self.noise_mean + self.noise_std * torch.randn(3) 
                c2w[:3, -1] += noise_src
                
                # Normalization of the scene (TODO: this shall be done in post-processing)
                c2w[:3, -1] /= 2.

                yield c2w, out_frame


# Run for usage example
if __name__ == "__main__":
    
    # Instantiate the object
    vsens = VirtualSensors(datapath="data/transforms.json",
                           roll_cfgs=["roll_0"])

    # Set generator
    measurements = vsens.get_measurement()

    # Acquire three measurements
    for measurement in measurements:
        
        # Unpack
        pose, frame = measurement

        # Visualize c2w matrix
        print(f"Camera to World Matrix:\n{pose}\n\n")
        
        # Visualize 
        exit()
    