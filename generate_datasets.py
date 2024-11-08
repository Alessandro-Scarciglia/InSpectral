'''
This script is meant to generate the datasets of the trajectories directly in rays shape.
It is an abstraction of the whole process of acquiring data from cameras and preprocessing.
The script read all the pictures and create a bunch of samples made like [ray_o, ray_d, label, cfg_code].

> ray_o and ray_d are ray origin (3x1) and direction (3x1)
> label is the channel(s) truth
> cfg_code the configuration code (0 for roll_0, 1 for roll_120, 2 for roll_240).
'''

# Parameters
DEG_RES = 10
RGB = False
SRC_IMG_CALIB = 1024
DST_IMG_SIZE = 256
DATASET_NAME = f"rgb_{RGB}_res_{DEG_RES}_size_{DST_IMG_SIZE}.npy"

# Paths
ROOT = "data/"
DATA_PATH = "data/transforms.json"
CALIB_PATH = "calibration/calibration.json"
DATASET_DST = ROOT + "preprocessed_data/" + DATASET_NAME


# Import modules
import numpy as np
import torch
import cv2
import json
from tqdm import tqdm
import time

from parameters import *
from rays_generator import RaysGenerator


# Define a scaling factor for the images
scale = SRC_IMG_CALIB // DST_IMG_SIZE

# Read and store calibration matrix of cameras
# Default calibration is from a 1024x1024 resolution
with open(CALIB_PATH, "r") as fopen:
    K = json.load(fopen)["mtx"]
    K = np.array(K).reshape(3, 3)
    K[:2, :3] /= scale

# Define ray helper object
ray_gen = RaysGenerator(**rays_parameters, K=K)

# Read annotation file and store in a dictionary
with open(DATA_PATH, "r") as fopen:
    annotations = json.load(fopen)

# Create a configuration code, i.e. progressive integer for each element
cfgs = list(annotations.keys())
cfgs_codes = {cfg_name: i for i, cfg_name in enumerate(cfgs)}

# Initialize the dataset list
dataset = list()

# Iterate through each configuration
for cfg in cfgs:

    # Re-define the reference folder
    img_df = annotations[cfg]

    # Generate the list of all images in the given configuration
    imgs_list = list(img_df.keys())

    # Iterate through every image
    print(f"Config: {cfg}")
    for i, img_path in tqdm(enumerate(imgs_list)):

        # Take frame every n_frame
        if i % DEG_RES:
            continue

        # Load the image
        if RGB:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (DST_IMG_SIZE, DST_IMG_SIZE))
        else:
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (DST_IMG_SIZE, DST_IMG_SIZE))
            img = np.expand_dims(img, axis=-1)

        # Retrieve the c2w matrix
        c2w_tensor = torch.tensor(img_df[img_path])

        # Generate rays for such c2w
        rays = ray_gen(c2w_tensor).numpy()

        # Generate configuration code
        cfg_code = np.ones(shape=(DST_IMG_SIZE, DST_IMG_SIZE, 1)) * cfgs_codes[cfg]

        # Concatenate data and reshape as a list of pixels
        # Target length is: 6 (3 for 3D origin + 3 for 3D direction) + 1 (cfg code) + Number of Channels 
        target_length = 6 + 1 + img.shape[-1]
        rays_cfg_labels = np.concatenate([rays, cfg_code, img/255.], axis=-1)
        rays_cfg_labels = rays_cfg_labels.reshape(-1, target_length)

        # Append to dataset list
        dataset += rays_cfg_labels.tolist()
        time.sleep(0.1)
    
# In the end, transform the dataset in a numpy array and store it
dataset = np.array(dataset)
print(f"Taken {len(dataset) / (DST_IMG_SIZE**2)} frames.")
np.random.shuffle(dataset)
np.save(DATASET_DST, dataset)

