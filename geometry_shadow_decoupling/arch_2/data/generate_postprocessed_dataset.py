'''
This script transforms the images and labels of a custom NeRF Dataset to produce another dataset directly 
injectable in NeRF training. In essence, from images and metadata one obtains rays origin, direction, channels gt and light direction.
Everything is transformed in a numpy output file .npy where each sample refers to a single pixel.
'''

# Import modules
import numpy as np
import torch
import json
import cv2
import os
import time
from tqdm import tqdm
import mathutils

# Input parameters to produce the dataset: input folder and output destination
"""
Guidelines for naming the postprocessed dataset:
[name_of_original_dataset]_[degree_resolution].npy
e.g. "colour_256_XY_12_1d5_training_3.npy" means that the dataset is sampled one frame each 3 degrees, i.e. 360/3=120 frames.
"""
SRC_DATASET = "colour_256_XY_12_1d5_training"
ARCH = "arch_2"
DEG_RES = 3

DATA_PATH = f"/home/vision/Desktop/Datasets/CloudSat_NeRF_Datasets/{SRC_DATASET}"
DATA_DST = f"/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/{ARCH}/data/postprocessed_dataset/{SRC_DATASET}_{DEG_RES}_{ARCH}.npy"


def calculate_intrinsic_matrix(
        fov: float,
        resolution: tuple[int, int]
) -> np.ndarray:
    """
    Calculate the intrinsic camera matrix K for an ideal pinhole camera.
    
    Parameters:
    ----------
    fov: float
        Field of view in degrees (assumes symmetric FOV for both axes).
    
    resolution: Tuple[int]
        width and height of the image in pixels.
    
    Returns:
    -------
    intrinsic_matrix: np.ndarray
        A 3x3 matrix with intrinsic camera parameters.
    """

    # Unpack image resolution
    width, height = resolution
    
    # Compute focal lengths
    fx = width / (2 * np.tan(fov / 2))
    fy = height / (2 * np.tan(fov / 2))
    
    # Compute the principal point (assumed to be the image center)
    cx = width / 2.
    cy = height / 2.
    
    # Construct the intrinsic matrix
    intrinsic_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return intrinsic_matrix

def main():
    '''
    Pipeline to produce and store the postprocessed datasets, from NeRF images/metadata to rays.
    '''
    
    # Load data
    with open(DATA_PATH + "/transforms.json", "r") as fopen:
        df = json.load(fopen)
        fov = df["camera_angle_x"]
        resolution = int(df["resolution"])
        num_ch = int(df["num_ch"])
        samples = df["frames"]

    # Set the rays generator object
    intrinsic_matrix = calculate_intrinsic_matrix(fov=fov, resolution=(resolution, resolution))
    raygen = RaysGeneratorSynth(height=resolution, width=resolution, num_ch=num_ch, intrinsic_matrix=intrinsic_matrix)

    # Dataset object
    dataset = list()

    # Store pixels from each training image
    print("Generating Training Dataset...")
    for i, sample in tqdm(enumerate(samples)):

        # Jump frames
        if i % DEG_RES:
            continue

        # Load the image
        img_path = os.path.join(DATA_PATH, sample["file_path"])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resolution, resolution))

        # Load the mask
        mask_path = os.path.join(DATA_PATH, sample["mask_path"])
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256, 256))
        mask = np.expand_dims(mask, axis=-1)
        
        # Generate rays
        c2w = torch.tensor(sample["transform_matrix"])
        rays = raygen(c2w)

        # Get light direction from quaternion
        sun_dir = mathutils.Matrix(sample["light_direction"]) @ mathutils.Vector([0, 0, -1])
        bc_sun_dir = np.ones((resolution, resolution, num_ch)) * sun_dir

        # Compose output: each sample is made of 6 elements (3 for rays origin + 3 for rays direction) + 3 elements for
        # light direction + 3 elements for RGB color channels + 1 element for binary mask = 13 elements per pixel.
        rays_cfg_labels = np.concatenate([rays, bc_sun_dir, img/255., mask/255.], axis=-1)
        rays_cfg_labels = rays_cfg_labels.reshape(-1, 13)

        # Append to dataset list
        dataset += rays_cfg_labels.tolist()
        time.sleep(0.1)

    # Store in a single dataset numpy file
    dataset = np.array(dataset)
    print(f"Taken {len(dataset) / (resolution**2)} frames.")
    np.random.shuffle(dataset)
    np.save(DATA_DST, dataset)


if __name__ == "__main__":

    from rays_generator import RaysGeneratorSynth
    main()