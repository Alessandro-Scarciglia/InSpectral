# Import modules
import numpy as np
import torch
from rays_generator_synth import RaysGeneratorSynth
import json
import cv2
import os
import time
from tqdm import tqdm


# Parameters
DATA_PATH = "/home/visione/Projects/BlenderScenarios/Sat/Dataset/Orbit_VR_256_fulllight/VIS_Training"
DATA_DST = "/home/visione/Projects/InSpectral/data/preprocessed_data/sat_vis_full_vr_60.npy"


# Generate K from resolution and FOV
def calculate_intrinsic_matrix(fov, resolution):
    """
    Calculate the intrinsic camera matrix K for an ideal pinhole camera.
    
    Parameters:
    - fov: Field of view in degrees (assumes symmetric FOV for both axes).
    - resolution: Tuple (width, height) of the image in pixels.
    
    Returns:
    - K: 3x3 intrinsic camera matrix.
    """

    # Image resolution
    width, height = resolution
    
    # Compute focal lengths
    fx = width / (2 * np.tan(fov / 2))
    fy = height / (2 * np.tan(fov / 2))
    
    # Compute the principal point (assumed to be the image center)
    cx = width / 2
    cy = height / 2
    
    # Construct the intrinsic matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K


# Format conversion
def main():
    
    # Load data
    with open(DATA_PATH + "/transforms.json", "r") as train_fopen:
        train_df = json.load(train_fopen)
        train_fov = train_df["camera_angle_x"]
        train_samples = train_df["frames"]

    # Define the rays generator
    K = calculate_intrinsic_matrix(fov=train_fov, resolution=(256, 256))
    raygen = RaysGeneratorSynth(H=256, W=256, CH=1, K=K)

    # Dataset object
    dataset = list()

    # Store pixels from each training image
    print("Generating Training Dataset...")
    for i, sample in tqdm(enumerate(train_samples)):

        # if i % 4:
        #     continue

        # Load the image
        img_path = os.path.join(DATA_PATH, sample["file_path"])
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=-1)
        
        # Generate rays
        c2w = torch.tensor(sample["transform_matrix"])
        rays = raygen(c2w)

        # Compose output
        rays_cfg_labels = np.concatenate([rays, img / 255.], axis=-1)
        rays_cfg_labels = rays_cfg_labels.reshape(-1, 7)

        # Append to dataset list
        dataset += rays_cfg_labels.tolist()
        time.sleep(0.1)

    # Store in a single dataset numpy file
    dataset = np.array(dataset)
    print(f"Taken {len(dataset) / (256**2)} frames.")
    np.random.shuffle(dataset)
    np.save(DATA_DST, dataset)


# Main
if __name__ == "__main__":
    main()