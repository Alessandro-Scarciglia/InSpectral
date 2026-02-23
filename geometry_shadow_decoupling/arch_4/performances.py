"""
This script is meant to assess performances of a NeRF trained model (PSNR, MAE)
"""

# Import standard modules
import torch
import json
from tqdm import tqdm
import os
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from data.rays_generator import RaysGeneratorSynth
from model.rendering import NeuralRenderer
from data.generate_postprocessed_dataset import calculate_intrinsic_matrix
from model.metrics import *

# Parameters
from config_parameters import *


# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **sh_parameters,
        **app_parameters,
        **nerf_parameters,
        device=cfg_parameters["device"]
    )
model.to(device)

# Load the test dataset
with open(dataset_parameters["test_path"] + "/transforms.json", "r") as train_fopen:
    test_df = json.load(train_fopen)
    test_samples = test_df["frames"]

# Compute camera intrinsics for training
intrinsic_matrix = calculate_intrinsic_matrix(
    fov=test_df["camera_angle_x"],
    resolution=(cfg_parameters["resolution"], cfg_parameters["resolution"])
)

# Define the rays generator object
raygen = RaysGeneratorSynth(
    **rays_parameters,
    intrinsic_matrix=intrinsic_matrix
    )

# Define epochs metrics
mae_epochs, psnr_epochs = list(), list()

# Load models for each checkpoint 
# for epoch in range(15):

# Load specific checkpoint of the model
checkpoint_path = f"/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/test_128_4/epoch_14/chkpt.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Define metrics
test_full_psnr_set, test_obj_psnr_set, test_bkg_psnr_set = list(), list(), list()
test_full_mae_set, test_obj_mae_set, test_bkg_mae_set = list(), list(), list()
cnt = 0

with torch.no_grad():
    for m_iter, test_sample in tqdm(enumerate(test_samples)):

        # Test 120 views out of 360
        # if m_iter not in [80]:
        #     continue

        if m_iter % 36:
            continue
        
        # for cnt in range(15):

        # Generate rays and appearance idx
        test_c2w = torch.tensor(test_sample["transform_matrix"])
        test_rays = raygen(test_c2w).reshape(-1, 6)
        test_rays[:, :3] /= 10.
        test_rays = test_rays.to(cfg_parameters["device"])
        test_idx = torch.ones((cfg_parameters["resolution"] * cfg_parameters["resolution"], 1)) * (m_iter / 3)

        # Retrieve labels
        target_image = cv2.imread(os.path.join(dataset_parameters["test_path"], test_sample["file_path"]))
        target_image = cv2.resize(target_image, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
        target_image = torch.tensor(target_image).reshape(-1, cfg_parameters["channels"]).to(cfg_parameters["device"])

        target_depth = cv2.imread(os.path.join(dataset_parameters["test_path"], test_sample["depth_path"]), 0)
        target_depth = cv2.resize(target_depth, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
        target_depth = torch.tensor(target_depth).reshape(-1, 1).to(cfg_parameters["device"])

        obj_mask = cv2.imread(os.path.join(dataset_parameters["test_path"], test_sample["mask_path"]), 0)
        obj_mask = cv2.resize(obj_mask, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
        obj_mask = torch.tensor(obj_mask).reshape(-1, 1).to(cfg_parameters["device"])

        # Estimate rendering: split the inference in N steps in chunks of 256x256.
        test_rgb, test_depth = list(), list()
        
        for n in range(((cfg_parameters["resolution"] // 64) ** 2)):
            test_rgb_n, test_depth_n, _, _ = model(
                test_rays[n * (64**2): (n+1) * (64**2), :],
                test_idx[n * (64**2): (n+1) * (64**2), :]
            )

            test_rgb.append(test_rgb_n)
            test_depth.append(test_depth_n)

        # Concatenate chunks
        test_rgb = torch.cat(test_rgb, dim=0).to(cfg_parameters["device"])
        test_depth = torch.cat(test_depth, dim=0).to(cfg_parameters["device"])

        # Evaluate test PSNR and MAE
        test_full_psnr, test_obj_psnr, test_bkg_psnr = compute_psnr(img1=test_rgb, img2=target_image, mask=obj_mask.repeat(1, cfg_parameters["channels"]))
        test_full_mae, test_obj_mae, test_bkg_mae = compute_mae(img1=test_depth.unsqueeze(-1), img2=target_depth, mask=obj_mask)
    
        # Collect test batch metrics about PSNR and MAE
        test_full_psnr_set.append(test_full_psnr)
        test_obj_psnr_set.append(test_obj_psnr)
        test_bkg_psnr_set.append(test_bkg_psnr)

        # Reconstruct images and store them
        frame = test_rgb.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])
        depth = test_depth.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], 1)

        sleep(0.1)
        cv2.imshow("Frame", depth)

        key = cv2.waitKey(0)
        if key == ord("q"):
            exit()

# Resume test for the epoch
psnr_epochs.append(np.mean(test_full_psnr_set))

# # Plot
# plt.figure(0)
# plt.plot(mae_epochs)

# plt.figure(1)
# plt.plot(psnr_epochs)

# plt.show()

print(f"PSNR: {test_full_psnr_set}")
