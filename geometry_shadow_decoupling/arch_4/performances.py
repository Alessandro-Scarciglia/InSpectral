"""
This script assesses the performance of a trained NeRF model by computing evaluation 
metrics (PSNR, MAE) on test samples and optionally displaying the rendered frames.
"""

import os
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

# Import custom modules
from data.rays_generator import RaysGeneratorSynth
from model.rendering import NeuralRenderer
from data.generate_postprocessed_dataset import calculate_intrinsic_matrix
from model.metrics import *

# Parameters
from config_parameters import *

# =======================================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =======================================================================================
CONFIG = {
    # Stride to subsample test frames (e.g., 36 means evaluate 1 frame every 36)
    "test_stride": 36,
    
    # Scale factor for normalization
    "scale_factor": 12.0,
    
    # Execution parameters
    "chunk_size": 4096,  # 64x64 chunks = 4096 pixels
    "interactive_display": True,
    
    # File paths
    "json_filename": "transforms.json",
    "checkpoint_path": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-03-02_12-03-09/epoch_14/chkpt.pt"
}


def main():
    # Define execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------------------------------------------------------------
    # INITIALIZE NEURAL ARCHITECTURE
    # -----------------------------------------------------------------------------------
    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **sh_parameters,
        **app_parameters,
        **nerf_parameters,
        device=cfg_parameters["device"]
    )
    
    # Load specific checkpoint of the model
    print(f"[INFO] Loading model checkpoint from: {CONFIG['checkpoint_path']}")
    if not os.path.exists(CONFIG["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found at specified path: {CONFIG['checkpoint_path']}")
        
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # -----------------------------------------------------------------------------------
    # DATASET METADATA LOADING & RAY SETUP
    # -----------------------------------------------------------------------------------
    json_path = os.path.join(dataset_parameters["test_path"], CONFIG["json_filename"])
    with open(json_path, "r") as train_fopen:
        test_df = json.load(train_fopen)
        test_samples = test_df["frames"]

    # Compute camera intrinsics
    intrinsic_matrix = calculate_intrinsic_matrix(
        fov=test_df["camera_angle_x"],
        resolution=(cfg_parameters["resolution"], cfg_parameters["resolution"])
    )

    # Initialize the rays generator object
    raygen = RaysGeneratorSynth(**rays_parameters, intrinsic_matrix=intrinsic_matrix)

    # -----------------------------------------------------------------------------------
    # METRICS EVALUATION LOOP
    # -----------------------------------------------------------------------------------
    test_full_psnr_set, test_obj_psnr_set, test_bkg_psnr_set = [], [], []
    test_full_mae_set, test_obj_mae_set, test_bkg_mae_set = [], [], []

    print("[INFO] Starting NeRF model quantitative validation...")
    
    with torch.no_grad():
        for m_iter, test_sample in tqdm(enumerate(test_samples), total=len(test_samples)):

            # Apply dataset filtering using the configured stride
            if m_iter % CONFIG["test_stride"] != 0:
                continue
            
            # Generate target rays from the view matrix and scale coordinates
            test_c2w = torch.tensor(test_sample["transform_matrix"], dtype=torch.float32, device=device)
            test_rays = raygen(test_c2w).reshape(-1, 6)
            test_rays[:, :3] /= CONFIG["scale_factor"]
            
            # Create appearance matching tracking indices
            num_rays = test_rays.shape[0]
            test_idx = torch.ones((num_rays, 1), dtype=torch.float32, device=device) * (m_iter / 4)

            # Load and preprocess Ground Truth reference annotations directly on device
            target_image_path = os.path.join(dataset_parameters["test_path"], test_sample["file_path"])
            target_image = cv2.imread(target_image_path)
            target_image = cv2.resize(target_image, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255.0
            target_image = torch.tensor(target_image, dtype=torch.float32, device=device).reshape(-1, cfg_parameters["channels"])

            target_depth_path = os.path.join(dataset_parameters["test_path"], test_sample["depth_path"])
            target_depth = cv2.imread(target_depth_path, 0)
            target_depth = cv2.resize(target_depth, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255.0
            target_depth = torch.tensor(target_depth, dtype=torch.float32, device=device).reshape(-1, 1)

            target_mask_path = os.path.join(dataset_parameters["test_path"], test_sample["mask_path"])
            obj_mask = cv2.imread(target_mask_path, 0)
            obj_mask = cv2.resize(obj_mask, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255.0
            obj_mask = torch.tensor(obj_mask, dtype=torch.float32, device=device).reshape(-1, 1)

            # Volumetric rendering split in chunks to maximize throughput and save VRAM
            test_rgb_list, test_depth_list = [], []
            
            rays_chunks = torch.split(test_rays, CONFIG["chunk_size"], dim=0)
            idx_chunks = torch.split(test_idx, CONFIG["chunk_size"], dim=0)

            for r_chunk, i_chunk in zip(rays_chunks, idx_chunks):
                rgb_chunk, depth_chunk, _, _ = model(r_chunk, i_chunk)
                test_rgb_list.append(rgb_chunk)
                test_depth_list.append(depth_chunk)

            # Assemble reconstructed frame slices
            test_rgb = torch.cat(test_rgb_list, dim=0)
            test_depth = torch.cat(test_depth_list, dim=0)

            # Quantify rendering errors
            test_full_psnr, test_obj_psnr, test_bkg_psnr = compute_psnr(
                img1=test_rgb, img2=target_image, mask=obj_mask.repeat(1, cfg_parameters["channels"])
            )
            test_full_mae, test_obj_mae, test_bkg_mae = compute_mae(
                img1=test_depth.unsqueeze(-1), img2=target_depth, mask=obj_mask
            )
        
            # Safely store numeric evaluations (handling float or Tensor instances)
            test_full_psnr_set.append(test_full_psnr.item() if hasattr(test_full_psnr, "item") else test_full_psnr)
            test_obj_psnr_set.append(test_obj_psnr.item() if hasattr(test_obj_psnr, "item") else test_obj_psnr)
            test_bkg_psnr_set.append(test_bkg_psnr.item() if hasattr(test_bkg_psnr, "item") else test_bkg_psnr)
            
            test_full_mae_set.append(test_full_mae.item() if hasattr(test_full_mae, "item") else test_full_mae)
            test_obj_mae_set.append(test_obj_mae.item() if hasattr(test_obj_mae, "item") else test_obj_mae)
            test_bkg_mae_set.append(test_bkg_mae.item() if hasattr(test_bkg_mae, "item") else test_bkg_mae)

            # -----------------------------------------------------------------------------------
            # INTERACTIVE FRAME VISUALIZATION
            # -----------------------------------------------------------------------------------
            if CONFIG["interactive_display"]:
                frame_np = test_rgb.cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])
                
                sleep(0.1)
                cv2.imshow("Evaluation Frame - Press 'q' to exit, any key to continue", frame_np)

                key = cv2.waitKey(0)
                if key == ord("q"):
                    print("[INFO] Evaluation terminated by user.")
                    cv2.destroyAllWindows()
                    return

    if CONFIG["interactive_display"]:
        cv2.destroyAllWindows()

    # Final execution logs summary
    print("\n" + "="*50 + "\nFINAL EVALUATION METRICS REPORT\n" + "="*50)
    print(f"Mean Full PSNR: {np.mean(test_full_psnr_set):.4f}")
    print(f"Mean Object PSNR: {np.mean(test_obj_psnr_set):.4f}")
    print(f"Mean Background PSNR: {np.mean(test_bkg_psnr_set):.4f}")
    print("-"*50)
    print(f"Mean Full MAE:  {np.mean(test_full_mae_set):.4f}")
    print(f"Mean Object MAE:  {np.mean(test_obj_mae_set):.4f}")
    print(f"Mean Background MAE:  {np.mean(test_bkg_mae_set):.4f}")
    print("="*50)


if __name__ == "__main__":
    main()