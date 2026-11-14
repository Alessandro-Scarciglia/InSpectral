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
    # Stride to subsample test frames during evaluation.
    # e.g. 36 means evaluate 1 frame every 36.
    "test_stride": 36*2,

    # Execution parameters
    "chunk_size": 4096,
    "interactive_display": False,

    # File paths
    "json_filename": "transforms_train.json",
    "checkpoint_path": (
        "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-08-14_10-41-06_quality8_free_app/epoch_29/chkpt_nerf.pt"
    ),
}


def main():

    # ===================================================================================
    # DEVICE CONFIGURATION
    # ===================================================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Using device: {device}")

    if device.type == "cuda":
        print(
            f"[INFO] CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ===================================================================================
    # INITIALIZE NEURAL ARCHITECTURE
    # ===================================================================================

    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **sh_parameters,
        **app_parameters,
        **nerf_parameters,
        device=device,
    )


    # ===================================================================================
    # LOAD CHECKPOINT
    # ===================================================================================

    print(
        f"[INFO] Loading model checkpoint from: "
        f"{CONFIG['checkpoint_path']}"
    )

    if not os.path.exists(CONFIG["checkpoint_path"]):
        raise FileNotFoundError(
            "Checkpoint not found at specified path: "
            f"{CONFIG['checkpoint_path']}"
        )

    checkpoint = torch.load(
        CONFIG["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("[INFO] Model loaded successfully.")


    # ===================================================================================
    # DATASET METADATA LOADING
    # ===================================================================================

    json_path = os.path.join(
        dataset_parameters["test_path"],
        CONFIG["json_filename"],
    )

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dataset metadata file not found: {json_path}"
        )

    with open(json_path, "r") as train_fopen:
        test_df = json.load(train_fopen)

    test_samples = test_df["frames"]


    # ===================================================================================
    # REPRODUCE TRAINING CAMERA NORMALIZATION
    # ===================================================================================
    #
    # IMPORTANT:
    #
    # The postprocessed training dataset was generated using:
    #
    #     samples = all_samples[::4]
    #
    #     max_dist = max(||camera_position||)
    #
    #     scale_factor = 1 / max_dist
    #
    #     c2w[:3, 3] *= scale_factor
    #
    # We must reproduce exactly the same transformation during evaluation.
    #
    # Do NOT use an arbitrary scale such as 12.0 here.
    #

    normalization_stride = 4

    normalization_samples = test_samples[
        ::normalization_stride
    ]

    if len(normalization_samples) == 0:
        raise ValueError(
            "No samples available to compute camera normalization."
        )

    c2w_matrices = np.array(
        [
            sample["transform_matrix"]
            for sample in normalization_samples
        ],
        dtype=np.float32,
    )

    original_camera_centers = c2w_matrices[:, :3, 3]

    camera_distances = np.linalg.norm(
        original_camera_centers,
        axis=1,
    )

    min_dist = float(
        camera_distances.min()
    )

    max_dist = float(
        camera_distances.max()
    )

    scale_factor = 1.0 / (
        max_dist + 1e-10
    )

    print("\n" + "=" * 70)
    print("CAMERA NORMALIZATION")
    print("=" * 70)
    print(
        f"[INFO] Normalization stride : "
        f"{normalization_stride}"
    )
    print(
        f"[INFO] Number of frames used: "
        f"{len(normalization_samples)}"
    )
    print(
        f"[INFO] Original min distance: "
        f"{min_dist:.8f}"
    )
    print(
        f"[INFO] Original max distance: "
        f"{max_dist:.8f}"
    )
    print(
        f"[INFO] Scale factor         : "
        f"{scale_factor:.10f}"
    )
    print(
        f"[INFO] Normalized max radius: "
        f"{max_dist * scale_factor:.8f}"
    )
    print("=" * 70 + "\n")


    # ===================================================================================
    # CAMERA INTRINSICS
    # ===================================================================================

    intrinsic_matrix = calculate_intrinsic_matrix(
        fov=test_df["camera_angle_x"],
        resolution=(
            cfg_parameters["resolution"],
            cfg_parameters["resolution"],
        ),
    )


    # ===================================================================================
    # INITIALIZE RAY GENERATOR
    # ===================================================================================

    raygen = RaysGeneratorSynth(
        **rays_parameters,
        intrinsic_matrix=intrinsic_matrix,
    )

    raygen = raygen.to(device)

    print("[INFO] Ray generator initialized.")

    if hasattr(raygen, "dirs"):
        print(
            f"[INFO] Ray generator dirs device: "
            f"{raygen.dirs.device}"
        )


    # ===================================================================================
    # METRICS STORAGE
    # ===================================================================================

    test_full_psnr_set = []
    test_obj_psnr_set = []
    test_bkg_psnr_set = []

    test_full_mae_set = []
    test_obj_mae_set = []
    test_bkg_mae_set = []


    # ===================================================================================
    # METRICS EVALUATION LOOP
    # ===================================================================================

    print(
        "[INFO] Starting NeRF model quantitative validation..."
    )

    with torch.no_grad():

        for m_iter, test_sample in tqdm(
            enumerate(test_samples),
            total=len(test_samples),
        ):

            # ---------------------------------------------------------------------------
            # DATASET FILTERING
            # ---------------------------------------------------------------------------

            if m_iter % CONFIG["test_stride"] != 0:
                continue


            # ---------------------------------------------------------------------------
            # CAMERA-TO-WORLD MATRIX
            # ---------------------------------------------------------------------------

            test_c2w = torch.tensor(
                test_sample["transform_matrix"],
                dtype=torch.float32,
                device=device,
            )


            # ---------------------------------------------------------------------------
            # APPLY THE SAME CAMERA NORMALIZATION USED DURING TRAINING
            # ---------------------------------------------------------------------------
            #
            # Training preprocessing:
            #
            #     c2w[:3, 3] *= scale_factor
            #
            # Therefore evaluation must perform exactly the same operation
            # before generating rays.
            #

            test_c2w[:3, 3] *= scale_factor


            # ---------------------------------------------------------------------------
            # GENERATE RAYS
            # ---------------------------------------------------------------------------

            test_rays = raygen(
                test_c2w
            ).reshape(-1, 6)


            # ---------------------------------------------------------------------------
            # CREATE APPEARANCE TRACKING INDICES
            # ---------------------------------------------------------------------------

            num_rays = test_rays.shape[0]

            test_idx = (
                torch.ones(
                    (num_rays, 1),
                    dtype=torch.float32,
                    device=device,
                )
                * (m_iter / 4)
            )


            # ---------------------------------------------------------------------------
            # LOAD TARGET RGB IMAGE
            # ---------------------------------------------------------------------------

            target_image_path = os.path.join(
                dataset_parameters["test_path"],
                test_sample["file_path"],
            )

            target_image = cv2.imread(
                target_image_path
            )

            if target_image is None:
                raise FileNotFoundError(
                    f"Unable to load target image: "
                    f"{target_image_path}"
                )

            target_image = cv2.resize(
                target_image,
                (
                    cfg_parameters["resolution"],
                    cfg_parameters["resolution"],
                ),
            )

            target_image = target_image / 255.0

            target_image = torch.tensor(
                target_image,
                dtype=torch.float32,
                device=device,
            ).reshape(
                -1,
                cfg_parameters["channels"],
            )


            # ---------------------------------------------------------------------------
            # LOAD TARGET DEPTH
            # ---------------------------------------------------------------------------

            target_depth_path = os.path.join(
                dataset_parameters["test_path"],
                test_sample["depth_path"],
            )

            target_depth = cv2.imread(
                target_depth_path,
                0,
            )

            if target_depth is None:
                raise FileNotFoundError(
                    f"Unable to load target depth: "
                    f"{target_depth_path}"
                )

            target_depth = cv2.resize(
                target_depth,
                (
                    cfg_parameters["resolution"],
                    cfg_parameters["resolution"],
                ),
            )

            target_depth = target_depth / 255.0

            target_depth = torch.tensor(
                target_depth,
                dtype=torch.float32,
                device=device,
            ).reshape(
                -1,
                1,
            )


            # ---------------------------------------------------------------------------
            # LOAD OBJECT MASK
            # ---------------------------------------------------------------------------

            target_mask_path = os.path.join(
                dataset_parameters["test_path"],
                test_sample["mask_path"],
            )

            obj_mask = cv2.imread(
                target_mask_path,
                0,
            )

            if obj_mask is None:
                raise FileNotFoundError(
                    f"Unable to load target mask: "
                    f"{target_mask_path}"
                )

            obj_mask = cv2.resize(
                obj_mask,
                (
                    cfg_parameters["resolution"],
                    cfg_parameters["resolution"],
                ),
            )

            obj_mask = obj_mask / 255.0

            obj_mask = torch.tensor(
                obj_mask,
                dtype=torch.float32,
                device=device,
            ).reshape(
                -1,
                1,
            )


            # ===================================================================================
            # VOLUMETRIC RENDERING
            # ===================================================================================

            test_rgb_list = []
            test_depth_list = []

            # Split rays into chunks to reduce VRAM usage.
            rays_chunks = torch.split(
                test_rays,
                CONFIG["chunk_size"],
                dim=0,
            )

            idx_chunks = torch.split(
                test_idx,
                CONFIG["chunk_size"],
                dim=0,
            )


            for r_chunk, i_chunk in zip(
                rays_chunks,
                idx_chunks,
            ):

                rgb_chunk, depth_chunk, _, _ = model(
                    r_chunk,
                    i_chunk,
                )

                test_rgb_list.append(
                    rgb_chunk
                )

                test_depth_list.append(
                    depth_chunk
                )


            # ---------------------------------------------------------------------------
            # REASSEMBLE FRAME
            # ---------------------------------------------------------------------------

            test_rgb = torch.cat(
                test_rgb_list,
                dim=0,
            )

            test_depth = torch.cat(
                test_depth_list,
                dim=0,
            )


            # ===================================================================================
            # IMAGE METRICS
            # ===================================================================================

            test_full_psnr, test_obj_psnr, test_bkg_psnr = compute_psnr(
                img1=test_rgb,
                img2=target_image,
                mask=obj_mask.repeat(
                    1,
                    cfg_parameters["channels"],
                ),
            )


            # ===================================================================================
            # DEPTH METRICS
            # ===================================================================================

            test_full_mae, test_obj_mae, test_bkg_mae = compute_mae(
                img1=test_depth.unsqueeze(-1),
                img2=target_depth,
                mask=obj_mask,
            )


            # ===================================================================================
            # STORE NUMERIC RESULTS
            # ===================================================================================

            test_full_psnr_set.append(
                test_full_psnr.item()
                if hasattr(test_full_psnr, "item")
                else test_full_psnr
            )

            test_obj_psnr_set.append(
                test_obj_psnr.item()
                if hasattr(test_obj_psnr, "item")
                else test_obj_psnr
            )

            test_bkg_psnr_set.append(
                test_bkg_psnr.item()
                if hasattr(test_bkg_psnr, "item")
                else test_bkg_psnr
            )

            test_full_mae_set.append(
                test_full_mae.item()
                if hasattr(test_full_mae, "item")
                else test_full_mae
            )

            test_obj_mae_set.append(
                test_obj_mae.item()
                if hasattr(test_obj_mae, "item")
                else test_obj_mae
            )

            test_bkg_mae_set.append(
                test_bkg_mae.item()
                if hasattr(test_bkg_mae, "item")
                else test_bkg_mae
            )


            # ===================================================================================
            # INTERACTIVE FRAME VISUALIZATION
            # ===================================================================================

            if CONFIG["interactive_display"]:

                frame_np = (
                    test_depth
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(
                        cfg_parameters["resolution"],
                        cfg_parameters["resolution"],
                        1,
                    )
                )

                sleep(0.1)

                cv2.imshow(
                    "Evaluation Frame - Press 'q' to exit, any key to continue",
                    frame_np,
                )

                key = cv2.waitKey(0)

                if key == ord("q"):

                    print(
                        "[INFO] Evaluation terminated by user."
                    )

                    cv2.destroyAllWindows()

                    return


    # ===================================================================================
    # CLOSE WINDOWS
    # ===================================================================================

    if CONFIG["interactive_display"]:
        cv2.destroyAllWindows()


    # ===================================================================================
    # FINAL EVALUATION REPORT
    # ===================================================================================

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS REPORT")
    print("=" * 50)

    print(
        f"Mean Full PSNR: "
        f"{np.mean(test_full_psnr_set):.4f}"
    )

    print(
        f"Mean Object PSNR: "
        f"{np.mean(test_obj_psnr_set):.4f}"
    )

    print(
        f"Mean Background PSNR: "
        f"{np.mean(test_bkg_psnr_set):.4f}"
    )

    print("-" * 50)

    print(
        f"Mean Full MAE:  "
        f"{np.mean(test_full_mae_set):.4f}"
    )

    print(
        f"Mean Object MAE:  "
        f"{np.mean(test_obj_mae_set):.4f}"
    )

    print(
        f"Mean Background MAE:  "
        f"{np.mean(test_bkg_mae_set):.4f}"
    )

    print("=" * 50)


# =======================================================================================
# ENTRY POINT
# =======================================================================================

if __name__ == "__main__":
    main()