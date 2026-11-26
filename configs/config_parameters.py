"""
Global hyperparameter configuration file for joint NeRF and Pose Optimization.
Updated to dynamically match the unit-box centered dataset normalization properties,
and incorporate the tunable bimodal driving loss parameters.
"""

import torch

from data.image_io import channels_for_mode

# =======================================================================================
# MODALITY FLAGS (the two switches to set for each experiment)
# ---------------------------------------------------------------------------------------
# IMAGE_MODE:       which images are rendered and supervised.
#                   "rgb" (3 channels), "ir" (1 channel) or "rgb_ir" (both, 4 channels).
# CORRUPTED_POSES:  True  -> camera poses are noisy: the dataset builder adds the configured
#                            pose noise (nominal trajectories) and training learns per-image
#                            pose corrections.
#                   False -> poses are trusted: no added noise, no pose refinement.
# Both are also read by data/generate_postprocessed_dataset.py, so the dataset and the
# training run always agree. Rebuild the dataset after changing either of them.
# =======================================================================================
IMAGE_MODE = "rgb"
CORRUPTED_POSES = True

# =======================================================================================
# DATASET PARAMETERS
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - data_path / test_path: Fixed paths pointing to your preprocessed .npy matrices and raw datasets.
# =======================================================================================
dataset_parameters = {
    "data_path": "data/postprocessed_dataset/Landsat8_v4_4_nominal_images_mask_sam2_grounded_bi_arch_4_res512_min22.88_max40.72_rgb_radial0.030_rhor0.90_point0.200_rhop0.90.npy",
    "test_path": "/home/vision/Desktop/Datasets/Landsat8_v4/",
    # Set through IMAGE_MODE at the top of this file
    "mode": IMAGE_MODE,
    # Distance in metres encoded as black (0) in the 8-bit depth labels (255 = camera).
    # The predicted depth uses the same black point, converted to scene units with the
    # dataset scale (1 / max camera distance), so prediction and label are comparable.
    "depth_max_meters": 40.0,
    # What the depth labels measure: "z" = depth along the camera's optical axis
    # (planar Z), "ray" = distance from the camera centre along each pixel's ray.
    "depth_type": "z",
}

# =======================================================================================
# GENERAL SETUP
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - resolution: Higher values (↑) increase spatial details but scale memory/computation quadratically.
# - channels: Dictated by data type (e.g., 3 for RGB, higher if working with multispectral/hyperspectral bands).
# =======================================================================================
cfg_parameters = {
    "resolution": 512,
    "channels": channels_for_mode(dataset_parameters["mode"]),
    "device": "cuda"
}

# =======================================================================================
# RAYS GENERATION CONFIGURATION
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - Matches cfg_parameters to map pixel-to-ray rendering ratios. Keep linked to main resolution.
# =======================================================================================
rays_parameters = {
    "height": cfg_parameters["resolution"],
    "width": cfg_parameters["resolution"],
    "num_ch": cfg_parameters["channels"],
}

# =======================================================================================
# SAMPLING & GEOMETRIC BOUNDS (UNIT BOX CONSTRAINED)
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - SCENE: Controls the world bounding radius. Must encapsulate the scaled payload/object.
# - n_ray_samples: Higher values (↑) resolve sharper geometry/edges but slow down training/inference.
# - near / far: Keep tightly bound around your target object scene to avoid wasting ray sampling capacity.
# =======================================================================================
SCALE = 1.0
SCENE = 0.25

sampler_parameters = {
    "n_ray_samples": 64,
    "near": 0.4,
    "far": 0.4,
    "enable_hierarchical_sampling": False,
    "n_importance_samples": 32,
    "importance_eps": 1e-5,
}

# =======================================================================================
# HASH EMBEDDER PARAMETERS
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - n_levels / high_res: Higher (↑) captures finer high-frequency details but may introduce noise/artifacts.
# - log2_hashmap_size: Increase (↑) if experiencing hash collisions (blurry shapes), at the cost of VRAM.
# - min_visible_levels: Higher values (↑) force high frequencies early; lower values (↓) ensure smooth warm-up.
# - level_warmup_start_epoch: Keep only the minimum levels active before this zero-based epoch.
# - level_warmup_steps: Ramp duration after the start epoch; 0 uses half of total training steps.
# =======================================================================================
hash_parameters = {
    "bbox": (torch.tensor([-SCENE, -SCENE, -SCENE]),
             torch.tensor([SCENE, SCENE, SCENE])),
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "low_res": 16,
    "high_res": 512,
    "min_visible_levels": 16,
    "enable_coarse_to_fine": False,
    "level_warmup_start_epoch": 3,
    "level_warmup_steps": 0
} 

# =======================================================================================
# OCCUPANCY GRID (EMPTY-SPACE SKIPPING)
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - enable_occupancy_grid: Skip the MLPs for samples in voxels the grid marks as empty.
#   Saves compute; off by default. Checkpoints trained with it on must be loaded with it on.
# - grid_resolution: Voxels per side inside the hash bounding box (memory grows cubically).
# - occupancy_threshold: Raw density below which a voxel counts as empty. Higher (↑) skips more.
# - occupancy_warmup_steps: Training steps before the first grid update (skipping starts then).
# - occupancy_update_interval: Steps between grid updates during training.
# =======================================================================================
occupancy_parameters = {
    "enable_occupancy_grid": False,
    "grid_resolution": 128,
    "occupancy_threshold": 0.01,
    "occupancy_warmup_steps": 500,
    "occupancy_update_interval": 16,
}

# =======================================================================================
# SPHERICAL HARMONICS EMBEDDER PARAMETERS
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - degree: Higher (↑) models sharper specular reflections and view-dependent effects but risks overfitting.
# =======================================================================================
sh_parameters = {
    "input_dim": 3,
    "degree": 4
}

# =======================================================================================
# APPEARANCE EMBEDDER PARAMETERS
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - embeddings_num: Must match or exceed the total number of unique frames/images in your training set.
# - embeddings_dim: Higher (↑) captures complex lighting/shadow fluctuations across frames but requires more data.
# =======================================================================================
app_parameters = {
    "embeddings_num": 90,
    "embeddings_dim": 16
}

# =======================================================================================
# CORE BACKBONE (SMALLNERF) ARCHITECTURE
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - n_layers / hidden_dim: Higher (↑) increases representation capacity for complex scenes but raises overfitting risks.
# - out_ch: Keep at 4 (RGB + Density/Sigma) to assure direct, native volumetric calculations.
# =======================================================================================
nerf_parameters = {
    "n_layers": 2,
    "hidden_dim": 128,
    "geo_feat_dim": 16,
    "n_layers_color": 2,
    "hidden_dim_color": 128,
    "input_ch": hash_parameters["n_levels"] * hash_parameters["n_features_per_level"],
    "input_ch_views": sh_parameters["degree"] ** 2,
    "out_ch": cfg_parameters["channels"],                        
    "input_app_dim": app_parameters["embeddings_dim"]
}

# =======================================================================================
# LEARNABLE CAMERA POSE RESIDUALS CONFIGURATION
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - lin_range / ang_range: Largest correction per axis (scene units / radians): the raw residuals
#   are clamped to [-1, 1] and multiplied by these. Keep them a few times above the expected pose
#   noise (radial 3% of a <= 1.0 distance; pointing 0.2 deg per axis).
#   Too high (↑) destabilizes trajectories; too low (↓) prevents correcting large camera registration errors.
# =======================================================================================
res_model_parameters = {
    "embeddings_num": 90,
    "embeddings_dim": 6,
    "lin_range": 0.1 / SCALE,            
    "ang_range": 1.0 * (torch.pi / 180)
}

# =======================================================================================
# MULTI-TASK TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - lr / emb_lr / res_lr: Main learning rates. If pose updates fail, increase res_lr (↑) or delay start_cam_opt.
# - start_cam_opt: Epoch to start camera tuning. Keep high enough to let the static NeRF geometry stabilize first.
# - tv_loss_weight: Higher (↑) reduces high-frequency noise and floats, but too high smooths out sharp textures.
# - sparsity_loss_weight: Higher (↑) cleans background artifacts but too high aggressively erases thin object geometry.
# - similarity_loss_weight: High values (↑) enforce smooth, consistent appearance adjustments across different frames.
# - stop_tv_epoch / start_seg_epoch: Temporal anchors to transition training from rough layout to precise masking.
# - driving_loss_softness / gamma: Controls segmentation sharpness. Lower softness (↓) yields sharper mask borders.
# - driving_loss_w_bg / w_fg: Imbalance weights. Increase w_bg (↑) if background noise sneaks into predictions.
# =======================================================================================
training_parameters = {
    "training_batch": 32*32*4,
    "epochs": 15,
    "lr": 1e-4,
    "res_lr": 2e-3,
    # Learning rate of the rotation residuals. A rotation of one raw unit (ang_range) moves the
    # image about as much as a lateral shift of ang_range * distance, which is ~10x less than one
    # raw unit of translation (lin_range), so rotation needs a ~10x higher rate to keep up.
    "res_lr_rot": 2e-2,
    "app_lr": 1e-4,
    "emb_lr": 5e-4,    
    # Pose refinement is opt-in; defaults preserve the original static-camera run.
    # Set through CORRUPTED_POSES at the top of this file
    "enable_camera_refinement": CORRUPTED_POSES,
    "camera_warmup_epochs": 1,
    "camera_update_interval": 2,
    "camera_only": True,
    "camera_pose_reg_weight": 1e-3,
    # Zero-prior on the rotation residuals (None = same as camera_pose_reg_weight). Keep it at 0:
    # the photometric gradient on rotation is ~25x weaker than on translation, so even 1e-3 sets
    # an equilibrium near the input attitude and rotation stops improving (any learning rate).
    "camera_rot_reg_weight": 0.0,
    # Trajectory smoothness prior on the corrected poses, aimed at white (frame-to-frame)
    # pose noise. Needs the dataset pose sidecar and densely sampled frames. 0 disables it.
    "camera_smoothness_weight": 0.0,
    "camera_smoothness_window": 5,
    "mixed_precision": True,
    "distortion_loss_weight": 5e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    
    # Core Reconstruction and Regularization Optimization Weights
    "photometric_loss_weight": 1,
    "tv_loss_weight": 1e-7, 
    "sparsity_loss_weight": 1e-7,
    "similarity_loss_weight": 0,     
    "emb_norm_loss_weight": 1e-1,
    
    # --- TUNABLE BI-MODAL DRIVING LOSS PARAMETERS ---
    "stop_tv_epoch": 10,
    "start_seg_epoch": 0,
    "bce_dice_loss_weight": 5e0,
    "driving_loss_softness": 0,       
    "driving_loss_gamma": 1.2,          
    "driving_loss_w_bg": 4.0,           
    "driving_loss_w_fg": 1.0,           
    "driving_loss_dice_weight": 0.5,    
    
    # Scheduling & Execution Flags
    "decay_rate": 0.99,
    "decay_steps": 3,
    "verbose": True,
    "pretrained": False,
    "checkpoint_nerf": "/home/vision/Desktop/InSpectral/results/folder_2026-07-31_10-08-00_segm1e1/epoch_9/chkpt_nerf.pt",
    "checkpoint_res": "/home/vision/Desktop/InSpectral/results/folder_2026-07-31_10-08-00_segm1e1/epoch_9/chkpt_res.pt"
}