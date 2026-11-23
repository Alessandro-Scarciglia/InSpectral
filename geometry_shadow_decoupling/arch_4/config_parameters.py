"""
Global hyperparameter configuration file for joint NeRF and Pose Optimization.
Updated to dynamically match the unit-box centered dataset normalization properties,
and incorporate the tunable bimodal driving loss parameters.
"""

import torch

# =======================================================================================
# DATASET PARAMETERS
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - data_path / test_path: Fixed paths pointing to your preprocessed .npy matrices and raw datasets.
# =======================================================================================
dataset_parameters = {
    "data_path": "geometry_shadow_decoupling/arch_4/data/postprocessed_dataset/Landsat8_v4_4_nominal_images_mask_sam2_grounded_bi_arch_4_res512_min22.88_max40.72_rgb_radial0.030_rhor0.90_point0.200_rhop0.90.npy",
    "test_path": "/home/vision/Desktop/Datasets/Landsat8_v4/",
    "mode": "rgb"
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
    "channels": 1 if dataset_parameters["mode"] == "ir" else 3,
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
    "out_ch": 1 if dataset_parameters["mode"] == "ir" else 3,                        
    "input_app_dim": app_parameters["embeddings_dim"]
}

# =======================================================================================
# LEARNABLE CAMERA POSE RESIDUALS CONFIGURATION
# ---------------------------------------------------------------------------------------
# Tuning Guide:
# - lin_range / ang_range: Limits bounds for camera optimization (BARF setup). 
#   Too high (↑) destabilizes trajectories; too low (↓) prevents correcting large camera registration errors.
# =======================================================================================
res_model_parameters = {
    "embeddings_num": 90,
    "embeddings_dim": 6,
    "lin_range": 0.1 / SCALE,            
    "ang_range": 0.5 * (torch.pi / 180)
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
    "app_lr": 1e-4,
    "emb_lr": 5e-4,    
    # Pose refinement is opt-in; defaults preserve the original static-camera run.
    "enable_camera_refinement": True,
    "camera_warmup_epochs": 1,
    "camera_update_interval": 2,
    "camera_only": True,
    "camera_pose_reg_weight": 1e-3,
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
    "checkpoint_nerf": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-07-31_10-08-00_segm1e1/epoch_9/chkpt_nerf.pt",
    "checkpoint_res": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-07-31_10-08-00_segm1e1/epoch_9/chkpt_res.pt"
}