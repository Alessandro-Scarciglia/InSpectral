"""
Global hyperparameter configuration file for joint NeRF and Pose Optimization.
Updated to dynamically match the unit-box centered dataset normalization properties.
"""

import torch

# =======================================================================================
# DATASET PARAMETERS
# =======================================================================================
dataset_parameters = {
    "data_path": "geometry_shadow_decoupling/arch_4/data/postprocessed_dataset/Landsat_4_nominal_masked_arch_4_res512_min23.43_max40.00.npy",
    "test_path": "/home/vision/Desktop/Datasets/Landsat"
}

# =======================================================================================
# GENERAL SETUP
# =======================================================================================
cfg_parameters = {
    "resolution": 512,
    "channels": 3,
    "device": "cuda:0"
}

# =======================================================================================
# RAYS GENERATION CONFIGURATION
# =======================================================================================
rays_parameters = {
    "height": cfg_parameters["resolution"],
    "width": cfg_parameters["resolution"],
    "num_ch": cfg_parameters["channels"],
}

# =======================================================================================
# SAMPLING & GEOMETRIC BOUNDS (UNIT BOX CONSTRAINED)
# =======================================================================================
SCALE = 1.0
SCENE = 0.25

sampler_parameters = {
    "n_ray_samples": 64,
    "near": 0.4,
    "far": 1.6,
}

# =======================================================================================
# HASH EMBEDDER PARAMETERS
# =======================================================================================
hash_parameters = {
    "bbox": (torch.tensor([-SCENE, -SCENE, -SCENE]),
             torch.tensor([SCENE, SCENE, SCENE])),
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "low_res": 16,
    "high_res": 512,
    "min_visible_levels": 8,  # Active levels at epoch 0
} 

# =======================================================================================
# SPHERICAL HARMONICS EMBEDDER PARAMETERS
# =======================================================================================
sh_parameters = {
    "input_dim": 3,
    "degree": 4
}

# =======================================================================================
# APPEARANCE EMBEDDER PARAMETERS
# =======================================================================================
app_parameters = {
    "embeddings_num": 90,
    "embeddings_dim": 4
}

# =======================================================================================
# CORE BACKBONE (SMALLNERF) ARCHITECTURE
# =======================================================================================
nerf_parameters = {
    "n_layers": 2,
    "hidden_dim": 128,
    "geo_feat_dim": 16,
    "n_layers_color": 2,
    "hidden_dim_color": 256,
    "input_ch": hash_parameters["n_levels"] * hash_parameters["n_features_per_level"],
    "input_ch_views": sh_parameters["degree"] ** 2,
    "out_ch": cfg_parameters["channels"],
}

# =======================================================================================
# LEARNABLE CAMERA POSE RESIDUALS CONFIGURATION
# =======================================================================================
res_model_parameters = {
    "embeddings_num": 90,
    "embeddings_dim": 6,
    "lin_range": 0.25 / SCALE,            
    "ang_range": 10 * (torch.pi / 180)    
}

# =======================================================================================
# MULTI-TASK TRAINING HYPERPARAMETERS
# =======================================================================================
training_parameters = {
    "training_batch": 32*32*8,
    "epochs": 10,
    "lr": 1e-4,
    "res_lr": 1e-2,
    "app_lr": 1e-4,
    "emb_lr": 1e-3,
    "start_cam_opt": 40,     
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "photometric_loss_weight": 1e-2,
    "tv_loss_weight": 1e-5,
    "stop_tv_epoch": 30,
    "start_seg_epoch": 0,
    "sparsity_loss_weight": 1e-4,
    "bce_dice_loss_weight": 1e-1,
    "bce_loss_weight": 0.5,
    "dice_loss_weight": 0.5,
    "similarity_loss_weight": 1e-2,
    "emb_norm_loss_weight": 1e-1,
    "decay_rate": 0.99,
    "decay_steps": 2,
    "verbose": True,
    "pretrained": False,
    "checkpoint_nerf": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-02-13_12-15-08/epoch_11/chkpt_nerf.pt",
    "checkpoint_res": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-02-13_12-15-08/epoch_11/chkpt_res.pt"
}