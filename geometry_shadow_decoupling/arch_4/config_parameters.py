# Import modules
import torch


# Dataset parameters
dataset_parameters = {
    "data_path": "geometry_shadow_decoupling/arch_4/data/postprocessed_dataset/colour_512_radialboost_m30_training_4_masked_arch_4_noise_baseline.npy",
    "test_path": "/home/vision/Desktop/Datasets/Landsat8/colour_512_radialboost_m30_training"
}


# General setup
cfg_parameters = {
    "resolution": 512,
    "channels": 3,
    "device": "cuda:0"
}


# Parameter dictionary for rays generation
rays_parameters = {
    "height": cfg_parameters["resolution"],
    "width": cfg_parameters["resolution"],
    "num_ch": cfg_parameters["channels"],
}


# Parameter dictionary for sampler
SCALE = 7.0
SCENE = 7.0 / SCALE
sampler_parameters = {
    "n_ray_samples": 64,
    "radius": SCENE * 1.73,
}


# Parameter dictionary for hash embedder
hash_parameters = {
    "bbox": (torch.tensor([-SCENE, -SCENE, -SCENE]),
             torch.tensor([SCENE, SCENE, SCENE])),
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "low_res": 16,
    "high_res": 512,
} 


# Parameter dictionary for sh embedder
sh_parameters = {
    "input_dim": 3,
    "degree": 4
}

# Parameters for appearance embedder
app_parameters = {
    "embeddings_num": 90,
    "embeddings_dim": 4
}


# Parameter dictionary for SmallNeRF 
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

# Residual MLP parameters
res_model_parameters = {
    "n_layers": 2,
    "hidden_dim": 128,
    "input_ch": 6,
    "output_ch": 6,
    "lin_range": 0.5 / SCALE,
    "ang_range": 0.03 * (torch.pi/180)
}


# Parameter dictionary for training 
training_parameters = {
    "training_batch": 32*32*8,
    "epochs": 10,
    "lr": 1e-3,
    "res_lr": 1e-2,
    "app_lr": 1e-4,
    "emb_lr": 1e-2,
    "start_cam_opt": 3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "tv_loss_weight": 1e-8,
    "stop_tv_epoch": 20,
    "start_seg_epoch": 0,
    "sparsity_loss_weight": 1e-7,
    "bce_dice_loss_weight": 5e-1,
    "bce_loss_weight": 0.5,
    "dice_loss_weight": 0.5,
    "similarity_loss_weight": 1e-2,
    "emb_norm_loss_weight": 1e-1,
    "decay_rate": 0.99,
    "decay_steps": 3,
    "verbose": True,
    "pretrained": False,
    "checkpoint_nerf": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-02-13_12-15-08/epoch_11/chkpt_nerf.pt",
    "checkpoint_nerf": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/results/folder_2026-02-13_12-15-08/epoch_11/chkpt_res.pt"
}