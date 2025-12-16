# Import modules
import torch


# Dataset parameters
dataset_parameters = {
    "data_path": "geometry_shadow_decoupling/arch_4/data/postprocessed_dataset/colour_256_radialboost_20_training_3_masked_arch_4.npy",
    "test_path": "/home/vision/Desktop/Datasets/Landsat8/colour_256_radialboost_20_training"
}


# General setup
cfg_parameters = {
    "resolution": 256,
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
SCALE = 1.0
SCENE = 8 / SCALE
sampler_parameters = {
    "n_ray_samples": 64,
    "near": 5 / SCALE,
    "far": 40 / SCALE
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
    "embeddings_num": 120,
    "embeddings_dim": 8
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


# Parameter dictionary for training 
training_parameters = {
    "training_batch": 32*32*16,
    "epochs": 10,
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "tv_loss_weight": 1e-8,
    "stop_tv_epoch": 20,
    "start_seg_epoch": 0,
    "sparsity_loss_weight": 1e-7,
    "bce_dice_loss_weight": 1e-1,
    "bce_loss_weight": 0.5,
    "dice_loss_weight": 0.5,
    "decay_rate": 0.9,
    "decay_steps": 5,
    "verbose": True
}