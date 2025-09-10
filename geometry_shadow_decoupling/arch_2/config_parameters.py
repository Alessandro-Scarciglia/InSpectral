# Import modules
import torch


# Dataset parameters
dataset_parameters = {
    "data_path": "geometry_shadow_decoupling/arch_2/data/postprocessed_dataset/colour_256_XY_12_1d5_training_3_arch_2.npy",
    "test_path": "/home/vision/Desktop/Datasets/CloudSat_NeRF_Datasets/colour_256_XY_12_1d5_training"
}


# General setup
cfg_parameters = {
    "resolution": 256,
    "channels": 3,
    "device": "cuda"
}


# Parameter dictionary for rays generation
rays_parameters = {
    "height": cfg_parameters["resolution"],
    "width": cfg_parameters["resolution"],
    "num_ch": cfg_parameters["channels"],
}


# Parameter dictionary for sampler
SCENE = 3.25
sampler_parameters = {
    "n_ray_samples": 32,
    "near": 9-1,
    "far": 15+1
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


# Parameter dictionary for SmallNeRF 
nerf_parameters = {
    "n_layers": 2,
    "hidden_dim": 128,
    "geo_feat_dim": 16,
    "n_layers_light": 2,
    "hidden_dim_light": 128,
    "n_layers_color": 2,
    "hidden_dim_color": 256,
    "input_ch": hash_parameters["n_levels"] * hash_parameters["n_features_per_level"],
    "input_ch_views": sh_parameters["degree"] ** 2,
    "out_ch": cfg_parameters["channels"] + 1,
}


# Parameter dictionary for training 
training_parameters = {
    "training_batch": 32*32*8,
    "epochs": 15,
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "tv_loss_weight": 1e-6,
    "stop_tv_epoch": 10,
    "start_seg_epoch": 0,
    "sparsity_loss_weight": 1e-7,
    "bce_dice_loss_weight": 1e-1,
    "bce_loss_weight": 0.5,
    "dice_loss_weight": 0.5,
    "decay_rate": 0.9,
    "decay_steps": 5,
    "verbose": True
}
