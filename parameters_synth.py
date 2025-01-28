# Import modules
import torch


# Dataset parameters
dataset_parameters = {
    "data_path": "data/preprocessed_data/ir.npy",
    "test_path": "/home/visione/Projects/BlenderScenarios/Asteroid/Dataset/Orbit_V_256/IR"
}


# General setup
cfg_parameters = {
    "resolution": 256,
    "channels": 1,
    "device": "cuda:1"
}


# Parameter dictionary for rays generation
rays_parameters = {
    "H": cfg_parameters["resolution"],
    "W": cfg_parameters["resolution"],
    "CH": cfg_parameters["channels"],
}


# Parameter dictionary for sampler
SCENE = 3
sampler_parameters = {
    "n_ray_samples": 128,
    "near": 3.,
    "far": 13. 
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


# Parameter dictionary for positional embedder
posenc_parameters = {
    "n_freq": 10
}


# Parameter dictionary for sh embedder
sh_parameters = {
    "input_dim": 3,
    "degree": 4,
    "out_dim": 4**2, # Always square of 'degree' 
}


# Parameter dictionary for SmallNeRF 
nerf_parameters = {
    "n_layers": 2,
    "hidden_dim": 64,
    "geo_feat_dim": 15,
    "n_layers_color": 2,
    "hidden_dim_color": 64,
    #"input_ch": posenc_parameters["n_freq"] * 6 + 3,
    "input_ch": hash_parameters["n_levels"] * hash_parameters["n_features_per_level"],          
    #"input_ch_views": posenc_parameters["n_freq"] * 6 + 3,
    "input_ch_views": sh_parameters["out_dim"],
    "out_ch": cfg_parameters["channels"],
}


# Parameter dictionary for training 
training_parameters = {
    "training_batch": 32*32*16,
    "epochs": 10,
    "lr": 0.05,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "tv_loss_weight": 1e-6,
    "stop_tv_epoch": 10,
    "sparsity_loss_weight": 1e-8,
    "decay_rate": 0.1,
    "decay_steps": 5,
    "verbose": True
}
