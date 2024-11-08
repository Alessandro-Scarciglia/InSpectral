# Import modules
import torch


# Dataset parameters
dataset_parameters = {
    "data_path": "data/preprocessed_data/rgb_False_res_3_size_256.npy",
    "valid_set_ratio": 0.2
}


# General setup
cfg_parameters = {
    "roll_cfgs": ["roll_0", "roll_120", "roll_240"],
    "resolution": int(dataset_parameters["data_path"].split(".")[0].split("_")[-1]),
    "channels": 1,
    "device": "cuda:0"
}


# Parameter dictionary for rays generation
rays_parameters = {
    "H": cfg_parameters["resolution"],
    "W": cfg_parameters["resolution"],
    "CH": cfg_parameters["channels"],
}


# Parameter dictionary for sampler
SCENE = 1.0
sampler_parameters = {
    "n_ray_samples": 64,
    "near": 0.,
    "far": SCENE * 1.73 # Diagonal of the scene cube
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


# Appearance embedding parameters
app_parameters = {
    "num_embeddings": len(cfg_parameters["roll_cfgs"]),
    "embedding_dim": 7
}


# Parameter dictionary for SmallNeRF 
nerf_parameters = {
    "n_layers": 3,
    "hidden_dim": 64,
    "geo_feat_dim": 10,
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
    "validation_batch": 32*32*16,
    "epochs": 10,
    "lr": 0.005,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "tv_loss_weight": 1e-7,
    "stop_tv_epoch": 5,
    "sparsity_loss_weight": 1e-9,
    "decay_rate": 1e-1,
    "decay_steps": 7,
    "test_trajectories": [
        "data/test_trajectories/nominal.json"
    ],
    "verbose": True
}