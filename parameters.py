# Import modules
import torch


# Parameter dictionary for rays generation
rays_parameters = {
    "H": 128,
    "W": 128,
    "CH": 3,
}


# Parameter dictionary for sampler
sampler_parameters = {
    "n_ray_samples": 200,
    "near": 0.,
    "far": 5.,
}


# Parameter dictionary for sh embedder
sh_parameters = {
    "input_dim": 3,
    "degree": 4,
    "out_dim": 16,
}


# Parameter dictionary for hash embedder
hash_parameters = {
    "bbox": (torch.tensor([-4., -4., -4.]),
             torch.tensor([4., 4., 4.])),
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "low_res": 16,
    "high_res": 512,
}           


# Parameter dictionary for SmallNeRF 
nerf_parameters = {
    "n_layers": 2,
    "hidden_dim": 64,
    "geo_feat_dim": 15,
    "n_layers_color": 3,
    "hidden_dim_color": 64,
    "input_ch": 32,          
    "input_ch_views": 16,    
    "out_ch": 3,
}


# Parameter dictionary for training 
training_parameters = {
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0,
    "degenerated_to_sgd": False,
    "tot_var_weight": 1e-6,
    "sparsity_loss_weight": 1e-6,
    "decay_rate": 10,
    "decay_steps": 1000
}
