# Import modules
import torch


# Training parameters
BATCH_SIZE = 32*32*16
EPOCHS = 10
SAMPLE_EVERY = 2 # (3 orbs 1080/5 = 216 | 2 orbs 720/3 = 250 | 1 orb 360/2 = 180)
TEST_EVERY = 12  # (3 orbs 1080/36 = 30 | 2 orbs 720/36 = 20 | 1 orb 360/12 = 30)
SCENE = 1.       # Set unitary for scaling scenes

# Display frame during rendez-vous
DISP = True

# Verbose during training
VERB = True

# Pics to jump
JUMP = []#["358", "359", "000", "001", "002"] + ["177", "178", "179", "180", "181"]


# General sat node setup
cfg_parameters = {
    "roll_cfgs": ["roll_0", "roll_120"],
    "resolution": 256,
    "channels": 1,
    "datapath": "data/transforms.json",
    "calibration_path": "calibration/calibration.json",
    "device": "cuda:0"
}


# Parameter dictionary for rays generation
rays_parameters = {
    "H": cfg_parameters["resolution"],
    "W": cfg_parameters["resolution"],
    "CH": cfg_parameters["channels"],
}


# Parameter dictionary for sampler
sampler_parameters = {
    "n_ray_samples": 64,
    "near": 0.,
    "far": SCENE * 1.73 # Diagonal of the scene cube
}


# Parameter dictionary for sh embedder
sh_parameters = {
    "input_dim": 3,
    "degree": 4,
    "out_dim": 4**2, # Always square of 'degree' 
}


# Parameter dictionary for positional embedder
posenc_parameters = {
    "n_freq": 10
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
optimizer_parameters = {
    "lr": 0.005,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "tot_var_weight": 1e-7,
    "sparsity_loss_weight": 1e-9,
    "decay_rate": 1e-1,
    "decay_steps": 7
}