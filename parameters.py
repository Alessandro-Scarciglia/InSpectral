# Import modules
import torch


class RenderingParameters:
    def __init__(self):

        # Parameters
        self.parameters = {
            "H": 256,
            "W": 256,
            "CH": 3,
            "K": torch.rand(3, 3),

            "n_ray_samples": 64,
            "near": 1.,
            "far": 7.,

            "input_dim": 3,
            "degree": 4,
            "out_dim": 16,

            "bbox": (torch.tensor([0., 0., 0.]),
                     torch.tensor([10., 10., 10.])),
            "n_levels": 16,
            "n_features_per_level": 10,
            "log2_hashmap_size": 19,
            "low_res": 64,
            "high_res": 512,

            "n_layers": 3,
            "hidden_dim": 64,
            "geo_feat_dim": 15,
            "n_layers_color": 3,
            "hidden_dim_color": 64,
            "input_ch": 10,          # It shall be equal to "n_features_per_level"
            "input_ch_views": 16,    # It shall be equal to "out_dim"
            "out_ch": 3,
        }

    # Function to get parameters
    def get_param(self, key: str):
        return self.parameters[key]
    
    # Function to get all parameters
    def get_all_params(self):
        return self.parameters
    
    # Function to change a parameter
    def set_param(self, key: str, val):
        self.parameters[key] = val


class TrainingParameters:
    def __init__(self):

        # Parameters
        self.parameters = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0,
            "degenerated_to_sgd": False,
            "tot_var_weight": 1e-6,
            "sparsity_loss_weight": 1e-6,
            "decay_rate": 1e-1,
            "decay_steps": 1000
            }

    # Function to get parameters
    def get_param(self, key: str):
        return self.parameters[key]
    
    # Function to get all parameters
    def get_all_params(self):
        return self.parameters
    
    # Function to change a parameter
    def set_param(self, key: str, val):
        self.parameters[key] = val


