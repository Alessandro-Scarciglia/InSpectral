# Import modules
import torch


class RenderingParameters:
    def __init__(self):

        # Parameters
        self.parameters = {
            "H": 100,
            "W": 100,
            "CH": 3,
            "K": torch.rand(3, 3),

            "n_ray_samples": 64,
            "near": 0.,
            "far": 5.,

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
            "device": "cpu",

            "n_layers": 10,
            "hidden_dim": 128,
            "geo_feat_dim": 15,
            "n_layers_color": 1,
            "hidden_dim_color": 128,
            "input_ch": 10,
            "input_ch_views": 16,
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


# Run for usage example
if __name__ == "__main__":

    # Define parameters object
    params = RenderingParameters()

    # Try methods
    print(params.get_param(key="hidden_dim"))
    params.set_param(key="hidden_dim", val=256)
    print(params.get_param(key="hidden_dim"))
    