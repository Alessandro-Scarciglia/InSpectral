# Import modules
import torch
import torch.nn as nn


class HashEmbedder(nn.Module):
    def __init__(self,
                 bbox: tuple,
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 low_resolution: int = 16,
                 high_resolution: int = 512):
        super(HashEmbedder, self).__init__()

        # Attributes
        self.bbox = bbox
        self.n_levels = n_levels
        self.n_feature_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.low_res = torch.tensor(low_resolution)
        self.high_res = torch.tensor(high_resolution)

        # Compute output dimension
        self.out_dim = n_levels * n_features_per_level

        # Compute exponential base for resolution usampling
        self.b = torch.exp(
            torch.log(high_resolution / low_resolution) / (n_levels -1)
        )

        # Initialize embeddings. An embedding per level is created. 
        # For each embedding (vocab, features) = (2**log2_hashmap_size, feat_per_level)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(2 ** log2_hashmap_size, n_features_per_level) for _ in range(n_levels)]
        )

        # Custom uniform inizialization of parameters
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=1e-4, b=-1e-4)

        
        def trilinear_interpolation(self,
                                    coords)
