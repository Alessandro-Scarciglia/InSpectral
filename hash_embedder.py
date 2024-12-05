# Import modules
import torch
import torch.nn as nn

torch.manual_seed(1234)

class HashEmbedder(nn.Module):
    def __init__(self,
                 bbox: tuple,
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 low_resolution: int = 16,
                 high_resolution: int = 512,
                 device: str = 'cpu'):
        super(HashEmbedder, self).__init__()

        # Attributes
        self.device = torch.device(device)
        self.bbox = bbox
        self.n_levels = n_levels
        self.n_feature_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.low_res = torch.tensor(low_resolution, device=device)
        self.high_res = torch.tensor(high_resolution, device=device)
        self.box_offset = torch.tensor([[[i,j,k] for i in [0, 1]
                                         for j in [0, 1]
                                         for k in [0, 1]]], device=device)
        
        self.primes = torch.tensor([1, 2654435761, 805459861,
                       3674653429, 2097192037, 1434869437, 2165219737], device=device)

        # Compute output dimension
        self.out_dim = n_levels * n_features_per_level

        # Compute exponential base for resolution usampling
        self.b = torch.exp(
            torch.log(self.high_res / self.low_res) / (n_levels -1)
        ).to(device)

        # Initialize embeddings. An embedding per level is created. 
        # For each embedding (vocab, features) = (2**log2_hashmap_size, feat_per_level)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(2 ** log2_hashmap_size, n_features_per_level).to(device) for _ in range(n_levels)]
        )

        # Custom uniform inizialization of parameters
        # for i in range(n_levels):
        #     nn.init.uniform_(self.embeddings[i].weight, a=-1e-4, b=1e-4)

        
    def trilinear_interpolation(self,
                                coords: torch.Tensor,
                                voxel_min: torch.Tensor,
                                voxel_max: torch.Tensor,
                                voxel_emb: torch.Tensor):
        '''
        It is an extension of the bilinear interpolation. The implementation
        follows in the steps of https://en.wikipedia.org/wiki/Trilinear_interpolation
        '''

        # Weights definition
        weights = (coords - voxel_min) / (voxel_max - voxel_min)

        # First step: from 8 points to 4 points
        c00 = voxel_emb[:, 0] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 4] * weights[:, 0][:, None]
        c01 = voxel_emb[:, 1] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 5] * weights[:, 0][:, None]
        c10 = voxel_emb[:, 2] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 6] * weights[:, 0][:, None]
        c11 = voxel_emb[:, 3] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 7] * weights[:, 0][:, None]

        # Second step: from 4 points to 2 points
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # Third step: from 2 points to 1 point
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c
    

    # Hashing function
    def hash(self,
             coords: torch.Tensor,
             log2_hashmap_size: int):
        
        # Initialize xor results
        xor_result = torch.zeros_like(coords)[..., 0]

        # Compute hashing for each coordinate dimension
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i] * self.primes[i]

        # 1) Apply the bitwise left shift operator.
        # 2) Apply the AND operator with xor_result
        hashing = torch.tensor((1 << log2_hashmap_size) - 1, device=self.device)
        hashing = hashing & xor_result

        return hashing 


    def get_voxel_vertices(self,
                           coords: torch.Tensor,
                           bbox: tuple,
                           resolution: torch.Tensor,
                           log2_hashmap_size: int):
        
        # Unpack the bbox extrema
        bbox_min, bbox_max = bbox
        bbox_min, bbox_max = bbox_min.to(self.device), bbox_max.to(self.device)

        # Build a binary mask to sign those coordinates out of the bbox.
        # Then, clamp eventual coordinates to the bbox domain limits.
        keep_mask = coords == torch.max(torch.min(coords, bbox_max), bbox_min)
        coords = torch.clamp(coords, min=bbox_min, max=bbox_max)

        # Compute for each coordinate the reference voxel extrema
        grid_size = (bbox_max - bbox_min) / resolution
        bottom_left_idx = torch.floor((coords - bbox_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + bbox_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1., 1., 1.], device=self.device) * grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offset
        hashed_voxel_indices = self.hash(voxel_indices, log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


    def forward(self, coords):
        '''
        In the forward overriding, coordinates in 3D are injected as input.
        In the end, embeddings and the binary mask of all valid coordinates
        are returned.
        '''

        # Bring coordinates to target hardware
        coords = coords.to(self.device)
        
        # Initialize the embeddings
        coords_embedded_all = list()

        # Iterate the forwarding for each resolution
        for i in range(self.n_levels):
            resolution = torch.floor(self.low_res * (self.b ** i))
            
            # Get voxels from coordinates
            voxel_min, voxel_max, hashed_idx, keep_mask = self.get_voxel_vertices(coords,
                                                                                  self.bbox,
                                                                                  resolution,
                                                                                  self.log2_hashmap_size)
            
            # Assign hashing to embeddings
            voxel_embedds = self.embeddings[i](hashed_idx)

            # Apply trilinear interpolation
            coords_embedded = self.trilinear_interpolation(coords,
                                                           voxel_min,
                                                           voxel_max,
                                                           voxel_embedds)
            
            coords_embedded_all.append(coords_embedded)

            # Adjust dimensions
            keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
            new_coords = torch.cat(coords_embedded_all, dim=-1)

        return new_coords, keep_mask



# Test
if __name__ == "__main__":
    
    # Instantiate the embedder object
    bbox = (torch.tensor([0., 0., 0.]), torch.tensor([2., 2., 2.]))

    embedder = HashEmbedder(bbox=bbox,
                            n_levels=1,
                            n_features_per_level=2,
                            log2_hashmap_size=4,
                            low_resolution=2,
                            high_resolution=8,
                            device="cuda")

    # Define a set of coordinates
    coords = torch.tensor([[2, 1.5, 1]])

    # Call
    emb_coords, mask = embedder(coords)
    print(emb_coords)