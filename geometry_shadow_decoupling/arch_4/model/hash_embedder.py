"""
This module implements a Multiresolution Spatial Hash Grid Embedder.
It transforms raw continuous Cartesian coordinates (x, y, z) into interpolated, 
learnable high-dimensional feature representations with support for temporal coarse-to-fine annealing.
"""

import torch
import torch.nn as nn

torch.manual_seed(1234)


class HashEmbedder(nn.Module):
    """
    Multiresolution spatial hashing preprocessing layer.
    
    Attributes:
    ----------
    bbox: tuple[torch.Tensor, torch.Tensor]
        Defines the 3D spatial boundaries of the target reconstruction volume.
    n_levels: int
        Number of multiresolution spatial embedding grids.
    n_features_per_level: int
        Dimensional length of the embedding feature vector per voxel vertex.
    log2_hashmap_size: int
        Exponent sizing the vocabulary hash dictionary (2**log2_hashmap_size slots).
    low_resolution: int
        Voxel cell side division frequency at the coarsest grid level.
    high_resolution: int
        Voxel cell side division frequency at the finest grid level.
    min_visible_levels: int
        Number of active baseline low-resolution levels present at epoch 0.
    device: str
        Target hardware acceleration device context ('cpu', 'cuda').
    """
    def __init__(
            self,
            bbox: tuple,
            n_levels: int = 16,
            n_features_per_level: int = 2,
            log2_hashmap_size: int = 19,
            low_resolution: int = 16,
            high_resolution: int = 512,
            min_visible_levels: int = 4,  
            device: str = 'cpu'
    ):
        super(HashEmbedder, self).__init__()

        # Attributes assignment
        self.device = torch.device(device)
        self.bbox = bbox
        self.n_levels = n_levels
        self.n_feature_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.min_visible_levels = min_visible_levels  
        
        self.low_res = torch.tensor(low_resolution, device=self.device)
        self.high_res = torch.tensor(high_resolution, device=self.device)
        
        # Spatial neighbor look-up voxel offsets
        self.box_offset = torch.tensor([[[i, j, k] for i in [0, 1]
                                         for j in [0, 1]
                                         for k in [0, 1]]], device=self.device)
        
        # Prime multipliers for spatial index scrambling
        self.primes = torch.tensor([1, 2654435761, 805459861,
                                    3674653429, 2097192037, 1434869437, 2165219737], device=self.device)

        # Compute combined output spatial dimensionality
        self.out_dim = n_levels * n_features_per_level

        # Calculate scale upsampling ratio base constant
        self.b = torch.exp(
            torch.log(self.high_res / self.low_res) / (n_levels - 1)
        ).to(self.device)

        # Pre-allocate separate parameter look-up tables per level
        self.embeddings = nn.ModuleList(
            [nn.Embedding(2 ** log2_hashmap_size, n_features_per_level).to(self.device) for _ in range(n_levels)]
        )
        
        # Initialize hash grid parameters with a very narrow uniform distribution to stabilize training bounds
        for embedding_layer in self.embeddings:
            nn.init.uniform_(embedding_layer.weight, a=-1e-4, b=1e-4)
        
    def trilinear_interpolation(
            self,
            coords: torch.Tensor,
            voxel_min: torch.Tensor,
            voxel_max: torch.Tensor,
            voxel_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates localized trilinear weights over 8 localized grid parameters.
        Follows standard coordinate interpolation: https://en.wikipedia.org/wiki/Trilinear_interpolation
        """
        # Linear tracking weights across dimensions
        weights = (coords - voxel_min) / (voxel_max - voxel_min)

        # First stage: Reductions from 8 structural vertices to 4 planar nodes
        c00 = voxel_emb[:, 0] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 4] * weights[:, 0][:, None]
        c01 = voxel_emb[:, 1] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 5] * weights[:, 0][:, None]
        c10 = voxel_emb[:, 2] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 6] * weights[:, 0][:, None]
        c11 = voxel_emb[:, 3] * (1 - weights[:, 0][:, None]) + voxel_emb[:, 7] * weights[:, 0][:, None]

        # Second stage: Reductions from 4 planar nodes to 2 axial points
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # Third stage: Reductions from 2 axial points to a single aggregated feature vector
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c
    
    def hash(
            self,
            coords: torch.Tensor,
            log2_hashmap_size: int
    ) -> torch.Tensor:
        """
        Transforms 3D discrete spatial voxel locations into wrapped table indices.
        Uses prime factor multiplication followed by a bitwise bitmask wrap.
        """
        xor_result = torch.zeros_like(coords)[..., 0]

        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i] * self.primes[i]

        hashing_mask = torch.tensor((1 << log2_hashmap_size) - 1, device=self.device)
        return hashing_mask & xor_result

    def get_voxel_vertices(
            self,
            coords: torch.Tensor,
            bbox: tuple,
            resolution: torch.Tensor,
            log2_hashmap_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes discrete local voxel bounds and wraps indices into look-up addresses.
        """
        bbox_min, bbox_max = bbox
        bbox_min, bbox_max = bbox_min.to(self.device), bbox_max.to(self.device)

        # Check for out-of-bounds ray intersections via boundary masks
        keep_mask = coords == torch.max(torch.min(coords, bbox_max), bbox_min)
        coords = torch.clamp(coords, min=bbox_min, max=bbox_max)

        grid_size = (bbox_max - bbox_min) / resolution
        bottom_left_idx = torch.floor((coords - bbox_min) / grid_size).int()
        
        # Calculate localized vertices from discrete index anchors to circumvent float drift
        voxel_min_vertex = bottom_left_idx * grid_size + bbox_min
        voxel_max_vertex = (bottom_left_idx + 1) * grid_size + bbox_min

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offset
        hashed_voxel_indices = self.hash(voxel_indices, log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

    def forward(
            self,
            coords: torch.Tensor,
            current_step: int = None,
            max_steps: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps raw coordinates onto multi-scale grids with soft coarse-to-fine linear masking.
        """
        coords = coords.to(self.device)
        coords_embedded_all = list()

        # -------------------------------------------------------------------------------
        # COARSE-TO-FINE COEFFICIENT CALCULATION
        # -------------------------------------------------------------------------------
        # weights = [1.0] * self.n_levels
        
        # if current_step is not None and max_steps is not None:
        #     # Complete the training structure expansion phase at exactly 50% training steps
        #     anneal_end_step = max_steps * 0.5
            
        #     # Identify the remaining resolution tiers to dynamically unlock
        #     remaining_levels = self.n_levels - self.min_visible_levels
            
        #     alpha = min(1.0, current_step / anneal_end_step)
        #     progress = self.min_visible_levels + (alpha * remaining_levels)
            
        #     for i in range(self.n_levels):
        #         if i < self.min_visible_levels:
        #             weights[i] = 1.0  
        #         else:
        #             # Ramps fine features smoothly between 0.0 and 1.0
        #             weights[i] = max(0.0, min(1.0, progress - i))

        # -------------------------------------------------------------------------------
        # INDEPENDENT GRID SCALE INTERPOLATION
        # -------------------------------------------------------------------------------
        for i in range(self.n_levels):
            resolution = torch.floor(self.low_res * (self.b ** i))
            
            voxel_min, voxel_max, hashed_idx, keep_mask = self.get_voxel_vertices(
                coords, self.bbox, resolution, self.log2_hashmap_size
            )
            
            voxel_embedds = self.embeddings[i](hashed_idx)

            coords_embedded = self.trilinear_interpolation(
                coords, voxel_min, voxel_max, voxel_embedds
            )
            
            # # Apply calculated structural fading windows
            # if weights[i] < 1.0:
            #     coords_embedded = coords_embedded * weights[i]
            
            coords_embedded_all.append(coords_embedded)

        # Aggregate spatial channel sequences
        keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        new_coords = torch.cat(coords_embedded_all, dim=-1)

        return new_coords, keep_mask