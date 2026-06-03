import torch
import torch.nn as nn
torch.manual_seed(1234)

class HashEmbedder(nn.Module):
    """
    This class implements the functionalities of the Hash Embedding preprocessing. It takes in input the 
    cartesian coordinates of the rendering volume and build a mapping from (x, y, z) to learnable features
    in a multiresolution fashion.

    Attributes:
    ----------
    bbox: tuple[torch.Tensor[float, float, float], torch.Tensor[float, float, float]]
        it defines the 3D bounding box which defines the rendering volume.
    n_levels: int
        it defines the number of intermediate levels between low and high resolution (included).
        The bigger n_levels, the greater the number of features resolutions taken into account.
    n_features_per_level: int
        it defines the length of a feature vector per each 'voxel'.
    log2_hashmap_size: int
        it is the exponent which defines the size of the vocabulary, i.e. 2**log2_hashmap_size is
        the number of reference points to be referred to in the rendering volume.
    low_resolution: int
        it is the number of division of the rendering volume side in the lowest resolution.
    high_resolution: int
        it is the number of division of the rendering volume side in the greatest resolution.
    device: str
        it is a string to target the hardware where to store tensors ("cpu", generic "cuda", or specific "cuda:x").
    """
    def __init__(
            self,
            bbox: tuple,
            n_levels: int = 16,
            n_features_per_level: int = 2,
            log2_hashmap_size: int = 19,
            low_resolution: int = 16,
            high_resolution: int = 512,
            device: str = 'cpu'
    ):
        super(HashEmbedder, self).__init__()

        # Attributes
        self.device = torch.device(device)
        self.bbox = bbox
        self.n_levels = n_levels
        self.n_feature_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.low_res = torch.tensor(low_resolution, device=device)
        self.high_res = torch.tensor(high_resolution, device=device)
        self.box_offset = torch.tensor([[[i, j, k] for i in [0, 1]
                                         for j in [0, 1]
                                         for k in [0, 1]]], device=device)
        
        self.primes = torch.tensor([1, 2654435761, 805459861,
                                    3674653429, 2097192037, 1434869437, 2165219737], device=device)

        # Compute output dimension
        self.out_dim = n_levels * n_features_per_level

        # Compute exponential base for resolution upsampling
        self.b = torch.exp(
            torch.log(self.high_res / self.low_res) / (n_levels - 1)
        ).to(device)

        # Initialize embeddings. An embedding per level is created. 
        self.embeddings = nn.ModuleList(
            [nn.Embedding(2 ** log2_hashmap_size, n_features_per_level).to(device) for _ in range(n_levels)]
        )
        
    def trilinear_interpolation(
            self,
            coords: torch.Tensor,
            voxel_min: torch.Tensor,
            voxel_max: torch.Tensor,
            voxel_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Trilinear interpolation extension. 
        Follows: https://en.wikipedia.org/wiki/Trilinear_interpolation
        """
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
    
    def hash(
            self,
            coords: torch.Tensor,
            log2_hashmap_size: int
    ) -> torch.Tensor:
        """
        Spatial coordinates hashing using prime numbers multiplication and XOR.
        """
        # Initialize xor results
        xor_result = torch.zeros_like(coords)[..., 0]

        # Compute hashing for each coordinate dimension
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i] * self.primes[i]

        hashing = torch.tensor((1 << log2_hashmap_size) - 1, device=self.device)
        hashing = hashing & xor_result

        return hashing 

    def get_voxel_vertices(
            self,
            coords: torch.Tensor,
            bbox: tuple,
            resolution: torch.Tensor,
            log2_hashmap_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes reference voxel extrema, hashed indices, and valid rendering region mask.
        """
        # Unpack the bbox extrema
        bbox_min, bbox_max = bbox
        bbox_min, bbox_max = bbox_min.to(self.device), bbox_max.to(self.device)

        # Build a binary mask to sign those coordinates out of the bbox.
        keep_mask = coords == torch.max(torch.min(coords, bbox_max), bbox_min)
        coords = torch.clamp(coords, min=bbox_min, max=bbox_max)

        # Compute for each coordinate the reference voxel extrema
        grid_size = (bbox_max - bbox_min) / resolution
        bottom_left_idx = torch.floor((coords - bbox_min) / grid_size).int()
        
        # CORREZIONE BUG FLOAT ALTE RISOLUZIONI:
        # Calcoliamo i vertici basandoci rigorosamente sugli indici discreti per evitare disallineamenti di floating-point
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
        Forward pass with support for Progressive Coarse-to-Fine Soft-Masking.
        
        Parameters:
        ----------
        coords: torch.Tensor
            3D input spatial coordinates.
        current_step: int (optional)
            The current training iteration step.
        max_steps: int (optional)
            The total number of training iterations scheduled.
        """
        # Bring coordinates to target hardware
        coords = coords.to(self.device)
        coords_embedded_all = list()

        # --- GESTIONE DEI PESI COARSE-TO-FINE ---
        # Di default tutti i livelli sono attivi (peso = 1.0)
        weights = [1.0] * self.n_levels
        
        if current_step is not None and max_steps is not None:
            # Definiamo che la transizione si completa al 50% del training totale
            anneal_end_step = max_steps * 0.5
            # Determina quanti livelli "sbloccare" proporzionalmente al progresso
            progress = (current_step / anneal_end_step) * self.n_levels
            
            for i in range(self.n_levels):
                # Il peso sfuma dolcemente tra 0.0 e 1.0 per il livello i-esimo
                weights[i] = max(0.0, min(1.0, progress - i))

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
            
            # --- APPLICAZIONE DEL SOFT-MASKING ---
            coords_embedded = coords_embedded * weights[i]
            
            coords_embedded_all.append(coords_embedded)

        # OTTIMIZZAZIONE PERFORMANCE: Estratti fuori dal loop for
        keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        new_coords = torch.cat(coords_embedded_all, dim=-1)

        return new_coords, keep_mask