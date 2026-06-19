# Import modules
import torch
import torch.nn as nn
import numpy as np


class RaysGeneratorSynth(nn.Module):
    """
    Implements optimized camera ray generation matching standard NeRF / OpenCV 
    conventions, starting from a camera-to-world (c2w) transformation matrix.
    """
    def __init__(
            self,
            height: int,
            width: int,
            num_ch: int,
            intrinsic_matrix: np.ndarray
    ):
        super(RaysGeneratorSynth, self).__init__()
        
        # Attributes
        self.height = height
        self.width = width
        self.ch = num_ch
        
        # Convert intrinsic matrix to tensor and register as buffer for device management
        intrinsic_tensor = torch.from_numpy(intrinsic_matrix).float()
        self.register_buffer("intrinsic_matrix", intrinsic_tensor)

        # Generate a static pixel grid 
        i_span = torch.linspace(0, self.width - 1, self.width)
        j_span = torch.linspace(0, self.height - 1, self.height)
        it, jt = torch.meshgrid(i_span, j_span, indexing='ij')
        i, j = it.t(), jt.t()
        
        # FIXED: Corrected reference frame signs to standard NeRF convention 
        # (X right, Y down/inverted, Z pointing backwards)
        dirs_static = torch.stack([
            (i - intrinsic_tensor[0, 2]) / intrinsic_tensor[0, 0],
            -(j - intrinsic_tensor[1, 2]) / intrinsic_tensor[1, 1],
            -torch.ones_like(i)
        ], dim=-1)
        
        # FIXED: Normalize directions to create unit vectors for precise volumetric sampling bounds
        dirs_static = dirs_static / torch.norm(dirs_static, dim=-1, keepdim=True)
        
        # Register static directions as buffer to automatically handle runtime device casting
        self.register_buffer("dirs", dirs_static)

    def get_rays(
            self,
            c2w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates an [H, W, 3] grid of ray origins and rotated unit directions.
        """
        # Force correct float32 precision
        c2w = c2w.float()

        # Extract the 3x3 rotation matrix
        rotation = c2w[:3, :3]  # Shape: [3, 3]
        
        # Rotate static unit directions using the camera extrinsics rotation tensor
        rays_d = torch.einsum('hwj,ij->hwi', self.dirs, rotation)
        
        # Extract the translation vector (camera center origin) -> Shape: [3]
        camera_origin = c2w[:3, -1]
        
        # Broadcast camera origin across the entire ray grid topology -> Shape: [H, W, 3]
        rays_o = camera_origin.expand(rays_d.shape)

        return rays_o, rays_d
    
    def forward(
            self,
            c2w: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass execution returning combined ray configurations during inference.
        """
        # Compute ray origins and directions
        rays_o, rays_d = self.get_rays(c2w)
        
        # Concatenate origins and directions along the last feature dimension -> Shape: [H, W, 6]
        out = torch.cat([rays_o, rays_d], dim=-1)
    
        return out