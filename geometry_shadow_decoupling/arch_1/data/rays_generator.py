# Import modules
import torch
import torch.nn as nn
import numpy as np


class RaysGeneratorSynth(nn.Module):
    """
    This class implements the methods for rays generation starting from a camera-to-world
    transformation and employing information regarding the camera.

    Attributes:
    ----------
    height: int
        the height of the image in pixels
    width: int
        the width of the image in pixels
    num_ch: int
        the number of channels (1 for grayscale, 3 for RGB/BGR, other for custom stacks)
    intrinsic_matrix: torch.Tensor
        the matrix containing the intrinsic parameters of a pinhole camera model
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
        self.intrinsic_matrix = intrinsic_matrix

    def get_rays(
            self,
            c2w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function generates HxW rays which start from the camera
        origin and proceed towards the rendering scene. Each ray is
        represented as a 6D vector: ray origin (3D) plus ray direction (3D).

        Parameters:
        ----------
        c2w: torch.Tensor
            it is the camera-to-world 4x4 transformation

        Returns:
        -------
        rays_o: torch.Tensor
            rays origin (x, y, z) in world reference frame (i.e. camera position)
        rays_d: torch.Tensor
            rays direction (ix, iy, iz) in world reference frame (i.e. a unit vector which starts from the camera
            origin and points somewhere in the rendering scene).
        """

        # Generate the meshgrid of all pixels on the image plane
        i_span = torch.linspace(0, self.width - 1, self.width)
        j_span = torch.linspace(0, self.height - 1, self.height)
        it, jt = torch.meshgrid(i_span, j_span, indexing='ij')
        i, j = it.t(), jt.t()

        # Compute directions
        dirs = torch.stack([(i - self.intrinsic_matrix[0][2]) / self.intrinsic_matrix[0][0],
                            -(j - self.intrinsic_matrix[1][2]) / self.intrinsic_matrix[1][1],
                            -torch.ones_like(i)],
                            dim=-1)
        
        # Rotate directions according to c2w
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], dim=-1)
        
        # Translate camera origin in word frame
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d
    
    def forward(
            self,
            c2w: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward function returns both true color channels and rays when in "train" mode.
        Viceversa, it returns only rays when in inference-only mode.
        """

        # Compute rays 
        rays_o, rays_d = self.get_rays(c2w)
        
        # Return rays origins and directions
        out = torch.concatenate([rays_o, rays_d], dim=-1)
    
        return out