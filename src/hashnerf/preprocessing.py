# Import modules
import torch
import torch.nn as nn
import numpy as np


class RaysGenerator(nn.Module):
    def __init__(self,
                 K: torch.Tensor):
        
        # Attributes
        self.K = K

    def get_rays(self,
                 H: int,
                 W: int,
                 c2w: torch.Tensor):
        '''
        This function generates HxW rays which start from the camera
        origin and proceed towards the rendering scene. Each ray is
        represented as a 6D vector: ray origin (3D) plus ray direction (3D).
        '''

        # Generate the meshgrid of all pixels on the image plane
        i_span = torch.linspace(0, W-1, W)
        j_span = torch.linspacve(0, H-1, H)
        i, j = torch.meshgrid(i_span, j_span).t()

        # Compute directions
        # TODO: check camera frame consistency
        dirs = torch.stack([(i - self.K[0][2]) / self.K[0][0],
                            -(j - self.K[1][2]) / self.K[1][1],
                            -torch.ones_like(i)],
                            dim=-1)
        
        # Rotate directions according to c2w
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], dim=-1)

        # Translate camera origin in word frame
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d