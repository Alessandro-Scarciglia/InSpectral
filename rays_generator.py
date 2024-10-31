# Import modules
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt


class RaysGenerator(nn.Module):
    def __init__(self,
                 H: int,
                 W: int,
                 CH: int,
                 K: torch.Tensor):
        super(RaysGenerator, self).__init__()
        
        # Attributes
        self.K = K
        self.H = H
        self.CH = CH
        self.W = W


    def get_rays(self,
                 c2w: torch.Tensor):
        '''
        This function generates HxW rays which start from the camera
        origin and proceed towards the rendering scene. Each ray is
        represented as a 6D vector: ray origin (3D) plus ray direction (3D).
        '''

        # Generate the meshgrid of all pixels on the image plane
        i_span = torch.linspace(0, self.W - 1, self.W)
        j_span = torch.linspace(0, self.H - 1, self.H)
        it, jt = torch.meshgrid(i_span, j_span)
        i, j = it.t(), jt.t()

        # Compute directions
        dirs = torch.stack([(i - self.K[0][2]) / self.K[0][0],
                            (j - self.K[1][2]) / self.K[1][1],
                            torch.ones_like(i)],
                            dim=-1)
        
        # Rotate directions according to c2w
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], dim=-1)
        
        # Translate camera origin in word frame
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d
    

    def forward(self,
                c2w: torch.Tensor):
        '''
        The forward function returns both true color channels and rays when in "train" mode.
        Viceversa, it returns only rays when in inference-only mode.
        '''

        # Compute rays 
        rays_o, rays_d = self.get_rays(c2w)
        
        # Return rays origins and directions
        out = torch.concatenate([rays_o, rays_d], dim=-1)
    
        return out