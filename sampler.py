# Import modules
import torch
import torch.nn as nn
from profiler import timing_decorator


class Sampler(nn.Module):
    def __init__(self,
                 n_samples: int,
                 near: int,
                 far: int,
                 device: str = 'cpu'):
        super(Sampler, self).__init__()

        # Attributes
        self.n_samples = n_samples
        self.device = torch.device(device)
        self.near = torch.tensor(near, device=device)
        self.far = torch.tensor(far, device=device)

    @timing_decorator
    def forward(self,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor):
        '''
        This function generate n_samples on a batch of rays which start from point
        rays_o (origins) and propagate toward rays_d (directions), within the
        rendering region [near, far].
        '''

        # Bring rays to target device
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        # Useful variables
        n_rays = rays_o.shape[0]

        # Generate equally spaced n_samples
        tvals = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
        zvals = self.near * (1 - tvals) + self.far * tvals
        zvals = zvals.expand([n_rays, self.n_samples])

        # Add a perturbation to the uniformly spaced samples
        mids = 0.5 * (zvals[..., 1:] + zvals[..., :-1])
        upper = torch.cat([mids, zvals[..., -1:]], dim=-1)
        lower = torch.cat([zvals[..., :1], mids], dim=-1)
        randvals = torch.rand(zvals.shape, device=self.device)

        zvals = lower + (upper - lower) * randvals
        
        # Define 3D points along each ray
        pts = rays_o[..., None, :] + rays_d[..., None, :] * zvals[..., :, None]

        return pts, zvals