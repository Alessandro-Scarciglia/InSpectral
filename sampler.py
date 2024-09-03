# Import modules
import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self,
                 n_samples: int,
                 near: int,
                 far: int):
        super(Sampler, self).__init__()

        # Attributes
        self.n_samples = n_samples
        self.near = near
        self.far = far


    def forward(self,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor):
        '''
        This function generate n_samples on a batch of rays which start from point
        rays_o (origins) and propagate toward rays_d (directions), within the
        rendering region [near, far].
        '''

        # Useful variables
        n_rays = rays_o.shape[0]

        # Generate equally spaced n_samples
        tvals = torch.linspace(0., 1., steps=self.n_samples)
        zvals = self.near * (1 - tvals) + self.far * tvals
        zvals = zvals.expand([n_rays, self.n_samples])

        # Add a perturbation to the uniformly spaced samples
        mids = 0.5 * (zvals[..., 1:] + zvals[..., :-1])
        upper = torch.cat([mids, zvals[..., -1:]], dim=-1)
        lower = torch.cat([zvals[..., :1], mids], dim=-1)
        randvals = torch.rand(zvals.shape)

        zvals = lower + (upper - lower) * randvals

        # Define 3D points along each ray
        pts = rays_o[..., None, :] + rays_d[..., None, :] * zvals[..., :, None]

        return pts, zvals



# Run for usage example
if __name__ == "__main__":

    # Istantiate the Sampler object
    sampler = Sampler(n_samples=10, near=0., far=5.)

    # Dummy input
    rays_o = torch.tensor([[0., 0., 0.]])
    rays_d = torch.tensor([[1., 1., 1.]])
    rays_d /= torch.norm(rays_d, dim=-1)
    
    # Test inference
    samples = sampler(rays_o, rays_d)

    # No params
    print(samples)