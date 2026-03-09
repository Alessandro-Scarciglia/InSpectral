"""
This module implements the MLP model in charge of estimating the residuals of camera poses.
"""

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self,
                 n_layers: int,
                 hidden_dim: int,
                 input_ch: int,
                 output_ch: int,
                 lin_range: float,
                 ang_range: float,
                 device: str = "cuda"
                 ):
        super(ResidualMLP, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.lin_range = lin_range
        self.ang_range = ang_range

        self.model = self.build_res_net()
        self.model.to(device)


    def build_res_net(self) -> nn.ModuleList:

        res_net = list()

        for layer in range(self.n_layers):

            if layer == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim

            if layer == self.n_layers - 1:
                out_dim = self.output_ch
            else:
                out_dim = self.hidden_dim

            res_net.append(nn.Linear(in_dim, out_dim, bias=False))

            # Activation only between hidden layers
            if layer < self.n_layers - 1:
                res_net.append(nn.ReLU())

        # Final activation (range [-1, 1])
        res_net.append(nn.Tanh())

        return nn.ModuleList(res_net)


    def forward(self,
                prior_rays: torch.Tensor
                ):

        # Ensure batch dimension
        if prior_rays.dim() == 1:
            prior_rays = prior_rays.unsqueeze(0)

        # Manual forward through ModuleList
        x = prior_rays
        for layer in self.model:
            x = layer(x)

        out = x

        # Residual translation
        dx = out[..., :3] * self.lin_range

        # Residual rotation vector (axis-angle)
        omega = out[..., 3:] * self.ang_range

        theta = torch.norm(omega, dim=-1, keepdim=True)

        eps = 1e-8
        omega_normalized = omega / (theta + eps)

        wx, wy, wz = omega_normalized.unbind(-1)

        zeros = torch.zeros_like(wx)

        K = torch.stack([
            torch.stack([ zeros, -wz,  wy], dim=-1),
            torch.stack([  wz, zeros, -wx], dim=-1),
            torch.stack([ -wy,  wx, zeros], dim=-1)
        ], dim=-2)

        I = torch.eye(3, device=omega.device).expand(K.shape[0], 3, 3)

        sin_theta = torch.sin(theta)[..., None]
        cos_theta = torch.cos(theta)[..., None]

        dR = (
            I
            + sin_theta * K
            + (1 - cos_theta) * (K @ K)
        )

        return dx, dR
