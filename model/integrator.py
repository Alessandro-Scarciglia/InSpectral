# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config_parameters import *


class Integrator(nn.Module):
    """
    Volumetric NeRF integrator.

    Outputs:
        chs_map:
            Integrated color / appearance channels.

        depth_map:
            Linear inverse depth in [0, 1]:

                near -> 1.0 -> white
                far  -> 0.0 -> black

            Background rays are forced to zero.

            Shape: [N]

        sparsity_loss:
            Shannon entropy sparsity regularization.

        mask:
            Accumulated volumetric opacity in [0, 1].

            Shape: [N, 1]
    """

    def __init__(
        self,
        near_depth: float = 0.000,
        far_depth: float = 1.125,
        device: str = "cpu",
    ):
        super(Integrator, self).__init__()

        self.near_depth = float(near_depth)
        self.far_depth = float(far_depth)
        self.device = torch.device(device)
        self.last_weights = None
        self.last_distortion_loss = torch.zeros((), device=self.device)
        # Optional camera viewing axis (unit vector, world frame). When set, depth is
        # reported as Z depth along this axis instead of distance along the ray.
        self.camera_axis = None
        self._check_range()

    def set_camera_axis(self, c2w: torch.Tensor = None) -> None:
        """Report Z depth for the camera c2w (looking along -z), or ray distance if None.

        Call it before rendering each full image; with None (the default) the depth
        is the distance along each ray.
        """
        if c2w is None:
            self.camera_axis = None
        else:
            axis = -c2w[:3, 2].detach().float()
            self.camera_axis = axis / axis.norm().clamp_min(1e-12)

    def set_depth_range(self, far_depth: float, near_depth: float = 0.0) -> None:
        """Set the depth mapped to 0 (black) and 1 (white), in normalised scene units.

        Depth is not used by any loss, so this only changes the rendered depth maps and
        the depth metrics, never the training itself.
        """
        self.near_depth = float(near_depth)
        self.far_depth = float(far_depth)
        self._check_range()

    def _check_range(self) -> None:
        if self.far_depth <= self.near_depth:
            raise ValueError(
                f"far_depth ({self.far_depth}) must be greater than "
                f"near_depth ({self.near_depth})."
            )

    def forward(
        self,
        raw: torch.Tensor,
        zvals: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ==========================================================================
        # 1. DEVICE / DTYPE
        # ==========================================================================

        device = raw.device
        # Integration is deliberately accumulated in fp32 even when the network
        # runs under autocast.  This prevents underflow in transmittance and
        # preserves stable depth/loss gradients.
        dtype = torch.float32
        raw = raw.float()

        if zvals.device != device or zvals.dtype != dtype:
            zvals = zvals.to(
                device=device,
                dtype=dtype,
            )

        if rays_d.device != device or rays_d.dtype != dtype:
            rays_d = rays_d.to(
                device=device,
                dtype=dtype,
            )

        # ==========================================================================
        # 2. SPLIT COLOR AND DENSITY
        # ==========================================================================

        num_color_channels = cfg_parameters["channels"]

        if raw.shape[-1] <= num_color_channels:
            raise ValueError(
                "Invalid raw tensor shape. Expected at least "
                f"{num_color_channels + 1} channels, but received "
                f"{raw.shape[-1]}."
            )

        chs = raw[..., :num_color_channels]
        density = raw[..., num_color_channels]

        # ==========================================================================
        # 3. DISTANCES BETWEEN CONSECUTIVE SAMPLES
        # ==========================================================================

        dists = zvals[..., 1:] - zvals[..., :-1]

        # Last sample gets a large integration interval.
        last_dist = torch.full_like(
            zvals[..., :1],
            1e10,
        )

        dists = torch.cat(
            [
                dists,
                last_dist,
            ],
            dim=-1,
        )

        # Convert z-distance to Euclidean distance.
        ray_norm = torch.norm(
            rays_d,
            dim=-1,
            keepdim=True,
        )

        dists = dists * ray_norm

        # ==========================================================================
        # 4. DENSITY -> ALPHA
        # ==========================================================================

        density = F.relu(density)

        alpha = 1.0 - torch.exp(
            -density * dists
        )

        alpha = torch.clamp(
            alpha,
            min=0.0,
            max=1.0,
        )

        # ==========================================================================
        # 5. TRANSMITTANCE
        # ==========================================================================

        eps = 1e-10

        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(alpha[..., :1]),
                    1.0 - alpha + eps,
                ],
                dim=-1,
            ),
            dim=-1,
        )[..., :-1]

        # ==========================================================================
        # 6. VOLUMETRIC WEIGHTS
        # ==========================================================================

        weights = alpha * transmittance
        self.last_weights = weights
        # mip-NeRF style distortion/concentration penalty, evaluated in FP32.
        pairwise = 2.0 * torch.sum(
            weights * (zvals * torch.cumsum(weights, dim=-1) -
                       torch.cumsum(weights * zvals, dim=-1)), dim=-1)
        intervals = torch.cat(
            [zvals[..., 1:] - zvals[..., :-1], torch.zeros_like(zvals[..., :1])],
            dim=-1,
        ).clamp_min(0)
        self.last_distortion_loss = (pairwise + (weights.square() * intervals).sum(dim=-1) / 3.0).mean().clamp_min(0)

        # ==========================================================================
        # 7. ACCUMULATED OPACITY / OBJECT MASK
        # ==========================================================================

        accumulated_opacity = torch.sum(
            weights,
            dim=-1,
        )

        accumulated_opacity = torch.clamp(
            accumulated_opacity,
            min=0.0,
            max=1.0,
        )

        # Shape: [N, 1]
        mask = accumulated_opacity.unsqueeze(-1)

        # ==========================================================================
        # 8. INTEGRATE COLOR / APPEARANCE
        # ==========================================================================

        chs_map = torch.sum(
            weights[..., None] * chs,
            dim=-2,
        )

        chs_map = torch.clamp(
            chs_map,
            min=0.0,
            max=1.0,
        )

        # ==========================================================================
        # 9. COMPUTE SURFACE DEPTH
        # ==========================================================================

        # IMPORTANT:
        #
        # Do NOT use only:
        #
        #     depth = sum(weights * zvals)
        #
        # because that makes low-opacity rays appear artificially close
        # to the camera.
        #
        # Instead, normalize by the accumulated opacity:
        #
        #     depth = sum(weights * zvals) / sum(weights)
        #
        # This gives the expected depth conditioned on the ray hitting
        # the reconstructed geometry.

        weighted_depth = torch.sum(
            weights * zvals,
            dim=-1,
        )

        depth = weighted_depth / (
            accumulated_opacity + eps
        )

        # zvals are distances along unit-length rays. For Z-depth labels, project the
        # distance onto the camera's viewing axis: z = t * cos(angle to the axis).
        if self.camera_axis is not None:
            unit_dirs = rays_d / ray_norm.clamp_min(eps)
            cos_axis = (unit_dirs * self.camera_axis.to(device)).sum(dim=-1)
            depth = depth * cos_axis

        # ==========================================================================
        # 10. NORMALIZE DEPTH TO [0, 1]
        # ==========================================================================

        depth_range = (
            self.far_depth - self.near_depth
        )

        normalized_depth = (
            depth - self.near_depth
        ) / depth_range

        normalized_depth = torch.clamp(
            normalized_depth,
            min=0.0,
            max=1.0,
        )

        # ==========================================================================
        # 11. INVERT DEPTH TO MATCH GROUND TRUTH
        # ==========================================================================

        # Your ground-truth convention is:
        #
        #     0 m       -> 1.0 -> white
        #     40 m      -> 0.0 -> black
        #
        # Therefore:
        #
        #     near -> 1
        #     far  -> 0
        #
        # Larger geometric depth means smaller output value.

        depth_map = 1.0 - normalized_depth

        # ==========================================================================
        # 12. DEPTH GAMMA
        # ==========================================================================

        depth_gamma = 1.0

        depth_map = torch.pow(
            depth_map,
            depth_gamma,
        )

        # ==========================================================================
        # 13. SUPPRESS BACKGROUND
        # ==========================================================================

        # Rays with essentially zero accumulated opacity do not contain
        # a reconstructed surface.
        #
        # Their depth must therefore be zero, matching the background
        # convention of the ground-truth depth image.

        depth_map = depth_map * accumulated_opacity

        # ==========================================================================
        # 14. FINAL CLAMP
        # ==========================================================================

        depth_map = torch.clamp(
            depth_map,
            min=0.0,
            max=1.0,
        )

        # Keep depth_map as [N].
        #
        # performances.py performs:
        #
        #     test_depth.unsqueeze(-1)
        #
        # before computing MAE.

        # ==========================================================================
        # 15. SHANNON ENTROPY SPARSITY LOSS
        # ==========================================================================

        entropy = -weights * torch.log(
            weights + eps
        )

        sparsity_loss = torch.mean(
            torch.sum(
                entropy,
                dim=-1,
            )
        )

        # ==========================================================================
        # 16. RETURN
        # ==========================================================================

        return (
            chs_map,          # [N, channels]
            depth_map,        # [N]
            sparsity_loss,    # scalar
            mask,             # [N, 1]
        )