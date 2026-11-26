# Import modules
import torch
import torch.nn as nn


class ResEmbedder(nn.Module):
    def __init__(
            self,
            embeddings_num: int,
            embeddings_dim: int,
            lin_range: float,
            ang_range: float,
            device: str
    ):
        """
        This class contains a table of embeddings whose purpose is to encode the pose residuals of cameras in
        the continuous space, and use this info to adjust the rendering estimation.

        Attributes:
        ----------
        embeddings_num: int
            it is the number of embedding vectors to be generated (i.e. the table length).
        embeddings_dim: int
            it is the number of elements per vector.
        device: str
            it can be either "cpu" or "cuda:x". It is the target device.
        """
        super().__init__()

        if embeddings_dim != 6:
            raise ValueError("ResEmbedder stores 3 translation + 3 rotation values per image (embeddings_dim=6).")

        # Attributes
        # Each raw value is clamped to [-1, 1] and multiplied by its range, so lin_range and
        # ang_range are the largest correction allowed per axis (scene units / radians).
        self.lin_range = lin_range
        self.ang_range = ang_range

        # Translation and rotation live in two tables so the optimizer can give them
        # different learning rates (see "res_lr_rot" in configs/config_parameters.py).
        self.trans_embeddings = nn.Embedding(embeddings_num, 3, device=device)
        self.rot_embeddings = nn.Embedding(embeddings_num, 3, device=device)

        # Inizializziamo esplicitamente a zero così i residui partono da una correzione nulla
        nn.init.zeros_(self.trans_embeddings.weight)
        nn.init.zeros_(self.rot_embeddings.weight)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints written before the split hold one [N, 6] table ("embeddings.weight").
        legacy = state_dict.pop(prefix + "embeddings.weight", None)
        if legacy is not None:
            state_dict[prefix + "trans_embeddings.weight"] = legacy[:, :3].clone()
            state_dict[prefix + "rot_embeddings.weight"] = legacy[:, 3:].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
            self,
            idx: int
    ) -> torch.Tensor:
        """
        It takes an integer as input (i.e. the image index) and returns the corresponding pose residual.

        Parameters:
        ----------
        idx: int
            an integer from 0 to (embeddings_num - 1).

        Returns:
        -------
        delta_pos: torch.Tensor[float]
            translation residual of the camera centre, [B, 3].
        dR: torch.Tensor[float]
            rotation residual applied to the ray directions, [B, 3, 3].
        """

        # Retrieve embedding vectors
        idx = idx.to(torch.int32)

        # Linear parameterisation (no tanh). The clamp to [-1, 1] makes lin_range / ang_range
        # the maximum correction per axis.
        delta_pos = torch.clamp(self.trans_embeddings(idx), -1.0, 1.0).reshape(-1, 3) * self.lin_range
        delta_theta = torch.clamp(self.rot_embeddings(idx), -1.0, 1.0).reshape(-1, 3) * self.ang_range

        # MODIFICA DI STABILITÀ (Evita i NAN causati dalla derivata di sqrt(0) all'inizio)
        # Sostituiamo torch.norm diretto con un calcolo protetto da un epsilon
        theta_sq = torch.sum(delta_theta ** 2, dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq + 1e-12)

        eps = 1e-8
        omega_normalized = delta_theta / (theta + eps)

        wx, wy, wz = omega_normalized.unbind(-1)

        zeros = torch.zeros_like(wx)

        K = torch.stack([
            torch.stack([ zeros, -wz,  wy], dim=-1),
            torch.stack([  wz, zeros, -wx], dim=-1),
            torch.stack([ -wy,  wx, zeros], dim=-1)
        ], dim=-2)

        I = torch.eye(3, device=delta_theta.device).expand(K.shape[0], 3, 3)

        sin_theta = torch.sin(theta)[..., None]
        cos_theta = torch.cos(theta)[..., None]

        dR = (
            I
            + sin_theta * K
            + (1 - cos_theta) * (K @ K)
        )
    
        return delta_pos, dR