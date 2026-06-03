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

        # Attributes
        self.lin_range = lin_range
        self.ang_range = ang_range
        self.embeddings = nn.Embedding(
            num_embeddings=embeddings_num,
            embedding_dim=embeddings_dim,
            device=device
        )
        
        # Inizializziamo esplicitamente a zero così i residui partono da una correzione nulla
        nn.init.zeros_(self.embeddings.weight)

    def forward(
            self,
            idx: int
    ) -> torch.Tensor:
        """
        It takes an integer as input (i.e. the image index) and returns the corresponding embedding.

        Parameters:
        ----------
        idx: int
            an integer from 0 to (embeddings_num - 1).

        Returns:
        -------
        embedding: torch.Tensor[float]
            it is the embedding corresponding to the query index.
        """

        # Retrieve embedding vector
        idx = idx.to(torch.int32)
        
        # MODIFICA: Eliminata la funzione Tanh. L'embedder ora è libero e lineare.
        # Inserito un clamp di sicurezza a +/- 5 per evitare esplosioni incontrollate, mantenendo la linearità.
        embedding = torch.clamp(self.embeddings(idx), -5.0, 5.0).reshape(-1, 6)
        
        delta_pos = embedding[..., :3] * self.lin_range
        delta_theta = embedding[..., 3:] * self.ang_range

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