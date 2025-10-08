# Import modules
import torch
import torch.nn as nn


class AppearanceEmbedder(nn.Module):
    def __init__(
            self,
            embeddings_num: int,
            embeddings_dim: int,
            device: str
    ):
        """
        This class contains a table of embeddings whose purpose is to encode the appearance of the scene in
        a continuous space, and use this info to enrich the photometric rendering estimation.

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
        self.embeddings = nn.Embedding(
            num_embeddings=embeddings_num,
            embedding_dim=embeddings_dim,
            device=device
        )

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
        embedding = self.embeddings(idx)

        return embedding
