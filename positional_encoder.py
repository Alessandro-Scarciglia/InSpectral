# Import modules
import torch
import torch.nn as nn


class PositionalEmbedder(nn.Module):
    def __init__(self, n_freq: int = 5):
        super(PositionalEmbedder, self).__init__()
        
        # Attributes
        self.n_freq = n_freq

    # Forward method for embeddings
    def forward(self, x):
        out = [x]
        for j in range(self.n_freq):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        
        return torch.cat(out, dim=1)


if __name__ == "__main__":
    enc = PositionalEmbedder(n_freq=5)
    viewdir = torch.tensor([[0, 0, 1],
                            [1, 0, 0]])
    out = enc(viewdir)

    print(out.shape)