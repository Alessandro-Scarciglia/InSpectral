# Import modules
import torch
import numpy as np

# Import custom modules
from radam import RAdam
from parameters import RenderingParameters
from loss import total_variation_loss


class Trainer:
    def __init__(self,
                 model,
                 lr: float = 0.001,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: int = 0,
                 degenerated_to_sgd: bool = False,
                 tot_var_weight: float = 1e-6,
                 tot_var_stop: int = 1000):
        
        # Attributes
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.degenerated_to_sgd = degenerated_to_sgd
        self.tot_var_w = tot_var_weight
        self.tot_var_stop = tot_var_stop
        self.params = RenderingParameters()

        # Lambdas
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

        # Instantiate the optimizer
        self.optimizer = RAdam(
            params=model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            degenerated_to_sgd=self.degenerated_to_sgd
        )


    def train_one_frame(self,
                       c2w: torch.Tensor,
                       frame: torch.Tensor,
                       niter: int):
        
        # Shift input and target to selected device
        c2w = c2w.to(self.params.get_param("device"))
        frame = frame.to(self.params.get_param("device"))

        # Forward pass
        chs_map, _, sparsity_loss = self.model(c2w)

        # Compute losses
        self.optimizer.zero_grad()
        loss_on_colors = self.img2mse(chs_map, frame)
        tot_var_loss = sum(
            total_variation_loss(
                self.model.embedder.embeddings[i], \
                self.params.get_param("low_res"), self.params.get_param("high_res"), \
                i, self.params.get_param("log2_hashmap_size"), \
                self.params.get_all_params("n_levels")
                ) for i in range(self.params.get_all_params("n_levels"))
        )

        # Compute combination of losses
        loss = loss_on_colors  + sparsity_loss + self.to_var_w * tot_var_loss

        # Reject the total variation after first N iterations
        if niter > self.tot_var_stop:
            self.tot_var_w = 0

        # Backprop
        loss.backward()
        self.optimizer.step()

        print(loss.shape)
        exit()