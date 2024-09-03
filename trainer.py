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
                 weight_decay: int = 1e-6,
                 degenerated_to_sgd: bool = False,
                 tot_var_weight: float = 1e-6,
                 sparsity_loss_weight: float = 1e-10,
                 tot_var_stop: int = 1000,
                 decay_rate: float = 1e-1,
                 decay_steps: int = 1000):
        
        # Attributes
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.degenerated_to_sgd = degenerated_to_sgd
        self.tot_var_w = tot_var_weight
        self.tot_var_stop = tot_var_stop
        self.sparsity_loss_w = sparsity_loss_weight
        self.decay_rate = decay_rate
        self.global_steps = 0
        self.decay_steps = decay_steps
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
        # c2w = c2w.to(self.params.get_param("device"))
        # frame = frame.to(self.params.get_param("device"))

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
                self.params.get_param("n_levels")
                ) for i in range(self.params.get_param("n_levels"))
        )

        # Compute combination of losses
        loss = loss_on_colors + self.sparsity_loss_w * sparsity_loss.sum() + self.tot_var_w * tot_var_loss

        # Reject the total variation after first N iterations
        if niter > self.tot_var_stop:
            self.tot_var_w = 0

        # Backprop
        loss.backward()
        self.optimizer.step()

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * (self.decay_rate ** (self.global_steps / self.decay_steps))

        # Increase the global steps
        self.global_steps += 1

        return loss

        
