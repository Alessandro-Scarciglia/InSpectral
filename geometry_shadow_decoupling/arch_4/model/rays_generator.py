# Import modules
import torch
import torch.nn as nn
import numpy as np


class RaysGeneratorSynth(nn.Module):
    """
    This class implements the methods for rays generation starting from a camera-to-world
    transformation and employing information regarding the camera.
    """
    def __init__(
            self,
            height: int,
            width: int,
            num_ch: int,
            intrinsic_matrix: np.ndarray
    ):
        super(RaysGeneratorSynth, self).__init__()
        
        # Attributes
        self.height = height
        self.width = width
        self.ch = num_ch
        
        # Convertiamo subito la matrice intrinseca in tensore
        intrinsic_tensor = torch.from_numpy(intrinsic_matrix).float()

        # Massima ottimizzazione: generiamo la griglia di pixel statica
        i_span = torch.linspace(0, self.width - 1, self.width)
        j_span = torch.linspace(0, self.height - 1, self.height)
        it, jt = torch.meshgrid(i_span, j_span, indexing='ij')
        i, j = it.t(), jt.t()
        
        # Calcoliamo le direzioni (dirs) della camera una volta sola qui nell'__init__
        # Questo elimina qualsiasi interazione con variabili CPU a runtime
        dirs_static = torch.stack([
            (i - intrinsic_tensor[0, 2]) / intrinsic_tensor[0, 0],
            (j - intrinsic_tensor[1, 2]) / intrinsic_tensor[1, 1],
            torch.ones_like(i)
        ], dim=-1)
        
        # Registriamo SOLO le direzioni statiche come unico buffer ufficiale.
        # Spostando questo modulo sul device, 'dirs' si sposterà istantaneamente su GPU.
        self.register_buffer("dirs", dirs_static)

    def get_rays(
            self,
            c2w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates HxW rays starting from camera origin towards the scene.
        """
        # Forza c2w sul tipo float32 corretto
        c2w = c2w.float()

        # Estraiamo la matrice di rotazione 3x3
        rotation = c2w[:3, :3]  # Shape: [3, 3]
        
        # Ruota le direzioni usando la rotazione della camera.
        # Usiamo torch.matmul o einsum che sono nativi e garantiscono l'allineamento dei device su CUDA.
        # self.dirs ha shape [H, W, 3], vogliamo moltiplicarlo per rotation [3, 3] -> esito [H, W, 3]
        rays_d = torch.einsum('hwj,ij->hwi', self.dirs, rotation)
        
        # Estraiamo la traslazione (origine della camera) -> Shape: [3]
        camera_origin = c2w[:3, -1]
        
        # Espandiamo l'origine per coprire l'intera griglia dei raggi [H, W, 3]
        rays_o = camera_origin.expand(rays_d.shape)

        return rays_o, rays_d
    
    def forward(
            self,
            c2w: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward function returns rays when in inference-only mode.
        """
        # Compute rays 
        rays_o, rays_d = self.get_rays(c2w)
        
        # Concatena Origini e Direzioni lungo l'ultimo canale -> Risultato: [H, W, 6]
        out = torch.cat([rays_o, rays_d], dim=-1)
    
        return out