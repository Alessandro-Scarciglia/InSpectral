'''
This file includes functions with all metrics needed to evaluate model performances.
'''

# Import modules
import torch


# Peak Signal To Noise Ratio (PSNR)
def compute_psnr(
        img1: torch.TensorType,
        img2: torch.TensorType,
        max_pixel_value: int = 1.0
):
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2)

    # Calculate PSNR
    psnr_value = 10. * torch.log10(max_pixel_value ** 2 / mse)

    return psnr_value.item()


