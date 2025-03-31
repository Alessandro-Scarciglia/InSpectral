'''
This file includes functions with all metrics needed to evaluate model performances.
'''

# Import modules
import torch
import numpy as np


# Peak Signal To Noise Ratio (PSNR)
def compute_psnr(
        img1: torch.TensorType,
        img2: torch.TensorType,
        mask: torch.TensorType,
        max_pixel_value: int = 1.0
):
        # Calculate Mean Squared Error (MSE)
        full_mse = torch.mean((img1 - img2) ** 2)
        obj_mse = torch.mean((img1[mask==1] - img2[mask==1]) ** 2)
        bkg_mse = torch.mean((img1[mask==0] - img2[mask==0]) ** 2)

        # Calculate full PSNRs
        full_psnr = 10. * torch.log10(max_pixel_value ** 2 / full_mse)
        obj_psnr = 10. * torch.log10(max_pixel_value ** 2 / obj_mse)
        bkg_psnr = 10. * torch.log10(max_pixel_value ** 2 / bkg_mse)

        return full_psnr.item(), obj_psnr.item(), bkg_psnr.item()


# Mean Absolute Error (MAE)
def compute_mae(
        img1: torch.TensorType,
        img2: torch.TensorType,
        mask: torch.TensorType
):
        # Compute Mean Absolute Error (MAE)
        full_mae = torch.mean(torch.abs(img1 - img2))
        obj_mae = torch.mean(torch.abs(img1[mask==1] - img2[mask==1]))
        bkg_mae = torch.mean(torch.abs(img1[mask==0] - img2[mask==0]))

        return full_mae.item(), obj_mae.item(), bkg_mae.item() 
