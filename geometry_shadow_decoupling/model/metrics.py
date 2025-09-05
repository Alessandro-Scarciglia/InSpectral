# Import modules
import torch


def compute_psnr(
        img1: torch.TensorType,
        img2: torch.TensorType,
        max_pixel_value: int = 1.0
) -> float:
    """
    This function implements the computation of the PSNR between two images.

    Parameters:
    ----------
    img1, img2: torch.Tensor[float], torch.Tensor[float]
        two images to be compared with the PSNR criterion.
    max_pixel_value: float
		it is the greatest value of pixel intensity per channel (generally 1 or 255).
        
    Returns:
    -------
    psnr_value: float
		the scalar value of the PSNR.
    """
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2)

    # Calculate PSNR
    psnr = 10. * torch.log10(max_pixel_value ** 2 / mse)
    psnr_value = psnr.item()

    return psnr_value


