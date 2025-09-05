# Import modules
import torch


def compute_psnr(
		img1: torch.TensorType,
		img2: torch.TensorType,
		mask: torch.TensorType,
		max_pixel_value: int = 1.0
) -> float:
	"""
	This function implements the computation of the PSNR between two images.

	Parameters:
	----------
	img1, img2: torch.Tensor[float], torch.Tensor[float]
		two images to be compared with the PSNR criterion.
	mask: torch.Tensor[bool]
		it is the segmentation mask object vs. background.
	max_pixel_value: float
		it is the greatest value of pixel intensity per channel (generally 1 or 255).
		
	Returns:
	-------
	full_psnr_val: float
		it is the PSNR score of the full image.
	obj_psnr_val: float
		it is the PSNR score of the object only (selected by mask).
	bkg_psnr_val: float
		it is the PSNR score of the background only (selected by mask).
	"""

	# Calculate Mean Squared Error (MSE)
	full_mse = torch.mean((img1 - img2) ** 2)
	obj_mse = torch.mean((img1[mask==1] - img2[mask==1]) ** 2)
	bkg_mse = torch.mean((img1[mask==0] - img2[mask==0]) ** 2)

	# Calculate full PSNRs
	full_psnr = 10. * torch.log10(max_pixel_value ** 2 / full_mse)
	obj_psnr = 10. * torch.log10(max_pixel_value ** 2 / obj_mse)
	bkg_psnr = 10. * torch.log10(max_pixel_value ** 2 / bkg_mse)
	
	full_psnr_val = full_psnr.item()
	obj_psnr_val = obj_psnr.item()
	bkg_psnr_val = bkg_psnr.item()

	return full_psnr_val, obj_psnr_val, bkg_psnr_val

def compute_mae(
		img1: torch.TensorType,
		img2: torch.TensorType,
		mask: torch.TensorType
):
		"""
		This function implements the Mean Absolute Error. It is a measure of how much a value
		shifts from a reference in absolute value. It is suitable for the evaluation of depth maps, where a shift
		towards or backwards shall be equally avoided.

		Parameters:
		----------
		img1, img2: torch.Tensor[float], torch.Tensor[float]
			they are two input images to be compared (target and label).
		mask: torch.Tensor[bool]
			it is the input mask to 
		"""
		
		# Compute Mean Absolute Error (MAE)
		full_mae = torch.mean(torch.abs(img1 - img2))
		obj_mae = torch.mean(torch.abs(img1[mask==1] - img2[mask==1]))
		bkg_mae = torch.mean(torch.abs(img1[mask==0] - img2[mask==0]))

		full_mae_val = full_mae.item()
		obj_mae_val = obj_mae.item()
		bkg_mae_val = bkg_mae.item()

		return full_mae_val, obj_mae_val, bkg_mae_val