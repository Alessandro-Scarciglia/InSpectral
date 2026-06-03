# Import modules
import torch


def compute_mse(
        val1: torch.Tensor,
        val2: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Mean Square Error (MSE) between two tensors.
    """
    return torch.mean((val1 - val2) ** 2)


def compute_psnr(
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        max_pixel_value: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the PSNR between two images (Full, Object-only, Background-only).
    Handles empty masks safely to prevent NaN propagation.
    """
    # 1. Full Image MSE & PSNR
    full_mse = compute_mse(img1, img2)
    # Aggiungiamo un eps (1e-10) per evitare divisioni per zero se le immagini fossero identiche
    full_psnr = 10. * torch.log10(max_pixel_value ** 2 / (full_mse + 1e-10))

    # 2. Object PSNR (Messa in sicurezza contro maschere vuote)
    obj_pixels1 = img1[mask == 1]
    obj_pixels2 = img2[mask == 1]
    
    if obj_pixels1.numel() > 0:
        obj_mse = torch.mean((obj_pixels1 - obj_pixels2) ** 2)
        obj_psnr = 10. * torch.log10(max_pixel_value ** 2 / (obj_mse + 1e-10))
    else:
        # Se l'oggetto non è visibile nel frame, assegna un valore neutro (0.0)
        obj_psnr = torch.tensor(0.0, device=img1.device)

    # 3. Background PSNR (Messa in sicurezza se l'immagine fosse occupata al 100% dall'oggetto)
    bkg_pixels1 = img1[mask == 0]
    bkg_pixels2 = img2[mask == 0]
    
    if bkg_pixels1.numel() > 0:
        bkg_mse = torch.mean((bkg_pixels1 - bkg_pixels2) ** 2)
        bkg_psnr = 10. * torch.log10(max_pixel_value ** 2 / (bkg_mse + 1e-10))
    else:
        bkg_psnr = torch.tensor(0.0, device=img1.device)

    # Restituiamo i tensori scalari di PyTorch, mantenendo il device coerente
    return full_psnr, obj_psnr, bkg_psnr


def compute_mae(
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Mean Absolute Error (Full, Object-only, Background-only).
    Handles empty masks safely to prevent NaN propagation.
    """
    # 1. Full Image MAE
    full_mae = torch.mean(torch.abs(img1 - img2))

    # 2. Object MAE
    obj_pixels1 = img1[mask == 1]
    obj_pixels2 = img2[mask == 1]
    
    if obj_pixels1.numel() > 0:
        obj_mae = torch.mean(torch.abs(obj_pixels1 - obj_pixels2))
    else:
        obj_mae = torch.tensor(0.0, device=img1.device)

    # 3. Background MAE
    bkg_pixels1 = img1[mask == 0]
    bkg_pixels2 = img2[mask == 0]
    
    if bkg_pixels1.numel() > 0:
        bkg_mae = torch.mean(torch.abs(bkg_pixels1 - bkg_pixels2))
    else:
        bkg_mae = torch.tensor(0.0, device=img1.device)

    return full_mae, obj_mae, bkg_mae