# Import modules
import torch
import torch.nn.functional as F


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
    full_psnr = 10. * torch.log10(max_pixel_value ** 2 / (full_mse + 1e-10))

    # 2. Object PSNR
    obj_pixels1 = img1[mask == 1]
    obj_pixels2 = img2[mask == 1]
    
    if obj_pixels1.numel() > 0:
        obj_mse = torch.mean((obj_pixels1 - obj_pixels2) ** 2)
        obj_psnr = 10. * torch.log10(max_pixel_value ** 2 / (obj_mse + 1e-10))
    else:
        obj_psnr = torch.tensor(0.0, device=img1.device)

    # 3. Background PSNR
    bkg_pixels1 = img1[mask == 0]
    bkg_pixels2 = img2[mask == 0]
    
    if bkg_pixels1.numel() > 0:
        bkg_mse = torch.mean((bkg_pixels1 - bkg_pixels2) ** 2)
        bkg_psnr = 10. * torch.log10(max_pixel_value ** 2 / (bkg_mse + 1e-10))
    else:
        bkg_psnr = torch.tensor(0.0, device=img1.device)

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


def compute_ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        resolution: int = 800,
        channels: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Structural Similarity Index (SSIM) (Full, Object-only, Background-only).
    Assumes img1 and img2 are flattened tensors of shape [H*W, C] or similar.
    Reshapes internally to 2D images [1, C, H, W] for structural kernel execution.
    """
    # Ripristiniamo la forma della griglia 2D dell'immagine [1, C, H, W]
    im1 = img1.view(1, resolution, resolution, channels).permute(0, 3, 1, 2)
    im2 = img2.view(1, resolution, resolution, channels).permute(0, 3, 1, 2)
    
    # Ricostruiamo la maschera spaziale speculare a 2D [1, 1, H, W]
    m_2d = mask.view(1, resolution, resolution, 1).permute(0, 3, 1, 2)

    # Generazione di una finestra Gaussiana standard 11x11 per l'SSIM
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

    # Calcolo delle medie locali (Mu) via convoluzione bidimensionale
    mu1 = F.conv2d(im1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(im2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calcolo delle covarianze locali (Sigma)
    sigma1_sq = F.conv2d(im1 * im1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(im2 * im2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(im1 * im2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    # Costanti empiriche standard per l'SSIM (assumendo pixel normalizzati a 1.0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Generazione della mappa di somiglianza strutturale locale
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 1. Full Image SSIM
    full_ssim = torch.mean(ssim_map)

    # Adattiamo la maschera sui canali della mappa per calcolare i sotto-target spaziali
    m_expanded = m_2d.repeat(1, channels, 1, 1)

    # 2. Object SSIM
    obj_pixels = ssim_map[m_expanded == 1]
    if obj_pixels.numel() > 0:
        obj_ssim = torch.mean(obj_pixels)
    else:
        obj_ssim = torch.tensor(1.0, device=img1.device) # 1.0 indica somiglianza perfetta (sfondo coerente vuoto)

    # 3. Background SSIM
    bkg_pixels = ssim_map[m_expanded == 0]
    if bkg_pixels.numel() > 0:
        bkg_ssim = torch.mean(bkg_pixels)
    else:
        bkg_ssim = torch.tensor(1.0, device=img1.device)

    return full_ssim, obj_ssim, bkg_ssim