import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as TM

from PIL import Image

import nerfstudio.data.datasets.base_dataset as ns_base_dataset

from nerfstudio.utils.eval_utils import eval_setup


# =============================================================
# DEPTH CONFIGURATION
# =============================================================

# Ground truth:
# 0.0 = distance >= 40 meters (far)
# 1.0 = camera origin (near)

DEPTH_MAX_METERS = 40.0

# Original camera translations in transforms.json must be in meters.
CAMERA_POSE_UNITS_ARE_METERS = True


# =============================================================
# MASK CONVERSION
# =============================================================

def install_mask_conversion_patch():

    original_loader = (
        ns_base_dataset.get_image_mask_tensor_from_path
    )

    def patched_mask_loader(filepath, scale_factor=1.0):

        source_path = Path(filepath)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Mask file not found: {source_path}"
            )

        with Image.open(source_path) as image:

            if len(image.getbands()) == 1:

                converted_path = source_path

            else:

                converted_path = source_path.with_name(
                    source_path.stem
                    + "_grayscale"
                    + source_path.suffix
                )

                needs_conversion = (
                    not converted_path.exists()
                    or source_path.stat().st_mtime
                    > converted_path.stat().st_mtime
                )

                if needs_conversion:

                    if "A" in image.getbands():

                        alpha = image.getchannel("A")
                        alpha_min, alpha_max = alpha.getextrema()

                        if alpha_min != alpha_max:
                            grayscale = alpha
                        else:
                            grayscale = image.convert("L")

                    else:
                        grayscale = image.convert("L")

                    grayscale.save(converted_path)

                    print(
                        f"[MASK CONVERSION] "
                        f"{source_path.name} -> "
                        f"{converted_path.name}"
                    )

        return original_loader(
            filepath=converted_path,
            scale_factor=scale_factor
        )

    ns_base_dataset.get_image_mask_tensor_from_path = (
        patched_mask_loader
    )

    print(
        "[MASK CONVERSION] "
        "Automatic single-channel conversion enabled."
    )


# =============================================================
# DATASET PATH UTILITIES
# =============================================================

def resolve_dataset_path(dataset_root, relative_path):

    path = Path(relative_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (dataset_root / path).resolve()


def load_depth_lookup(dataset_root):

    """
    Reads transforms.json.

    Expected JSON keys:

        file_path
        mask_path
        depth_path

    Example:

        file_path: images/frame_001.png
        mask_path: images/masks/frame_001.png
        depth_path: images/depth/frame_001.png

    Creates a mapping:

        absolute image path -> absolute depth path
    """

    transforms_path = dataset_root / "transforms.json"

    if not transforms_path.exists():

        raise FileNotFoundError(
            f"transforms.json not found: {transforms_path}"
        )

    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])

    if not frames:
        raise ValueError(
            "No frames found in transforms.json."
        )

    depth_lookup = {}

    frames_without_depth = []

    for frame in frames:

        if "file_path" not in frame:

            raise KeyError(
                "A frame in transforms.json has no file_path."
            )

        image_path = resolve_dataset_path(
            dataset_root,
            frame["file_path"]
        )

        if "depth_path" not in frame:

            frames_without_depth.append(frame["file_path"])
            continue

        depth_path = resolve_dataset_path(
            dataset_root,
            frame["depth_path"]
        )

        if not depth_path.exists():

            raise FileNotFoundError(
                f"Depth file not found:\n"
                f"Image: {image_path}\n"
                f"Depth: {depth_path}"
            )

        depth_lookup[str(image_path)] = depth_path

    print("\n[DEPTH LOADER] Dataset metadata loaded.")
    print(f"[DEPTH LOADER] Total frames: {len(frames)}")
    print(f"[DEPTH LOADER] Frames with depth: {len(depth_lookup)}")

    if frames_without_depth:

        print(
            f"[DEPTH LOADER] Frames without depth: "
            f"{len(frames_without_depth)}"
        )

    if not depth_lookup:

        raise RuntimeError(
            "No depth_path entries found in transforms.json."
        )

    return depth_lookup


# =============================================================
# NERFSTUDIO SCALE
# =============================================================

def get_dataparser_scale(eval_dataset):

    """
    Retrieves the scale applied by Nerfstudio to the camera poses.

    Nerfstudio may rescale camera translations through auto_scale_poses.

    Predicted depth is expressed in the resulting scene coordinates.
    To recover the original coordinate units:

        depth_original = depth_scene / dataparser_scale
    """

    dataparser_outputs = getattr(
        eval_dataset,
        "_dataparser_outputs",
        None
    )

    if dataparser_outputs is None:

        dataparser_outputs = getattr(
            eval_dataset,
            "dataparser_outputs",
            None
        )

    if dataparser_outputs is None:

        raise RuntimeError(
            "Cannot access eval_dataset dataparser outputs. "
            "Unable to recover the original depth scale."
        )

    scale = getattr(
        dataparser_outputs,
        "dataparser_scale",
        None
    )

    if scale is None:

        raise RuntimeError(
            "dataparser_scale not available in Nerfstudio dataset."
        )

    scale = float(scale)

    if not math.isfinite(scale) or scale <= 0:

        raise ValueError(
            f"Invalid Nerfstudio dataparser_scale: {scale}"
        )

    print(
        f"\n[DEPTH SCALE] Nerfstudio dataparser_scale: {scale}"
    )

    return scale


# =============================================================
# RGB METRICS
# =============================================================

def calculate_psnr(pred, gt):

    mse = F.mse_loss(pred, gt)

    if mse.item() == 0:
        return float("inf")

    return (
        10.0 * torch.log10(1.0 / mse)
    ).item()


def calculate_ssim(pred, gt):

    # Input format: [1, C, H, W]

    pred = pred.permute(2, 0, 1).unsqueeze(0)
    gt = gt.permute(2, 0, 1).unsqueeze(0)

    return TM.structural_similarity_index_measure(
        pred,
        gt,
        data_range=1.0
    ).item()


# =============================================================
# GROUND-TRUTH DEPTH LOADER
# =============================================================

def load_ground_truth_depth(
    depth_path,
    height,
    width,
    device,
    dtype
):

    """
    Loads an 8-bit grayscale depth image.

    Ground truth convention:

        0   = distance >= 40 meters
        255 = camera origin

    Normalization:

        depth = grayscale / 255

    No depth_unit_scale_factor is applied.
    """

    with Image.open(depth_path) as image:

        grayscale = image.convert("L")

        if grayscale.size != (width, height):

            grayscale = grayscale.resize(
                (width, height),
                Image.NEAREST
            )

        depth_array = np.asarray(
            grayscale,
            dtype=np.float32
        ).copy()

    # Normalize [0, 255] -> [0, 1]
    depth_array /= 255.0

    gt_depth = torch.from_numpy(
        depth_array
    ).unsqueeze(-1)

    return gt_depth.to(
        device=device,
        dtype=dtype
    )


# =============================================================
# PREDICTED DEPTH NORMALIZATION
# =============================================================

def normalize_predicted_depth(
    pred_depth,
    dataparser_scale,
    max_distance_meters=40.0
):

    """
    Converts Nerfacto predicted depth into the ground-truth convention.

    Nerfacto depth:
        expressed in Nerfstudio scene coordinates.

    Ground truth:
        0 = far (>= 40 m)
        1 = near (camera origin)

    Steps:

        1. Recover original coordinate units.
        2. Interpret original units as meters.
        3. Normalize using the fixed 40 m range.
        4. Invert the depth convention.
        5. Clamp to [0, 1].

    Formula:

        depth_meters = depth_scene / dataparser_scale

        depth_normalized =
            clamp(1 - depth_meters / 40, 0, 1)
    """

    if pred_depth.ndim == 3 and pred_depth.shape[-1] == 1:

        depth_scene = pred_depth.squeeze(-1)

    elif pred_depth.ndim == 2:

        depth_scene = pred_depth

    else:

        raise ValueError(
            f"Unexpected predicted depth shape: "
            f"{pred_depth.shape}"
        )

    if dataparser_scale <= 0:

        raise ValueError(
            f"Invalid dataparser scale: {dataparser_scale}"
        )

    # ---------------------------------------------------------
    # Recover original coordinate units (meters)
    # ---------------------------------------------------------

    depth_meters = depth_scene / dataparser_scale

    # ---------------------------------------------------------
    # Valid prediction pixels
    # ---------------------------------------------------------

    valid_mask = (
        torch.isfinite(depth_meters)
        & (depth_meters >= 0)
    )

    if not valid_mask.any():

        raise ValueError(
            "Predicted depth contains no valid pixels."
        )

    # ---------------------------------------------------------
    # Fixed inverse normalization
    # ---------------------------------------------------------

    normalized = torch.zeros_like(depth_meters)

    normalized[valid_mask] = (
        1.0
        -
        (
            depth_meters[valid_mask]
            / max_distance_meters
        )
    )

    normalized = torch.clamp(
        normalized,
        min=0.0,
        max=1.0
    )

    statistics = {

        "pred_depth_min_meters": (
            depth_meters[valid_mask].min().item()
        ),

        "pred_depth_max_meters": (
            depth_meters[valid_mask].max().item()
        ),

        "dataparser_scale": dataparser_scale,

        "max_distance_meters": max_distance_meters
    }

    return (
        normalized.unsqueeze(-1),
        valid_mask,
        statistics
    )


# =============================================================
# DEPTH MAE
# =============================================================

def calculate_depth_mae(
    pred_depth,
    gt_depth,
    dataparser_scale
):

    """
    Computes MAE between normalized inverse depth maps.

    IMPORTANT:

        Ground-truth zero values are INCLUDED.

        Zero represents a valid distance >= 40 meters,
        not an invalid depth pixel.
    """

    if pred_depth.shape != gt_depth.shape:

        raise ValueError(
            f"Depth dimension mismatch: "
            f"prediction={pred_depth.shape}, "
            f"ground truth={gt_depth.shape}"
        )

    normalized_pred, valid_mask, statistics = (
        normalize_predicted_depth(
            pred_depth=pred_depth,
            dataparser_scale=dataparser_scale,
            max_distance_meters=DEPTH_MAX_METERS
        )
    )

    pred = normalized_pred.squeeze(-1)
    gt = gt_depth.squeeze(-1)

    # Do not exclude gt == 0.
    valid_mask = (
        valid_mask
        & torch.isfinite(gt)
    )

    if not valid_mask.any():

        return None, 0, statistics

    errors = torch.abs(
        pred[valid_mask] - gt[valid_mask]
    )

    return (
        errors.mean().item(),
        errors.numel(),
        statistics
    )


# =============================================================
# JSON UTILITIES
# =============================================================

def sanitize_json(value):

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):

        return {
            k: sanitize_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            sanitize_json(v)
            for v in value
        ]

    return value


# =============================================================
# EVALUATION
# =============================================================

def evaluate(config_path, output_path):

    # ---------------------------------------------------------
    # 1. Initialize Nerfstudio
    # ---------------------------------------------------------

    config, pipeline, checkpoint_path, _ = eval_setup(
        config_path
    )

    pipeline.eval()
    pipeline.model.eval()

    datamanager = pipeline.datamanager

    dataloader = (
        datamanager.fixed_indices_eval_dataloader
    )

    eval_dataset = datamanager.eval_dataset

    # ---------------------------------------------------------
    # 2. Dataset configuration
    # ---------------------------------------------------------

    dataset_root = Path(
        config.pipeline.datamanager.dataparser.data
    ).expanduser()

    if dataset_root.suffix.lower() == ".json":
        dataset_root = dataset_root.parent

    dataset_root = dataset_root.resolve()

    depth_lookup = load_depth_lookup(dataset_root)

    # Recover Nerfstudio scene scaling
    dataparser_scale = get_dataparser_scale(
        eval_dataset
    )

    if not CAMERA_POSE_UNITS_ARE_METERS:

        raise ValueError(
            "The current depth conversion assumes camera poses "
            "are expressed in meters."
        )

    print(
        f"\n[EVALUATION] Images: "
        f"{len(eval_dataset.image_filenames)}"
    )

    print(
        f"[EVALUATION] Ground truth: "
        f"grayscale [0, 1]"
    )

    print(
        "[EVALUATION] Convention: "
        "0 = >= 40 m, 1 = camera origin"
    )

    # ---------------------------------------------------------
    # 3. Global accumulators
    # ---------------------------------------------------------

    results_per_image = []

    total_depth_error = 0.0
    total_valid_depth_pixels = 0

    # ---------------------------------------------------------
    # 4. Evaluation loop
    # ---------------------------------------------------------

    with torch.no_grad():

        for idx, (camera, batch) in enumerate(dataloader):

            # -------------------------------------------------
            # Render RGB and depth
            # -------------------------------------------------

            outputs = pipeline.model.get_outputs_for_camera(
                camera
            )

            pred_rgb = outputs["rgb"]
            pred_depth = outputs["depth"]

            # -------------------------------------------------
            # Ground-truth RGB
            # -------------------------------------------------

            if "image" not in batch:

                raise KeyError(
                    f"RGB image not found. "
                    f"Available keys: {list(batch.keys())}"
                )

            gt_rgb = batch["image"].to(
                pred_rgb.device
            )

            gt_rgb = (
                pipeline.model.renderer_rgb.blend_background(
                    gt_rgb
                )
            )

            # -------------------------------------------------
            # RGB metrics
            # -------------------------------------------------

            if pred_rgb.shape != gt_rgb.shape:

                raise ValueError(
                    f"RGB dimension mismatch: "
                    f"prediction={pred_rgb.shape}, "
                    f"ground truth={gt_rgb.shape}"
                )

            psnr = calculate_psnr(
                pred_rgb,
                gt_rgb
            )

            ssim = calculate_ssim(
                pred_rgb,
                gt_rgb
            )

            # -------------------------------------------------
            # Evaluation image index
            # -------------------------------------------------

            image_idx = batch["image_idx"]

            if torch.is_tensor(image_idx):

                image_idx = int(
                    image_idx.reshape(-1)[0].item()
                )

            else:

                image_idx = int(image_idx)

            if image_idx >= len(eval_dataset.image_filenames):

                raise IndexError(
                    f"Invalid evaluation image index: {image_idx}"
                )

            image_path = Path(
                eval_dataset.image_filenames[image_idx]
            ).resolve()

            # -------------------------------------------------
            # Retrieve corresponding depth path
            # -------------------------------------------------

            image_key = str(image_path)

            if image_key not in depth_lookup:

                raise KeyError(
                    f"No depth_path associated with image:\n"
                    f"{image_path}\n"
                    "Check transforms.json."
                )

            depth_path = depth_lookup[image_key]

            # -------------------------------------------------
            # Load ground-truth depth
            # -------------------------------------------------

            height = int(gt_rgb.shape[0])
            width = int(gt_rgb.shape[1])

            gt_depth = load_ground_truth_depth(
                depth_path=depth_path,
                height=height,
                width=width,
                device=pred_depth.device,
                dtype=pred_depth.dtype
            )

            # -------------------------------------------------
            # Calculate depth MAE
            # -------------------------------------------------

            depth_mae, valid_pixels, depth_stats = (
                calculate_depth_mae(
                    pred_depth=pred_depth,
                    gt_depth=gt_depth,
                    dataparser_scale=dataparser_scale
                )
            )

            if depth_mae is not None:

                total_depth_error += (
                    depth_mae * valid_pixels
                )

                total_valid_depth_pixels += valid_pixels

            # -------------------------------------------------
            # Store per-image results
            # -------------------------------------------------

            image_results = {

                "image_id": image_idx,

                "image_path": str(image_path),

                "depth_path": str(depth_path),

                "psnr": psnr,

                "ssim": ssim,

                "depth_mae": depth_mae,

                "valid_depth_pixels": valid_pixels,

                "depth_statistics": depth_stats
            }

            results_per_image.append(
                image_results
            )

            print(
                f"Image {image_idx:04d} | "
                f"PSNR: {psnr:.4f} | "
                f"SSIM: {ssim:.4f} | "
                f"Depth MAE: "
                f"{depth_mae if depth_mae is not None else 'N/A'}"
            )

    # ---------------------------------------------------------
    # 5. Global metrics
    # ---------------------------------------------------------

    if not results_per_image:

        raise RuntimeError(
            "Evaluation dataloader returned no images."
        )

    valid_results = [
        r for r in results_per_image
        if r["depth_mae"] is not None
    ]

    global_psnr = sum(
        r["psnr"] for r in results_per_image
    ) / len(results_per_image)

    global_ssim = sum(
        r["ssim"] for r in results_per_image
    ) / len(results_per_image)

    global_depth_mae = (
        total_depth_error / total_valid_depth_pixels
        if total_valid_depth_pixels > 0
        else None
    )

    mean_image_depth_mae = (
        sum(
            r["depth_mae"] for r in valid_results
        ) / len(valid_results)
        if valid_results
        else None
    )

    global_results = {

        "checkpoint": str(checkpoint_path),

        "dataset": str(dataset_root),

        "num_images": len(results_per_image),

        "depth_ground_truth_encoding": "grayscale_uint8_0_255",

        "depth_ground_truth_convention": "0=distance >= 40m, 1=camera_origin",

        "depth_max_distance_meters": DEPTH_MAX_METERS,

        "camera_pose_units": "meters",

        "nerfstudio_dataparser_scale": dataparser_scale,

        "depth_prediction_conversion": (
            "clamp(1 - (depth_scene / dataparser_scale) / 40, 0, 1)"
        ),

        "psnr": global_psnr,

        "ssim": global_ssim,

        "depth_mae": global_depth_mae,

        "depth_mae_mean_per_image": mean_image_depth_mae,

        "valid_depth_pixels": total_valid_depth_pixels,

        "per_image": results_per_image
    }

    global_results = sanitize_json(
        global_results
    )

    # ---------------------------------------------------------
    # 6. Save JSON
    # ---------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:

        json.dump(
            global_results,
            f,
            indent=4,
            allow_nan=False
        )

    # ---------------------------------------------------------
    # 7. Print results
    # ---------------------------------------------------------

    print("\n===================================")
    print("          FINAL RESULTS")
    print("===================================")

    print(f"Images: {global_results['num_images']}")

    print(f"PSNR: {global_results['psnr']}")

    print(f"SSIM: {global_results['ssim']}")

    print(f"Depth MAE: {global_results['depth_mae']}")

    print(
        "Depth MAE mean per image: "
        f"{global_results['depth_mae_mean_per_image']}"
    )

    print(
        f"Valid depth pixels: "
        f"{global_results['valid_depth_pixels']}"
    )

    print(f"\nResults saved to: {output_path}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # PATH CONFIGURATION
    # ---------------------------------------------------------

    CONFIG_PATH = Path(
        "/home/vision/Desktop/InSpectral/benchmarks/unnamed/instant-ngp/2026-09-24_165749/config.yml"
    )

    OUTPUT_PATH = Path(
        "/home/vision/Desktop/InSpectral/benchmarks/unnamed/instant-ngp/metrics.json"
    )

    # ---------------------------------------------------------
    # INSTALL MASK CONVERSION
    # ---------------------------------------------------------

    install_mask_conversion_patch()

    # ---------------------------------------------------------
    # RUN EVALUATION
    # ---------------------------------------------------------

    evaluate(
        config_path=CONFIG_PATH,
        output_path=OUTPUT_PATH
    )