"""Render a trained geometry-shadow-decoupled NeRF checkpoint."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config_parameters import (
    app_parameters,
    cfg_parameters,
    dataset_parameters,
    hash_parameters,
    nerf_parameters,
    rays_parameters,
    sampler_parameters,
    sh_parameters,
    training_parameters,
)
from data.generate_postprocessed_dataset import calculate_intrinsic_matrix
from data.rays_generator import RaysGeneratorSynth
from model.rendering import NeuralRenderer


# Edit these values, then run this file directly from the IDE.
INFERENCE_CONFIG = {
    # Model and input data
    "checkpoint_path": "results/folder_2026-11-05_18-24-13/epoch_4/chkpt_nerf.pt",
    "transforms_path": None,  # None selects transforms_test.json, then transforms_train.json.
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "resolution": cfg_parameters["resolution"],
    "chunk_size": 4096,
    "normalization_stride": 4,  # Must match the stride used to normalize the training camera poses.

    # Choose "appearance" to keep the camera fixed, or "camera" to move it with fixed appearance.
    "mode": "camera",  # "appearance" or "camera".

    # Appearance sweep (mode="appearance")
    "camera_index": 120,  # Frame index in transforms_path.
    "appearance_start": 0.0,
    "appearance_end": 89,  # None uses the last embedding in the checkpoint.
    "appearance_steps": 30,

    # Camera sweep (mode="camera")
    "camera_source": "trajectory",  # "trajectory" or "sphere".
    "appearance_position": 0,  # Fractional positions interpolate adjacent appearance embeddings.
    "views": 10,  # Number of camera poses when camera_source="sphere".
    "sphere_radius": 1.0,  # Radius in the normalized scene coordinates.

    # Optional 3D geometry/appearance visualization for one camera view.
    "visualize_3d": True,
    "visualize_3d_index": 1,
    "pixel_stride": 4,
    "sigma_threshold": 5,
    "show_3d": True,
}


def make_camera_pose(position: np.ndarray) -> torch.Tensor:
    """Build an OpenGL-style camera-to-world pose with its local -Z aimed at the origin."""
    backward = position / np.linalg.norm(position)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(backward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.column_stack((right, up, backward))
    pose[:3, 3] = position
    return torch.from_numpy(pose)


def sphere_poses(radius: float, count: int) -> list[torch.Tensor]:
    if radius <= 0:
        raise ValueError("--sphere-radius must be positive.")
    if count < 1:
        raise ValueError("--views must be at least 1.")
    indices = np.arange(count, dtype=np.float64)
    y = 1.0 - 2.0 * (indices + 0.5) / count
    radial = np.sqrt(1.0 - y * y)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    positions = radius * np.column_stack((radial * np.cos(theta), y, radial * np.sin(theta)))
    return [make_camera_pose(position.astype(np.float32)) for position in positions]


def read_dataset(path: Path) -> tuple[dict, list[dict]]:
    with path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No camera frames found in {path}.")
    return metadata, frames


def dataset_camera_poses(frames: list[dict], normalization_stride: int) -> list[torch.Tensor]:
    if normalization_stride < 1:
        raise ValueError("--normalization-stride must be positive.")
    source_poses = []
    for frame in frames:
        pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        if pose.shape != (4, 4):
            raise ValueError("Every transform_matrix must have shape 4x4.")
        source_poses.append(pose)
    normalization_poses = torch.stack(source_poses[::normalization_stride])
    max_radius = torch.linalg.vector_norm(normalization_poses[:, :3, 3], dim=-1).max()
    if max_radius <= 0:
        raise ValueError("Cannot normalize a camera trajectory whose poses all lie at the origin.")
    scale = 1.0 / max_radius
    for pose in source_poses:
        pose[:3, 3] *= scale
    return source_poses


def get_appearance_embedding(model: NeuralRenderer, position: float) -> torch.Tensor:
    embeddings = model.app_embedder.embeddings.weight
    if not 0 <= position <= embeddings.shape[0] - 1:
        raise ValueError(
            f"Appearance position must be between 0 and {embeddings.shape[0] - 1}, got {position}."
        )
    lower = int(np.floor(position))
    upper = min(lower + 1, embeddings.shape[0] - 1)
    fraction = position - lower
    return embeddings[lower] * (1.0 - fraction) + embeddings[upper] * fraction


def infer_rays(
    model: NeuralRenderer,
    rays: torch.Tensor,
    appearance_embedding: torch.Tensor,
    chunk_size: int,
    include_samples: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    rgb_chunks = []
    depth_chunks = []
    point_chunks = []
    sigma_chunks = []
    sample_rgb_chunks = []
    device = model.device

    for start in range(0, len(rays), chunk_size):
        ray_chunk = rays[start : start + chunk_size].to(device)
        samples, zvals = model.sampler(ray_chunk[:, :3], ray_chunk[:, 3:6], perturb=False)
        n_rays, n_samples, _ = samples.shape
        encoded_points, keep_mask = model.embedder(samples.reshape(-1, 3))
        encoded_points = encoded_points * keep_mask.unsqueeze(-1).float()
        encoded_dirs = model.sh_encoder(ray_chunk[:, 3:6])
        encoded_dirs = encoded_dirs[:, None, :].expand(-1, n_samples, -1).reshape(-1, encoded_dirs.shape[-1])
        encoded_app = appearance_embedding.to(device).view(1, -1).expand(n_rays, -1)
        encoded_app = encoded_app[:, None, :].expand(-1, n_samples, -1).reshape(-1, encoded_app.shape[-1])
        raw = model.nerf(torch.cat((encoded_points, encoded_dirs, encoded_app), dim=-1))
        raw = raw.reshape(n_rays, n_samples, -1)
        rgb, depth, _, _ = model.integrator(raw, zvals, ray_chunk[:, 3:6])

        if model.enable_hierarchical_sampling and model.n_importance_samples > 0:
            fine_z = model.sampler.sample_pdf(
                zvals,
                model.integrator.last_weights.detach(),
                model.n_importance_samples,
                model.importance_eps,
            )
            zvals = torch.cat((zvals.float(), fine_z), dim=-1).sort(dim=-1).values
            samples = ray_chunk[:, None, :3] + ray_chunk[:, None, 3:6] * zvals[..., None]
            n_samples = zvals.shape[-1]
            encoded_points, keep_mask = model.embedder(samples.reshape(-1, 3))
            encoded_points = encoded_points * keep_mask.unsqueeze(-1).float()
            encoded_dirs = model.sh_encoder(ray_chunk[:, 3:6])
            encoded_dirs = encoded_dirs[:, None, :].expand(-1, n_samples, -1).reshape(-1, encoded_dirs.shape[-1])
            encoded_app = appearance_embedding.to(device).view(1, -1).expand(n_rays, -1)
            encoded_app = encoded_app[:, None, :].expand(-1, n_samples, -1).reshape(-1, encoded_app.shape[-1])
            raw = model.nerf(torch.cat((encoded_points, encoded_dirs, encoded_app), dim=-1))
            raw = raw.reshape(n_rays, n_samples, -1)
            rgb, depth, _, _ = model.integrator(raw, zvals, ray_chunk[:, 3:6])

        rgb_chunks.append(rgb.cpu())
        depth_chunks.append(depth.cpu())

        if include_samples:
            point_chunks.append(samples.cpu())
            sigma_chunks.append(raw[..., -1].cpu())
            sample_rgb_chunks.append(raw[..., : cfg_parameters["channels"]].cpu())

    return (
        torch.cat(rgb_chunks),
        torch.cat(depth_chunks),
        torch.cat(point_chunks) if include_samples else None,
        torch.cat(sigma_chunks) if include_samples else None,
        torch.cat(sample_rgb_chunks) if include_samples else None,
    )


def save_render(rgb: torch.Tensor, depth: torch.Tensor, resolution: int, output: Path, name: str) -> None:
    rgb_image = rgb.numpy().reshape(resolution, resolution, cfg_parameters["channels"])
    depth_image = depth.numpy().reshape(resolution, resolution)
    if cfg_parameters["channels"] == 1:
        rgb_image = rgb_image[..., 0]
    if not cv2.imwrite(str(output / f"{name}.png"), np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)):
        raise OSError(f"Could not write RGB render to {output / f'{name}.png'}.")
    if not cv2.imwrite(
        str(output / f"{name}_depth.png"),
        np.clip(depth_image * 255.0, 0, 255).astype(np.uint8),
    ):
        raise OSError(f"Could not write depth render to {output / f'{name}_depth.png'}.")


def save_3d_visualization(
    points: torch.Tensor,
    sigmas: torch.Tensor,
    colors: torch.Tensor,
    output: Path,
    name: str,
    sigma_threshold: float,
    show: bool,
) -> None:
    points_np = points.reshape(-1, 3).numpy()
    sigma_np = sigmas.reshape(-1).numpy()
    colors_np = colors.reshape(-1, cfg_parameters["channels"]).numpy()
    visible = sigma_np >= sigma_threshold
    if not np.any(visible):
        raise ValueError(
            f"No 3D samples meet --sigma-threshold {sigma_threshold}; lower the threshold and try again."
        )
    selected_points = points_np[visible]
    selected_sigma = sigma_np[visible]
    # Training images are loaded with OpenCV, so the model's channels are BGR.
    selected_colors = np.clip(colors_np[visible][:, ::-1], 0.0, 1.0)
    if selected_colors.shape[-1] == 1:
        selected_colors = np.repeat(selected_colors, 3, axis=-1)
    size_scale = selected_sigma / max(float(selected_sigma.max()), 1e-8)

    np.savez_compressed(
        output / f"{name}_samples.npz",
        points=points_np,
        sigma=sigma_np,
        rgb=np.clip(colors_np[:, ::-1], 0.0, 1.0),
        visible=visible,
    )
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        selected_points[:, 0],
        selected_points[:, 1],
        selected_points[:, 2],
        c=selected_colors,
        s=2.0 + 18.0 * size_scale,
        alpha=0.8,
        linewidths=0,
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(f"Predicted sample colors; point size follows sigma (threshold {sigma_threshold:g})")
    ranges = np.ptp(selected_points, axis=0)
    half_extent = max(float(ranges.max()) / 2.0, 1e-3)
    centers = (selected_points.min(axis=0) + selected_points.max(axis=0)) / 2.0
    axis.set_xlim(centers[0] - half_extent, centers[0] + half_extent)
    axis.set_ylim(centers[1] - half_extent, centers[1] + half_extent)
    axis.set_zlim(centers[2] - half_extent, centers[2] + half_extent)
    figure.tight_layout()
    figure.savefig(output / f"{name}_3d.png", dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    config = INFERENCE_CONFIG
    checkpoint_path = Path(config["checkpoint_path"])
    resolution = int(config["resolution"])
    chunk_size = int(config["chunk_size"])
    pixel_stride = int(config["pixel_stride"])
    mode = config["mode"]
    camera_source = config["camera_source"]
    if resolution < 1 or chunk_size < 1 or pixel_stride < 1:
        raise ValueError("resolution, chunk_size, and pixel_stride must be positive.")
    if config["appearance_steps"] < 1:
        raise ValueError("appearance_steps must be at least 1.")
    if mode not in ("appearance", "camera"):
        raise ValueError("mode must be 'appearance' or 'camera'.")
    if camera_source not in ("trajectory", "sphere"):
        raise ValueError("camera_source must be 'trajectory' or 'sphere'.")
    if mode == "appearance" and camera_source != "trajectory":
        raise ValueError("camera_source is only used when mode is 'camera'.")
    if config["show_3d"] and not config["visualize_3d"]:
        raise ValueError("show_3d requires visualize_3d to be enabled.")

    device = torch.device(config["device"])
    transforms_path = config["transforms_path"]
    if transforms_path is None:
        transforms_dir = Path(dataset_parameters["test_path"])
        candidates = (transforms_dir / "transforms_test.json", transforms_dir / "transforms_train.json")
        transforms_path = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        transforms_path = Path(transforms_path)
    metadata, frames = read_dataset(transforms_path)
    intrinsic = calculate_intrinsic_matrix(
        fov=metadata["camera_angle_x"],
        resolution=(resolution, resolution),
    )
    raygen = RaysGeneratorSynth(
        height=resolution,
        width=resolution,
        num_ch=cfg_parameters["channels"],
        intrinsic_matrix=intrinsic,
    )

    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **sh_parameters,
        **app_parameters,
        **nerf_parameters,
        device=str(device),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inference_root = Path(__file__).resolve().parent.parent / "renders"
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = inference_root / run_timestamp
    output.mkdir(parents=True, exist_ok=True)
    print(f"Saving inference outputs to {output}")
    dataset_poses = dataset_camera_poses(frames, int(config["normalization_stride"]))

    if mode == "appearance":
        camera_index = int(config["camera_index"])
        if not 0 <= camera_index < len(dataset_poses):
            raise ValueError(f"camera_index must be between 0 and {len(dataset_poses) - 1}.")
        end = config["appearance_end"]
        if end is None:
            end = model.app_embedder.embeddings.num_embeddings - 1
        positions = np.linspace(
            config["appearance_start"],
            end,
            int(config["appearance_steps"]),
        )
        poses = [dataset_poses[camera_index]] * len(positions)
    elif camera_source == "trajectory":
        positions = np.full(len(dataset_poses), config["appearance_position"])
        poses = dataset_poses
    else:
        views = int(config["views"])
        positions = np.full(views, config["appearance_position"])
        poses = sphere_poses(config["sphere_radius"], views)

    visualize_3d_index = int(config["visualize_3d_index"])
    if config["visualize_3d"] and not 0 <= visualize_3d_index < len(poses):
        raise ValueError(f"visualize_3d_index must be between 0 and {len(poses) - 1}.")

    with torch.inference_mode():
        for view_index, (pose, appearance_position) in enumerate(zip(poses, positions)):
            rays_image = raygen(pose)
            include_samples = config["visualize_3d"] and view_index == visualize_3d_index
            appearance_embedding = get_appearance_embedding(model, float(appearance_position))
            rgb, depth, _, _, _ = infer_rays(
                model,
                rays_image.reshape(-1, 6),
                appearance_embedding,
                chunk_size,
            )
            name = (
                f"appearance_{view_index:04d}_embedding_{appearance_position:.3f}"
                if mode == "appearance"
                else f"view_{view_index:04d}"
            )
            if include_samples:
                sampled_rays = rays_image[::pixel_stride, ::pixel_stride].reshape(-1, 6)
                _, _, points, sigmas, sample_colors = infer_rays(
                    model,
                    sampled_rays,
                    appearance_embedding,
                    chunk_size,
                    include_samples=True,
                )
                save_3d_visualization(
                    points,
                    sigmas,
                    sample_colors,
                    output,
                    name,
                    config["sigma_threshold"],
                    config["show_3d"],
                )

            save_render(rgb, depth, resolution, output, name)
            print(f"Saved {name} ({view_index + 1}/{len(poses)})")


if __name__ == "__main__":
    main()
