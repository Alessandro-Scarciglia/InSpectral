"""
Camera-pose utilities for joint NeRF and pose refinement.

The training table stores rays, not poses, so the poses used to build it are read from
the sidecar file written by data/generate_postprocessed_dataset.py
(<dataset>_poses.npz, same name as the .npy plus "_poses").

Conventions follow training/training_pipeline.py, where a pose residual (dt, dR) is
applied to a ray as:  origin' = origin + dt,  direction' = dR @ direction.
For a camera-to-world pose (R, c) this is the corrected pose (dR @ R, c + dt).
"""

import os
import re

import numpy as np
import torch


def dataset_scale_factor(data_path: str):
    """
    Scene scale used to build a ray table: 1 / (largest camera distance), read from the
    "max<distance>" tag in the dataset filename, exactly as training validation does.

    The builder computes that distance from the (possibly noisy) training poses, so it
    can differ by a few percent from the clean poses. Evaluation and inference must use
    this value, not one recomputed from transforms_*.json. Returns None if the filename
    has no such tag.
    """
    match = re.search(r"max(-?\d+\.\d+)", os.path.basename(data_path))
    return None if match is None else 1.0 / (float(match.group(1)) + 1e-10)


def sidecar_path(data_path: str) -> str:
    """Path of the pose sidecar that belongs to a ray-table .npy file."""
    root, _ = os.path.splitext(data_path)
    return f"{root}_poses.npz"


def load_pose_sidecar(data_path: str, device: torch.device):
    """
    Load the input and ground-truth camera poses saved next to a ray table.

    Returns (input_poses, ground_truth_poses): [N, 4, 4] float32 tensors on `device`.
    Either is None when unavailable (older datasets have no sidecar at all).
    """
    path = sidecar_path(data_path)

    if not os.path.exists(path):
        return None, None

    sidecar = np.load(path)
    input_poses = torch.tensor(sidecar["input_poses"], dtype=torch.float32, device=device)

    ground_truth = sidecar["ground_truth_poses"]
    ground_truth_poses = (
        torch.tensor(ground_truth, dtype=torch.float32, device=device)
        if len(ground_truth) == len(input_poses)
        else None
    )

    return input_poses, ground_truth_poses


def corrected_poses(res_model, input_poses: torch.Tensor) -> torch.Tensor:
    """Apply the learned residual of every image to its input pose. Returns [N, 4, 4]."""
    indices = torch.arange(len(input_poses), device=input_poses.device)
    delta_t, delta_r = res_model(indices)

    poses = input_poses.clone()
    poses[:, :3, :3] = delta_r @ input_poses[:, :3, :3]
    poses[:, :3, 3] = input_poses[:, :3, 3] + delta_t

    return poses


def _savitzky_golay_center_weights(window: int, device, dtype) -> torch.Tensor:
    """Weights that give the centre value of a local quadratic least-squares fit."""
    t = torch.arange(window, dtype=torch.float64) - window // 2
    design = torch.stack([torch.ones_like(t), t, t * t], dim=1)
    weights = (design @ torch.linalg.pinv(design))[window // 2]
    return weights.to(device=device, dtype=dtype)


def trajectory_smoothness_loss(poses: torch.Tensor, window: int = 5) -> torch.Tensor:
    """
    High-frequency energy of a camera trajectory sampled at consecutive frames.

    Every camera position and orientation is compared with a local quadratic
    (Savitzky-Golay) fit through its `window` neighbours. A quadratic follows the slow
    curvature of an orbit almost exactly, so the residual is mostly the frame-to-frame
    (white) pose noise that this prior is meant to remove. Simpler priors such as
    "second difference = 0" or a moving average also penalise the orbit's own curvature
    and pull the poses away from the truth.

    poses:   corrected poses [N, 4, 4], ordered along the trajectory
    window:  odd number of frames in each local fit (>= 5)

    Works when consecutive frames are a few degrees apart (e.g. 90 frames per orbit);
    on very coarse orbits (15 degrees or more between frames) the quadratic fit no
    longer follows the trajectory and the prior should stay off.
    """
    window = max(5, int(window) | 1)

    if len(poses) < window:
        return poses.new_zeros(())

    # Each pose as a 12-vector: position (3) + rotation matrix entries (9).
    features = torch.cat([poses[:, :3, 3], poses[:, :3, :3].reshape(len(poses), 9)], dim=1)

    weights = _savitzky_golay_center_weights(window, features.device, features.dtype)
    local_fit = (features.unfold(0, window, 1) * weights).sum(dim=-1)
    centre = features[window // 2: len(features) - window // 2]

    return (centre - local_fit).square().sum(dim=-1).mean()


def _umeyama_similarity(source: torch.Tensor, target: torch.Tensor):
    """Similarity transform (s, R, t) minimising ||s R source + t - target||. Inputs [N, 3]."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t

    covariance = xt.T @ xs / len(source)
    u, d, vt = torch.linalg.svd(covariance)

    sign = torch.eye(3, device=source.device, dtype=source.dtype)
    if torch.det(u) * torch.det(vt) < 0:
        sign[2, 2] = -1.0

    rotation = u @ sign @ vt
    variance = xs.square().sum() / len(source)
    scale = (d * sign.diagonal()).sum() / variance.clamp_min(1e-12)
    translation = mu_t - scale * rotation @ mu_s

    return scale, rotation, translation


def pose_errors(poses: torch.Tensor, ground_truth: torch.Tensor, align: bool = True) -> dict:
    """
    Mean translation error (normalised scene units) and rotation error (degrees).

    Joint pose/NeRF optimisation can only recover poses up to a global similarity
    transform, so by default the estimated camera centres are aligned to the ground
    truth (Umeyama) before measuring, as in BARF.
    """
    poses = poses.double()
    ground_truth = ground_truth.double()

    centers = poses[:, :3, 3]
    rotations = poses[:, :3, :3]

    if align and len(poses) >= 3:
        scale, rotation, translation = _umeyama_similarity(centers, ground_truth[:, :3, 3])
        centers = scale * centers @ rotation.T + translation
        rotations = rotation @ rotations

    translation_error = torch.linalg.norm(centers - ground_truth[:, :3, 3], dim=-1)

    relative = rotations.transpose(-1, -2) @ ground_truth[:, :3, :3]
    cos_angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_error = torch.rad2deg(torch.arccos(cos_angle))

    return {
        "translation": translation_error.mean().item(),
        "rotation_deg": rotation_error.mean().item(),
    }
