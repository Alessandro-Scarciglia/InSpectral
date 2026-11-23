"""
Export the geometry learned by a trained checkpoint as a coloured point cloud and,
optionally, a triangle mesh (PLY files that open in MeshLab, CloudCompare or Blender).

The density field is sampled on a regular grid inside the hash-grid bounding box.
Voxels whose density exceeds `density_threshold` become points; the mesh is the
iso-surface of the density at the same level (marching cubes, needs scikit-image).
Colours are queried from the colour branch with a fixed appearance embedding and a
viewing direction pointing from outside the object towards its centre.

Edit EXPORT_CONFIG below, then run from the repository root:

    python -m tools.export_geometry
"""

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from configs.config_parameters import (
    app_parameters,
    cfg_parameters,
    dataset_parameters,
    hash_parameters,
    nerf_parameters,
    occupancy_parameters,
    sampler_parameters,
    sh_parameters,
)
from model.rendering import NeuralRenderer


EXPORT_CONFIG = {
    "checkpoint_path": "results/folder_2026-11-05_18-24-13/epoch_4/chkpt_nerf.pt",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    # Samples per side of the density grid inside the bounding box (memory grows cubically).
    "grid_resolution": 256,
    # Raw density above which space counts as occupied (also the mesh iso-level).
    "density_threshold": 5.0,
    # Appearance used for the colours: an embedding index, or None for the mean embedding.
    "appearance_index": None,
    "export_point_cloud": True,
    "export_mesh": True,
    # "normalized": scene units used in training. "dataset": units of the transforms.json
    # (e.g. metres), using the max camera distance encoded in the dataset filename.
    "output_units": "normalized",
    "chunk_size": 65536,
}


def dataset_unit_scale(data_path: str) -> float:
    """Factor from normalised scene units back to the units of transforms.json."""
    match = re.search(r"max(-?\d+\.\d+)", os.path.basename(data_path))
    if match is None:
        raise ValueError(
            "Cannot read the max camera distance from the dataset filename; "
            "use output_units = 'normalized'."
        )
    return float(match.group(1))


@torch.no_grad()
def query_field(model: NeuralRenderer, points: torch.Tensor, appearance: torch.Tensor,
                chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw density [N] and colour [N, C] at the given points."""
    densities, colours = [], []
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size].to(model.device)
        encoded, keep = model.embedder(chunk)
        encoded = encoded * keep.unsqueeze(-1).float()
        view_dirs = -chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        encoded_dirs = model.sh_encoder(view_dirs)
        encoded_app = appearance.view(1, -1).expand(len(chunk), -1)
        raw = model.nerf(torch.cat((encoded, encoded_dirs, encoded_app), dim=-1))
        densities.append(raw[:, -1].clamp_min(0).cpu())
        colours.append(raw[:, :-1].cpu())
    return torch.cat(densities), torch.cat(colours)


def colours_to_rgb_and_ir(colours: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Turn model channels (BGR, IR or BGR+IR) into uint8 RGB plus an optional IR value."""
    channels = colours.shape[-1]
    values = np.clip(colours, 0.0, 1.0)
    if channels == 1:
        rgb = np.repeat(values, 3, axis=-1)
        ir = values[:, 0]
    else:
        rgb = values[:, 2::-1]
        ir = values[:, 3] if channels >= 4 else None
    return (rgb * 255).round().astype(np.uint8), ir


def write_ply(path: Path, vertices: np.ndarray, rgb: np.ndarray, ir: np.ndarray | None = None,
              faces: np.ndarray | None = None, density: np.ndarray | None = None) -> None:
    """Binary little-endian PLY with per-vertex colour (and optional IR / density / faces)."""
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if ir is not None:
        fields.append(("ir", "<f4"))
    if density is not None:
        fields.append(("density", "<f4"))

    data = np.empty(len(vertices), dtype=fields)
    data["x"], data["y"], data["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if ir is not None:
        data["ir"] = ir
    if density is not None:
        data["density"] = density

    ply_types = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
    header += [f"property {ply_types[kind]} {name}" for name, kind in fields]
    if faces is not None:
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")

    with open(path, "wb") as target:
        target.write(("\n".join(header) + "\n").encode("ascii"))
        target.write(data.tobytes())
        if faces is not None:
            face_data = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
            face_data["n"] = 3
            face_data["v"] = faces
            target.write(face_data.tobytes())


def main() -> None:
    config = EXPORT_CONFIG
    device = torch.device(config["device"])
    resolution = int(config["grid_resolution"])
    threshold = float(config["density_threshold"])

    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **sh_parameters,
        **app_parameters,
        **nerf_parameters,
        **occupancy_parameters,
        device=str(device),
    ).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    embeddings = model.app_embedder.embeddings.weight.detach()
    index = config["appearance_index"]
    appearance = embeddings.mean(dim=0) if index is None else embeddings[int(index)]

    # Regular grid of sample positions inside the bounding box
    bbox_min, bbox_max = (bound.float() for bound in hash_parameters["bbox"])
    axes = [torch.linspace(float(bbox_min[i]), float(bbox_max[i]), resolution) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    print(f"[INFO] Sampling the density on a {resolution}^3 grid...")
    density, colours = query_field(model, grid, appearance.to(device), int(config["chunk_size"]))

    unit_scale = 1.0
    if config["output_units"] == "dataset":
        unit_scale = dataset_unit_scale(dataset_parameters["data_path"])
    elif config["output_units"] != "normalized":
        raise ValueError("output_units must be 'normalized' or 'dataset'.")

    output = Path("exports") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output.mkdir(parents=True, exist_ok=True)

    occupied = density > threshold
    print(f"[INFO] {int(occupied.sum())} of {len(density)} grid samples above density {threshold:g}.")

    if config["export_point_cloud"]:
        if not occupied.any():
            print("[WARNING] No occupied samples: lower density_threshold to export a point cloud.")
        else:
            rgb, ir = colours_to_rgb_and_ir(colours[occupied].numpy())
            write_ply(output / "point_cloud.ply", grid[occupied].numpy() * unit_scale, rgb, ir,
                      density=density[occupied].numpy())
            print(f"[INFO] Point cloud saved to {output / 'point_cloud.ply'}")

    if config["export_mesh"]:
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            print("[WARNING] scikit-image is not installed: skipping the mesh (pip install scikit-image).")
            marching_cubes = None

        volume = density.reshape(resolution, resolution, resolution).numpy()
        if marching_cubes is not None and volume.min() < threshold < volume.max():
            spacing = tuple(float(bbox_max[i] - bbox_min[i]) / (resolution - 1) for i in range(3))
            vertices, faces, _, _ = marching_cubes(volume, level=threshold, spacing=spacing)
            vertices = vertices + bbox_min.numpy()
            _, vertex_colours = query_field(model, torch.tensor(vertices, dtype=torch.float32),
                                            appearance.to(device), int(config["chunk_size"]))
            rgb, ir = colours_to_rgb_and_ir(vertex_colours.numpy())
            write_ply(output / "mesh.ply", vertices * unit_scale, rgb, ir, faces=faces.astype(np.int32))
            print(f"[INFO] Mesh with {len(vertices)} vertices and {len(faces)} faces saved to {output / 'mesh.ply'}")
        elif marching_cubes is not None:
            print("[WARNING] The density never crosses density_threshold: no surface to mesh.")


if __name__ == "__main__":
    main()
