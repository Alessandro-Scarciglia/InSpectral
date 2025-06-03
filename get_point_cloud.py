# Import modules
import torch
import open3d as o3d
import json
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist


# Import custom modules
from generate_nerf_datasets import calculate_intrinsic_matrix
from rays_generator_synth import RaysGeneratorSynth
from rendering import NeuralRenderer
from parameters_synth import *


# Generate and store ground truth point cloud from CAD model
def pc_from_cad(src_path: str,
                dst_path: str,
                num_points: int = 5000):

    # Load .stl file
    mesh = o3d.io.read_triangle_mesh(src_path)

    # Sample points from the surface of the mesh
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # Store the ground truth (it does not need to be computed dinamically)
    o3d.io.write_point_cloud(dst_path, pcd)

    return pcd


# Compute point cloud from NeRF model
def pc_from_model(model_path: str,
                  calib_path: str,
                  wps_path: str,
                  resolution: tuple = (256, 256),
                  sigma_th: float = 3.):
    
    # Load calibration parameters
    with open(calib_path, "r") as fopen:
        fov = json.load(fopen)["camera_angle_x"]
    
    # Compute camera calibration
    K = calculate_intrinsic_matrix(fov=fov, resolution=resolution)

    # Define the rays generator object
    raygen = RaysGeneratorSynth(**rays_parameters, K=K)

    # Define the model
    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **posenc_parameters,
        **sh_parameters,
        **nerf_parameters,
        device=cfg_parameters["device"]
    ).eval()

    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load waypoints and rescale distances
    wps = np.load(wps_path)

    # Disable gradient ret
    with torch.no_grad():

        # Compute inference on all waypoints and collect samples
        cloud_collector = torch.tensor([], dtype=torch.float32)
        for i, wp in tqdm(enumerate(wps)):

            # Transform in torch tensor
            wp = torch.from_numpy(wp.astype(np.float32)).reshape(4, 4)

            # Generate rays from waypoint
            rays = raygen(wp)
            rays = rays.reshape(-1, 6)

            # Inference (batching)
            samples = torch.tensor([], dtype=torch.float32, device=cfg_parameters["device"])
            sigmas = torch.tensor([], dtype=torch.float32, device=cfg_parameters["device"])
            for i in range(4):

                # Predict color-sigmas
                _, _, _, batch_samples, batch_raw = model(rays[i*(128**2): (i+1)*(128**2)])
                time.sleep(0.1)

                # Collect samples of reference (N x 3)
                batch_samples = batch_samples.reshape(-1, 3)
                samples = torch.cat([samples, batch_samples], dim=0)

                # Select sigmas only and reshape (N x 1)
                batch_sigmas = batch_raw[..., -1].reshape(-1)
                sigmas = torch.cat([sigmas, batch_sigmas], dim=0)
                
            # Filter out samples with zero sigma
            samples = samples[sigmas > sigma_th].cpu()
            sigmas = sigmas[sigmas > sigma_th].cpu()
            
            # Attach to cloud
            sample_sigma = torch.cat([samples, sigmas.unsqueeze(-1)], dim=-1)
            cloud_collector = torch.cat([cloud_collector, sample_sigma * 30.], dim=0)

        # Transform the cloud collector in numpy array to handle the plotting
        cloud_collector = cloud_collector.numpy()

        # Estrai le coordinate e le densità
        points = cloud_collector[:, :3]

        # Crea un oggetto PointCloud di Open3D
        pcd = o3d.geometry.PointCloud()

        # Aggiungi le coordinate al point cloud
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd
    

def voxelize_pointcloud(pointcloud, voxel_size, bounds):
    """
    Convert pointcloud to a binary voxel grid representation.
    
    :param pointcloud: numpy array of shape (N, 3), where N is the number of points.
    :param voxel_size: the size of each voxel (e.g., 1x1x1 or 0.5x0.5x0.5).
    :param bounds: the (min, max) bounds of the space, e.g., [(x_min, y_min, z_min), (x_max, y_max, z_max)].
    :return: voxel grid (binary), shape (x_voxels, y_voxels, z_voxels).
    """
    # Create a 3D grid of voxel indices
    min_bound, max_bound = np.array(bounds)
    voxel_grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    
    # Normalize pointcloud coordinates to the voxel grid
    normalized_points = (pointcloud - min_bound) / voxel_size
    voxel_indices = np.floor(normalized_points).astype(int)
    
    # Ensure the points are within bounds
    voxel_indices = np.clip(voxel_indices, 0, voxel_grid_shape - 1)
    
    # Create a binary voxel grid
    voxel_grid = np.zeros(voxel_grid_shape, dtype=bool)
    voxel_grid[tuple(voxel_indices.T)] = True
    
    return voxel_grid


def iou(voxel_grid1, voxel_grid2):
    """
    Calculate the Intersection over Union (IoU) between two voxel grids.
    
    :param voxel_grid1: First voxel grid (binary).
    :param voxel_grid2: Second voxel grid (binary).
    :return: IoU score.
    """
    intersection = np.logical_and(voxel_grid1, voxel_grid2).sum()
    
    return intersection


def chamfer_distance(pc1, pc2):
    """
    Calcola la Chamfer Distance tra due point cloud pc1 e pc2
    pc1: (N, 3) torch.Tensor
    pc2: (M, 3) torch.Tensor
    """
    # Distanze punto-a-punto tra tutte le coppie
    dist_matrix = torch.cdist(pc1, pc2, p=2)

    # Somma delle distanze minime in entrambe le direzioni
    dist1 = torch.mean(torch.min(dist_matrix, dim=1)[0])  # da pc1 a pc2
    dist2 = torch.mean(torch.min(dist_matrix, dim=0)[0])  # da pc2 a pc1

    return dist1 + dist2


# Main
if __name__ == "__main__":

    # Create and load pc from stl
    pcd_gt = pc_from_cad(src_path="/home/visione/Projects/BlenderScenarios/Sat/cloudsat1.stl",
                         dst_path="./gt.ply",
                         num_points=10000)
    
    for i in range(10):

        # Compute pc from model
        pcd_est = pc_from_model(model_path="training_logs/folder_2025-04-23_15-15-39/epoch_29/chkpt.pt",
                                calib_path="/home/visione/Projects/BlenderScenarios/Asteroid/Dataset/Orbit_V_256/IR/transforms.json",
                                wps_path="data/test_wps.npy",
                                resolution=(256, 256),
                                sigma_th=i*0.2)
        
        # Get np.arrays
        pcd_est_np = np.array(pcd_est.points)
        pcd_gt_np = np.array(pcd_gt.points)
        
        # Compute voxel grids
        vg_est = voxelize_pointcloud(pointcloud=pcd_est_np, voxel_size=0.5, bounds=[[-5, -5, -5], [5, 5, 5]])
        vg_gt = voxelize_pointcloud(pointcloud=pcd_gt_np, voxel_size=0.5, bounds=[[-5, -5, -5], [5, 5, 5]])

        # Compute score
        int_score = iou(voxel_grid1=vg_est, voxel_grid2=vg_gt)
        int_coverage = int_score / (5 / 0.5)**3
        
        # Get torch.tensors
        pcd_est_t = torch.tensor(np.asarray(pcd_est.points), dtype=torch.float32)
        pcd_gt_t = torch.tensor(np.asarray(pcd_gt.points), dtype=torch.float32)

        # Compute Chamfer Distance
        chmf_dist = chamfer_distance(pc1=pcd_est_t, pc2=pcd_gt_t)

        # Show results
        print(f"IoU score: {int_coverage:.5f} | Chamfer Distance: {chmf_dist.item():.5f} | Score: {int_coverage + 1/(chmf_dist.item()+1e-5)}")


        # Apply color to pcd1 (shades of blue)
        pcd_gt.paint_uniform_color([0, 0, 1])  

        # Apply color to pcd2 (shades of red)
        pcd_est.paint_uniform_color([1, 0, 0])  

        # Visualize both point clouds together
        o3d.visualization.draw_geometries([pcd_est, pcd_gt], window_name="Point Clouds in Blue and Red")