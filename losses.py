# Import modules
from math import exp, log, floor
import torch
from parameters_custom import *


# Hashing function
def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''

    # A set of prime numbers is employed in order to distribute uniformly the indexes
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    # Initialize xor results with zero
    xor_result = torch.zeros_like(coords)[..., 0]
    
    # Fill each entry according to prime distribution
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


# Discrete version of the intergral of the L1 norm of the embeddings: it promotes smoothness
def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
    
    # Get resolution
    b = exp((log(max_resolution)-log(min_resolution))/(n_levels-1))
    resolution = torch.tensor(floor(min_resolution * b**level)).to(cfg_parameters["device"])

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 100
    cube_size = torch.floor(torch.clip(resolution/10.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution-cube_size, (3,)).to(cfg_parameters["device"])
    idx = min_vertex + torch.stack([torch.arange(cube_size+1).to(cfg_parameters["device"]) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)

    # Compute loss
    tv_x = torch.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()

    return ((tv_x + tv_y + tv_z)/cube_size)


# Sparsity loss for volumetric desnity estimation (sigmas)
def sigma_sparsity_loss(sigmas):
    
    # Using Cauchy Sparsity loss on sigma values
    return torch.log(1.0 + 2*sigmas**2).sum(dim=-1)
