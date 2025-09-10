# Import modules
from math import exp, log, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_parameters import *


# Hashing function
def hash(
        coords: torch.Tensor,
        log2_hashmap_size: int
) -> torch.Tensor:
    '''
    This function takes as input multidimensional coordinates (at most 7D) and compress them in a single
    discrete index for a hash table of size 2^log2_hashmap_size. In essence, it maps a multidimensional vector
    in a single value which ca be employed for querying on a lookup table.

    Each coordinate is multiplied by a prime number in order to mitigate the risk of collisions (same mapping of two
    different input vectors). It is a way to guarantee a uniform distribution of the mapping output. Likewise, the XOR shuffles
    the coordinates bits in a way that simple sums/products could not.

    Finally, 1 << log2_hashmap_size is equivalent to 2^log2_hashmap_size. When subtracting 1, one obtains a bit mask with 1s
    with the same length of log2_hashmap_size. When compared with & xor_results, it keeps the most significant digits.

    Example:
        log2_hashmap_size = 4, then (1 << 4) = 4**2 (x<<n means to shift the bit of x of n-steps to the left).
        Thus 1 -> '00001', 1<<4 = '10000' = 1*(2**4) + 0*(2**3) + ... + 0 = 16. Finally (1<<4)-1 = 15 = '01111'.
        This way we get a boolean mask which cut out the hash result from 0 to 15.

    Parameters:
    ----------
    coords: torch.Tensor[float]
        input coordinates, up to (Nx7) because of the hardcoded primes.
    log2_hasmap_size: int
        it is the dimension of the hash table, defined in the config file.

    Returns:
    -------
    hash_index: torch.Tensor[int]
        it is the hash-mapping of the input coordinates in indexes. Given (Nx7) coordinates
        it is expected to retrieve (Nx1) indexes. 
    '''

    # A set of prime numbers is employed in order to distribute uniformly the indexes
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    # Initialize xor results with zero
    xor_result = torch.zeros_like(coords)[..., 0]
    
    # Fill each entry according to prime distribution
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    # Compute the final hash indexes
    hash_indexes = torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

    return hash_indexes

def total_variation_loss(
        embeddings: torch.Tensor,
        min_resolution: int,
        max_resolution: int,
        level: int,
        log2_hashmap_size: int,
        n_levels: int
) -> torch.Tensor:
    """
    Total Variation Loss (TVL) is a cost which increases when there are sharp changes in the input values.
    In essence, penalizing the TVL promote the smoothness in the optimization process. As a drawback, if the loss gain
    is particulary high, blurry output might happen.

    Parameters:
    ----------
    embeddings: torch.Tensor[float]
        they are the values of the hash table which refer to the input coordinates. They are optimizable parameters
        and are introduced in the TVL in order to smooth variations of close embeddings while they are getting optimized.
    min_resolution: int
        the minimum size of voxel in which the rendering volume dimension is split.
    max_resolution: int
        the maximum size of voxel in which the rendering volume dimension is split.
    level: int
        the current level of resolution (to understand where we are between min_resolution and max_resolution).
    n_levels: int
        the levels of resolutions between min_resolution and max_resolution (set in the configuration file).

    Returns:
    -------
    tvl_val: torch.Tensor[float]
        it is the computed scalar for Total Variation Loss.
    """
    
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
    tvl_val = ((tv_x + tv_y + tv_z) / cube_size)

    return tvl_val

def sigma_sparsity_loss(
        sigmas: torch.Tensor
) -> torch.Tensor:
    """
    Simple implementation of sparsity loss according to Cauchy.

    Parameters:
    ----------
    sigmas: torch.Tensor[float]
        it is the output of the first branch of NeRF model (i.e. volumetric occupancy estimate).
    
    Returns:
    sparsity_loss_val: torch.Tensor[float]
        the value of the computed sparsity loss.
    """
    
    # Using Cauchy Sparsity loss on sigma values
    sparsity_loss_val = torch.log(1.0 + 2*sigmas**2).sum(dim=-1)

    return sparsity_loss_val
